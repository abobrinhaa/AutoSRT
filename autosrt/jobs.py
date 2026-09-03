"""Fila de trabalhos com um único operário.

A fila é serial de propósito, não por simplicidade. A GPU não comporta duas
coisas ao mesmo tempo: numa placa de 5 GB, o Whisper e um modelo de
linguagem local já não cabem juntos, e duas transcrições simultâneas
disputam a mesma memória e ficam mais lentas que se fossem em sequência.
"""

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PENDENTE = "pendente"
RODANDO = "rodando"
CONCLUIDO = "concluido"
ERRO = "erro"
CANCELADO = "cancelado"

FINAIS = {CONCLUIDO, ERRO, CANCELADO}

# Piso para arriscar uma previsão de tempo: antes disso a extrapolação é ruído.
MIN_PROGRESSO_PARA_ESTIMAR = 3
MIN_SEGUNDOS_PARA_ESTIMAR = 20


@dataclass
class Job:
    """Um trabalho na fila."""

    id: str
    nome: str
    entrada: str
    estado: str = PENDENTE
    etapa: str = "Na fila..."
    progresso: int = 0
    resultado: str = None
    erro: str = None
    detalhes: dict = field(default_factory=dict)
    criado_em: float = field(default_factory=time.time)
    iniciado_em: float = None
    cancelar: threading.Event = field(default_factory=threading.Event)

    def segundos_restantes(self, agora=None):
        """Quanto ainda falta, medindo o ritmo do que já andou.

        Devolve ``None`` enquanto não há o que medir. Nos primeiros segundos a
        conta pula de uma hora para dois minutos a cada atualização, e número
        que dança desse jeito é pior que número nenhum -- por isso só responde
        depois que o trabalho andou um pouco.
        """
        if self.estado != RODANDO or not self.iniciado_em:
            return None
        if self.progresso < MIN_PROGRESSO_PARA_ESTIMAR:
            return None

        decorrido = (agora or time.time()) - self.iniciado_em
        if decorrido < MIN_SEGUNDOS_PARA_ESTIMAR:
            return None

        total = decorrido * 100 / self.progresso
        return max(0, round(total - decorrido))

    def para_json(self) -> dict:
        return {
            "id": self.id,
            "nome": self.nome,
            "estado": self.estado,
            "etapa": self.etapa,
            "progresso": self.progresso,
            "erro": self.erro,
            "detalhes": self.detalhes,
            "pronto": self.estado in FINAIS,
            "baixavel": self.estado == CONCLUIDO and bool(self.resultado),
            "restante_seg": self.segundos_restantes(),
        }


class JobQueue:
    """Fila serial. Aceita trabalhos de qualquer thread; executa um por vez."""

    def __init__(self, worker):
        """
        Args:
            worker: função ``worker(job)`` que executa o trabalho. Deve
                preencher ``job.resultado`` e pode atualizar ``job.etapa`` e
                ``job.progresso``.
        """
        self._worker = worker
        self._lock = threading.Lock()
        self._jobs = {}
        self._ordem = []
        self._fila = []
        self._thread = None

    def enviar(self, nome, entrada, **detalhes) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], nome=nome, entrada=entrada,
                  detalhes=detalhes)
        with self._lock:
            self._jobs[job.id] = job
            self._ordem.append(job.id)
            self._fila.append(job)
            self._garantir_operario()
        return job

    def _garantir_operario(self):
        """Sobe o operário se não houver um vivo. Chamado com o lock preso."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._rodar, daemon=True)
            self._thread.start()

    def _rodar(self):
        while True:
            with self._lock:
                if not self._fila:
                    self._thread = None
                    return
                job = self._fila.pop(0)

            if job.cancelar.is_set():
                job.estado = CANCELADO
                job.etapa = "Cancelado antes de começar."
                continue

            job.estado = RODANDO
            job.etapa = "Começando..."
            # A previsão conta a partir daqui, não de quando entrou na fila:
            # tempo parado atrás de outro trabalho não diz nada sobre este.
            job.iniciado_em = time.time()
            try:
                self._worker(job)
            except Exception as exc:
                if job.cancelar.is_set():
                    job.estado = CANCELADO
                    job.etapa = "Cancelado."
                else:
                    job.estado = ERRO
                    job.erro = str(exc) or exc.__class__.__name__
                    job.etapa = "Falhou."
                    logger.exception("trabalho %s falhou", job.id)
            else:
                if job.cancelar.is_set():
                    job.estado = CANCELADO
                    job.etapa = "Cancelado."
                else:
                    job.estado = CONCLUIDO
                    job.etapa = "Pronto!"
                    job.progresso = 100

    def buscar(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def listar(self, limite=20) -> list:
        """Os trabalhos que valem a tela agora, na ordem em que importam.

        Primeiro o que roda, depois a fila na ordem em que será atendida,
        depois os terminados do mais novo para o mais velho.

        Cortar pela ordem de chegada, como antes, escondia justamente o
        trabalho em andamento assim que a fila passava do limite: com 50
        filmes enviados de uma vez, a página mostrava 20 cartões "na fila"
        parados e nenhuma barra andando -- o único que andava era o mais
        antigo, o primeiro a ser cortado.

        As vagas são disputadas: metade fica reservada aos terminados
        quando a fila é maior que a lista, senão os botões de baixar
        sumiriam atrás de uma parede de "na fila".
        """
        with self._lock:
            todos = [self._jobs[i] for i in self._ordem]

        ativos = ([j for j in todos if j.estado == RODANDO]
                  + [j for j in todos if j.estado == PENDENTE])
        recentes = [j for j in reversed(todos) if j.estado in FINAIS]

        vagas = min(len(recentes), max(limite - len(ativos), limite // 2))
        return ativos[:limite - vagas] + recentes[:vagas]

    def remover(self, job_id) -> bool:
        """Tira um trabalho já terminado da lista. Devolve se removeu.

        Trabalho que ainda roda não sai: o operário continuaria mexendo em
        algo que a página não mostra mais.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.estado not in FINAIS:
                return False
            del self._jobs[job_id]
            self._ordem.remove(job_id)
            return True

    def limpar_terminados(self) -> int:
        """Remove todos os trabalhos terminados. Devolve quantos saíram."""
        with self._lock:
            terminados = [i for i in self._ordem
                          if self._jobs[i].estado in FINAIS]
            for job_id in terminados:
                del self._jobs[job_id]
                self._ordem.remove(job_id)
            return len(terminados)

    def cancelar(self, job_id) -> bool:
        job = self.buscar(job_id)
        if not job or job.estado in FINAIS:
            return False
        job.cancelar.set()
        return True

    @property
    def ocupado(self) -> bool:
        with self._lock:
            return bool(self._fila) or any(
                j.estado == RODANDO for j in self._jobs.values())

    def em_uso(self, caminho) -> bool:
        """Diz se algum trabalho não terminado tem esse arquivo como entrada."""
        alvo = os.path.abspath(caminho)
        with self._lock:
            return any(os.path.abspath(j.entrada) == alvo
                       for j in self._jobs.values()
                       if j.estado not in FINAIS)
