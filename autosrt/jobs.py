"""Fila de trabalhos com um ou mais operários.

Um operário só é o padrão, não por simplicidade: numa GPU com pouca VRAM
(a placa de 5 GB que motivou esse padrão), o Whisper e um modelo de
linguagem local já não cabem juntos, e duas transcrições simultâneas
disputam a mesma memória e ficam mais lentas que se fossem em sequência.

Em uma GPU com VRAM de sobra (ex.: RTX 3060 12 GB) rodando modelos que
cabem em memória duas vezes, ``max_workers=2`` processa dois arquivos ao
mesmo tempo em vez de deixar a placa ociosa entre um trabalho e outro --
mas continua sendo escolha de quem configura o servidor, não algo que este
módulo decide sozinho pelo hardware.
"""

import logging
import os
import queue
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


#: Ocioso por mais que isso, o operário encerra em vez de ficar parado para
#: sempre -- o próximo ``enviar()`` sobe outro. Só importa para o teto de
#: threads vivas; não afeta quanto tempo um trabalho já em andamento leva.
OPERARIO_OCIOSO_SEG = 30


class JobQueue:
    """Fila com um teto de operários simultâneos (padrão: 1, serial).

    Aceita trabalhos de qualquer thread. Até ``max_workers`` deles rodam ao
    mesmo tempo, cada um em sua própria thread, consumindo uma fila
    (``queue.Queue``) compartilhada -- é o padrão produtor/consumidor de
    sempre, escolhido justamente para não ter que adivinhar "tem operário
    ocioso ou não" na hora de decidir se sobe mais um.
    """

    def __init__(self, worker, max_workers=1):
        """
        Args:
            worker: função ``worker(job)`` que executa o trabalho. Deve
                preencher ``job.resultado`` e pode atualizar ``job.etapa`` e
                ``job.progresso``.
            max_workers: quantos trabalhos rodam ao mesmo tempo. Valores
                menores que 1 viram 1 -- zero operário deixaria a fila
                parada para sempre.
        """
        self._worker = worker
        self._max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._jobs = {}
        self._ordem = []
        self._fila = queue.Queue()
        self._threads = []

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @max_workers.setter
    def max_workers(self, valor):
        """Ajusta o teto em tempo de execução, sem derrubar quem já roda.

        Vale a partir do próximo ``enviar()``: baixar o teto não interrompe
        operário em andamento (ele só não é reposto quando termina); subir
        libera vaga para os próximos trabalhos enfileirados.
        """
        self._max_workers = max(1, int(valor))

    def enviar(self, nome, entrada, **detalhes) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], nome=nome, entrada=entrada,
                  detalhes=detalhes)
        with self._lock:
            self._jobs[job.id] = job
            self._ordem.append(job.id)
        self._fila.put(job)
        self._garantir_operarios()
        return job

    def _garantir_operarios(self):
        """Sobe operários até o teto configurado. Threads ociosas demais já
        se encerraram sozinhas (ver ``_rodar``), então subir até o teto aqui
        nunca duplica um operário que já está de pé."""
        with self._lock:
            self._threads = [t for t in self._threads if t.is_alive()]
            while len(self._threads) < self._max_workers:
                t = threading.Thread(target=self._rodar, daemon=True)
                self._threads.append(t)
                t.start()

    def _rodar(self):
        while True:
            try:
                job = self._fila.get(timeout=OPERARIO_OCIOSO_SEG)
            except queue.Empty:
                return

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
        if not self._fila.empty():
            return True
        with self._lock:
            return any(j.estado == RODANDO for j in self._jobs.values())

    def em_uso(self, caminho) -> bool:
        """Diz se algum trabalho não terminado tem esse arquivo como entrada."""
        alvo = os.path.abspath(caminho)
        with self._lock:
            return any(os.path.abspath(j.entrada) == alvo
                       for j in self._jobs.values()
                       if j.estado not in FINAIS)
