"""Fila de trabalhos com um único operário.

A fila é serial de propósito, não por simplicidade. A GPU não comporta duas
coisas ao mesmo tempo: numa placa de 5 GB, o Whisper e um modelo de
linguagem local já não cabem juntos, e duas transcrições simultâneas
disputam a mesma memória e ficam mais lentas que se fossem em sequência.
"""

import logging
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
    cancelar: threading.Event = field(default_factory=threading.Event)

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
        with self._lock:
            ids = self._ordem[-limite:]
            return [self._jobs[i] for i in reversed(ids)]

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
