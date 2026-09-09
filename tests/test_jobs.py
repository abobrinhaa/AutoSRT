"""Concorrência da fila (autosrt.jobs.JobQueue).

O padrão continua sendo um operário só (ver test_web.TestFila) -- a mudança
aqui é permitir mais, para quem tem VRAM sobrando (ex.: RTX 3060 12 GB) e
quer processar mais de um arquivo ao mesmo tempo em vez de deixar a placa
ociosa entre um trabalho e outro.
"""

import threading
import time
import unittest

from autosrt import jobs


def _esperar_todos(trabalhos, limite=5):
    fim = time.time() + limite
    while time.time() < fim and any(
            j.estado not in jobs.FINAIS for j in trabalhos):
        time.sleep(0.02)
    return trabalhos


class TestConcorrenciaDaFila(unittest.TestCase):
    def test_dois_workers_processam_ao_mesmo_tempo(self):
        simultaneos = []
        pico = []
        lock = threading.Lock()

        def worker(job):
            with lock:
                simultaneos.append(1)
                pico.append(len(simultaneos))
            time.sleep(0.1)
            with lock:
                simultaneos.pop()

        fila = jobs.JobQueue(worker, max_workers=2)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(4)]
        _esperar_todos(enviados)

        self.assertTrue(all(j.estado == jobs.CONCLUIDO for j in enviados))
        self.assertEqual(max(pico), 2)

    def test_nao_passa_do_teto_configurado(self):
        simultaneos = []
        pico = []
        lock = threading.Lock()

        def worker(job):
            with lock:
                simultaneos.append(1)
                pico.append(len(simultaneos))
            time.sleep(0.08)
            with lock:
                simultaneos.pop()

        fila = jobs.JobQueue(worker, max_workers=2)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(6)]
        _esperar_todos(enviados)

        self.assertTrue(all(j.estado == jobs.CONCLUIDO for j in enviados))
        self.assertLessEqual(max(pico), 2)

    def test_padrao_sem_max_workers_continua_serial(self):
        simultaneos = []
        pico = []
        lock = threading.Lock()

        def worker(job):
            with lock:
                simultaneos.append(1)
                pico.append(len(simultaneos))
            time.sleep(0.05)
            with lock:
                simultaneos.pop()

        fila = jobs.JobQueue(worker)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(3)]
        _esperar_todos(enviados)

        self.assertEqual(max(pico), 1)

    def test_max_workers_minimo_e_um(self):
        # Zero ou negativo não pode significar "nenhum operário" -- a fila
        # trabalharia para sempre com trabalhos parados nela.
        fila = jobs.JobQueue(lambda job: None, max_workers=0)
        job = fila.enviar("j", "/tmp/j")
        _esperar_todos([job])
        self.assertEqual(job.estado, jobs.CONCLUIDO)

    def test_cancelar_funciona_com_varios_workers(self):
        liberar = threading.Event()

        def worker(job):
            liberar.wait(timeout=5)

        fila = jobs.JobQueue(worker, max_workers=2)
        a = fila.enviar("a", "/tmp/a")
        b = fila.enviar("b", "/tmp/b")
        # Os dois entram em RODANDO quase juntos, com 2 operários livres.
        fim = time.time() + 5
        while time.time() < fim and (a.estado != jobs.RODANDO or b.estado != jobs.RODANDO):
            time.sleep(0.02)

        fila.cancelar(b.id)
        liberar.set()
        _esperar_todos([a, b])

        self.assertEqual(a.estado, jobs.CONCLUIDO)
        # Cancelado no meio: o worker não sabe cooperar sozinho aqui (é um
        # cancelamento de teste), mas o estado final não pode ficar "rodando".
        self.assertIn(b.estado, (jobs.CONCLUIDO, jobs.CANCELADO))

    def test_max_workers_pode_ser_ajustado_em_tempo_de_execucao(self):
        simultaneos = []
        pico = []
        lock = threading.Lock()

        def worker(job):
            with lock:
                simultaneos.append(1)
                pico.append(len(simultaneos))
            time.sleep(0.08)
            with lock:
                simultaneos.pop()

        fila = jobs.JobQueue(worker)
        self.assertEqual(fila.max_workers, 1)

        fila.max_workers = 2
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(4)]
        _esperar_todos(enviados)

        self.assertEqual(max(pico), 2)

    def test_fila_com_varios_workers_volta_a_funcionar_depois_de_esvaziar(self):
        fila = jobs.JobQueue(lambda job: None, max_workers=2)
        for _ in range(2):
            job = fila.enviar("j", "/tmp/j")
            _esperar_todos([job])
            self.assertEqual(job.estado, jobs.CONCLUIDO)


if __name__ == "__main__":
    unittest.main()
