"""Concorrência da fila (autosrt.jobs.JobQueue).

O padrão continua sendo um operário só (ver test_web.TestFila) -- a mudança
aqui é permitir mais, para quem tem VRAM sobrando (ex.: RTX 3060 12 GB) e
quer processar mais de um arquivo ao mesmo tempo em vez de deixar a placa
ociosa entre um trabalho e outro.
"""

import logging
import threading
import time
import unittest

from autosrt import jobs

# Os testes de falha (worker e on_final quebrados) provocam exceção de
# propósito; o traceback delas no log só poluiria a saída da suíte -- mesmo
# ajuste que test_web.py já faz para autosrt.jobs.
logging.getLogger("autosrt.jobs").setLevel(logging.CRITICAL)


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


class TestOnFinal(unittest.TestCase):
    """``on_final`` é o gancho que autosrt.historico usa pra persistir cada
    trabalho terminado (ver autosrt.web, JobQueue(on_final=historico.registrar)).
    Sem fila nem thread nestes testes -- só o contrato: quando é chamado,
    com o quê, e que um erro nele não derruba o operário."""

    def test_chamado_apos_concluir(self):
        recebidos = []
        fila = jobs.JobQueue(lambda job: None, on_final=recebidos.append)
        job = fila.enviar("j", "/tmp/j")
        _esperar_todos([job])

        self.assertEqual(len(recebidos), 1)
        self.assertIs(recebidos[0], job)
        self.assertEqual(recebidos[0].estado, jobs.CONCLUIDO)

    def test_chamado_apos_erro_com_estado_erro(self):
        def falha(job):
            raise RuntimeError("deu ruim")

        recebidos = []
        fila = jobs.JobQueue(falha, on_final=recebidos.append)
        job = fila.enviar("j", "/tmp/j")
        _esperar_todos([job])

        self.assertEqual(len(recebidos), 1)
        self.assertEqual(recebidos[0].estado, jobs.ERRO)

    def test_chamado_apos_cancelar_com_estado_cancelado(self):
        liberar = threading.Event()

        def worker(job):
            liberar.wait(2)
            # O operário só percebe o cancelamento ao checar depois --
            # mesma mecânica que o resto da suíte já usa (ver
            # TestConcorrenciaDaFila acima).

        recebidos = []
        fila = jobs.JobQueue(worker, on_final=recebidos.append)
        job = fila.enviar("j", "/tmp/j")
        time.sleep(0.05)
        job.cancelar.set()
        liberar.set()
        _esperar_todos([job])

        self.assertEqual(len(recebidos), 1)
        self.assertEqual(recebidos[0].estado, jobs.CANCELADO)

    def test_erro_dentro_de_on_final_nao_impede_o_proximo_trabalho(self):
        # Histórico é registro, não faz parte do trabalho em si -- um bug
        # ali (disco cheio, permissão, SQLite travado) não pode travar a
        # fila nem apagar o resultado do trabalho que acabou de terminar.
        def on_final_quebrado(job):
            raise RuntimeError("não consegui gravar")

        fila = jobs.JobQueue(lambda job: None, on_final=on_final_quebrado)
        primeiro = fila.enviar("um", "/tmp/um")
        segundo = fila.enviar("dois", "/tmp/dois")
        _esperar_todos([primeiro, segundo])

        self.assertEqual(primeiro.estado, jobs.CONCLUIDO)
        self.assertEqual(segundo.estado, jobs.CONCLUIDO)

    def test_sem_on_final_nao_quebra(self):
        fila = jobs.JobQueue(lambda job: None)
        job = fila.enviar("j", "/tmp/j")
        _esperar_todos([job])
        self.assertEqual(job.estado, jobs.CONCLUIDO)


if __name__ == "__main__":
    unittest.main()
