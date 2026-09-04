"""Barra única para o trabalho todo, e previsão de tempo medida."""

import unittest
from unittest import mock

from autosrt import jobs, pipeline
from autosrt.cue import Cue


class TestBarraUnica(unittest.TestCase):
    """Antes, transcrição e tradução contavam de 0 a 100 cada uma na sua vez.

    A barra enchia, voltava para trás e enchia de novo, e quem olhava não
    tinha como saber quanto faltava para o fim de verdade.
    """

    def rodar(self, *, translate=True, cues_traduzidos=4):
        """Roda o pipeline com Whisper e tradutor de mentira, colhendo a barra."""
        vistos = []

        cues = [Cue.from_source(index=i + 1, start=i * 2000,
                                end=i * 2000 + 1000,
                                source_text=f"linha {i}")
                for i in range(4)]

        def whisper_falso(media_path, **kwargs):
            progresso = kwargs.get("progress")
            if progresso:
                for pct in (0, 25, 50, 75, 100):
                    progresso(pct)
            return cues

        def tradutor_falso(cues_, lang, progress=None, **kwargs):
            if progress:
                for feitas in range(1, cues_traduzidos + 1):
                    progress(feitas, cues_traduzidos)
            return cues_traduzidos, []

        with mock.patch.object(pipeline, "_translate_with_llm", tradutor_falso), \
             mock.patch.object(pipeline, "_safe_detect", lambda c: "en"), \
             mock.patch.object(pipeline.srt_io, "save_cues", lambda *a, **k: None), \
             mock.patch("autosrt.transcribe.transcribe", whisper_falso):
            pipeline.process_media(
                "/tmp/filme.mkv", "/tmp/filme.srt", translate=translate,
                progress=lambda feitas, total: vistos.append(feitas))
        return vistos

    def test_a_barra_nunca_anda_para_tras(self):
        vistos = self.rodar()
        for anterior, seguinte in zip(vistos, vistos[1:]):
            self.assertLessEqual(anterior, seguinte,
                                 f"a barra voltou de {anterior} para {seguinte}")

    def test_transcricao_para_no_peso_e_traducao_termina_em_100(self):
        vistos = self.rodar()
        self.assertIn(pipeline.PESO_TRANSCRICAO, vistos)
        self.assertEqual(vistos[-1], 100)
        # A transcrição sozinha não pode encostar em 100: ainda falta traduzir.
        self.assertLessEqual(max(vistos[:5]), pipeline.PESO_TRANSCRICAO)

    def test_so_transcrever_usa_a_barra_inteira(self):
        # Sem tradução não há segunda fase para reservar espaço.
        vistos = self.rodar(translate=False)
        self.assertEqual(vistos[-1], 100)

    def test_barra_fica_dentro_da_faixa(self):
        for visto in self.rodar():
            self.assertGreaterEqual(visto, 0)
            self.assertLessEqual(visto, 100)


class TestPrevisao(unittest.TestCase):
    def job(self, progresso, rodando_ha, estado=jobs.RODANDO):
        job = jobs.Job(id="x", nome="filme.mkv", entrada="/tmp/filme.mkv")
        job.estado = estado
        job.progresso = progresso
        job.iniciado_em = 1000.0
        return job, 1000.0 + rodando_ha

    def test_metade_em_dez_minutos_preve_mais_dez(self):
        job, agora = self.job(progresso=50, rodando_ha=600)
        self.assertEqual(job.segundos_restantes(agora), 600)

    def test_um_quarto_em_cinco_minutos_preve_quinze(self):
        job, agora = self.job(progresso=25, rodando_ha=300)
        self.assertEqual(job.segundos_restantes(agora), 900)

    def test_quase_no_fim_preve_pouco(self):
        job, agora = self.job(progresso=99, rodando_ha=990)
        self.assertEqual(job.segundos_restantes(agora), 10)

    def test_sem_previsao_no_comeco(self):
        # Extrapolar de 1% dá número que pula de uma hora para dois minutos a
        # cada atualização: pior que não mostrar nada.
        job, agora = self.job(progresso=1, rodando_ha=600)
        self.assertIsNone(job.segundos_restantes(agora))

    def test_sem_previsao_nos_primeiros_segundos(self):
        job, agora = self.job(progresso=50, rodando_ha=5)
        self.assertIsNone(job.segundos_restantes(agora))

    def test_sem_previsao_para_quem_nao_esta_rodando(self):
        for estado in (jobs.PENDENTE, jobs.CONCLUIDO, jobs.ERRO, jobs.CANCELADO):
            with self.subTest(estado=estado):
                job, agora = self.job(progresso=50, rodando_ha=600, estado=estado)
                self.assertIsNone(job.segundos_restantes(agora))

    def test_sem_previsao_antes_de_sair_da_fila(self):
        # Tempo parado atrás de outro trabalho não diz nada sobre este.
        job = jobs.Job(id="x", nome="a.mkv", entrada="/tmp/a.mkv")
        job.estado = jobs.RODANDO
        job.progresso = 50
        self.assertIsNone(job.segundos_restantes())

    def test_previsao_sai_no_json(self):
        job, _ = self.job(progresso=50, rodando_ha=600)
        self.assertIn("restante_seg", job.para_json())

    def test_json_de_trabalho_novo_nao_tem_previsao(self):
        job = jobs.Job(id="x", nome="a.mkv", entrada="/tmp/a.mkv")
        self.assertIsNone(job.para_json()["restante_seg"])


if __name__ == "__main__":
    unittest.main()
