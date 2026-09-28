"""A legenda precisa ficar na tela o tempo de alguém ler.

O Whisper fecha a legenda no instante em que a fala acaba: "Sim." dura
300 ms e some antes de ser lida. O ajuste estende o fim para dentro do
silêncio que já existe, sem nunca invadir a legenda seguinte.
"""

import unittest

from autosrt import timing
from autosrt.cue import Cue


def cue(inicio, fim, texto="Sim."):
    return Cue.from_source(index=1, start=inicio, end=fim, source_text=texto)


class TestDuracaoMinima(unittest.TestCase):
    def test_legenda_curta_ganha_a_duracao_minima(self):
        cues = [cue(0, 300), cue(10000, 12000)]
        timing.ajustar_exibicao(cues, duracao_minima=1.5)
        self.assertEqual(cues[0].end, 1500)

    def test_padrao_ja_vem_ligado(self):
        cues = [cue(0, 300), cue(10000, 12000)]
        timing.ajustar_exibicao(cues)
        self.assertEqual(cues[0].end,
                         round(timing.DURACAO_MINIMA_PADRAO * 1000))

    def test_none_quer_dizer_padrao(self):
        cues = [cue(0, 300), cue(10000, 12000)]
        timing.ajustar_exibicao(cues, duracao_minima=None,
                                caracteres_por_segundo=None)
        self.assertEqual(cues[0].end,
                         round(timing.DURACAO_MINIMA_PADRAO * 1000))

    def test_nunca_invade_a_legenda_seguinte(self):
        cues = [cue(0, 300), cue(1000, 3000)]
        timing.ajustar_exibicao(cues, duracao_minima=1.5)
        self.assertEqual(cues[0].end, 1000 - timing.INTERVALO_MINIMO_MS)

    def test_nunca_encurta(self):
        cues = [cue(0, 5000)]
        timing.ajustar_exibicao(cues, duracao_minima=1.5)
        self.assertEqual(cues[0].end, 5000)

    def test_sobreposicao_que_ja_existia_nao_piora(self):
        cues = [cue(0, 1200), cue(1000, 3000)]
        timing.ajustar_exibicao(cues, duracao_minima=1.5)
        self.assertEqual(cues[0].end, 1200)

    def test_ultima_legenda_tambem_e_ajustada(self):
        cues = [cue(0, 300)]
        timing.ajustar_exibicao(cues, duracao_minima=2)
        self.assertEqual(cues[0].end, 2000)

    def test_zero_desliga(self):
        cues = [cue(0, 300, "Uma frase bem mais comprida que o normal.")]
        estendidas = timing.ajustar_exibicao(
            cues, duracao_minima=0, caracteres_por_segundo=0)
        self.assertEqual(cues[0].end, 300)
        self.assertEqual(estendidas, 0)

    def test_devolve_quantas_foram_estendidas(self):
        cues = [cue(0, 300), cue(5000, 9000), cue(20000, 20200)]
        self.assertEqual(timing.ajustar_exibicao(cues, duracao_minima=1.5), 2)


class TestVelocidadeDeLeitura(unittest.TestCase):
    def test_texto_longo_fica_mais_tempo(self):
        texto = "a" * 60  # 60 caracteres a 15 por segundo = 4 s
        cues = [cue(0, 2000, texto)]
        timing.ajustar_exibicao(cues, duracao_minima=0,
                                caracteres_por_segundo=15)
        self.assertEqual(cues[0].end, 4000)

    def test_tags_nao_contam_como_texto(self):
        self.assertEqual(timing.duracao_de_leitura_ms("<i>abc</i>", 3), 1000)

    def test_quebra_de_linha_conta_como_espaco(self):
        self.assertEqual(timing.duracao_de_leitura_ms("ab\ncd", 5), 1000)

    def test_teto_de_duracao(self):
        cues = [cue(0, 2000, "a" * 500)]
        timing.ajustar_exibicao(cues, duracao_minima=0,
                                caracteres_por_segundo=15)
        self.assertEqual(cues[0].end, timing.DURACAO_MAXIMA_MS)

    def test_legenda_ja_maior_que_o_teto_nao_encurta(self):
        cues = [cue(0, 9000, "a" * 500)]
        timing.ajustar_exibicao(cues)
        self.assertEqual(cues[0].end, 9000)

    def test_mede_o_texto_de_trabalho(self):
        # Depois da tradução o que vai para a tela é o português, que
        # costuma sair mais longo que o original.
        legenda = cue(0, 1000, "Yes.")
        legenda.text = "a" * 45  # 3 s a 15 caracteres por segundo
        timing.ajustar_exibicao([legenda], duracao_minima=0,
                                caracteres_por_segundo=15)
        self.assertEqual(legenda.end, 3000)


if __name__ == "__main__":
    unittest.main()
