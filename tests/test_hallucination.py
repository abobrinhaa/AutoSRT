"""O Whisper inventa frase de encerramento onde não há fala.

Sobre silêncio e trilha sonora ele "chuta" o clichê de legenda automática
de YouTube -- "Obrigado por assistir", "Até a próxima", "Legendas pela
comunidade Amara.org". Não é erro de tradução: a frase já nasce assim na
transcrição, e o tradutor só a repassa fielmente.
"""

import unittest

from autosrt import hallucination
from autosrt.cue import Cue


def cue(indice, inicio, fim, texto):
    """Uma legenda, com os tempos em segundos para caber na leitura."""
    return Cue.from_source(index=indice, start=int(inicio * 1000),
                           end=int(fim * 1000), source_text=texto)


def textos(cues):
    return [c.source_text for c in cues]


#: Diálogo de verdade, sem clichê nenhum, longo o bastante para haver meio.
DIALOGO = [
    cue(1, 1, 3, "Você viu o que aconteceu ontem?"),
    cue(2, 3, 5, "Vi, e não acreditei."),
    cue(3, 5, 7, "Ninguém acreditou."),
    cue(4, 7, 9, "O delegado ainda está lá?"),
    cue(5, 9, 11, "Está, desde cedo."),
]


class TestNormalizar(unittest.TestCase):
    def test_ignora_acento_caixa_e_pontuacao(self):
        self.assertEqual(hallucination.normalizar("Até a próxima!!!"),
                         "ate a proxima")

    def test_ignora_tags_de_formatacao(self):
        self.assertEqual(hallucination.normalizar("<i>Obrigado</i>"), "obrigado")

    def test_apostrofo_nao_parte_a_palavra(self):
        self.assertEqual(hallucination.normalizar("Don't forget"), "dont forget")

    def test_ponto_de_dominio_vira_espaco(self):
        self.assertEqual(hallucination.normalizar("Amara.org"), "amara org")

    def test_constantes_ja_estao_normalizadas(self):
        """Uma constante escrita com acento nunca casaria com nada."""
        for frase in (hallucination.FRASES_INEQUIVOCAS
                      | hallucination.FRASES_AMBIGUAS):
            self.assertEqual(hallucination.normalizar(frase), frase)


class TestFrasesInequivocas(unittest.TestCase):
    """Clichê de canal de vídeo não é fala de filme em lugar nenhum."""

    def test_sai_mesmo_cercado_de_fala_de_verdade(self):
        cues = DIALOGO[:2] + [cue(3, 5, 7, "Obrigado por assistir!")] + DIALOGO[2:]
        resultado = hallucination.filtrar_alucinacoes(cues)
        self.assertNotIn("Obrigado por assistir!", textos(resultado))
        self.assertEqual(len(resultado), 5)

    def test_amara_sai_onde_estiver(self):
        cues = DIALOGO[:2] + [
            cue(3, 5, 7, "Legendas pela comunidade Amara.org")] + DIALOGO[2:]
        resultado = hallucination.filtrar_alucinacoes(cues)
        self.assertEqual(len(resultado), 5)

    def test_inscreva_se_no_canal(self):
        cues = DIALOGO[:2] + [
            cue(3, 5, 7, "Inscreva-se no canal!")] + DIALOGO[2:]
        self.assertEqual(len(hallucination.filtrar_alucinacoes(cues)), 5)

    def test_thanks_for_watching(self):
        cues = DIALOGO + [cue(6, 30, 33, "Thanks for watching!")]
        self.assertEqual(len(hallucination.filtrar_alucinacoes(cues)), 5)

    def test_legenda_toda_feita_de_clichê_em_duas_linhas(self):
        cues = DIALOGO + [
            cue(6, 30, 33, "Obrigado por assistir!\nAté a próxima!")]
        self.assertEqual(len(hallucination.filtrar_alucinacoes(cues)), 5)


class TestFraseDentroDeFalaMaior(unittest.TestCase):
    """Só sai a legenda que é *inteiramente* o clichê.

    O mesmo texto dentro de uma fala maior é fala de verdade -- remover por
    substring apagaria a frase junto.
    """

    def test_nao_sai_por_substring(self):
        fala = "Obrigado por assistir ao filme comigo ontem, foi ótimo."
        cues = DIALOGO[:2] + [cue(3, 5, 7, fala)] + DIALOGO[2:]
        self.assertIn(fala, textos(hallucination.filtrar_alucinacoes(cues)))


class TestFrasesAmbiguas(unittest.TestCase):
    """"Obrigado", "Até a próxima" e "Tchau" são fala legítima de filme.

    Só saem com sinal corroborante: borda do arquivo, isolamento por
    silêncio longo, ou repetição idêntica.
    """

    def test_obrigado_no_meio_do_dialogo_fica(self):
        cues = DIALOGO[:2] + [cue(3, 5, 7, "Obrigado.")] + DIALOGO[2:]
        self.assertIn("Obrigado.", textos(hallucination.filtrar_alucinacoes(cues)))

    def test_ate_a_proxima_no_fim_do_arquivo_sai(self):
        cues = DIALOGO + [cue(6, 11, 13, "Até a próxima!")]
        self.assertEqual(len(hallucination.filtrar_alucinacoes(cues)), 5)

    def test_no_comeco_do_arquivo_sai(self):
        cues = [cue(1, 0, 2, "Obrigado!")] + [
            cue(i + 2, c.start / 1000 + 2, c.end / 1000 + 2, c.source_text)
            for i, c in enumerate(DIALOGO)]
        resultado = hallucination.filtrar_alucinacoes(cues)
        self.assertNotIn("Obrigado!", textos(resultado))

    def test_isolada_por_silencio_longo_sai(self):
        # Sete segundos de silêncio antes e nada por perto: não é diálogo.
        cues = DIALOGO[:3] + [cue(4, 18, 20, "Tchau.")] + [
            cue(5, 30, 32, "O delegado ainda está lá?"),
            cue(6, 32, 34, "Está, desde cedo."),
            cue(7, 34, 36, "Desde as seis da manhã."),
        ]
        self.assertNotIn("Tchau.", textos(hallucination.filtrar_alucinacoes(cues)))

    def test_repeticao_identica_conta_como_sinal(self):
        cues = (DIALOGO[:2] + [cue(3, 5, 7, "Obrigado.")] + DIALOGO[2:4]
                + [cue(6, 12, 14, "Obrigado.")] + DIALOGO[4:]
                + [cue(8, 20, 22, "E aí, tudo certo?"),
                   cue(9, 22, 24, "Tudo certo.")])
        self.assertNotIn("Obrigado.", textos(hallucination.filtrar_alucinacoes(cues)))


class TestSalvaguardas(unittest.TestCase):
    def test_lista_vazia(self):
        self.assertEqual(hallucination.filtrar_alucinacoes([]), [])

    def test_nunca_esvazia_o_arquivo(self):
        """Filtro de segurança não pode transformar transcrição em nada.

        Um arquivo curto que é só agradecimento (um vídeo de encerramento,
        por exemplo) sairia vazio -- e legenda vazia é pior que legenda com
        uma frase a mais.
        """
        cues = [cue(1, 1, 3, "Obrigado por assistir!"),
                cue(2, 4, 6, "Até a próxima!")]
        self.assertEqual(len(hallucination.filtrar_alucinacoes(cues)), 2)

    def test_reindexa_o_que_sobrou(self):
        cues = DIALOGO[:2] + [cue(3, 5, 7, "Obrigado por assistir!")] + DIALOGO[2:]
        resultado = hallucination.filtrar_alucinacoes(cues)
        self.assertEqual([c.index for c in resultado], [1, 2, 3, 4, 5])

    def test_nao_mexe_na_lista_recebida(self):
        cues = DIALOGO + [cue(6, 30, 33, "Thanks for watching!")]
        hallucination.filtrar_alucinacoes(cues)
        self.assertEqual(len(cues), 6)


class TestFrasesExtras(unittest.TestCase):
    """Cada acervo tem o seu clichê; a lista embutida não cobre todos."""

    def test_frase_do_usuario_sai(self):
        cues = DIALOGO[:2] + [cue(3, 5, 7, "Legendas: João da Silva")] + DIALOGO[2:]
        resultado = hallucination.filtrar_alucinacoes(
            cues, extras=["Legendas: João da Silva"])
        self.assertEqual(len(resultado), 5)

    def test_extra_tambem_ignora_acento_e_caixa(self):
        cues = DIALOGO[:2] + [cue(3, 5, 7, "ASSISTA O PRÓXIMO EPISÓDIO")] + DIALOGO[2:]
        resultado = hallucination.filtrar_alucinacoes(
            cues, extras=["assista o proximo episodio"])
        self.assertEqual(len(resultado), 5)

    def test_extra_vazio_nao_apaga_tudo(self):
        """Uma linha em branco na lista viraria "casa com qualquer coisa"."""
        resultado = hallucination.filtrar_alucinacoes(DIALOGO, extras=["", "   "])
        self.assertEqual(len(resultado), 5)


if __name__ == "__main__":
    unittest.main()
