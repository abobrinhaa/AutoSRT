import unittest

from autosrt import limpeza
from autosrt.cue import Cue


class TestRemoverSDH(unittest.TestCase):
    def test_parenteses(self):
        self.assertEqual(limpeza.remover_sdh("(TIROS) Corre!").strip(), "Corre!")

    def test_colchetes(self):
        self.assertEqual(limpeza.remover_sdh("[música tensa]").strip(), "")

    def test_chaves(self):
        self.assertEqual(limpeza.remover_sdh("{RISOS} Que engraçado").strip(),
                         "Que engraçado")

    def test_rotulo_de_locutor_em_maiuscula(self):
        self.assertEqual(limpeza.remover_sdh("JOHN: Não vá embora"),
                         "Não vá embora")

    def test_nao_confunde_horario_com_rotulo(self):
        # "14:30" não é um rótulo de locutor (não começa com letra maiúscula).
        self.assertEqual(limpeza.remover_sdh("14:30 é a hora combinada"),
                         "14:30 é a hora combinada")

    def test_nao_confunde_fala_normal_com_rotulo(self):
        self.assertEqual(limpeza.remover_sdh("Espera: eu vou com você"),
                         "Espera: eu vou com você")

    def test_descricao_no_meio_da_fala(self):
        # remover_sdh só tira a descrição; sobra de espaço é trabalho de
        # limpar_espacos (as duas rodam juntas em preparar_texto).
        import re
        resultado = re.sub(r"\s+", " ", limpeza.remover_sdh(
            "Não posso (SUSPIRA) fazer isso agora")).strip()
        self.assertEqual(resultado, "Não posso fazer isso agora")


class TestConsertarTags(unittest.TestCase):
    def test_bem_formado_nao_muda(self):
        self.assertEqual(limpeza.consertar_tags("<i>Olá</i>"), "<i>Olá</i>")

    def test_fecha_abertura_sem_par(self):
        self.assertEqual(limpeza.consertar_tags("<i>Olá"), "<i>Olá</i>")

    def test_remove_fechamento_orfao(self):
        self.assertEqual(limpeza.consertar_tags("Olá</i>"), "Olá")

    def test_remove_par_vazio(self):
        self.assertEqual(limpeza.consertar_tags("<i></i>"), "")

    def test_remove_par_vazio_com_espaco(self):
        self.assertEqual(limpeza.consertar_tags("<i> </i>texto"), "texto")

    def test_duas_aberturas_uma_so_fecha(self):
        self.assertEqual(limpeza.consertar_tags("<i>A</i> e <i>B"),
                         "<i>A</i> e <i>B</i>")

    def test_negrito_e_sublinhado_tambem(self):
        self.assertEqual(limpeza.consertar_tags("<b>forte"), "<b>forte</b>")
        self.assertEqual(limpeza.consertar_tags("<u>sublinhado"),
                         "<u>sublinhado</u>")


class TestLimparEspacos(unittest.TestCase):
    def test_espaco_antes_de_pontuacao(self):
        self.assertEqual(limpeza.limpar_espacos("Oi , tudo bem ?"), "Oi, tudo bem?")

    def test_espaco_duplo(self):
        self.assertEqual(limpeza.limpar_espacos("Muito   espaço   aqui"),
                         "Muito espaço aqui")

    def test_espaco_inquebravel(self):
        self.assertEqual(limpeza.limpar_espacos("Fim de linha"), "Fim de linha")

    def test_sobra_no_fim_da_linha(self):
        self.assertEqual(limpeza.limpar_espacos("Primeira linha   \nSegunda   "),
                         "Primeira linha\nSegunda")

    def test_abertura_de_interrogacao_e_exclamacao(self):
        self.assertEqual(limpeza.limpar_espacos("¿ Qué pasa ?"), "¿Qué pasa?")


class TestJuntarLinhasCurtas(unittest.TestCase):
    def test_junta_frase_unica_partida_ao_meio(self):
        self.assertEqual(limpeza.juntar_linhas_curtas("Não\nvá."), "Não vá.")

    def test_mantem_dialogo_com_travessao(self):
        original = "- Você viu?\n- Vi."
        self.assertEqual(limpeza.juntar_linhas_curtas(original), original)

    def test_mantem_duas_frases_completas(self):
        original = "Isso é uma frase.\nE aqui outra frase completa."
        self.assertEqual(limpeza.juntar_linhas_curtas(original), original)

    def test_nao_junta_se_o_resultado_estourar_a_largura(self):
        original = ("Esta primeira parte da frase é bem comprida\n"
                    "e continua ainda mais longa sem parar em lugar nenhum")
        self.assertEqual(limpeza.juntar_linhas_curtas(original), original)

    def test_texto_de_uma_linha_so_nao_muda(self):
        self.assertEqual(limpeza.juntar_linhas_curtas("Só uma linha."),
                         "Só uma linha.")


class TestQuebrarLinhasLongas(unittest.TestCase):
    def test_quebra_o_que_passa_da_largura(self):
        texto = ("Esta é uma frase bem comprida que definitivamente "
                 "ultrapassa o limite de caracteres por linha de uma legenda")
        resultado = limpeza.quebrar_linhas_longas(texto)
        linhas = resultado.split("\n")
        # rewrap() escolhe o ponto de corte mais equilibrado; não crava um
        # teto rígido (ver TestRewrap em test_textfmt.py), então o que se
        # confere aqui é que quebrou em duas e nenhuma palavra se perdeu.
        self.assertEqual(len(linhas), 2)
        self.assertEqual(" ".join(linhas), texto)

    def test_nao_mexe_no_que_ja_cabe(self):
        self.assertEqual(limpeza.quebrar_linhas_longas("Texto curto"), "Texto curto")

    def test_mantem_dialogo_intacto(self):
        original = ("- Uma fala de travessão bem grande que passaria do "
                    "limite normal de leitura em uma legenda\n- Outra fala")
        self.assertEqual(limpeza.quebrar_linhas_longas(original), original)

    def test_tag_nao_conta_no_comprimento(self):
        # <i></i> não ocupa espaço na tela, então não deve empurrar a
        # legenda para a quebra de linha.
        texto = "<i>" + "a" * 40 + "</i>"
        self.assertEqual(limpeza.quebrar_linhas_longas(texto), texto)


class TestPrepararTexto(unittest.TestCase):
    def test_combina_sdh_tags_e_espacos(self):
        resultado = limpeza.preparar_texto("(TIROS) <i>Corre,   agora!</i>")
        self.assertEqual(resultado, "<i>Corre, agora!</i>")

    def test_legenda_so_de_descricao_fica_vazia(self):
        self.assertEqual(limpeza.preparar_texto("(TIROS)"), "")

    def test_legenda_so_de_espaco_fica_vazia(self):
        self.assertEqual(limpeza.preparar_texto("   "), "")

    def test_sem_remover_descricoes_preserva_parenteses(self):
        resultado = limpeza.preparar_texto("(TIROS) Corre!",
                                           remover_descricoes=False)
        self.assertIn("TIROS", resultado)

    def test_nao_junta_nem_quebra_linha_aqui(self):
        # Isso só acontece depois da tradução, em finalizar_texto -- antes
        # dela não dá para saber o tamanho final do texto.
        texto = "Não\nvá."
        self.assertEqual(limpeza.preparar_texto(texto), texto)


class TestPrepararCues(unittest.TestCase):
    def cue(self, texto, index=1):
        return Cue.from_source(index=index, start=0, end=1000, source_text=texto)

    def test_remove_cue_que_era_so_descricao(self):
        cues = [self.cue("(TIROS)"), self.cue("Corre!", index=2)]
        limpos, removidas = limpeza.preparar_cues(cues)
        self.assertEqual(removidas, 1)
        self.assertEqual([c.source_text for c in limpos], ["Corre!"])

    def test_texto_e_source_text_ficam_iguais_apos_preparar(self):
        cues = [self.cue("(TIROS) Corre!")]
        limpos, _ = limpeza.preparar_cues(cues)
        self.assertEqual(limpos[0].text, limpos[0].source_text)

    def test_lista_toda_limpa_nao_perde_nenhuma(self):
        cues = [self.cue("Olá"), self.cue("Tudo bem?", index=2)]
        limpos, removidas = limpeza.preparar_cues(cues)
        self.assertEqual(removidas, 0)
        self.assertEqual(len(limpos), 2)


class TestFinalizarCues(unittest.TestCase):
    def cue(self, texto_traduzido, index=1):
        c = Cue.from_source(index=index, start=0, end=1000, source_text="x")
        c.text = texto_traduzido
        return c

    def test_junta_e_conserta_depois_da_traducao(self):
        cues = [self.cue("<i>Não\nvá.")]
        limpeza.finalizar_cues(cues)
        self.assertEqual(cues[0].text, "<i>Não vá.</i>")

    def test_quebra_traducao_longa(self):
        original = ("Esta tradução para português ficou bem mais "
                    "comprida do que o texto original em inglês era")
        cues = [self.cue(original)]
        limpeza.finalizar_cues(cues)
        linhas = cues[0].text.split("\n")
        self.assertEqual(len(linhas), 2)
        self.assertEqual(" ".join(linhas), original)

    def test_nao_mexe_em_texto_vazio(self):
        cues = [self.cue("")]
        limpeza.finalizar_cues(cues)
        self.assertEqual(cues[0].text, "")


if __name__ == "__main__":
    unittest.main()
