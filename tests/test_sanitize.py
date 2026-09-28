"""O arquivo gravado não pode ter lixo que o tocador mostra ao pé da letra.

Casos reais: o modelo de tradução devolvendo a quebra de linha como
``</br>`` ou ``</n>`` (eco do ``<N>texto</N>`` do prompt), e frases em
chinês no meio da legenda em português -- tanto o clichê de encerramento
que o Whisper inventa quanto o modelo que escorrega de idioma no meio da
frase.
"""

import unittest

from autosrt import sanitize
from autosrt.cue import Cue


def limpar(texto, **kwargs):
    return sanitize.limpar_texto(texto, **kwargs)


class TestQuebraDeLinhaEscritaComoTexto(unittest.TestCase):
    def test_br_em_todas_as_grafias_vira_quebra(self):
        for tag in ("<br>", "</br>", "<br/>", "<br />", "<BR>"):
            with self.subTest(tag=tag):
                self.assertEqual(limpar(f"Olá{tag}mundo"), "Olá\nmundo")

    def test_barra_n_literal_vira_quebra(self):
        self.assertEqual(limpar("Olá\\nmundo"), "Olá\nmundo")
        self.assertEqual(limpar("Olá\\Nmundo"), "Olá\nmundo")
        self.assertEqual(limpar("Olá\\r\\nmundo"), "Olá\nmundo")

    def test_eco_da_tag_n_do_prompt_some(self):
        self.assertEqual(limpar("Olá mundo</n>"), "Olá mundo")
        self.assertEqual(limpar("<N>Olá mundo</N>"), "Olá mundo")

    def test_quebra_repetida_nao_vira_linha_vazia(self):
        self.assertEqual(limpar("Olá<br>\n<br>mundo"), "Olá\nmundo")


class TestTags(unittest.TestCase):
    def test_numeracao_de_bloco_some(self):
        self.assertEqual(limpar("<12>Olá mundo</12>"), "Olá mundo")

    def test_tags_que_o_srt_conhece_ficam(self):
        for texto in ("<i>Olá mundo</i>", "<b>Olá</b>", "<u>Olá</u>",
                      '<font color="#ffffff">Olá</font>'):
            with self.subTest(texto=texto):
                self.assertEqual(limpar(texto), texto)

    def test_tag_desconhecida_some(self):
        self.assertEqual(limpar("<span>Olá</span> mundo"), "Olá mundo")

    def test_tag_sem_par_some(self):
        # O "</B>" solto que o llama3.1 deixava vazar na legenda final.
        self.assertEqual(limpar("Olá mundo</B>"), "Olá mundo")
        self.assertEqual(limpar("<i>Olá mundo"), "Olá mundo")

    def test_posicionamento_do_ssa_fica_e_o_resto_sai(self):
        self.assertEqual(limpar("{\\an8}Olá"), "{\\an8}Olá")
        self.assertEqual(limpar("{\\i1}Olá"), "Olá")

    def test_sinal_de_menor_solto_nao_e_tag(self):
        self.assertEqual(limpar("Eu <3 você"), "Eu <3 você")
        self.assertEqual(limpar("se a < b > c"), "se a < b > c")


class TestOutrosResiduos(unittest.TestCase):
    def test_entidades_html_viram_o_caractere(self):
        self.assertEqual(limpar("Tom &amp; Jerry"), "Tom & Jerry")
        self.assertEqual(limpar("&quot;Oi&quot;"), '"Oi"')

    def test_caracteres_de_controle_somem(self):
        self.assertEqual(limpar("Olá\x00\x07 mundo�"), "Olá mundo")

    def test_cerca_de_codigo_some(self):
        self.assertEqual(limpar("```\nOlá mundo\n```"), "Olá mundo")

    def test_espacos_sobrando_e_linhas_vazias(self):
        self.assertEqual(limpar("  Olá   mundo  \n\n  tchau "), "Olá mundo\ntchau")

    def test_mais_de_duas_linhas_e_reagrupado(self):
        texto = ("Isto veio quebrado\nem linhas demais\npelo modelo, "
                 "e precisa\nvoltar a caber em duas")
        resultado = limpar(texto)
        self.assertEqual(len(resultado.split("\n")), 2)
        self.assertEqual(resultado.split(), texto.split())

    def test_mais_de_duas_linhas_curtas_viram_uma(self):
        self.assertEqual(limpar("uma\nduas\ntrês"), "uma duas três")

    def test_sobra_de_quebra_nao_gruda_palavras(self):
        self.assertEqual(limpar("Olá<span>mundo</span>"), "Olá mundo")

    def test_dialogo_de_tres_turnos_fica(self):
        texto = "- Oi.\n- Tudo bem?\n- Tudo."
        self.assertEqual(limpar(texto), texto)

    def test_texto_limpo_passa_intacto(self):
        for texto in ("Cheguei em 1º lugar!", "Custa R$ 5,00.",
                      'Ele disse: "não".', "- Você viu?\n- Vi.",
                      "Isso é... estranho.", "♪ Música ♪"):
            with self.subTest(texto=texto):
                self.assertEqual(limpar(texto), texto)


class TestEscritaDoLesteAsiatico(unittest.TestCase):
    def test_ideogramas_no_meio_da_frase_saem(self):
        self.assertEqual(limpar("Eu não sei o que 发生了 aqui."),
                         "Eu não sei o que aqui.")

    def test_linha_toda_em_chines_sai_inteira(self):
        self.assertEqual(limpar("Obrigado.\n请不吝点赞 订阅 转发"), "Obrigado.")

    def test_linha_com_mais_ideograma_que_letra_sai_inteira(self):
        # Sobrar só o "by" do crédito seria trocar um lixo por outro.
        self.assertEqual(limpar("字幕by索兰娅"), "")

    def test_japones_e_coreano_tambem(self):
        self.assertEqual(limpar("ご視聴ありがとうございました"), "")
        self.assertEqual(limpar("시청해주셔서 감사합니다"), "")

    def test_pontuacao_de_largura_cheia_vira_a_comum(self):
        self.assertEqual(limpar("Não！"), "Não!")
        self.assertEqual(limpar("Sim，claro。"), "Sim,claro.")

    def test_sobra_de_pontuacao_nao_vira_legenda(self):
        self.assertEqual(limpar("你好！"), "")

    def test_legenda_em_idioma_cjk_fica_intacta(self):
        self.assertEqual(limpar("你好，世界", remover_cjk=False), "你好，世界")

    def test_tem_cjk(self):
        self.assertTrue(sanitize.tem_cjk("Olá 世界"))
        self.assertFalse(sanitize.tem_cjk("Olá mundo, ação!"))

    def test_predominio_cjk(self):
        self.assertTrue(sanitize.predominio_cjk("字幕by索兰娅"))
        self.assertFalse(sanitize.predominio_cjk("I went to 北京 yesterday."))
        self.assertFalse(sanitize.predominio_cjk("..."))

    def test_idioma_cjk(self):
        for codigo in ("zh", "zh-CN", "zh_TW", "ja", "ko", "yue"):
            with self.subTest(codigo=codigo):
                self.assertTrue(sanitize.idioma_cjk(codigo))
        for codigo in ("en", "pt", "", None):
            with self.subTest(codigo=codigo):
                self.assertFalse(sanitize.idioma_cjk(codigo))


def cue(indice, texto):
    return Cue.from_source(index=indice, start=indice * 1000,
                           end=indice * 1000 + 900, source_text=texto)


class TestLimparLegendas(unittest.TestCase):
    def test_limpa_o_texto_de_trabalho(self):
        cues = [cue(1, "Olá</br>mundo")]
        resultado = sanitize.limpar_legendas(cues)
        self.assertEqual(resultado[0].text, "Olá\nmundo")

    def test_texto_de_origem_nao_e_tocado(self):
        cues = [cue(1, "Olá</br>mundo")]
        sanitize.limpar_legendas(cues)
        self.assertEqual(cues[0].source_text, "Olá</br>mundo")

    def test_legenda_que_era_so_lixo_sai_e_as_outras_sao_renumeradas(self):
        cues = [cue(1, "Oi."), cue(2, "请不吝点赞 订阅"), cue(3, "Tchau.")]
        resultado = sanitize.limpar_legendas(cues)
        self.assertEqual([c.text for c in resultado], ["Oi.", "Tchau."])
        self.assertEqual([c.index for c in resultado], [1, 2])

    def test_nao_esvazia_o_arquivo_inteiro(self):
        # Arquivo vazio é pior do que arquivo com lixo: some sem aviso.
        cues = [cue(1, "请不吝点赞 订阅")]
        resultado = sanitize.limpar_legendas(cues)
        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0].text, "请不吝点赞 订阅")

    def test_lista_vazia(self):
        self.assertEqual(sanitize.limpar_legendas([]), [])


if __name__ == "__main__":
    unittest.main()
