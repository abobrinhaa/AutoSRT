"""Guarda o JavaScript da página contra escapes quebrados.

PAGINA é uma string Python comum, então ``\\n`` escrito com uma barra só
vira quebra de linha de verdade e parte o literal JavaScript ao meio --
a página inteira morre com "Invalid or unexpected token" e nenhum botão
funciona. O teste lê o template sem importar o módulo, para rodar mesmo
onde o Flask não está instalado.
"""

import os
import re
import unittest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_PY = os.path.join(RAIZ, "autosrt", "web.py")


def render_pagina():
    """Devolve o HTML final, sem depender do Flask."""
    with open(WEB_PY, encoding="utf-8") as arquivo:
        fonte = arquivo.read()
    trecho = re.search(r'^PAGINA = """.*?^"""', fonte, re.S | re.M)
    escopo = {}
    exec(trecho.group(0), escopo)
    return escopo["PAGINA"].replace("{{ACCEPT}}", ".mp4,.mkv,.srt")


def strings_com_quebra(js):
    """Aponta as linhas onde um literal de string engole uma quebra de linha."""
    quebradas = []
    delimitador = None
    linha = 1
    i = 0
    while i < len(js):
        c = js[i]
        if c == "\n":
            if delimitador in ("'", '"'):
                quebradas.append(linha)
                delimitador = None
            linha += 1
        elif delimitador and c == "\\":
            i += 1  # escape: pula o próximo caractere
        elif delimitador:
            if c == delimitador:
                delimitador = None
        elif c in ("'", '"', "`"):
            delimitador = c
        i += 1
    return quebradas


class TestPaginaJS(unittest.TestCase):
    def setUp(self):
        self.html = render_pagina()
        self.js = re.search(r"<script>(.*?)</script>", self.html, re.S).group(1)

    def test_sem_string_partida_por_quebra_de_linha(self):
        quebradas = strings_com_quebra(self.js)
        self.assertEqual(
            quebradas, [],
            "literal de string cortado por quebra de linha (use \\\\n, não \\n) "
            f"nas linhas {quebradas} do <script>")

    def test_sem_placeholder_esquecido(self):
        sobrou = re.findall(r"\{\{[^}]*\}\}", self.html)
        self.assertEqual(sobrou, [], f"placeholder não substituído: {sobrou}")

    def test_alertas_usam_newline_escapado(self):
        # As três mensagens multi-linha precisam do \n literal no JS.
        self.assertEqual(self.js.count(r"\n"), 3)


if __name__ == "__main__":
    unittest.main()
