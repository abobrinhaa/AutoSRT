"""Testes das ferramentas do acervo de filmes (autosrt.filmes)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autosrt import filmes


class TestExtractTitleYear(unittest.TestCase):
    def test_titulo_com_ano_entre_parenteses(self):
        titulo, ano = filmes.extract_title_year("Blackwell's Island (1939) - DVDRip")
        self.assertEqual(titulo, "Blackwell's Island")
        self.assertEqual(ano, 1939)

    def test_titulo_com_ano_e_dublado(self):
        titulo, ano = filmes.extract_title_year("Submarine Base - 1943 DUBLADO")
        self.assertEqual(titulo, "Submarine Base")
        self.assertEqual(ano, 1943)

    def test_padrao_de_release_com_pontos(self):
        titulo, ano = filmes.extract_title_year("Titulo.Do.Filme.1994.1080p.BluRay.x264-GRUPO")
        self.assertEqual(titulo, "Titulo Do Filme")
        self.assertEqual(ano, 1994)

    def test_sem_ano_usa_o_nome_inteiro(self):
        titulo, ano = filmes.extract_title_year("gravacao da reuniao")
        self.assertEqual(titulo, "gravacao da reuniao")
        self.assertIsNone(ano)

    def test_nome_so_com_ano_nao_tem_titulo(self):
        self.assertIsNone(filmes.extract_title_year("1984"))


class TestCorrigirGenero(unittest.TestCase):
    def test_ja_correto_nao_muda_nada(self):
        texto = "<movie>\n  <title>X</title>\n  <genre>TERROR</genre>\n</movie>\n"
        novo, mudou = filmes._corrigir_genero(texto, "TERROR")
        self.assertFalse(mudou)
        self.assertEqual(novo, texto)

    def test_genero_errado_e_substituido(self):
        texto = "<movie>\n  <title>X</title>\n  <genre>CULT</genre>\n  <year>1999</year>\n</movie>\n"
        novo, mudou = filmes._corrigir_genero(texto, "TERROR")
        self.assertTrue(mudou)
        self.assertIn("<genre>TERROR</genre>", novo)
        self.assertNotIn("CULT", novo)
        self.assertIn("<title>X</title>", novo)
        self.assertIn("<year>1999</year>", novo)

    def test_sem_genero_e_inserido(self):
        texto = "<movie>\n  <title>X</title>\n  <year>1999</year>\n</movie>\n"
        novo, mudou = filmes._corrigir_genero(texto, "TERROR")
        self.assertTrue(mudou)
        self.assertIn("<genre>TERROR</genre>", novo)
        self.assertIn("<title>X</title>", novo)

    def test_arquivo_sem_tag_movie_nao_e_tocado(self):
        texto = "<tvshow><title>X</title></tvshow>\n"
        novo, mudou = filmes._corrigir_genero(texto, "TERROR")
        self.assertFalse(mudou)
        self.assertEqual(novo, texto)


class TestDiscoverUnits(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _arquivo(self, *partes):
        caminho = self.tmp.joinpath(*partes)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(b"\0")
        return caminho

    def test_video_solto_na_pasta_de_genero(self):
        self._arquivo("TERROR", "Filme Solto (1999).mkv")
        units, review = filmes.discover_units(self.tmp)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "file")
        self.assertEqual(units[0].genre_folder, "TERROR")
        self.assertFalse(review)

    def test_pasta_com_um_video_usa_movie_nfo(self):
        self._arquivo("CULT", "Algum Filme", "video.mkv")
        units, review = filmes.discover_units(self.tmp)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].kind, "folder")
        self.assertEqual(units[0].nfo_path.name, "movie.nfo")

    def test_pasta_com_varios_videos_vai_pra_revisao(self):
        self._arquivo("CULT", "Documentario", "parte1.mkv")
        self._arquivo("CULT", "Documentario", "parte2.mkv")
        units, review = filmes.discover_units(self.tmp)
        self.assertEqual(len(units), 0)
        self.assertEqual(len(review), 1)

    def test_skip_dirs_ignorados(self):
        self._arquivo("$RECYCLE.BIN", "lixo.mkv")
        units, review = filmes.discover_units(self.tmp)
        self.assertEqual(units, [])
        self.assertEqual(review, [])


class TestAtualizarGeneros(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _nfo(self, genero_pasta, nome_arquivo, genero_atual):
        pasta = self.tmp / genero_pasta
        pasta.mkdir(parents=True, exist_ok=True)
        caminho = pasta / nome_arquivo
        caminho.write_text(
            f"<movie>\n  <title>X</title>\n  <genre>{genero_atual}</genre>\n</movie>\n",
            encoding="utf-8")
        return caminho

    def test_simulacao_nao_grava(self):
        caminho = self._nfo("TERROR", "a.nfo", "CULT")
        original = caminho.read_text(encoding="utf-8")
        with mock.patch.object(filmes, "base_dir", return_value=self.tmp):
            relatorio = filmes.atualizar_generos(execute=False)
        self.assertEqual(relatorio["corrigidos"], 1)
        self.assertTrue(relatorio["simulado"])
        self.assertEqual(caminho.read_text(encoding="utf-8"), original)

    def test_execucao_corrige_de_verdade(self):
        caminho = self._nfo("TERROR", "a.nfo", "CULT")
        with mock.patch.object(filmes, "base_dir", return_value=self.tmp):
            relatorio = filmes.atualizar_generos(execute=True)
        self.assertEqual(relatorio["corrigidos"], 1)
        self.assertIn("<genre>TERROR</genre>", caminho.read_text(encoding="utf-8"))

    def test_ja_correto_nao_conta_como_corrigido(self):
        self._nfo("TERROR", "a.nfo", "TERROR")
        with mock.patch.object(filmes, "base_dir", return_value=self.tmp):
            relatorio = filmes.atualizar_generos(execute=True)
        self.assertEqual(relatorio["corrigidos"], 0)
        self.assertEqual(relatorio["ja_corretos"], 1)


class TestAvisarJellyfin(unittest.TestCase):
    def test_sem_configuracao_avisa_com_mensagem_clara(self):
        with mock.patch.object(filmes.config, "get_jellyfin_url", return_value=None), \
             mock.patch.object(filmes.config, "get_jellyfin_api_key", return_value="x"):
            with self.assertRaises(filmes.JellyfinNaoConfigurado):
                filmes.avisar_jellyfin()

    def test_chama_library_refresh_com_a_chave(self):
        resposta = mock.Mock(status_code=204)
        with mock.patch.object(filmes.config, "get_jellyfin_url", return_value="http://jelly:8096"), \
             mock.patch.object(filmes.config, "get_jellyfin_api_key", return_value="chave123"), \
             mock.patch.object(filmes.requests, "post", return_value=resposta) as post:
            resultado = filmes.avisar_jellyfin()
        self.assertEqual(resultado, {"ok": True})
        post.assert_called_once_with("http://jelly:8096/Library/Refresh",
                                     params={"api_key": "chave123"}, timeout=10)

    def test_erro_http_do_jellyfin_vira_jellyfin_falhou(self):
        resposta = mock.Mock(status_code=500)
        with mock.patch.object(filmes.config, "get_jellyfin_url", return_value="http://jelly:8096"), \
             mock.patch.object(filmes.config, "get_jellyfin_api_key", return_value="chave123"), \
             mock.patch.object(filmes.requests, "post", return_value=resposta):
            with self.assertRaises(filmes.JellyfinFalhou):
                filmes.avisar_jellyfin()


if __name__ == "__main__":
    unittest.main()
