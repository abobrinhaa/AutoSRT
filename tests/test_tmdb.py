"""Testes do reconhecimento de filme via TMDB."""

import unittest
from unittest import mock

from autosrt import tmdb


class TestAdivinharTituloEAno(unittest.TestCase):
    def test_padrao_de_release_com_pontos(self):
        titulo, ano = tmdb.guess_title_and_year(
            "Duna.Parte.Dois.2024.2160p.BluRay.x265-GRUPO.mkv")
        self.assertEqual(titulo, "Duna Parte Dois")
        self.assertEqual(ano, 2024)

    def test_padrao_com_underscore(self):
        titulo, ano = tmdb.guess_title_and_year("O_Poderoso_Chefao_1972.mp4")
        self.assertEqual(titulo, "O Poderoso Chefao")
        self.assertEqual(ano, 1972)

    def test_sem_ano_usa_o_nome_inteiro(self):
        titulo, ano = tmdb.guess_title_and_year("gravacao_da_reuniao.mkv")
        self.assertEqual(titulo, "gravacao da reuniao")
        self.assertIsNone(ano)

    def test_nao_confunde_resolucao_com_ano(self):
        titulo, ano = tmdb.guess_title_and_year("Filme.2160p.mkv")
        self.assertIsNone(ano)
        self.assertIn("2160p", titulo)

    def test_pega_o_ultimo_ano_quando_ha_mais_de_um(self):
        titulo, ano = tmdb.guess_title_and_year("Serie.1984.Temporada.2019.mkv")
        self.assertEqual(ano, 2019)


class TestAdivinharSerieEEpisodio(unittest.TestCase):
    def test_padrao_sxxexx_com_pontos(self):
        titulo, temporada, episodio = tmdb.guess_series_title_and_episode(
            "Nome.Da.Serie.S01E05.1080p.WEB-DL-GRUPO.mkv")
        self.assertEqual(titulo, "Nome Da Serie")
        self.assertEqual(temporada, 1)
        self.assertEqual(episodio, 5)

    def test_temporada_e_episodio_de_dois_digitos(self):
        titulo, temporada, episodio = tmdb.guess_series_title_and_episode(
            "Outra.Serie.S12E108.mkv")
        self.assertEqual(titulo, "Outra Serie")
        self.assertEqual(temporada, 12)
        self.assertEqual(episodio, 108)

    def test_sem_padrao_devolve_none(self):
        titulo, temporada, episodio = tmdb.guess_series_title_and_episode(
            "Duna.Parte.Dois.2024.mkv")
        self.assertIsNone(titulo)
        self.assertIsNone(temporada)
        self.assertIsNone(episodio)


class TestSearchMovie(unittest.TestCase):
    def test_sem_chave_nao_busca(self):
        self.assertIsNone(tmdb.search_movie("Duna", api_key=None))

    def test_sem_consulta_nao_busca(self):
        self.assertIsNone(tmdb.search_movie("", api_key="chave"))

    def test_devolve_primeiro_resultado(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": [{"title": "Duna"}, {"title": "Outro"}]}
        sessao.get.return_value = resposta

        resultado = tmdb.search_movie("Duna", api_key="chave", session=sessao)
        self.assertEqual(resultado["title"], "Duna")

    def test_sem_resultado_devolve_none(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": []}
        sessao.get.return_value = resposta

        self.assertIsNone(tmdb.search_movie("Filme Inexistente", api_key="chave",
                                            session=sessao))

    def test_falha_de_rede_nao_levanta(self):
        import requests
        sessao = mock.Mock()
        sessao.get.side_effect = requests.ConnectionError("sem rede")

        self.assertIsNone(tmdb.search_movie("Duna", api_key="chave", session=sessao))

    def test_http_erro_devolve_none(self):
        sessao = mock.Mock()
        sessao.get.return_value = mock.Mock(status_code=401)

        self.assertIsNone(tmdb.search_movie("Duna", api_key="chave-invalida",
                                            session=sessao))


class TestLookupForFile(unittest.TestCase):
    def test_sem_chave_devolve_none(self):
        self.assertIsNone(tmdb.lookup_for_file("Duna.2021.mkv", api_key=None))

    def test_monta_resumo_compacto(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": [{
            "title": "Duna",
            "release_date": "2021-09-15",
            "poster_path": "/abc.jpg",
            "original_language": "en",
            "overview": "Um herdeiro nobre precisa viajar até o planeta mais perigoso.",
        }]}
        sessao.get.return_value = resposta

        resultado = tmdb.lookup_for_file("Duna.2021.1080p.mkv", api_key="chave",
                                         session=sessao)
        self.assertEqual(resultado["tipo"], "filme")
        self.assertEqual(resultado["titulo"], "Duna")
        self.assertEqual(resultado["ano"], 2021)
        self.assertEqual(resultado["poster"], "https://image.tmdb.org/t/p/w342/abc.jpg")
        self.assertEqual(resultado["idioma_original"], "en")
        self.assertEqual(resultado["idioma_original_nome"], "Inglês")
        self.assertEqual(resultado["sinopse"],
                         "Um herdeiro nobre precisa viajar até o planeta mais perigoso.")

    def test_sem_correspondencia_devolve_none(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": []}
        sessao.get.return_value = resposta

        self.assertIsNone(tmdb.lookup_for_file("qualquer_coisa.mkv", api_key="chave",
                                               session=sessao))

    def test_reconhece_serie_pelo_padrao_sxxexx(self):
        sessao = mock.Mock()
        resposta_busca = mock.Mock(status_code=200)
        resposta_busca.json.return_value = {"results": [{
            "id": 42,
            "name": "Nome Da Serie",
            "poster_path": "/serie.jpg",
            "original_language": "en",
            "overview": "Uma serie qualquer.",
        }]}
        resposta_detalhes = mock.Mock(status_code=200)
        resposta_detalhes.json.return_value = {
            "number_of_seasons": 3, "number_of_episodes": 24,
        }
        sessao.get.side_effect = [resposta_busca, resposta_detalhes]

        resultado = tmdb.lookup_for_file("Nome.Da.Serie.S01E05.mkv", api_key="chave",
                                         session=sessao)
        self.assertEqual(resultado["tipo"], "serie")
        self.assertEqual(resultado["titulo"], "Nome Da Serie")
        self.assertEqual(resultado["poster"], "https://image.tmdb.org/t/p/w342/serie.jpg")
        self.assertEqual(resultado["sinopse"], "Uma serie qualquer.")
        self.assertEqual(resultado["temporada"], 1)
        self.assertEqual(resultado["episodio"], 5)
        self.assertEqual(resultado["temporadas_total"], 3)
        self.assertEqual(resultado["episodios_total"], 24)

    def test_serie_sem_correspondencia_devolve_none(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": []}
        sessao.get.return_value = resposta

        self.assertIsNone(tmdb.lookup_for_file("Serie.Inexistente.S01E01.mkv",
                                               api_key="chave", session=sessao))


class TestCache(unittest.TestCase):
    def setUp(self):
        tmdb.limpar_cache()

    def tearDown(self):
        tmdb.limpar_cache()

    def test_so_busca_uma_vez_por_arquivo(self):
        sessao = mock.Mock()
        resposta = mock.Mock(status_code=200)
        resposta.json.return_value = {"results": [{
            "title": "Duna", "release_date": "2021", "poster_path": None,
            "original_language": "en",
        }]}
        sessao.get.return_value = resposta

        tmdb.lookup_cached("Duna.2021.mkv", api_key="chave", session=sessao)
        tmdb.lookup_cached("Duna.2021.mkv", api_key="chave", session=sessao)

        sessao.get.assert_called_once()

    def test_cacheia_resultado_negativo(self):
        sessao = mock.Mock()
        sessao.get.return_value = mock.Mock(status_code=404)

        primeiro = tmdb.lookup_cached("nao_existe.mkv", api_key="chave", session=sessao)
        segundo = tmdb.lookup_cached("nao_existe.mkv", api_key="chave", session=sessao)

        self.assertIsNone(primeiro)
        self.assertIsNone(segundo)
        sessao.get.assert_called_once()

    def test_detalhes_da_serie_sao_compartilhados_entre_episodios(self):
        # Dois episódios da mesma série (nomes de arquivo diferentes, mesmo
        # tv_id) não devem repetir a chamada de detalhes (number_of_seasons/
        # number_of_episodes) -- só a busca por nome varia por episódio.
        sessao = mock.Mock()
        resposta_busca = mock.Mock(status_code=200)
        resposta_busca.json.return_value = {"results": [{
            "id": 42, "name": "Nome Da Serie", "poster_path": None,
            "original_language": "en", "overview": "",
        }]}
        resposta_detalhes = mock.Mock(status_code=200)
        resposta_detalhes.json.return_value = {
            "number_of_seasons": 3, "number_of_episodes": 24,
        }
        sessao.get.side_effect = [
            resposta_busca, resposta_detalhes,  # episódio 1: busca + detalhes
            resposta_busca,                     # episódio 2: só busca
        ]

        tmdb.lookup_cached("Nome.Da.Serie.S01E01.mkv", api_key="chave", session=sessao)
        tmdb.lookup_cached("Nome.Da.Serie.S01E02.mkv", api_key="chave", session=sessao)

        self.assertEqual(sessao.get.call_count, 3)


if __name__ == "__main__":
    unittest.main()
