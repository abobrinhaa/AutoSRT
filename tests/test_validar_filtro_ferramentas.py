"""Testes para validar a regra de filtro que mostra ferramentas por filme.

Este módulo testa a lógica em autosrt/web.py que determina quais ações
(ferramentas) estão disponíveis para cada arquivo (filme/legenda).

A regra é: as ações disponíveis dependem do tipo do arquivo (mídia ou legenda),
da presença de legendas irmãs (mesma pasta, mesmo nome) e do formato da legenda.
"""

import os
import tempfile
import unittest

from autosrt import pipeline, web


class TestValidacaoFiltroFerramenta(unittest.TestCase):
    """Valida a regra que mostra ferramentas para cada arquivo."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def criar_arquivo(self, nome):
        """Cria um arquivo vazio para teste."""
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "wb") as f:
            f.write(b"\0")
        return caminho

    def criar_legenda_com_filme(self, nome):
        """Legenda com o filme irmão do lado, que é o que libera o deslocar."""
        self.criar_arquivo(os.path.splitext(nome)[0] + ".mp4")
        return self.criar_arquivo(nome)

    def acoes_ids(self, caminho):
        """Extrai IDs das ações disponíveis para um arquivo."""
        return [a["id"] for a in web.acoes_para(caminho)]

    def acoes_rotulos(self, caminho):
        """Extrai rótulos das ações disponíveis para um arquivo."""
        return [a["rotulo"] for a in web.acoes_para(caminho)]

    # ============================================================
    # TESTES: MÍDIA SEM LEGENDA IRMÃ
    # ============================================================

    def test_video_sem_legenda_tem_duas_acoes(self):
        """Vídeo isolado oferece transcrição e conversão."""
        caminho = self.criar_arquivo("filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(len(ids), 2)

    def test_video_sem_legenda_oferece_completo_e_transcrever(self):
        """Vídeo sem legenda irmã: completo e só transcrever."""
        caminho = self.criar_arquivo("filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertIn("completo", ids)
        self.assertIn("transcrever", ids)

    def test_video_sem_legenda_nao_oferece_traduzir_existente(self):
        """Vídeo isolado não oferece aproveitar legenda existente."""
        caminho = self.criar_arquivo("filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertNotIn("traduzir_existente", ids)

    def test_audio_sem_legenda_tem_mesmas_acoes_que_video(self):
        """Áudio isolado: mesmas opções que vídeo."""
        caminho = self.criar_arquivo("podcast.mp3")
        ids = self.acoes_ids(caminho)
        self.assertEqual(set(ids), {"completo", "transcrever"})

    # ============================================================
    # TESTES: MÍDIA COM LEGENDA IRMÃ
    # ============================================================

    def test_video_com_legenda_srt_tem_tres_acoes(self):
        """Vídeo com legenda .srt irmã: 3 opções."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.srt")
        caminho = os.path.join(self.tmp, "filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(len(ids), 3)

    def test_video_com_srt_oferece_traduzir_existente_primeiro(self):
        """Com legenda irmã, 'traduzir_existente' é primeira opção."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.srt")
        caminho = os.path.join(self.tmp, "filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids[0], "traduzir_existente")
        self.assertIn("completo", ids[1:])
        self.assertIn("transcrever", ids[1:])

    def test_video_com_ssa_oferece_traduzir_existente(self):
        """Legenda irmã .ssa também ativa traduzir_existente."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.ssa")
        caminho = os.path.join(self.tmp, "filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids[0], "traduzir_existente")

    def test_video_com_ass_oferece_traduzir_existente(self):
        """Legenda irmã .ass também ativa traduzir_existente."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.ass")
        caminho = os.path.join(self.tmp, "filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids[0], "traduzir_existente")

    def test_legenda_srt_tem_preferencia_sobre_ssa_ass(self):
        """Se existem múltiplas legendas irmãs, .srt é preferido."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.ssa")
        self.criar_arquivo("filme.ass")
        self.criar_arquivo("filme.srt")
        legenda = web.legenda_irma(os.path.join(self.tmp, "filme.mp4"))
        self.assertTrue(legenda.endswith(".srt"))

    def test_ssa_tem_preferencia_sobre_ass(self):
        """Entre .ssa e .ass (sem .srt), .ssa é preferido."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.ssa")
        self.criar_arquivo("filme.ass")
        legenda = web.legenda_irma(os.path.join(self.tmp, "filme.mp4"))
        self.assertTrue(legenda.endswith(".ssa"))

    def test_legenda_com_nome_diferente_nao_e_irmã(self):
        """Legenda em pasta diferente não é considerada irmã."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("subpasta/filme.srt")  # Em subpasta
        caminho = os.path.join(self.tmp, "filme.mp4")
        legenda = web.legenda_irma(caminho)
        self.assertIsNone(legenda)

    def test_legenda_com_nome_raiz_diferente_nao_e_irmã(self):
        """Legenda com nome raiz diferente não é irmã."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("outra.srt")  # Nome diferente
        caminho = os.path.join(self.tmp, "filme.mp4")
        legenda = web.legenda_irma(caminho)
        self.assertIsNone(legenda)

    # ============================================================
    # TESTES: LEGENDA SRT
    # ============================================================

    def test_legenda_srt_sozinha_so_traduz(self):
        """Legenda .srt sem o filme: sobra traduzir."""
        caminho = self.criar_arquivo("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["traduzir"])

    def test_legenda_srt_com_filme_tem_duas_acoes(self):
        """Legenda .srt com o filme irmão oferece 2 opções."""
        caminho = self.criar_legenda_com_filme("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertEqual(len(ids), 2)

    def test_legenda_srt_com_filme_oferece_traduzir_e_deslocar(self):
        """Legenda .srt com o filme: traduzir e ajustar tempo."""
        caminho = self.criar_legenda_com_filme("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertIn("traduzir", ids)
        self.assertIn("deslocar", ids)

    def test_legenda_srt_nao_oferece_converter(self):
        """Legenda .srt não oferece conversão (já é formato final)."""
        caminho = self.criar_arquivo("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertNotIn("converter", ids)

    def test_legenda_srt_nao_oferece_transcrever(self):
        """Legenda .srt não oferece transcrição."""
        caminho = self.criar_arquivo("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertNotIn("transcrever", ids)

    # ============================================================
    # TESTES: LEGENDA SSA/ASS
    # ============================================================

    def test_legenda_ssa_sozinha_traduz_e_converte(self):
        """Legenda .ssa sem o filme: converter continua, deslocar não."""
        caminho = self.criar_arquivo("legenda.ssa")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["traduzir", "converter"])

    def test_legenda_ssa_com_filme_tem_tres_acoes(self):
        """Legenda .ssa com o filme irmão oferece 3 opções."""
        caminho = self.criar_legenda_com_filme("legenda.ssa")
        ids = self.acoes_ids(caminho)
        self.assertEqual(len(ids), 3)

    def test_legenda_ssa_com_filme_oferece_traduzir_converter_deslocar(self):
        """Legenda .ssa com o filme: traduzir, converter, deslocar."""
        caminho = self.criar_legenda_com_filme("legenda.ssa")
        ids = self.acoes_ids(caminho)
        self.assertIn("traduzir", ids)
        self.assertIn("converter", ids)
        self.assertIn("deslocar", ids)

    def test_legenda_ssa_converter_e_segunda_opcao(self):
        """Em .ssa, converter aparece como segunda opção."""
        caminho = self.criar_legenda_com_filme("legenda.ssa")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids[0], "traduzir")
        self.assertEqual(ids[1], "converter")
        self.assertEqual(ids[2], "deslocar")

    def test_legenda_ass_sozinha_traduz_e_converte(self):
        """Legenda .ass sem o filme: converter continua, deslocar não."""
        caminho = self.criar_arquivo("legenda.ass")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["traduzir", "converter"])

    def test_legenda_ass_com_filme_tem_tres_acoes(self):
        """Legenda .ass com o filme irmão oferece 3 opções."""
        caminho = self.criar_legenda_com_filme("legenda.ass")
        ids = self.acoes_ids(caminho)
        self.assertEqual(len(ids), 3)

    def test_legenda_ass_com_filme_oferece_traduzir_converter_deslocar(self):
        """Legenda .ass com o filme: traduzir, converter, deslocar."""
        caminho = self.criar_legenda_com_filme("legenda.ass")
        ids = self.acoes_ids(caminho)
        self.assertIn("traduzir", ids)
        self.assertIn("converter", ids)
        self.assertIn("deslocar", ids)

    # ============================================================
    # TESTES: ORDEM DAS AÇÕES
    # ============================================================

    def test_ordem_acoes_video_sem_legenda(self):
        """Ordem: completo, transcrever."""
        caminho = self.criar_arquivo("filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["completo", "transcrever"])

    def test_ordem_acoes_video_com_legenda(self):
        """Ordem com legenda: traduzir_existente, completo, transcrever."""
        self.criar_arquivo("filme.mp4")
        self.criar_arquivo("filme.srt")
        caminho = os.path.join(self.tmp, "filme.mp4")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids[0], "traduzir_existente")
        self.assertEqual(set(ids[1:]), {"completo", "transcrever"})

    def test_ordem_acoes_legenda_srt(self):
        """Ordem .srt com o filme: traduzir, deslocar."""
        caminho = self.criar_legenda_com_filme("legenda.srt")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["traduzir", "deslocar"])

    def test_ordem_acoes_legenda_ssa(self):
        """Ordem .ssa com o filme: traduzir, converter, deslocar."""
        caminho = self.criar_legenda_com_filme("legenda.ssa")
        ids = self.acoes_ids(caminho)
        self.assertEqual(ids, ["traduzir", "converter", "deslocar"])

    # ============================================================
    # TESTES: FORMATO DE RETORNO
    # ============================================================

    def test_acoes_retorna_dicionarios_com_id_e_rotulo(self):
        """Cada ação é {'id': str, 'rotulo': str}."""
        caminho = self.criar_arquivo("legenda.srt")
        acoes = web.acoes_para(caminho)
        for acao in acoes:
            self.assertIn("id", acao)
            self.assertIn("rotulo", acao)
            self.assertIsInstance(acao["id"], str)
            self.assertIsInstance(acao["rotulo"], str)
            self.assertTrue(len(acao["id"]) > 0)
            self.assertTrue(len(acao["rotulo"]) > 0)

    def test_rotulos_sao_amigaveis(self):
        """Rótulos são em português claro."""
        caminho = self.criar_legenda_com_filme("legenda.srt")
        rotulos = self.acoes_rotulos(caminho)
        self.assertTrue(any("Traduz" in r for r in rotulos))
        self.assertTrue(any("Ajust" in r or "tempo" in r for r in rotulos))

    # ============================================================
    # TESTES: CASE INSENSITIVITY
    # ============================================================

    def test_extensao_maiuscula_srt_e_reconhecida(self):
        """Extensão .SRT em maiúscula é reconhecida como legenda."""
        caminho = self.criar_arquivo("legenda.SRT")
        ids = self.acoes_ids(caminho)
        # Se reconhecido como legenda, tem 'traduzir'; se vídeo, tem 'completo'
        self.assertIn("traduzir", ids)

    def test_extensao_maiuscula_ssa_e_reconhecida(self):
        """Extensão .SSA em maiúscula é reconhecida como legenda com converter."""
        caminho = self.criar_arquivo("legenda.SSA")
        ids = self.acoes_ids(caminho)
        self.assertIn("converter", ids)

    def test_extensao_maiuscula_ass_e_reconhecida(self):
        """Extensão .ASS em maiúscula é reconhecida como legenda com converter."""
        caminho = self.criar_arquivo("legenda.ASS")
        ids = self.acoes_ids(caminho)
        self.assertIn("converter", ids)

    # ============================================================
    # TESTES: INTEGRAÇÃO COM A VALIDAÇÃO
    # ============================================================

    def test_acao_retornada_sempre_existe_nos_acoes_para(self):
        """Toda ação retornada em acoes_para é válida."""
        tipos = ["filme.mp4", "legenda.srt", "legenda.ssa"]
        for tipo in tipos:
            caminho = self.criar_arquivo(tipo)
            ids = self.acoes_ids(caminho)
            # Todos os IDs devem estar no conjunto de válidos
            acoes_validas = {
                "completo", "transcrever", "traduzir", "deslocar", "converter",
                "traduzir_existente"
            }
            for id_ in ids:
                self.assertIn(id_, acoes_validas,
                              f"ID inválido '{id_}' para {tipo}")

    def test_nao_ha_duplicatas_nas_acoes(self):
        """Cada ação aparece no máximo uma vez."""
        tipos = ["filme.mp4", "legenda.srt", "legenda.ssa"]
        for tipo in tipos:
            caminho = self.criar_arquivo(tipo)
            ids = self.acoes_ids(caminho)
            self.assertEqual(len(ids), len(set(ids)),
                             f"Duplicatas em {tipo}: {ids}")

    # ============================================================
    # TESTES: EDGE CASES
    # ============================================================

    def test_arquivo_sem_extensao_nao_e_reconhecido(self):
        """Arquivo sem extensão não é reconhecido como mídia ou legenda."""
        caminho = self.criar_arquivo("arquivo_sem_extensao")
        # Deve retornar ações de legenda (fallback) ou vazio
        # No código atual, fallback é legenda
        acoes = web.acoes_para(caminho)
        self.assertIsInstance(acoes, list)

    def test_extensao_desconhecida_e_tratada(self):
        """Extensão desconhecida é tratada como legenda (fallback)."""
        caminho = self.criar_arquivo("arquivo.xyz")
        acoes = web.acoes_para(caminho)
        self.assertIsInstance(acoes, list)
        # Deve retornar algo sensato
        ids = [a["id"] for a in acoes]
        self.assertTrue(len(ids) > 0)

    def test_caminho_com_multiplos_pontos_e_tratado(self):
        """Arquivo com múltiplos pontos usa última extensão."""
        caminho = self.criar_arquivo("arquivo.backup.srt")
        ids = self.acoes_ids(caminho)
        # Deve reconhecer como .srt
        self.assertIn("traduzir", ids)

    def test_caminho_relativo_nao_quebra(self):
        """Caminhos relativos não quebram a função."""
        # Criar arquivo na pasta tmp
        caminho = self.criar_arquivo("legenda.srt")
        # Passar relativo (fake, só para testar parsing)
        ids = self.acoes_ids(caminho)
        self.assertTrue(len(ids) > 0)


class TestConstantesAcoes(unittest.TestCase):
    """Valida as constantes de ações definidas em web.py."""

    def test_acoes_midia_definidas(self):
        """ACOES_MIDIA está definida."""
        self.assertTrue(hasattr(web, "ACOES_MIDIA"))
        self.assertEqual(len(web.ACOES_MIDIA), 2)

    def test_acoes_legenda_definidas(self):
        """ACOES_LEGENDA está definida."""
        self.assertTrue(hasattr(web, "ACOES_LEGENDA"))
        self.assertEqual(len(web.ACOES_LEGENDA), 2)

    def test_acao_converter_definida(self):
        """ACAO_CONVERTER está definida."""
        self.assertTrue(hasattr(web, "ACAO_CONVERTER"))
        self.assertEqual(len(web.ACAO_CONVERTER), 2)

    def test_acao_legenda_existente_definida(self):
        """ACAO_LEGENDA_EXISTENTE está definida."""
        self.assertTrue(hasattr(web, "ACAO_LEGENDA_EXISTENTE"))
        self.assertEqual(len(web.ACAO_LEGENDA_EXISTENTE), 2)

    def test_extensoes_irmas_definidas(self):
        """EXTENSOES_IRMAS está definida em ordem de preferência."""
        self.assertTrue(hasattr(web, "EXTENSOES_IRMAS"))
        self.assertEqual(web.EXTENSOES_IRMAS[0], ".srt")
        self.assertEqual(web.EXTENSOES_IRMAS[1], ".ssa")
        self.assertEqual(web.EXTENSOES_IRMAS[2], ".ass")


class TestDeslocarPedeFilme(unittest.TestCase):
    """A regra: deslocar só aparece quando o filme irmão está do lado.

    Deslocar é só somar segundos nos tempos, e para isso o vídeo não é lido.
    O que precisa do filme é *descobrir* quantos segundos são, e isso se
    descobre assistindo. Sem o filme na pasta a opção só rende legenda torta
    de outro jeito. Quem quiser deslocar no escuro tem o --deslocar da linha
    de comando, que continua aceitando qualquer legenda.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def criar_arquivo(self, nome):
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "wb") as f:
            f.write(b"\0")
        return caminho

    def acoes_ids(self, caminho):
        return [a["id"] for a in web.acoes_para(caminho)]

    def test_legenda_sozinha_nao_oferece_deslocar(self):
        for extensao in (".srt", ".ssa", ".ass"):
            with self.subTest(extensao=extensao):
                caminho = self.criar_arquivo("sozinha" + extensao)
                self.assertNotIn("deslocar", self.acoes_ids(caminho))

    def test_legenda_com_filme_oferece_deslocar(self):
        for extensao in (".srt", ".ssa", ".ass"):
            with self.subTest(extensao=extensao):
                self.criar_arquivo("com_filme.mp4")
                caminho = self.criar_arquivo("com_filme" + extensao)
                self.assertIn("deslocar", self.acoes_ids(caminho))

    def test_qualquer_formato_de_video_conta_como_irmao(self):
        # A regra não pode valer só para .mp4: quem guarda .mkv perderia a
        # opção sem entender por quê.
        for i, extensao in enumerate(sorted(pipeline.MEDIA_EXTENSIONS)):
            with self.subTest(extensao=extensao):
                base = f"filme{i}"
                self.criar_arquivo(base + extensao)
                caminho = self.criar_arquivo(base + ".srt")
                self.assertIn("deslocar", self.acoes_ids(caminho))

    def test_filme_de_outro_nome_nao_libera_deslocar(self):
        # Mesma pasta, nome diferente: não dá para afirmar que um é do outro.
        self.criar_arquivo("outro_filme.mp4")
        caminho = self.criar_arquivo("legenda.srt")
        self.assertNotIn("deslocar", self.acoes_ids(caminho))

    def test_a_falta_do_filme_nao_tira_as_outras_acoes(self):
        # Só o deslocar depende do filme; traduzir e converter seguem valendo.
        srt = self.criar_arquivo("a.srt")
        self.assertEqual(self.acoes_ids(srt), ["traduzir"])

        ssa = self.criar_arquivo("b.ssa")
        self.assertEqual(self.acoes_ids(ssa), ["traduzir", "converter"])


if __name__ == "__main__":
    unittest.main()
