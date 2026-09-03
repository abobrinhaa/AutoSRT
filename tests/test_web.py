import io
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

from autosrt import jobs, pipeline, web

# Os testes de erro provocam falhas de propósito; o traceback delas no log
# só poluiria a saída da suíte.
logging.getLogger("autosrt.jobs").setLevel(logging.CRITICAL)

SRT = """1
00:00:01,000 --> 00:00:03,000
This is an English sentence for the test.
"""


class EchoLLM:
    def complete(self, system, user):
        from autosrt import llm_translate
        return "\n".join(f"<{n}>PT:{c.strip()}</{n}>"
                         for n, c in llm_translate.BLOCK_RE.findall(user))


class BaseWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.app = web.create_app(media_dir=self.tmp)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def escrever(self, nome, conteudo=SRT):
        caminho = os.path.join(self.tmp, nome)
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(conteudo)
        return caminho

    def tocar(self, nome):
        caminho = os.path.join(self.tmp, nome)
        with open(caminho, "wb") as handle:
            handle.write(b"\0")
        return caminho

    def esperar(self, job_id, limite=5):
        fim = time.time() + limite
        while time.time() < fim:
            dados = self.client.get(f"/api/trabalho/{job_id}").get_json()
            if dados["pronto"]:
                return dados
            time.sleep(0.05)
        self.fail("o trabalho não terminou a tempo")


class TestPagina(BaseWeb):
    def test_pagina_abre(self):
        resposta = self.client.get("/")
        self.assertEqual(resposta.status_code, 200)
        self.assertIn("AutoSRT", resposta.get_data(as_text=True))

    def test_pagina_nao_puxa_recurso_externo(self):
        # Sem internet na rede local, um CDN deixaria a página quebrada. O que
        # importa é o que a página CARREGA - endereço em placeholder de campo
        # é só texto e não busca nada.
        html = self.client.get("/").get_data(as_text=True)
        for padrao in ("<script src=", "<link ", "@import", "url(http",
                       'src="http', "src='http"):
            self.assertNotIn(padrao, html, f"a página carrega algo externo: {padrao}")

    def test_filtro_do_seletor_reflete_o_que_o_servidor_aceita(self):
        # Filtro desatualizado esconde arquivo válido no explorador sem dar
        # nenhuma pista de que ele existe.
        html = self.client.get("/").get_data(as_text=True)
        aceitas = re.search(r'id="arquivo" accept="([^"]+)"', html).group(1)
        do_seletor = set(aceitas.split(","))
        do_servidor = pipeline.MEDIA_EXTENSIONS | pipeline.SUBTITLE_EXTENSIONS
        self.assertEqual(do_seletor, do_servidor)

    def test_seletor_aceita_mais_de_um_arquivo(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("multiple", html)

    def test_nenhum_marcador_de_template_sobra_na_pagina(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn("{{", html)

    def test_requisicoes_do_script_sao_todas_relativas(self):
        html = self.client.get("/").get_data(as_text=True)
        for chamada in re.findall(r"fetch\(\s*['\"]([^'\"]+)", html):
            self.assertTrue(chamada.startswith("/"),
                            f"fetch para endereço não relativo: {chamada}")

    def test_exemplo_do_modelo_reage_ao_modo(self):
        # Regressão: escolher "Local" trocava a URL mas o exemplo do campo
        # Modelo continuava sugerindo um slug do OpenRouter.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("marcarModoAtivo", html)
        self.assertIn('data-exemplo-openrouter="deepseek/deepseek-chat"', html)
        self.assertIn('data-exemplo-local="llama3.1"', html)

    def test_barra_de_acoes_esconde_de_verdade_com_pasta_vazia(self):
        # Mesma classe de bug do campo-chave: "display: flex" de autor
        # vencia o "display: none" do atributo hidden vindo do navegador --
        # verificado com um navegador de verdade que a barra de ações
        # continuava visível numa pasta vazia até este override existir.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn(".barra-acoes[hidden] { display: none; }", html)

    def test_campo_chave_some_no_modo_local(self):
        # Regressão: o campo "Chave" continuava visível (só com rótulo
        # diferente) no modo Local, onde normalmente não faz sentido nenhum
        # -- e um "hidden" via JS não bastava, porque a regra
        # "#config label { display: grid }" vencia o hidden sem um
        # override explícito (verificado num navegador de verdade).
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="campo-chave"', html)
        self.assertIn("campo-chave').hidden = local", html)
        self.assertIn("#config label[hidden] { display: none; }", html)

    def test_selo_de_chave_diferencia_o_modo(self):
        # Regressão: o selo "chave configurada"/"sem chave" do topo era o
        # mesmo texto/estilo nos dois modos, mesmo chave sendo opcional
        # em modo local -- lia como aviso onde não era.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("chave opcional (modo local)", html)
        self.assertIn("chave configurada (opcional aqui)", html)

    def test_modelo_e_limpo_ao_trocar_para_modo_incompativel(self):
        # Regressão: trocar para "Local" mantinha um slug do OpenRouter
        # (com barra, ex. deepseek/deepseek-chat) no campo Modelo, que não
        # existe num servidor local -- risco de salvar um modelo inválido.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("function limparModeloSeIncompativel", html)
        self.assertIn("limparModeloSeIncompativel(true)", html)
        self.assertIn("limparModeloSeIncompativel(false)", html)

    def test_palavra_api_some_no_modo_local(self):
        # Regressão: "Endereço da API" continuava falando de "API" mesmo
        # depois de trocar para um servidor local, o que não fazia sentido
        # para quem está lendo.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="rotulo-endereco"', html)
        self.assertIn("Endereço do servidor", html)

    def test_salvar_recusa_modelo_vazio_fora_do_openrouter(self):
        # Regressão: salvar com o campo Modelo vazio (ex.: só trocou para
        # "Local" e não digitou nada) apagava o modelo configurado, e a
        # tela seguinte mostrava o modelo padrão do OpenRouter como se
        # fosse o que estava configurado -- parecia que o nome do Ollama
        # tinha "sumido" e virado outro modelo sozinho.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("baseUrl !== configAtual.openrouter_base_url", html)
        self.assertIn("Informe um modelo antes de salvar", html)

    def test_modo_ativo_tem_marcacao_textual_alem_da_cor(self):
        # Regressão: só a cor do botão indicava qual modo estava ativo --
        # sem contraste suficiente (ou lendo em preto e branco), não dava
        # para saber qual dos dois estava selecionado.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("marca-selecionado", html)
        self.assertIn("selecionado", html)

    def test_salvar_mostra_confirmacao_de_sucesso(self):
        # Regressão: salvar sem erro nem aviso não dava nenhum sinal de que
        # a gravação aconteceu -- parecia que o botão não tinha feito nada.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="mensagem-salvar"', html)
        self.assertIn("function mostrarSucesso", html)
        self.assertIn("mostrarSucesso('mensagem-salvar')", html)
        self.assertIn("Configuração salva.", html)

    def test_painel_de_transcricao_tem_campos_de_vad(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="config-transcricao"', html)
        self.assertIn('id="vad_threshold"', html)
        self.assertIn('id="vad_min_silence_ms"', html)
        self.assertIn('id="estado-vad"', html)
        self.assertIn("mostrarSucesso('mensagem-salvar-vad')", html)

    def test_painel_de_transcricao_tem_campos_de_anti_alucinacao(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="condition_on_previous_text"', html)
        self.assertIn('id="hallucination_silence_threshold"', html)

    def test_botao_de_configuracao_nao_mora_no_painel_da_automacao(self):
        # Configuração de tradução e transcrição não tem nada a ver com o
        # "processar ao enviar": o botão fica no cabeçalho, ao lado do tema,
        # e não dentro do painel do toggle.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="abrir-config"', html)
        self.assertIn('aria-haspopup="dialog"', html)
        self.assertIn('aria-controls="painel-config"', html)
        botao = html.index('id="abrir-config"')
        dock = html.index('id="dock-auto"')
        self.assertLess(botao, dock,
                        "o botão de configuração voltou para dentro do toggle")
        self.assertLess(html.index('class="cabecalho"'), botao)

    def test_cada_secao_de_configuracao_tem_a_sua_aba(self):
        # Regressão: as duas seções foram empilhadas numa tela só dentro do
        # diálogo -- viravam uma rolagem sem fim. Cada uma é uma aba.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('role="tablist"', html)
        self.assertIn('id="aba-traducao"', html)
        self.assertIn('id="aba-transcricao"', html)
        self.assertEqual(html.count('role="tab"'), 2)
        self.assertEqual(html.count('role="tabpanel"'), 2)
        self.assertIn('aria-controls="config"', html)
        self.assertIn('aria-controls="config-transcricao"', html)

    def test_so_uma_aba_aparece_por_vez(self):
        # A seção escondida precisa sumir de verdade: "display: grid" de autor
        # vence o hidden do navegador -- mesmo bug já visto no campo-chave.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("[role=tabpanel][hidden] { display: none; }", html)
        self.assertIn("function mostrarAba", html)
        painel = html.index('id="config-transcricao"')
        self.assertIn("hidden", html[painel:painel + 200],
                      "a segunda aba começa aberta junto com a primeira")

    def test_abas_andam_com_as_setas_do_teclado(self):
        # Padrão APG para abas: seta anda entre elas, só a ativa fica no Tab.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("ArrowRight", html)
        self.assertIn("ArrowLeft", html)
        self.assertIn("tabIndex", html)

    def test_abas_de_configuracao_moram_dentro_do_dialogo(self):
        html = self.client.get("/").get_data(as_text=True)
        abre = html.index('<dialog id="painel-config"')
        fecha = html.index("</dialog>")
        for painel in ('id="config"', 'id="config-transcricao"'):
            self.assertTrue(abre < html.index(painel) < fecha,
                            f"{painel} ficou fora do diálogo")

    def test_dialogo_de_configuracao_da_saida(self):
        # Heurística 3 (controle e liberdade): botão de fechar explícito,
        # além do Esc que o <dialog> nativo já dá, e clique no fundo.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="fechar-config"', html)
        self.assertIn("dialogoConfig.close()", html)
        self.assertIn("showModal()", html)

    def test_falta_de_chave_avisa_no_botao_que_abre_a_configuracao(self):
        # Antes o painel sem chave se abria sozinho no fim da página. Dentro
        # do diálogo isso ficaria invisível -- o aviso migra para o botão.
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="aviso-config"', html)
        self.assertIn("function avisarChaveFaltando", html)


class TestListagem(BaseWeb):
    def test_lista_videos_e_legendas(self):
        self.escrever("a.srt")
        self.tocar("b.mkv")
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual({i["nome"] for i in itens}, {"a.srt", "b.mkv"})

    def test_classifica_o_tipo(self):
        self.escrever("a.srt")
        self.tocar("b.mkv")
        tipos = {i["nome"]: i["tipo"] for i in self.client.get("/api/arquivos").get_json()}
        self.assertEqual(tipos, {"a.srt": "legenda", "b.mkv": "video"})

    def test_esconde_backup_e_original(self):
        self.escrever("filme.srt")
        self.escrever("filme_backup.srt")
        self.escrever("filme.original.srt")
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual([i["nome"] for i in itens], ["filme.srt"])

    def test_ignora_arquivo_de_outro_tipo(self):
        self.tocar("nota.txt")
        self.assertEqual(self.client.get("/api/arquivos").get_json(), [])

    def test_encontra_em_subpasta(self):
        self.escrever(os.path.join("classicos", "a.srt"))
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual(itens[0]["nome"], os.path.join("classicos", "a.srt"))

    def test_esconde_a_pasta_de_originais(self):
        self.escrever("filme.srt")
        self.escrever(os.path.join(pipeline.ORIGINALS_DIRNAME, "filme.srt"))
        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual([i["nome"] for i in itens], ["filme.srt"])


class TestReconhecimentoTMDB(BaseWeb):
    """O selo de filme na listagem é um extra: nunca deve travar nem exigir
    rede quando não há chave configurada, e some sozinho quando o TMDB não
    reconhece nada."""

    def setUp(self):
        super().setUp()
        from autosrt import config, tmdb
        self.tmdb = tmdb
        self._chave_original = config.get_tmdb_api_key
        self._lookup_original = tmdb.lookup_cached
        tmdb.limpar_cache()

    def tearDown(self):
        from autosrt import config
        config.get_tmdb_api_key = self._chave_original
        self.tmdb.lookup_cached = self._lookup_original
        self.tmdb.limpar_cache()

    def test_sem_chave_nao_tem_campo_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: None
        self.tocar("Duna.2021.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])

    def test_com_chave_e_reconhecimento_traz_o_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: {
            "titulo": "Duna", "ano": 2021,
            "poster": "https://image.tmdb.org/t/p/w154/x.jpg",
            "idioma_original": "en", "idioma_original_nome": "Inglês",
        }
        self.tocar("Duna.2021.1080p.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertEqual(itens[0]["filme"]["titulo"], "Duna")
        self.assertEqual(itens[0]["filme"]["ano"], 2021)

    def test_legenda_nunca_recebe_campo_filme(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: {"titulo": "Duna", "ano": 2021,
                                                     "poster": None,
                                                     "idioma_original": "en",
                                                     "idioma_original_nome": "Inglês"}
        self.escrever("legenda.srt")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])

    def test_sem_correspondencia_o_selo_nao_aparece(self):
        from autosrt import config
        config.get_tmdb_api_key = lambda: "chave-teste"
        self.tmdb.lookup_cached = lambda *a, **kw: None
        self.tocar("video_qualquer.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        self.assertIsNone(itens[0]["filme"])


class TestSeguranca(BaseWeb):
    def test_recusa_caminho_para_fora_da_pasta(self):
        resposta = self.client.post("/api/processar",
                                    json={"arquivo": "../../etc/passwd"})
        self.assertEqual(resposta.status_code, 400)

    def test_recusa_caminho_absoluto(self):
        resposta = self.client.post("/api/processar",
                                    json={"arquivo": "/etc/passwd"})
        self.assertIn(resposta.status_code, (400, 404))

    def test_recusa_nome_vazio(self):
        self.assertEqual(
            self.client.post("/api/processar", json={"arquivo": ""}).status_code,
            400)

    def test_arquivo_inexistente(self):
        resposta = self.client.post("/api/processar", json={"arquivo": "nada.srt"})
        self.assertEqual(resposta.status_code, 404)


class TestEnvio(BaseWeb):
    def enviar(self, nome, conteudo=SRT):
        return self.client.post(
            "/api/enviar",
            data={"arquivo": (io.BytesIO(conteudo.encode("utf-8")), nome)},
            content_type="multipart/form-data")

    def enviar_varios(self, *pares):
        return self.client.post(
            "/api/enviar",
            data={"arquivo": [(io.BytesIO(c.encode("utf-8")), n)
                              for n, c in pares]},
            content_type="multipart/form-data")

    def test_aceita_legenda(self):
        resposta = self.enviar("filme.srt")
        self.assertEqual(resposta.status_code, 201)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.srt")))

    def test_aceita_video(self):
        # Em rede local, enviar filme pelo navegador leva um ou dois minutos,
        # irrelevante perto do tempo de transcrição.
        resposta = self.enviar("filme.mkv", "conteudo")
        self.assertEqual(resposta.status_code, 201)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.mkv")))

    def test_aceita_filme_e_legenda_de_uma_vez(self):
        resposta = self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        self.assertEqual(resposta.status_code, 201)
        self.assertEqual(sorted(resposta.get_json()["guardados"]),
                         ["filme.mkv", "filme.srt"])
        for nome in ("filme.mkv", "filme.srt"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, nome)))

    def test_o_par_enviado_junto_e_reconhecido(self):
        self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        itens = self.client.get("/api/arquivos").get_json()
        video = next(i for i in itens if i["nome"] == "filme.mkv")
        self.assertTrue(video["tem_legenda"])
        self.assertEqual(video["acoes"][0]["id"], "traduzir_existente")

    def test_enviar_nao_processa_nada(self):
        # Enfileirar no envio traduziria a legenda sozinha antes de o vídeo
        # chegar, desperdiçando justamente o pareamento.
        self.enviar_varios(("filme.mkv", "video"), ("filme.srt", SRT))
        self.assertEqual(self.client.get("/api/trabalhos").get_json(), [])

    def test_arquivo_invalido_no_meio_nao_impede_os_validos(self):
        resposta = self.enviar_varios(("filme.srt", SRT), ("nota.txt", "x"))
        self.assertEqual(resposta.status_code, 201)
        dados = resposta.get_json()
        self.assertEqual(dados["guardados"], ["filme.srt"])
        self.assertEqual(len(dados["recusados"]), 1)

    def test_recusa_tipo_desconhecido(self):
        resposta = self.enviar("planilha.xlsx", "conteudo")
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("legenda", resposta.get_json()["erro"])

    def test_limite_de_tamanho_explica_a_alternativa(self):
        app = web.create_app(media_dir=self.tmp, max_upload_gb=0.000001)
        app.config["TESTING"] = True
        cliente = app.test_client()
        resposta = cliente.post(
            "/api/enviar",
            data={"arquivo": (io.BytesIO(b"x" * 5000), "grande.srt")},
            content_type="multipart/form-data")
        self.assertEqual(resposta.status_code, 413)
        self.assertIn("pasta do servidor", resposta.get_json()["erro"])

    def test_recusa_sem_arquivo(self):
        resposta = self.client.post("/api/enviar", data={},
                                    content_type="multipart/form-data")
        self.assertEqual(resposta.status_code, 400)

    def test_nome_com_caminho_e_reduzido(self):
        self.enviar("../../fora.srt")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "fora.srt")))


class TestAutomatico(BaseWeb):
    """O "processar ao enviar": filme entra na fila sozinho, legenda não.

    Roda contra o config.json de verdade (redirecionado para a pasta do
    teste), porque metade do valor está justamente em o estado sobreviver e
    valer para todo mundo na rede -- não é uma caixinha do navegador.
    """

    def setUp(self):
        super().setUp()
        from autosrt import config
        self.config = config
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self._env = os.environ.pop("AUTOSRT_AUTO_PROCESSAR", None)
        # O operário pega o trabalho de verdade; sem isto ele tentaria
        # chamar o Whisper num .mkv de um byte e o teste dependeria de GPU.
        self._patch = mock.patch.object(
            pipeline, "process_media", side_effect=RuntimeError("sem whisper"))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.config.app_directory = self._app_dir
        if self._env is None:
            os.environ.pop("AUTOSRT_AUTO_PROCESSAR", None)
        else:
            os.environ["AUTOSRT_AUTO_PROCESSAR"] = self._env

    def ligar(self, ligado=True):
        resposta = self.client.post(
            "/api/config", json={"auto_processar": "true" if ligado else "false"})
        self.assertEqual(resposta.status_code, 200)

    def enviar(self, *pares):
        return self.client.post(
            "/api/enviar",
            data={"arquivo": [(io.BytesIO(c.encode("utf-8")), n)
                              for n, c in pares]},
            content_type="multipart/form-data")

    def trabalhos(self):
        return self.client.get("/api/trabalhos").get_json()

    def test_filme_enviado_entra_na_fila_sozinho(self):
        self.ligar()
        resposta = self.enviar(("filme.mkv", "video"))
        self.assertEqual(resposta.status_code, 201)
        self.assertTrue(resposta.get_json()["automatico"])
        self.assertEqual(len(resposta.get_json()["enfileirados"]), 1)

        trabalhos = self.trabalhos()
        self.assertEqual(len(trabalhos), 1)
        self.assertEqual(trabalhos[0]["nome"], "filme.mkv")

    def test_o_filme_vai_para_a_corrente_inteira(self):
        # "Processar ao enviar" quer dizer transcrever E traduzir: parar na
        # transcrição deixaria o segundo passo que o botão veio eliminar.
        self.ligar()
        self.enviar(("filme.mkv", "video"))
        self.assertEqual(self.trabalhos()[0]["detalhes"]["acao"], "completo")

    def test_legenda_sozinha_fica_fora_da_regra(self):
        self.ligar()
        resposta = self.enviar(("filme.srt", SRT))
        self.assertEqual(resposta.get_json()["enfileirados"], [])
        self.assertEqual(self.trabalhos(), [])
        # Guardada, sim: ela só não vira trabalho sem alguém pedir.
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "filme.srt")))

    def test_legenda_junto_do_filme_nao_vira_trabalho(self):
        self.ligar()
        self.enviar(("filme.mkv", "video"), ("filme.srt", SRT))
        trabalhos = self.trabalhos()
        self.assertEqual([t["nome"] for t in trabalhos], ["filme.mkv"])

    def test_legenda_do_lado_nao_desvia_o_filme(self):
        # A legenda irmã é ignorada de propósito: o filme chega sem legenda
        # boa, e o pedido é transcrever de novo, não reaproveitar.
        self.escrever("filme.srt")
        self.ligar()
        self.enviar(("filme.mkv", "video"))
        self.assertEqual(self.trabalhos()[0]["detalhes"]["acao"], "completo")

    def test_lote_inteiro_entra_de_uma_vez(self):
        # O caso que o botão existe para atender: descarregar a pasta de
        # filmes num arrastar só e sair de perto.
        self.ligar()
        resposta = self.enviar(*[(f"filme{i}.mkv", "video") for i in range(5)])
        self.assertEqual(len(resposta.get_json()["enfileirados"]), 5)
        self.assertEqual(len(self.trabalhos()), 5)

    def test_o_que_roda_nao_some_atras_do_lote(self):
        # Enfileirar 30 de uma vez não pode empurrar para fora da lista
        # justamente o que está sendo transcrito agora.
        self.ligar()
        self.enviar(*[(f"filme{i:02d}.mkv", "video") for i in range(30)])
        nomes = [t["nome"] for t in self.trabalhos()]
        self.assertIn("filme00.mkv", nomes)

    def test_desligado_nao_enfileira_nada(self):
        self.ligar(False)
        resposta = self.enviar(("filme.mkv", "video"))
        self.assertFalse(resposta.get_json()["automatico"])
        self.assertEqual(self.trabalhos(), [])

    def test_padrao_e_desligado(self):
        # Uma transcrição custa meia hora de GPU; ligar isso por omissão
        # gastaria a placa de quem só quis guardar um arquivo.
        self.assertFalse(self.client.get("/api/config").get_json()["auto_processar"])
        self.enviar(("filme.mkv", "video"))
        self.assertEqual(self.trabalhos(), [])

    def test_estado_sobrevive_e_volta_na_config(self):
        self.ligar()
        self.assertTrue(self.client.get("/api/config").get_json()["auto_processar"])
        self.ligar(False)
        self.assertFalse(self.client.get("/api/config").get_json()["auto_processar"])

    def test_reenvio_nao_duplica_o_trabalho_em_andamento(self):
        self.ligar()
        self.enviar(("filme.mkv", "video"))
        self.enviar(("filme.mkv", "video"))
        self.assertEqual(len(self.trabalhos()), 1)

    def test_valor_invalido_e_recusado(self):
        resposta = self.client.post("/api/config", json={"auto_processar": "talvez"})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("verdadeiro ou falso", resposta.get_json()["erro"])


class TestAcaoAutomatica(unittest.TestCase):
    """A regra em si, sem HTTP no meio."""

    def test_midia_transcreve_e_traduz(self):
        for nome in ("filme.mkv", "filme.mp4", "audio.mp3"):
            self.assertEqual(web.acao_automatica(nome), "completo", nome)

    def test_legenda_nao_tem_acao_automatica(self):
        for nome in ("filme.srt", "filme.ssa", "filme.ass"):
            self.assertIsNone(web.acao_automatica(nome), nome)


class TestProcessamento(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = self._fake_translate

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    @staticmethod
    def _fake_translate(cues, lang, **kwargs):
        for cue in cues:
            cue.text = "PT:" + cue.source_text
        return len(cues), []

    def test_processa_e_disponibiliza_download(self):
        self.escrever("filme.srt")
        job = self.client.post("/api/processar",
                               json={"arquivo": "filme.srt"}).get_json()
        final = self.esperar(job["id"])

        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        self.assertTrue(final["baixavel"])
        self.assertEqual(final["detalhes"]["total"], 1)

        baixado = self.client.get(f"/api/baixar/{job['id']}")
        try:
            self.assertEqual(baixado.status_code, 200)
            self.assertIn("PT:", baixado.get_data(as_text=True))
        finally:
            baixado.close()

    def test_trabalho_aparece_na_lista(self):
        self.escrever("filme.srt")
        self.client.post("/api/processar", json={"arquivo": "filme.srt"})
        trabalhos = self.client.get("/api/trabalhos").get_json()
        self.assertEqual(len(trabalhos), 1)

    def test_erro_vira_mensagem_e_nao_derruba(self):
        self.escrever("vazio.srt", "")
        job = self.client.post("/api/processar",
                               json={"arquivo": "vazio.srt"}).get_json()
        final = self.esperar(job["id"])
        self.assertEqual(final["estado"], jobs.ERRO)
        self.assertTrue(final["erro"])
        # O servidor continua de pé.
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_download_de_trabalho_inexistente(self):
        self.assertEqual(self.client.get("/api/baixar/naoexiste").status_code, 404)


class TestSensibilidadeDaVadNaFila(BaseWeb):
    """A sensibilidade da VAD configurada no painel vale para todo trabalho
    de mídia que passa pela fila -- não é por vídeo."""

    def setUp(self):
        super().setUp()
        from autosrt import config
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)

    def test_repassa_a_sensibilidade_configurada(self):
        self.client.post("/api/config", json={
            "vad_threshold": "0.2", "vad_min_silence_ms": "300"})
        video = self.tocar("filme.mkv")

        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=1, failed=[], detected_lang="en")
            job = self.client.post("/api/processar",
                                   json={"arquivo": "filme.mkv"}).get_json()
            self.esperar(job["id"])

        self.assertEqual(fake.call_args.kwargs["vad_threshold"], 0.2)
        self.assertEqual(fake.call_args.kwargs["vad_min_silence_ms"], 300)

    def test_sem_configuracao_nao_manda_nada(self):
        video = self.tocar("filme.mkv")

        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=1, failed=[], detected_lang="en")
            job = self.client.post("/api/processar",
                                   json={"arquivo": "filme.mkv"}).get_json()
            self.esperar(job["id"])

        self.assertIsNone(fake.call_args.kwargs["vad_threshold"])
        self.assertIsNone(fake.call_args.kwargs["vad_min_silence_ms"])


class TestAntiAlucinacaoNaFila(BaseWeb):
    """condition_on_previous_text e hallucination_silence_threshold
    configurados no painel valem para todo trabalho de mídia da fila."""

    def setUp(self):
        super().setUp()
        from autosrt import config
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)

    def test_repassa_a_configuracao(self):
        self.client.post("/api/config", json={
            "condition_on_previous_text": "true",
            "hallucination_silence_threshold": "2"})
        video = self.tocar("filme.mkv")

        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=1, failed=[], detected_lang="en")
            job = self.client.post("/api/processar",
                                   json={"arquivo": "filme.mkv"}).get_json()
            self.esperar(job["id"])

        self.assertTrue(fake.call_args.kwargs["condition_on_previous_text"])
        self.assertEqual(
            fake.call_args.kwargs["hallucination_silence_threshold"], 2.0)

    def test_sem_configuracao_nao_manda_nada(self):
        video = self.tocar("filme.mkv")

        with mock.patch.object(pipeline, "process_media") as fake:
            fake.return_value = pipeline.PipelineResult(
                total=1, translated=1, failed=[], detected_lang="en")
            job = self.client.post("/api/processar",
                                   json={"arquivo": "filme.mkv"}).get_json()
            self.esperar(job["id"])

        self.assertIsNone(fake.call_args.kwargs["condition_on_previous_text"])
        self.assertIsNone(fake.call_args.kwargs["hallucination_silence_threshold"])


class TestAcoesDisponiveis(unittest.TestCase):
    def ids(self, caminho):
        return [a["id"] for a in web.acoes_para(caminho)]

    def test_video_pode_so_transcrever(self):
        self.assertEqual(self.ids("filme.mkv"), ["completo", "transcrever"])

    def test_legenda_nao_oferece_transcrever(self):
        self.assertNotIn("transcrever", self.ids("filme.srt"))

    def test_legenda_sozinha_so_traduz(self):
        # Sem o filme irmão na pasta, ajustar o tempo não é oferecido.
        self.assertEqual(self.ids("filme.srt"), ["traduzir"])

    def test_legenda_com_filme_tambem_ajusta_o_tempo(self):
        pasta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, pasta, ignore_errors=True)
        for nome in ("filme.mkv", "filme.srt"):
            open(os.path.join(pasta, nome), "wb").close()

        ids = self.ids(os.path.join(pasta, "filme.srt"))
        self.assertIn("traduzir", ids)
        self.assertIn("deslocar", ids)

    def test_converter_so_aparece_para_ssa(self):
        self.assertIn("converter", self.ids("filme.ssa"))
        self.assertIn("converter", self.ids("filme.ass"))
        self.assertNotIn("converter", self.ids("filme.srt"))


class TestLegendaJaExistente(unittest.TestCase):
    """Vídeo com legenda ao lado não deveria ser transcrito por engano.

    Transcrever leva dezenas de minutos de GPU para produzir o que já está
    no disco; traduzir a legenda existente leva pouco mais de um minuto.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "filme.mkv")
        with open(self.video, "wb") as handle:
            handle.write(b"\0")

    def criar_legenda(self, extensao=".srt"):
        caminho = os.path.join(self.tmp, "filme" + extensao)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(SRT)
        return caminho

    def test_sem_legenda_ao_lado(self):
        self.assertIsNone(web.legenda_irma(self.video))

    def test_encontra_srt_de_mesmo_nome(self):
        esperado = self.criar_legenda(".srt")
        self.assertEqual(web.legenda_irma(self.video), esperado)

    def test_encontra_ssa(self):
        esperado = self.criar_legenda(".ssa")
        self.assertEqual(web.legenda_irma(self.video), esperado)

    def test_srt_tem_preferencia_sobre_ssa(self):
        self.criar_legenda(".ssa")
        srt = self.criar_legenda(".srt")
        self.assertEqual(web.legenda_irma(self.video), srt)

    def test_nome_diferente_nao_conta(self):
        outro = os.path.join(self.tmp, "outro-filme.srt")
        with open(outro, "w", encoding="utf-8") as handle:
            handle.write(SRT)
        self.assertIsNone(web.legenda_irma(self.video))

    def test_acao_de_aproveitar_aparece_primeiro(self):
        self.criar_legenda()
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertEqual(ids[0], "traduzir_existente")

    def test_sem_legenda_a_acao_nao_aparece(self):
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertNotIn("traduzir_existente", ids)

    def test_transcrever_continua_disponivel(self):
        self.criar_legenda()
        ids = [a["id"] for a in web.acoes_para(self.video)]
        self.assertIn("completo", ids)


class TestAcoesIndividuais(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    def test_deslocar_move_os_tempos_sem_traduzir(self):
        self.escrever("filme.srt")
        self.tocar("filme.mkv")  # o filme irmão é o que libera o deslocar
        job = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "deslocar", "segundos": 2.5,
        }).get_json()
        self.esperar(job["id"])

        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "filme.srt"))
        self.assertEqual(cues[0].start, 3500)
        # O texto continua em inglês: deslocar não traduz.
        self.assertIn("English", cues[0].source_text)

    def test_deslocar_legenda_sem_filme_e_recusado(self):
        # A regra não é só esconder a opção do menu: o pedido feito na mão,
        # ou vindo de uma página velha aberta antes, também é barrado.
        self.escrever("sozinha.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "sozinha.srt", "acao": "deslocar", "segundos": 2.5,
        })
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("deslocar", resposta.get_json()["erro"])

        # E a legenda fica como estava.
        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "sozinha.srt"))
        self.assertEqual(cues[0].start, 1000)

    def test_converter_gera_srt_sem_traduzir(self):
        ssa = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
               "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
               "MarginV, Effect, Text\n"
               "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Hello there\n")
        self.escrever("filme.ssa", ssa)
        job = self.client.post("/api/processar", json={
            "arquivo": "filme.ssa", "acao": "converter"}).get_json()
        final = self.esperar(job["id"])

        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        destino = os.path.join(self.tmp, "filme.srt")
        self.assertTrue(os.path.exists(destino))
        with open(destino, encoding="utf-8") as handle:
            self.assertIn("Hello there", handle.read())

    def test_acao_invalida_para_o_tipo_e_recusada(self):
        self.escrever("filme.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "transcrever"})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("não vale", resposta.get_json()["erro"])

    def test_acao_inexistente_e_recusada(self):
        self.escrever("filme.srt")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.srt", "acao": "explodir"})
        self.assertEqual(resposta.status_code, 400)

    def test_aproveita_a_legenda_em_vez_de_transcrever(self):
        self.escrever("filme.srt")
        self.tocar("filme.mkv")

        itens = self.client.get("/api/arquivos").get_json()
        video = next(i for i in itens if i["nome"] == "filme.mkv")
        self.assertTrue(video["tem_legenda"])
        self.assertEqual(video["acoes"][0]["id"], "traduzir_existente")

        job = self.client.post("/api/processar", json={
            "arquivo": "filme.mkv", "acao": "traduzir_existente"}).get_json()
        final = self.esperar(job["id"])
        self.assertEqual(final["estado"], jobs.CONCLUIDO)
        self.assertEqual(final["detalhes"]["total"], 1)

    def test_aproveitar_e_recusado_sem_legenda_ao_lado(self):
        self.tocar("filme.mkv")
        resposta = self.client.post("/api/processar", json={
            "arquivo": "filme.mkv", "acao": "traduzir_existente"})
        self.assertEqual(resposta.status_code, 400)


class TestLote(BaseWeb):
    def setUp(self):
        super().setUp()
        self._original = pipeline._translate_with_llm
        pipeline._translate_with_llm = lambda cues, lang, **kw: (len(cues), [])

    def tearDown(self):
        pipeline._translate_with_llm = self._original

    def test_enfileira_varios_de_uma_vez(self):
        for nome in ("a.srt", "b.srt", "c.srt"):
            self.escrever(nome)
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "a.srt", "acao": "traduzir"},
            {"arquivo": "b.srt", "acao": "traduzir"},
            {"arquivo": "c.srt", "acao": "traduzir"},
        ]})
        self.assertEqual(resposta.status_code, 202)
        self.assertEqual(len(resposta.get_json()["enfileirados"]), 3)

    def test_pasta_inteira_com_acoes_diferentes(self):
        self.escrever("legenda.srt")
        self.escrever("outra.srt")
        self.tocar("outra.mkv")  # o filme irmão é o que libera o deslocar
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "legenda.srt", "acao": "traduzir"},
            {"arquivo": "outra.srt", "acao": "deslocar", "segundos": 1},
        ]})
        jobs_criados = resposta.get_json()["enfileirados"]
        self.assertEqual(len(jobs_criados), 2)
        for job in jobs_criados:
            self.esperar(job["id"])
        # A deslocada não foi traduzida.
        cues = pipeline.srt_io.load_cues(os.path.join(self.tmp, "outra.srt"))
        self.assertEqual(cues[0].start, 2000)

    def test_arquivo_ruim_no_meio_nao_impede_os_outros(self):
        self.escrever("bom.srt")
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "bom.srt", "acao": "traduzir"},
            {"arquivo": "sumiu.srt", "acao": "traduzir"},
        ]})
        dados = resposta.get_json()
        self.assertEqual(len(dados["enfileirados"]), 1)
        self.assertEqual(len(dados["recusados"]), 1)

    def test_lote_vazio_e_recusado(self):
        self.assertEqual(
            self.client.post("/api/processar-lote", json={"itens": []}).status_code,
            400)

    def test_lote_recusa_caminho_para_fora(self):
        resposta = self.client.post("/api/processar-lote", json={"itens": [
            {"arquivo": "../../etc/passwd", "acao": "traduzir"}]})
        self.assertEqual(len(resposta.get_json()["recusados"]), 1)


class TestConfiguracao(BaseWeb):
    def setUp(self):
        super().setUp()
        from autosrt import config
        self.config = config
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self._env = os.environ.pop("OPENROUTER_API_KEY", None)

    def tearDown(self):
        self.config.app_directory = self._app_dir
        if self._env is not None:
            os.environ["OPENROUTER_API_KEY"] = self._env

    def test_sem_chave_configurada(self):
        dados = self.client.get("/api/config").get_json()
        self.assertFalse(dados["tem_chave"])

    def test_grava_a_chave(self):
        resposta = self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        self.assertEqual(resposta.status_code, 200)
        self.assertTrue(self.client.get("/api/config").get_json()["tem_chave"])

    def test_a_chave_nunca_volta_para_o_navegador(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-segredo"})
        corpo = self.client.get("/api/config").get_data(as_text=True)
        self.assertNotIn("sk-or-v1-segredo", corpo)

    def test_grava_modelo_e_endereco(self):
        self.client.post("/api/config", json={
            "modelo": "deepseek/deepseek-chat",
            "base_url": "http://localhost:11434/v1"})
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["modelo"], "deepseek/deepseek-chat")
        self.assertEqual(dados["base_url"], "http://localhost:11434/v1")

    def test_endereco_customizado_sem_modelo_nao_finge_um_modelo_do_openrouter(self):
        # Regressão: salvar um endereço local sem modelo (ex.: o campo de
        # modelo ficou vazio) fazia a resposta mostrar o modelo padrão do
        # OpenRouter como se fosse "o que está configurado" -- na prática,
        # ninguém digitou aquilo, e um servidor local não tem esse modelo.
        self.client.post("/api/config", json={"base_url": "http://localhost:11434/v1"})
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["modelo"], "")

    def test_endereco_do_openrouter_sem_modelo_ainda_mostra_o_padrao(self):
        from autosrt import llm
        self.client.post("/api/config", json={"base_url": llm.DEFAULT_BASE_URL})
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["modelo"], llm.DEFAULT_MODEL)

    def test_arquivo_gravado_e_so_do_dono(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        caminho = os.path.join(self.tmp, "config.json")
        self.assertEqual(os.stat(caminho).st_mode & 0o777, 0o600)

    def test_valor_vazio_limpa_a_configuracao(self):
        self.client.post("/api/config", json={"chave": "sk-or-v1-teste"})
        self.client.post("/api/config", json={"chave": ""})
        self.assertFalse(self.client.get("/api/config").get_json()["tem_chave"])

    def test_pedido_sem_nada_e_recusado(self):
        self.assertEqual(
            self.client.post("/api/config", json={}).status_code, 400)

    def test_avisa_quando_o_ambiente_tem_prioridade(self):
        os.environ["OPENROUTER_API_KEY"] = "do-ambiente"
        try:
            resposta = self.client.post("/api/config", json={"chave": "do-arquivo"})
            self.assertIn("prioridade", resposta.get_json()["aviso"])
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_sem_chave_tmdb_configurada(self):
        dados = self.client.get("/api/config").get_json()
        self.assertFalse(dados["tem_chave_tmdb"])

    def test_grava_a_chave_tmdb(self):
        resposta = self.client.post("/api/config", json={"chave_tmdb": "abc123"})
        self.assertEqual(resposta.status_code, 200)
        self.assertTrue(self.client.get("/api/config").get_json()["tem_chave_tmdb"])

    def test_chave_tmdb_nunca_volta_para_o_navegador(self):
        self.client.post("/api/config", json={"chave_tmdb": "segredo-tmdb"})
        corpo = self.client.get("/api/config").get_data(as_text=True)
        self.assertNotIn("segredo-tmdb", corpo)

    def test_expoe_os_enderecos_padrao_dos_dois_modos(self):
        from autosrt import llm
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["openrouter_base_url"], llm.DEFAULT_BASE_URL)
        self.assertEqual(dados["local_base_url"], llm.LOCAL_BASE_URL)

    def test_sensibilidade_da_vad_comeca_vazia(self):
        dados = self.client.get("/api/config").get_json()
        self.assertIsNone(dados["vad_threshold"])
        self.assertIsNone(dados["vad_min_silence_ms"])

    def test_grava_sensibilidade_da_vad(self):
        resposta = self.client.post("/api/config", json={
            "vad_threshold": "0.2", "vad_min_silence_ms": "300"})
        self.assertEqual(resposta.status_code, 200)
        dados = self.client.get("/api/config").get_json()
        self.assertEqual(dados["vad_threshold"], 0.2)
        self.assertEqual(dados["vad_min_silence_ms"], 300)

    def test_vad_threshold_invalido_e_recusado(self):
        resposta = self.client.post("/api/config",
                                    json={"vad_threshold": "nao-e-numero"})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn("número", resposta.get_json()["erro"])
        # Não gravou nada de errado.
        self.assertIsNone(self.client.get("/api/config").get_json()["vad_threshold"])

    def test_vad_min_silence_invalido_e_recusado(self):
        resposta = self.client.post("/api/config",
                                    json={"vad_min_silence_ms": "trezentos"})
        self.assertEqual(resposta.status_code, 400)

    def test_vazio_limpa_a_sensibilidade_da_vad(self):
        self.client.post("/api/config", json={"vad_threshold": "0.2"})
        self.client.post("/api/config", json={"vad_threshold": ""})
        self.assertIsNone(self.client.get("/api/config").get_json()["vad_threshold"])

    def test_anti_alucinacao_comeca_vazia(self):
        dados = self.client.get("/api/config").get_json()
        self.assertIsNone(dados["condition_on_previous_text"])
        self.assertIsNone(dados["hallucination_silence_threshold"])

    def test_grava_anti_alucinacao(self):
        resposta = self.client.post("/api/config", json={
            "condition_on_previous_text": "true",
            "hallucination_silence_threshold": "2"})
        self.assertEqual(resposta.status_code, 200)
        dados = self.client.get("/api/config").get_json()
        self.assertTrue(dados["condition_on_previous_text"])
        self.assertEqual(dados["hallucination_silence_threshold"], 2.0)

    def test_condition_on_previous_text_invalido_e_recusado(self):
        resposta = self.client.post(
            "/api/config", json={"condition_on_previous_text": "talvez"})
        self.assertEqual(resposta.status_code, 400)

    def test_hallucination_silence_threshold_invalido_e_recusado(self):
        resposta = self.client.post(
            "/api/config",
            json={"hallucination_silence_threshold": "nao-e-numero"})
        self.assertEqual(resposta.status_code, 400)


class TestModelos(BaseWeb):
    """/api/modelos alimenta o botão "Buscar modelos" do painel: nunca
    expõe a chave ao navegador, e delega a chamada de rede a llm.list_models
    (já testado à parte)."""

    def test_lista_modelos_do_endereco_informado(self):
        from autosrt import web as web_module
        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = [{"id": "modelo-a", "preco": "grátis"}]
            resposta = self.client.get(
                "/api/modelos?base_url=http://localhost:11434/v1")

        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["modelos"],
                         [{"id": "modelo-a", "preco": "grátis"}])
        self.assertEqual(fake.call_args.args[0], "http://localhost:11434/v1")

    def test_sem_base_url_usa_o_configurado(self):
        from autosrt import config, web as web_module
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self.client.post("/api/config", json={"base_url": "http://meu-servidor/v1"})

        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = []
            self.client.get("/api/modelos")

        self.assertEqual(fake.call_args.args[0], "http://meu-servidor/v1")

    def test_falha_vira_502_com_mensagem(self):
        from autosrt import web as web_module
        from autosrt.llm import LLMError
        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.side_effect = LLMError("servidor local fora do ar")
            resposta = self.client.get("/api/modelos")

        self.assertEqual(resposta.status_code, 502)
        self.assertIn("fora do ar", resposta.get_json()["erro"])

    def test_nao_expoe_a_chave_ao_navegador(self):
        from autosrt import config, web as web_module
        self._app_dir = config.app_directory
        config.app_directory = lambda: self.tmp
        self.addCleanup(setattr, config, "app_directory", self._app_dir)
        self.client.post("/api/config", json={"chave": "sk-or-v1-segredo"})

        with mock.patch.object(web_module.llm, "list_models") as fake:
            fake.return_value = []
            resposta = self.client.get("/api/modelos")

        self.assertNotIn("sk-or-v1-segredo", resposta.get_data(as_text=True))
        self.assertEqual(fake.call_args.kwargs["api_key"], "sk-or-v1-segredo")


class TestListaDeTrabalhos(unittest.TestCase):
    """O corte da lista tem que preservar o que a pessoa foi olhar."""

    def fila_parada(self, quantos):
        """Fila com `quantos` trabalhos, nenhum deles andando."""
        travar = threading.Event()
        fila = jobs.JobQueue(lambda job: travar.wait(5))
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(quantos)]
        # Espera o operário pegar o primeiro, senão o teste corre antes de
        # existir qualquer trabalho em andamento para procurar.
        fim = time.time() + 5
        while time.time() < fim and enviados[0].estado != jobs.RODANDO:
            time.sleep(0.01)
        self.addCleanup(travar.set)
        return fila, enviados

    def test_o_que_roda_aparece_mesmo_com_a_fila_cheia(self):
        # Regressão: cortando pela ordem de chegada, mandar 50 filmes de uma
        # vez deixava na tela 20 cartões "na fila" parados -- o único que
        # andava era o mais antigo, e era o primeiro a ser cortado.
        fila, enviados = self.fila_parada(50)
        listados = fila.listar()
        self.assertEqual(listados[0].id, enviados[0].id)
        self.assertEqual(len(listados), 20)

    def test_a_fila_aparece_na_ordem_em_que_sera_atendida(self):
        fila, enviados = self.fila_parada(50)
        listados = fila.listar()
        self.assertEqual([j.id for j in listados],
                         [j.id for j in enviados[:20]])

    def test_terminados_nao_somem_atras_da_fila(self):
        # Sem vaga reservada, uma fila longa engoliria os botões de baixar.
        fila = jobs.JobQueue(lambda job: None)
        prontos = [fila.enviar(f"pronto{i}", f"/tmp/p{i}") for i in range(5)]
        fim = time.time() + 5
        while time.time() < fim and any(j.estado not in jobs.FINAIS for j in prontos):
            time.sleep(0.02)

        travar = threading.Event()
        self.addCleanup(travar.set)
        fila._worker = lambda job: travar.wait(5)
        for i in range(50):
            fila.enviar(f"espera{i}", f"/tmp/e{i}")

        listados = fila.listar()
        self.assertEqual(len(listados), 20)
        terminados = [j for j in listados if j.estado in jobs.FINAIS]
        self.assertEqual(len(terminados), 5)
        # Do mais novo para o mais velho, que é a ordem em que se procura
        # o que acabou de sair.
        self.assertEqual([j.id for j in terminados],
                         [j.id for j in reversed(prontos)])

    def test_lista_curta_cabe_inteira(self):
        fila = jobs.JobQueue(lambda job: None)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(3)]
        fim = time.time() + 5
        while time.time() < fim and any(j.estado not in jobs.FINAIS for j in enviados):
            time.sleep(0.02)
        self.assertEqual({j.id for j in fila.listar()}, {j.id for j in enviados})


class TestFila(unittest.TestCase):
    def test_executa_um_de_cada_vez(self):
        simultaneos = []
        pico = []

        def worker(job):
            simultaneos.append(1)
            pico.append(len(simultaneos))
            time.sleep(0.05)
            simultaneos.pop()

        fila = jobs.JobQueue(worker)
        enviados = [fila.enviar(f"j{i}", f"/tmp/{i}") for i in range(5)]

        fim = time.time() + 5
        while time.time() < fim and any(
                j.estado not in jobs.FINAIS for j in enviados):
            time.sleep(0.02)

        self.assertTrue(all(j.estado == jobs.CONCLUIDO for j in enviados))
        # A GPU nao comporta dois trabalhos ao mesmo tempo.
        self.assertEqual(max(pico), 1)

    def test_falha_no_worker_vira_estado_de_erro(self):
        def worker(job):
            raise RuntimeError("quebrou")

        fila = jobs.JobQueue(worker)
        job = fila.enviar("j", "/tmp/j")
        fim = time.time() + 5
        while time.time() < fim and job.estado not in jobs.FINAIS:
            time.sleep(0.02)
        self.assertEqual(job.estado, jobs.ERRO)
        self.assertIn("quebrou", job.erro)

    def test_cancelar_antes_de_comecar(self):
        def worker(job):
            time.sleep(0.2)

        fila = jobs.JobQueue(worker)
        primeiro = fila.enviar("a", "/tmp/a")
        segundo = fila.enviar("b", "/tmp/b")
        fila.cancelar(segundo.id)

        fim = time.time() + 5
        while time.time() < fim and segundo.estado not in jobs.FINAIS:
            time.sleep(0.02)
        self.assertEqual(segundo.estado, jobs.CANCELADO)
        self.assertEqual(primeiro.estado, jobs.CONCLUIDO)

    def test_fila_volta_a_funcionar_depois_de_esvaziar(self):
        fila = jobs.JobQueue(lambda job: None)
        for _ in range(2):
            job = fila.enviar("j", "/tmp/j")
            fim = time.time() + 5
            while time.time() < fim and job.estado not in jobs.FINAIS:
                time.sleep(0.02)
            self.assertEqual(job.estado, jobs.CONCLUIDO)


if __name__ == "__main__":
    unittest.main()
