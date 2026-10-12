"""Isolamento de config.app_directory para a sessão de teste inteira.

Job em fila é processado por uma thread em segundo plano (ver
autosrt.jobs.JobQueue) que pode terminar depois que o teste que a disparou
já voltou e desfez seu próprio redirecionamento de
autosrt.config.app_directory (ver BaseWeb.setUp em test_web.py e
equivalentes). Nessa janela, autosrt.historico.registrar -- ligado via
JobQueue(on_final=...) -- gravaria no ~/.config/autosrt/historico.db de
verdade em vez do descartável do teste.

Isto fecha a janela inteira: define um diretório descartável para a sessão
de testes inteira, antes de qualquer teste rodar e antes de qualquer
thread de trabalho poder existir. Os redirecionamentos que testes
individuais já fazem continuam valendo por cima disto enquanto o teste
roda -- isto só é a rede de segurança para quando uma thread atrasada
termina fora da janela deles.
"""

import os
import tempfile

os.environ.setdefault(
    "AUTOSRT_CONFIG_DIR", tempfile.mkdtemp(prefix="autosrt-tests-config-"))

# O worker automático das ferramentas de filmes (ver web._worker_filmes_periodico)
# sobe uma thread de verdade a cada create_app(). Nos testes isso só deixaria
# threads dormindo pra trás sem rodar nada útil (o intervalo é de horas) --
# "0" desliga a thread por completo.
os.environ.setdefault("AUTOSRT_FILMES_WORKER_INTERVALO_S", "0")
