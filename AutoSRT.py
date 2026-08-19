"""AutoSRT - ponto de entrada.

A lógica vive no pacote ``autosrt``. Este arquivo existe para permitir
``python AutoSRT.py filme.mkv`` sem instalar nada; instalado, o comando
``autosrt`` chama a mesma função.
"""

import sys

from autosrt.cli import main

if __name__ == "__main__":
    sys.exit(main())
