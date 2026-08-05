"""AutoSRT - ponto de entrada.

A lógica vive no pacote ``autosrt``. Este arquivo apenas abre a interface,
e é o alvo do PyInstaller ao gerar o executável.
"""

from autosrt.gui import main

if __name__ == "__main__":
    main()
