"""Erros compartilhados entre os módulos de leitura de legendas."""


class UnsupportedSubtitleError(ValueError):
    """O arquivo não é uma legenda que o AutoSRT saiba ler."""
