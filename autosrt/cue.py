"""Representação de uma legenda individual.

A distinção entre ``source_text`` e ``text`` é o que viabiliza o passe de
correção de gênero: a tradução precisa ser comparada com o texto de origem,
porque é dele que vêm os marcadores de gênero ("sir", "Mr.", "he").
"""

from dataclasses import dataclass


@dataclass
class Cue:
    """Uma legenda, com tempos em milissegundos.

    Attributes:
        index: número sequencial da legenda no arquivo (base 1).
        start: início em milissegundos.
        end: fim em milissegundos.
        source_text: texto no idioma de origem. Nunca é modificado depois
            da leitura do arquivo.
        text: texto de trabalho. Começa igual a ``source_text`` e passa a
            conter a tradução depois do passe de tradução.
    """

    index: int
    start: int
    end: int
    source_text: str
    text: str

    @classmethod
    def from_source(cls, index: int, start: int, end: int, source_text: str) -> "Cue":
        return cls(index=index, start=start, end=end,
                   source_text=source_text, text=source_text)

    @property
    def duration(self) -> int:
        """Duração em milissegundos."""
        return self.end - self.start
