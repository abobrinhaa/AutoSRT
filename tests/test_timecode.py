import unittest

from autosrt.sync import SyncError, format_timecode, parse_timecode


class TestParseTimecode(unittest.TestCase):
    def test_formato_completo_de_legenda(self):
        self.assertEqual(parse_timecode("00:01:23,456"), 83456)

    def test_aceita_ponto_como_separador_decimal(self):
        self.assertEqual(parse_timecode("00:01:23.456"), 83456)

    def test_sem_hora(self):
        self.assertEqual(parse_timecode("01:23"), 83000)

    def test_fracao_curta_e_milesimo(self):
        # "5" é meio segundo, não cinco milissegundos.
        self.assertEqual(parse_timecode("00:00:01,5"), 1500)

    def test_segundos_puros(self):
        self.assertEqual(parse_timecode("83.5"), 83500)

    def test_vazio_e_rejeitado(self):
        with self.assertRaises(SyncError):
            parse_timecode("")

    def test_texto_invalido_e_rejeitado(self):
        with self.assertRaises(SyncError):
            parse_timecode("abc")


class TestFormatTimecode(unittest.TestCase):
    def test_formata(self):
        self.assertEqual(format_timecode(83456), "00:01:23,456")

    def test_negativo_vira_zero(self):
        self.assertEqual(format_timecode(-10), "00:00:00,000")

    def test_ida_e_volta(self):
        for valor in (0, 1500, 83456, 3661001):
            self.assertEqual(parse_timecode(format_timecode(valor)), valor)


if __name__ == "__main__":
    unittest.main()
