import unittest

from autosrt.cue import Cue
from autosrt.sync import SyncError, fps_sync, linear_sync, shift_cues


def make_cues(*spans):
    return [Cue.from_source(i, start, end, f"linha {i}")
            for i, (start, end) in enumerate(spans, start=1)]


class TestShift(unittest.TestCase):
    def test_atrasa_todas_as_legendas(self):
        cues = make_cues((1000, 2000), (3000, 4000))
        shift_cues(cues, 500)
        self.assertEqual([(c.start, c.end) for c in cues],
                         [(1500, 2500), (3500, 4500)])

    def test_adianta_com_valor_negativo(self):
        cues = make_cues((1000, 2000))
        shift_cues(cues, -300)
        self.assertEqual((cues[0].start, cues[0].end), (700, 1700))

    def test_nao_produz_tempo_negativo(self):
        cues = make_cues((100, 200))
        shift_cues(cues, -5000)
        self.assertEqual((cues[0].start, cues[0].end), (0, 0))


class TestLinearSync(unittest.TestCase):
    def test_corrige_deslocamento_puro(self):
        cues = make_cues((1000, 2000), (5000, 6000))
        # Ambos os pontos precisam de +1000.
        linear_sync(cues, 1000, 2000, 5000, 6000)
        self.assertEqual(cues[0].start, 2000)
        self.assertEqual(cues[1].start, 6000)

    def test_corrige_deriva_progressiva(self):
        # Legenda de 25 fps sobre vídeo de 23,976: fator 1,0427.
        cues = make_cues((0, 1000), (100000, 101000))
        scale, _ = linear_sync(cues, 0, 0, 100000, 104270)
        self.assertAlmostEqual(scale, 1.0427, places=4)
        self.assertEqual(cues[1].start, 104270)

    def test_pontos_de_origem_iguais_sao_rejeitados(self):
        cues = make_cues((0, 1000))
        with self.assertRaises(SyncError):
            linear_sync(cues, 5000, 0, 5000, 1000)

    def test_ancoras_caem_exatamente_no_alvo(self):
        cues = make_cues((2000, 3000), (60000, 61000))
        linear_sync(cues, 2000, 1500, 60000, 62000)
        self.assertEqual(cues[0].start, 1500)
        self.assertEqual(cues[1].start, 62000)


class TestFpsSync(unittest.TestCase):
    def test_pal_para_cinema_estica_os_tempos(self):
        cues = make_cues((0, 1000), (100000, 101000))
        scale = fps_sync(cues, 25.0, 23.976)
        self.assertGreater(scale, 1.0)
        self.assertEqual(cues[1].start, round(100000 * scale))

    def test_framerate_invalido_e_rejeitado(self):
        with self.assertRaises(SyncError):
            fps_sync(make_cues((0, 1000)), 0, 25.0)


if __name__ == "__main__":
    unittest.main()
