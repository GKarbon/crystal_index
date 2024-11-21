import unittest
from numpy import array, allclose, square, sqrt
from src.crystal_index.crystal import (
    get_possible_index_combinations,
    CrystalSystem,
    CrystalPlane,
    Crystal,
)


class TestCrystal(unittest.TestCase):
    def test_get_possible_index_combinations(self):
        result = get_possible_index_combinations(2)
        expected = [
            (0, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, 1, 2),
            (0, 2, 2),
            (1, 2, 2),
            (2, 2, 2),
        ]
        self.assertEqual(result, expected)
        self.assertEqual(len(get_possible_index_combinations(5)), 55)

    def test_crystal_system_enum(self):
        self.assertEqual(CrystalSystem.FCC.value, "FCC")
        self.assertEqual(CrystalSystem.BCC.value, "BCC")
        self.assertEqual(CrystalSystem.SC.value, "SC")

    def test_crystal_plane_initialization(self):
        plane = CrystalPlane(1, 2, 3)
        self.assertEqual(plane.h, 1)
        self.assertEqual(plane.k, 2)
        self.assertEqual(plane.l, 3)
        self.assertTrue(allclose(plane.planenormal, array([1, 2, 3])))

    def test_crystal_plane_angle_between(self):
        plane1 = CrystalPlane(1, 0, 0)
        plane2 = CrystalPlane(0, 1, 0)
        angle = CrystalPlane.angle_between(plane1, plane2)
        self.assertAlmostEqual(angle, 90, places=4)

    def test_crystal_plane_get_zone_axis(self):
        plane1 = CrystalPlane(1, 0, 0)
        plane2 = CrystalPlane(0, 1, 0)
        zone_axis = CrystalPlane.get_zone_axis(plane1, plane2)
        self.assertTrue(allclose(zone_axis, array([0, 0, 1])))

    def test_crystal_plane_get_d_spacing(self):
        plane = CrystalPlane(1, 1, 1)
        d_spacing = plane.get_d_spacing()
        self.assertEqual(d_spacing, 1 / sqrt(1 + 1 + 1))
        plane = CrystalPlane(2, 2, 2)
        d_spacing = plane.get_d_spacing()
        self.assertEqual(d_spacing, 1 / sqrt(square(2) + square(2) + square(2)))

    def test_crystal_plane_get_equivalent_planes(self):
        plane = CrystalPlane(1, 2, 3)
        equivalents = plane.get_equivalent_planes()
        self.assertEqual(len(equivalents), 48)  # 6 permutations * 8 sign variations

    def check_crystal_initialization(self, crystal_system, order, expected_ratios):
        crystal = Crystal(crystal_system, order)
        self.assertEqual(crystal.crystal_system.value, crystal_system)
        self.assertEqual(crystal.order, order)
        self.assertEqual(len(crystal.planes), order)
        calculated_ratios = [
            crystal.planes[0].get_d_spacing() / plane.get_d_spacing()
            for plane in crystal.planes
        ]
        for calculated, expected in zip(calculated_ratios, expected_ratios):
            self.assertAlmostEqual(calculated, expected, places=4)

    def test_SC_crystal_initialization(self):
        expected_ratios = [float(sqrt(i) / sqrt(1)) for i in [1, 2, 3, 4, 5, 6, 8, 9]]
        self.check_crystal_initialization("SC", 8, expected_ratios)

    def test_BCC_crystal_initialization(self):
        expected_ratios = [float(sqrt(i) / sqrt(1)) for i in [1, 2, 3, 4, 5, 6, 7, 8]]
        self.check_crystal_initialization("BCC", 8, expected_ratios)

    def test_FCC_crystal_initialization(self):
        expected_ratios = [
            float(sqrt(i) / sqrt(3)) for i in [3, 4, 8, 11, 12, 16, 19, 20]
        ]
        self.check_crystal_initialization("FCC", 8, expected_ratios)

    def test_crystal_get_d_ratio(self):
        crystal = Crystal("FCC", 8)
        d_ratios = crystal.get_d_ratio()
        self.assertTrue(len(d_ratios) > 0)
        for ratio, pair in d_ratios:
            self.assertIsInstance(ratio, float)
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], CrystalPlane)
            self.assertIsInstance(pair[1], CrystalPlane)


if __name__ == "__main__":
    unittest.main()
