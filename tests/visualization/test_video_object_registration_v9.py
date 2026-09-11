"""Registration must use source geometry and phase, never robot tracking error."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[2] / "mimicx/visualization/video_object_registration.py"


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "the source-based registration module is missing")
        spec = importlib.util.spec_from_file_location("video_object_registration", MODULE)
        self.registration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.registration)

    def test_similarity_recovers_metric_camera_basis(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(60, 3))
        rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        target = 0.7 * source @ rotation.T + [1, 2, 3]
        scale, basis, origin = self.registration.fit_similarity(source, target)
        np.testing.assert_allclose(scale, 0.7, atol=1e-12)
        np.testing.assert_allclose(basis, rotation, atol=1e-12)
        np.testing.assert_allclose(origin, [1, 2, 3], atol=1e-12)

    def test_degenerate_calibration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registration.fit_similarity(np.zeros((4, 3)), np.ones((4, 3)))

    def test_phase_matching_recovers_resets_without_fullstretch(self):
        reference = np.random.default_rng(4).normal(size=(519, 29))
        expected = np.r_[np.arange(1, 519), np.arange(282)]
        phase, rms = self.registration.match_reference_phase(reference[expected], reference)
        np.testing.assert_array_equal(phase, expected)
        np.testing.assert_allclose(rms, 0)
        frames = self.registration.source_frames_from_phase(phase, 50.0, 30.0, 312)
        self.assertEqual(frames[100], 61)
        self.assertEqual(frames[518], 0)
        self.assertEqual(frames[-1], 169)

    def test_unmatched_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registration.match_reference_phase(np.ones((2, 3)), np.zeros((4, 3)))

    def test_source_time_outside_horizon_is_not_stretched_or_clamped(self):
        frames = self.registration.source_frames_from_phase(np.array([0, 100, 600]), 50, 30, 312)
        self.assertEqual(frames, [0, 60, None])

    def test_ray_plane_lifting_and_reprojection(self):
        intrinsics = np.array([[800, 0, 400], [0, 800, 360], [0, 0, 1]])
        point = self.registration.lift_to_camera_depth([480, 520], intrinsics, 4.0)
        np.testing.assert_allclose(point, [0.4, 0.8, 4.0])
        np.testing.assert_allclose(self.registration.project_points(point[None], intrinsics), [[480, 520]])
        with self.assertRaises(ValueError):
            self.registration.lift_to_camera_depth([480, 520], intrinsics, -1)

    def test_ground_calibration_solves_depth_not_forward_offset(self):
        ray_origin = np.array([0.0, 0.0, 2.0])
        ray_direction = np.array([0.0, 1.0, -0.5])
        depth = self.registration.ground_ray_depth(ray_origin, ray_direction, 0.1)
        self.assertAlmostEqual(depth, 3.8)
        self.assertIsNone(self.registration.ground_ray_depth(ray_origin, [0, 1, 0], 0.1))

    def test_camera_translation_fit_uses_source_skeleton(self):
        points = np.random.default_rng(8).normal(size=(14, 3)) * 0.2 + [0, 0, 4]
        intrinsics = np.array([[800, 0, 400], [0, 800, 360], [0, 0, 1]])
        expected = np.array([0.03, -0.13, 0.08])
        pixels = self.registration.project_points(points + expected, intrinsics)
        actual = self.registration.fit_camera_translation(points, pixels, intrinsics, np.ones(14))
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_missing_detections_stay_null_and_shared_phase_is_identical(self):
        trajectory = [[1, 2, 3], None, [3, 4, 5]]
        first = self.registration.sample_source_trajectory(trajectory, [0, 1, 2, None])
        second = self.registration.sample_source_trajectory(trajectory, [2, 1, 0])
        self.assertEqual(first, [[1, 2, 3], None, [3, 4, 5], None])
        self.assertEqual(first[0], second[2])
        self.assertEqual(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
