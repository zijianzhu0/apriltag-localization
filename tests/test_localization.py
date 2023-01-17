import unittest

import gtsam
import numpy as np

# Import localization library
import sys
sys.path.insert(1, './')
from localize import TagInfo, localize_multi_tags


class MockDetection:
    def __init__(self, tag_id, translation, rotation_zyx, pose_err=0.0):
        self.tag_id = tag_id
        self.pose_t = translation
        self.pose_R = gtsam.Rot3.RzRyRx(rotation_zyx[0], rotation_zyx[1], rotation_zyx[2]).matrix()
        self.pose_err = pose_err


class TestMultiTags(unittest.TestCase):

    def test_no_detections(self):
        with self.assertRaises(AssertionError):
            localize_multi_tags([], {}, 0)

    def test_one_detection(self):
        detections = [MockDetection(0, [0, 0, 0], [0, 0, 0])]
        with self.assertRaises(AssertionError):
            localize_multi_tags(detections, {}, 0)

    def test_outlier_detections(self):
        detections = [
            MockDetection(-1, np.zeros(3), np.zeros(3)),
            MockDetection(-2, np.zeros(3), np.zeros(3)),
        ]

        with self.assertRaises(AssertionError):
            localize_multi_tags(detections, {}, 0)

    def test_duplicate_detections(self):
        detections = [
            MockDetection(0, np.zeros(3), np.zeros(3)),
            MockDetection(0, np.zeros(3), np.zeros(3)),
        ]

        tag_map = { 0: TagInfo(gtsam.Pose3(), 0) }

        with self.assertRaises(AssertionError):
            localize_multi_tags(detections, tag_map, 0)

    def test_two_detections(self):
        noise = 0.1

        # detected two tags left and right in front of camera
        # remember detections are in camera frame!
        detections = [
            MockDetection(1, np.array([-1, 0, 1]), np.zeros(3)),
            MockDetection(2, np.array([1, 0, 1]), np.zeros(3)),
        ]

        tag_map = {
            1: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, 1, 0])), noise),
            2: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, -1, 0])), noise),
        }

        # camera should be at origin
        cam_pose = localize_multi_tags(detections, tag_map, noise)
        self.assertTrue(np.allclose(np.identity(3), cam_pose.rotation().matrix()))
        self.assertTrue(np.allclose(np.zeros(3), cam_pose.translation()))
    
    def test_two_noisy_detections(self):
        noise = 0.1

        # corrupt tag detections with camera still at origin
        # remember detections are in camera frame!
        detections = [
            MockDetection(1, np.array([-1.1, 0, 1]), np.zeros(3)),
            MockDetection(2, np.array([1.1, 0, 1]), np.zeros(3)),
        ]

        tag_map = {
            1: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, 1, 0])), noise),
            2: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, -1, 0])), noise),
        }

        # camera should be at origin
        cam_pose = localize_multi_tags(detections, tag_map, noise)
        self.assertTrue(np.allclose(np.identity(3), cam_pose.rotation().matrix()))
        self.assertTrue(np.allclose(np.zeros(3), cam_pose.translation()))

    def test_four_detections(self):
        noise = 0.01

        # four symmetric detections
        # remember detections are in camera frame!
        detections = [
            MockDetection(1, np.array([-2, 0, 1]), np.zeros(3)),
            MockDetection(2, np.array([-1, 0, 1]), np.zeros(3)),
            MockDetection(3, np.array([1, 0, 1]), np.zeros(3)),
            MockDetection(4, np.array([2, 0, 1]), np.zeros(3)),
        ]

        tag_map = {
            1: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, -2, 0])), noise),
            2: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, -1, 0])), noise),
            3: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, 1, 0])), noise),
            4: TagInfo(gtsam.Pose3(gtsam.Rot3(), np.array([1, 2, 0])), noise),
        }

        # camera should be at origin
        cam_pose = localize_multi_tags(detections, tag_map, noise)
        self.assertTrue(np.allclose(np.identity(3), cam_pose.rotation().matrix()))
        self.assertTrue(np.allclose(np.zeros(3), cam_pose.translation()))

    def test_rotated_detections(self):
        noise = 0.1

        # apply rotation to detections, one on either side of
        # camera at origin
        # remember detections are in camera frame!
        detections = [
            MockDetection(1, np.array([-1, 0, 0]), np.array([0, -1.23, 0])),
            MockDetection(2, np.array([1, 0, 0]), np.array([0, 1.23, 0])),
        ]

        tag_map = {
            1: TagInfo(gtsam.Pose3(gtsam.Rot3.Ypr(1.23, 0, 0), np.array([0, 1, 0])), noise),
            2: TagInfo(gtsam.Pose3(gtsam.Rot3.Ypr(-1.23, 0, 0), np.array([0, -1, 0])), noise),
        }

        # camera should be at origin
        cam_pose = localize_multi_tags(detections, tag_map, noise)
        self.assertTrue(np.allclose(np.identity(3), cam_pose.rotation().matrix()))
        self.assertTrue(np.allclose(np.zeros(3), cam_pose.translation()))


if __name__ == '__main__':
    unittest.main()
