import gtsam
from gtsam.symbol_shorthand import P, T
import numpy as np


class TagInfo:
    def __init__(self, field_pose, noise_sigma):
        self.pose = field_pose      # in field2tag frame
        self.prior_noise_model = gtsam.noiseModel.Isotropic.Sigma(6, noise_sigma)


def to_veh_frame(detection):
    # remap translation
    # cam +Z -> veh +X
    # cam +Y -> veh -Z
    # cam +X -> veh -Y
    tf = np.array([[0.0, 0.0, 1.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]])
    new_t = tf @ detection.pose_t

    # remap rotation
    cam_ypr = gtsam.Rot3(detection.pose_R).ypr()
    new_rot = gtsam.Rot3.Ypr(-cam_ypr[1], -cam_ypr[2], cam_ypr[0])

    veh2tag = gtsam.Pose3(new_rot, new_t)
    return veh2tag


def localize_single_tag(detections, tag_map):
    assert len(detections) == 1

    # get cam2tag from detection
    det = detections[0]
    cam2tag = to_veh_frame(det)

    # lookup tag pose in map
    field2tag = tag_map[det.tag_id].pose

    return field2tag.transformPoseFrom(cam2tag.inverse())


def generate_initial_camera_pose(detections, tag_map):
    # pick the detection with lowest error
    assert len(detections) > 0

    best_detection = None
    best_error = 1e99
    for det in detections:
        if det.pose_err < best_error:
            best_detection = det
            best_error = det.pose_err

    return localize_single_tag([best_detection], tag_map)


def localize_multi_tags(detections, tag_map, detection_noise):
    # Use the factor graph to average results if there's more than 1 detection
    # Otherwise, just return none.
    assert len(detections) > 1

    # setup optimization problem
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # Load a prior map of detected tags
    processed_tags = set()
    for det in detections:
        tag_id = det.tag_id
        assert tag_id in tag_map

        # don't process the same tag more than once
        if tag_id in processed_tags:
            continue

        tag_info = tag_map[tag_id]
        tag_key = T(tag_id)
        initial_values.insert(tag_key, tag_info.pose)
        graph.add(gtsam.PriorFactorPose3(tag_key, tag_info.pose, tag_info.prior_noise_model))
        processed_tags.add(tag_id)

    # No point in optimizing just one confirmed detection
    if processed_tags.size() < 2:
        print('[localize_camera] not enough mapped tags found')
        return None

    # compute initial camera pose
    init_cam_pose = generate_initial_camera_pose(detections, tag_map)

    # then add the camera pose node
    camera_key = P(0)
    initial_values.insert(camera_key, init_cam_pose)

    # finally add the detection factors
    for det in detections:
        assert det.tag_id in processed_tags

        # TODO update this to scale noise based on detection pose error
        detection_nm = gtsam.noiseModel.Isotropic.Sigma(6, detection_noise)
        cam_tag_measurement = to_veh_frame(det)
        graph.push_back(gtsam.BetweenFactorPose3(camera_key, T(tag_id), cam_tag_measurement, detection_nm))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    # TODO apply some sanity checks
    # i.e. ensure the optimization completed, no high residuals, tag map didn't deviate too much, etc...

    return result.at(camera_key)


# tag_map is a dict of tag_id -> TagInfo
def build_map(tag_map_yaml):
    assert len(tag_map_yaml) > 0

    tag_map = {}
    for tag_yaml in tag_map_yaml:
        position = np.array(tag_yaml['position'])
        quat = tag_yaml['orientation']
        orientation = gtsam.Rot3.Quaternion(quat[0], quat[1], quat[2], quat[3])
        field_pose = gtsam.Pose3(orientation, position)
        tag_map[tag_yaml['tag_id']] = TagInfo(field_pose, tag_yaml['noise'])

    return tag_map


if __name__ == '__main__':
    print('Testing localization...')
