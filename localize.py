import gtsam
from gtsam.symbol_shorthand import P, T
import numpy as np


class TagInfo:
    def __init__(self):
        self.pose = gtsam.Pose3()  # field2tag
        self.prior_noise_model = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)


def to_veh_frame(detection):
    cam2pose = gtsam.Pose3(detection.pose_R, detection.pose_t)

    # cam +Z -> veh +X
    # cam +Y -> veh -Z
    # cam +X -> veh -Y
    tf = np.array([[ 0.0,  0.0,  1.0, 0.0],
                   [-1.0,  0.0,  0.0, 0.0],
                   [ 0.0, -1.0,  0.0, 0.0],
                   [ 0.0,  0.0,  0.0, 1.0]])
    veh2cam = gtsam.Pose3(tf)
    return veh2cam * cam2pose


def generate_initial_camera_pose(detections, tag_map):
    # pick the detection with lowest error
    assert len(detections) > 0

    best_detection = None
    best_error = 1e99
    for det in detections:
        if det.pose_err < best_error:
            best_detection = det
            best_error = det.pose_err

    cam2tag = to_veh_frame(best_detection)
    field2tag = tag_map[best_detection.tag_id].pose
    field2cam = field2tag * cam2tag.inverse()
    return field2cam


# tag_map is a dict of tag_id -> TagInfo
def localize_camera(detections, tag_map):
    # Use the factor graph to average results if there's more than 1 detection
    # Otherwise, just return none.
    if len(detections) < 2:
        return None

    # setup optimization problem
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    # Load a prior map of detected tags
    processed_tags = set()
    for det in detections:
        tag_id = det.tag_id
        if tag_id not in tag_map:
            continue
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
        if det.tag_id not in processed_tags:
            continue

        # TODO update this to scale noise based on detection pose error
        detection_nm = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        cam_tag_measurement = to_veh_frame(det)
        graph.push_back(gtsam.BetweenFactorPose3(camera_key, T(tag_id), cam_tag_measurement, detection_nm))

    # optimize using Levenberg-Marquardt optimization
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    # TODO apply some sanity checks
    # i.e. ensure the optimization completed, no high residuals, tag map didn't deviate too much, etc...

    return result.at(camera_key)
