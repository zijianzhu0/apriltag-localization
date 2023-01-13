import base64
import json

import cv2
import gtsam
import numpy as np


def detections_channel_def():
    return {
        "topic": "/frontend/detections",
        "encoding": "json",
        "schemaName": "apriltag.detections",
        "schema": json.dumps(
            {
                "type": "object",
                "properties": {
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag_id": {"type": "integer"},
                                "center_x": {"type": "number"},
                                "center_y": {"type": "number"},
                                "decision_margin": {"type": "number"},
                                "pose_err": {"type": "number"},
                                "hamming": {"type": "integer"},
                                "pose_x": {"type": "number"},
                                "pose_y": {"type": "number"},
                                "pose_z": {"type": "number"},
                            }
                        }
                    },
                }
            }
        ),
    }


def serialize_detections(detections):
    det_json = []
    for det in detections:
        pose_t = det.pose_t.flatten()
        det_json.append({
            "tag_id": det.tag_id,
            "center_x": det.center[0],
            "center_y": det.center[1],
            "decision_margin": det.decision_margin,
            "pose_err": det.pose_err,
            "hamming": det.hamming,
            "pose_x": pose_t[0],
            "pose_y": pose_t[1],
            "pose_z": pose_t[2],
        })

    return json.dumps({"detections": det_json}).encode("utf8")


def overlay_channel_def(topic_name):
    return {
        "topic": topic_name,
        "encoding": "json",
        "schemaName": "ros.sensor_msgs.CompressedImage",
        "schema": json.dumps({
            "type": "object",
            "properties": {
                "header": {
                    "type": "object",
                    "properties": {
                        "stamp": {
                            "type": "object",
                            "properties": {
                                "sec": {"type": "integer"},
                                "nsec": {"type": "integer"},
                            }
                        }
                    }
                },
                "encoding": {"type": "string"},
                "data": {"type": "string", "contentEncoding": "base64"},
            }
        }),
    }


def serialize_overlay(overlay_img):
    img_encode = cv2.imencode('.jpg', overlay_img)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()
    b64_encode = base64.b64encode(byte_encode).decode("utf8")
    return json.dumps({
        "header": {
            "stamp": {
                "sec": 0,
                "nsec": 0,
            }
        },
        "encoding": "jpeg",
        "data": b64_encode
    }).encode("utf8")


def frame_tf_def():
    with open("schemas/FrameTransform.json", "r") as fp:
        schema_str = fp.read()

    return {
        "topic": "/tf",
        "encoding": "json",
        "schemaName": "foxglove.FrameTransform",
        "schema": schema_str,
    }


def serialize_tf():
    return json.dumps({
        "timestamp": {
            "sec": 0,
            "nsec": 0,
        },
        "parent_frame_id": "world",
        "child_frame_id": "camera_frame",
        "translation": {
            "x": 0,
            "y": 0,
            "z": 0,
        },
        "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0,
        }
    }).encode("utf8")


def camera_scene_def():
    with open("schemas/SceneUpdate.json", "r") as fp:
        schema_str = fp.read()

    return {
        "topic": "/frontend/camera/scene",
        "encoding": "json",
        "schemaName": "foxglove.SceneUpdate",
        "schema": schema_str,
    }


def serialize_cam_scene(detections):
    tag_models = []
    tag_labels = []
    for det in detections:
        pose_t = det.pose_t.flatten()
        quat = gtsam.Rot3(det.pose_R).quaternion()
        tag_models.append({
            "pose": {
                "position": {
                    "x": pose_t[0],
                    "y": pose_t[1],
                    "z": pose_t[2],
                },
                "orientation": {
                    "x": quat[1],
                    "y": quat[2],
                    "z": quat[3],
                    "w": quat[0],
                },
            },
            "size": {
                "x": 0.15,
                "y": 0.15,
                "z": 0.01,
            },
            "color": {
                "r": 1.0,
                "g": 1.0,
                "b": 1.0,
                "a": 1.0,
            },
        })
        tag_labels.append({
            "pose": {
                "position": {
                    "x": pose_t[0],
                    "y": pose_t[1]+0.15,
                    "z": pose_t[2],
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
            "billboard": True,
            "font_size": 15.0,
            "scale_invariant": True,
            "color": {
                "r": 0.0,
                "g": 0.0,
                "b": 0.0,
                "a": 1.0,
            },
            "text": f'tag_{det.tag_id}',
        })

    return json.dumps({
        "entities": [{
            "timestamp": {
                "sec": 0,
                "nsec": 0,
            },
            "frame_id": "camera_frame",
            "id": "camera_scene",
            "lifetime": {
                "sec": 0,
                "nsec": 0,
            },
            "frame_locked": True,
            "metadata": [],
            "arrows": [],
            "cubes": tag_models,
            "spheres": [],
            "cylinders": [],
            "lines": [],
            "triangles": [],
            "texts": tag_labels,
            "models": [],
        }],
        "deletions": [],
    }).encode("utf8")


def pose_def(topic_name):
    with open("schemas/Pose.json", "r") as fp:
        schema_str = fp.read()

    return {
        "topic": topic_name,
        "encoding": "json",
        "schemaName": "foxglove.Pose",
        "schema": schema_str,
    }


def serialize_pose(pose):
    pose_t = pose.translation()
    quat = pose.rotation().quaternion()
    return json.dumps({
        "position": {
            "x": pose_t[0],
            "y": pose_t[1],
            "z": pose_t[2],
        },
        "orientation": {
            "x": quat[1],
            "y": quat[2],
            "z": quat[3],
            "w": quat[0],
        },
    }).encode("utf8")


def draw_minimap(localizer_yaml, field2cam, base_img):
    h, w, _ = base_img.shape
    x_scale = float(w) / localizer_yaml['field_length']
    y_scale = float(h) / localizer_yaml['field_width']
    pix_coord = field2cam.translation()[:2] * np.array([x_scale, y_scale])
    pix_coord[1] = h - pix_coord[1]
    return cv2.circle(np.copy(base_img), (int(pix_coord[0]), int(pix_coord[1])), radius=20, color=(255, 0, 0), thickness=-1)
