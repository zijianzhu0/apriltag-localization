import base64
import json

import numpy as np
import cv2


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
        det_json.append({
            "tag_id": det.tag_id,
            "center_x": det.center[0],
            "center_y": det.center[1],
            "decision_margin": det.decision_margin,
            "pose_err": det.pose_err,
        })

    return json.dumps({"detections": det_json}).encode("utf8")


def overlay_channel_def():
    return {
        "topic": "/frontend/overlay",
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
