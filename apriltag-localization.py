import asyncio
import sys
import time

import cv2
import gtsam
import yaml

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId

from frontend import TagDetector
from localize import build_map, localize_single_tag, localize_multi_tags
from websocket_serdes import detections_channel_def, serialize_detections, overlay_channel_def, serialize_overlay, \
    camera_scene_def, serialize_cam_scene, frame_tf_def, serialize_tf, draw_minimap, pose_def, serialize_pose


async def main():
    class Listener(FoxgloveServerListener):
        def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("First client subscribed to", channel_id)

        def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("Last client unsubscribed from", channel_id)

    async with FoxgloveServer("0.0.0.0", 8765, "apriltag-localization") as server:
        server.set_listener(Listener())
        detections_chan = await server.add_channel(detections_channel_def())
        overlay_chan = await server.add_channel(overlay_channel_def("/frontend/overlay"))
        cam_scene_chan = await server.add_channel(camera_scene_def())
        tf_chan = await server.add_channel(frame_tf_def())
        minimap_chan = await server.add_channel(overlay_channel_def("/minimap"))
        cam_pose_chan = await server.add_channel(pose_def("/cam_pose"))

        with open(sys.argv[1], 'r') as fp:
            yaml_config = yaml.load(fp, yaml.Loader)

        base_img = cv2.imread("minimap.png")
        localizer_yaml = yaml_config["localizer"]
        tag_map = build_map(localizer_yaml["tag_map"])
        tag_detector = TagDetector(yaml_config["frontend"], tag_map)
        while True:
            frontend_result = tag_detector.process_frame()
            if frontend_result is None:
                continue

            detections, overlay = frontend_result

            if len(detections) == 0:
                cam_pose = None
            elif len(detections) == 1:
                cam_pose = localize_single_tag(detections, tag_map)
            else:
                cam_pose = localize_multi_tags(detections, tag_map, localizer_yaml["detections_noise"])

            await asyncio.sleep(0.05)

            if cam_pose is not None:
                await server.send_message(
                    cam_pose_chan,
                    time.time_ns(),
                    serialize_pose(cam_pose),
                )

            await server.send_message(
                detections_chan,
                time.time_ns(),
                serialize_detections(detections),
            )

            await server.send_message(
                overlay_chan,
                time.time_ns(),
                serialize_overlay(overlay),
            )

            await server.send_message(
                cam_scene_chan,
                time.time_ns(),
                serialize_cam_scene(detections)
            )

            await server.send_message(
                tf_chan,
                time.time_ns(),
                serialize_tf()
            )

            if cam_pose is None:
                minimap_img = base_img
            else:
                minimap_img = draw_minimap(localizer_yaml, cam_pose, base_img)

            await server.send_message(
                minimap_chan,
                time.time_ns(),
                serialize_overlay(minimap_img)
            )


if __name__ == "__main__":
    run_cancellable(main())
