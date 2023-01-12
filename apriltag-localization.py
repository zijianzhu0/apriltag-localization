import asyncio
import sys
import time

import yaml

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId

from frontend import TagDetector
from websocket_serdes import detections_channel_def, serialize_detections, overlay_channel_def, serialize_overlay


async def main():
    class Listener(FoxgloveServerListener):
        def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("First client subscribed to", channel_id)

        def on_unsubscribe(self, server: FoxgloveServer, channel_id: ChannelId):
            print("Last client unsubscribed from", channel_id)

    async with FoxgloveServer("0.0.0.0", 8765, "apriltag-localization") as server:
        server.set_listener(Listener())
        detections_chan = await server.add_channel(detections_channel_def())
        overlay_chan = await server.add_channel(overlay_channel_def())

        with open(sys.argv[1], 'r') as fp:
            yaml_config = yaml.load(fp, yaml.Loader)

        tag_detector = TagDetector(yaml_config["frontend"])
        while True:
            frontend_result = tag_detector.process_frame()
            if frontend_result is None:
                continue

            detections, overlay = frontend_result

            await asyncio.sleep(0.05)

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


if __name__ == "__main__":
    run_cancellable(main())
