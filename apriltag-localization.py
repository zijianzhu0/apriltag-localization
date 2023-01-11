import asyncio
import time

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId

from frontend import TagDetector
from websocket import detections_channel_def, serialize_detections, overlay_channel_def, serialize_overlay


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

        tag_detector = TagDetector()
        while True:
            detections, overlay = tag_detector.process_frame()

            await asyncio.sleep(0.1)

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
