import cv2
from pupil_apriltags import Detector


class TagDetector:
    def __init__(self):
        self.detector = Detector(
            families="tag16h5",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0)

        self.capture = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.capture.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            img,
            # camera_params=[336.7755634193813, 333.3575643300718, 336.02729840829176, 212.77376312080065],
            # tag_size=0.065,
            # estimate_tag_pose=True,
        )
        # print(f'{len(detections)} detections')

        overlay = frame
        for tag in detections:
            overlay = cv2.circle(overlay, (int(tag.center[0]), int(tag.center[1])), radius=5, color=(0, 255, 0),
                                 thickness=-1)
        return detections, overlay

    def __del__(self):
        self.capture.release()
