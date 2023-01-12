import cv2
from pupil_apriltags import Detector


def resize_image(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim)


class TagDetector:
    def __init__(self, frontend_yaml):
        self.detector = Detector(
            families=frontend_yaml["tag_family"],
            nthreads=frontend_yaml["num_threads"],
            quad_decimate=frontend_yaml["quad_decimate"],
            quad_sigma=frontend_yaml["quad_sigma"],
            refine_edges=frontend_yaml["refine_edges"],
            decode_sharpening=frontend_yaml["decode_sharpening"],
            debug=0)

        self.capture = cv2.VideoCapture(frontend_yaml["capture_id"])
        self.tag_size = frontend_yaml["tag_size"]

        cam_yaml = frontend_yaml["camera_params"]
        assert cam_yaml["cal_model"] == "RADIAL"
        K = cam_yaml["calibration"]["K"]
        self.camera_params = (K[0][0], K[1][1], K[0][2], K[1][2])

        self.max_pose_err = frontend_yaml["max_pose_err"]
        self.decision_threshold = frontend_yaml["decision_threshold"]

    def filter_detections(self, detections):
        result = []
        for det in detections:
            if det.pose_err > self.max_pose_err:
                continue
            if det.decision_margin < self.decision_threshold:
                continue

            result.append(det)

        return result

    def process_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return None

        # 1) resize image
        frame = resize_image(frame, 50)

        # TODO 2) undistort https://amroamroamro.github.io/mexopencv/matlab/cv.undistort.html

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self.detector.detect(
            img,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
            estimate_tag_pose=True,
        )

        detections = self.filter_detections(detections)

        overlay = frame
        for tag in detections:
            overlay = cv2.circle(overlay, (int(tag.center[0]), int(tag.center[1])), radius=5, color=(0, 255, 0),
                                 thickness=-1)
        return detections, overlay

    def __del__(self):
        self.capture.release()
