import cv2
from pupil_apriltags import Detector
# import sys


# img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

at_detector = Detector(
    families="tag16h5",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = at_detector.detect(
       img,
       # camera_params=[336.7755634193813, 333.3575643300718, 336.02729840829176, 212.77376312080065],
       # tag_size=0.065,
       # estimate_tag_pose=True,
    )
    # print(f'{len(detections)} detections')

    for tag in detections:
        frame = cv2.circle(frame, (int(tag.center[0]), int(tag.center[1])), radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
