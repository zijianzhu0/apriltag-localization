import sys

from frontend import resize_image
import cv2 as cv
import numpy as np
import yaml

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_CHECK_COND + cv.fisheye.CALIB_FIX_SKEW

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)


def main(config_path):
    with open(config_path, 'r') as fp:
        config_yaml = yaml.load(fp, yaml.Loader)

    cam_yaml = config_yaml['frontend']['camera_params']
    scale_percent = cam_yaml['scale_percent']
    cal_model = cam_yaml['cal_model']

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points (2.1mm size of checkerboard)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * 0.0021

    capture = cv.VideoCapture(config_yaml['frontend']['capture_id'])

    cal_frames = 100
    while cal_frames > 0:
        ret, frame = capture.read()
        frame = resize_image(frame, scale_percent)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret:
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if cal_frames % 5 == 0:
                objpoints.append(objp)
                imgpoints.append(corners2)

            # Draw and display the corners
            frame = cv.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
            cal_frames -= 1

        cv.imshow('frame', frame)
        cv.waitKey(100)

    cv.destroyAllWindows()

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    print("Found " + str(N_OK) + " valid images for calibration")
    print(f'Image dims: {gray.shape[::-1]}')

    result = None
    if cal_model == "FISHEYE":
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            criteria
        )
        print(f'K={K}')
        print(f'D={D}')

        result = {
            "K": K.tolist(),
            "D": D.tolist(),
        }
    elif cal_model == "RADIAL":
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f'K={K}')
        print(f'dist={dist}')

        result = {
            "K": K.tolist(),
            "dist": dist.tolist(),
        }

    config_yaml["frontend"]["camera_params"]["calibration"] = result
    with open("config.yaml", "w") as fp:
        yaml.dump(config_yaml, fp, yaml.Dumper)


if __name__ == '__main__':
    exit(main(sys.argv[1]))
