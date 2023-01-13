# AprilTag Localization
The 2023 FIRST FRC season [introduced AprilTag fiducial markers](https://docs.wpilib.org/en/stable/docs/software/vision-processing/apriltag/apriltag-intro.html) on the field for robots to localize themselves on the competition field. This repo is an open-source implementation of a soft-real-time AprilTag localization algorithm for use on an embedded target like a Raspberry Pi. Combined with a hard-real-time robot microcontroller (roboRio for example) teams can fuse odometry and vision data to robustly estimate robot pose over the course of a match.

## Quick Start
This implementation is coded entirely in Python 3, so it can run on virutally all single-board-computers with at least 2GB of RAM and 1GHz processor. To get started setup a virtual environment using Python 3:
1. `python3 -m venv venv` and source the activation script `source venv/bin/activate`.
2. Then install the required python packages `pip install -r requirements.txt`.
3. You'll need a webcam connected to your machine for the following steps. Any USB or raspicam would be fine as long as it's OpenCV compatible.
4. Calibrate the camera by running `python calibrate.py config.yaml`. This opens a window where you can move a checkerboard calibration pattern around the camera's field-of-view. The calibration script will automatically collect a sample of 20 images to calculate intrinsic projection and distortion parameters. The results are saved as `config.yaml` with the camera parameters field populated. If you don't already have a checkerboard, print [the 6x9 example from the OpenCV repository](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png) on letter-sized paper and tape it to a rigid surface.
5. Examine the contents of `config.yaml` for any details you'd like to adjust. Most of the default values should work out-of-the-box more-or-less for the 2023 FRC AprilTag setup.
6. Run the localizer `python apriltag-localizer.py` which will start a websocket interface on port `8765` for Foxglove Studio to connect. Foxglove studio is like an advanced version of smartdashboard/networktables that provides a variety of robotics visualization and debugging tools. Download and install the desktop version from [their website here](https://foxglove.dev/download). You can also just use the online web interface at https://studio.foxglove.dev/ if you don't want to install more software.
7. If you are running `python apriltag-localizer.py` on an embedded platform like the Raspberry Pi, I'd advise against running Foxglove on the same machine. Use a development laptop or desktop connected to the same network instead, since Foxglove uses a lot of browser WebGL and Javascript.
8. Hit "Open Connection" and select "Foxglove WebSocket". Enter `ws://localhost:8765` or (`ws://ip.address.of.coprocessor:8765` if you're connecting from a remote dev machine) and connect.
9. Import the supplied layout JSON file `apriltag-localization.frontend.json` to get a view of the camera, 3D view, and list of detections.
10. Try moving an AprilTag in front of the camera to see it's detection in 3D.

## Advanced Topics
The default camera distortion model is `RADIAL` which should work for most camera setups. If your camera uses a fisheye lens, you'll need to switch the `cal_model` field to `FISHEYE`.
