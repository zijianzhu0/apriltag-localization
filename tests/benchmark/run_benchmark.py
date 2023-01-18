import glob
import os
import time
import yaml

import cv2
import numpy as np

# Set toplevel dir as import path
import sys
sys.path.insert(1, './')
from frontend import TagDetector
from localize import build_map, localize_single_tag, localize_multi_tags


if __name__ == '__main__':
    # load config
    with open('tests/benchmark/benchmark_config.yaml', 'r') as fp:
        yaml_config = yaml.load(fp, yaml.Loader)

    localizer_yaml = yaml_config["localizer"]
    tag_map = build_map(localizer_yaml["tag_map"])
    tag_detector = TagDetector(yaml_config["frontend"], tag_map)

    print('loading sample images...')
    image_paths = glob.glob('tests/benchmark/dataset/*.jpg')
    print(f'globbed {len(image_paths)} test images')

    all_errors = []
    frontend_latencies = []
    estimation_latencies = []
    for path in image_paths:
        img_basename = os.path.basename(path)
        print(f'testing {img_basename}...')
        img = cv2.imread(path)

        # parse x y field ground truth coordinates and convert to meters
        truth = np.array(list(map(float, img_basename.split('_')[0:2])))
        truth *= 0.0254
        print(f'  ground truth: {truth}')

        # process frame through frontend
        start = time.time()
        detections, overlay = tag_detector.process_frame(img)
        frontend_latency = time.time() - start
        assert len(detections) > 0
        print(f'  detected {len(detections)} tags latency {frontend_latency} seconds')
        frontend_latencies.append(frontend_latency)

        # perform pose estimation
        start = time.time()
        if len(detections) == 1:
            cam_pose = localize_single_tag(detections, tag_map)
        else:
            cam_pose = localize_multi_tags(detections, tag_map, localizer_yaml["detections_noise"])
        estimation_latency = time.time() - start
        measured_pos = cam_pose.translation()[0:2]
        print(f'  localized position: {measured_pos} latency {estimation_latency} seconds')
        print(f'  attitude (ypr degrees): {np.rad2deg(cam_pose.rotation().ypr())}')
        estimation_latencies.append(estimation_latency)

        # evaluate error
        error = np.linalg.norm(truth - measured_pos)
        print(f'  localization error: {error} meters')
        all_errors.append(error)

        # cv2.imshow('overlay', overlay)
        # cv2.waitKey(0)

    frontend_latencies = np.array(frontend_latencies)
    estimation_latencies = np.array(estimation_latencies)
    end2end_latencies = frontend_latencies + estimation_latencies
    print('\n\n'+'='*20+' Benchmark Report '+'='*20)
    print('localization error:')
    print(f'  mean={np.mean(all_errors)} meters')
    print(f'  min={np.min(all_errors)} meters')
    print(f'  max={np.max(all_errors)} meters')
    print('performance:')
    print(f'  mean frontend latency: {np.mean(frontend_latencies)} seconds')
    print(f'  mean estimation latency: {np.mean(estimation_latencies)} seconds')
    print(f'  mean end2end latency: {np.mean(end2end_latencies)} seconds')
    print('='*58)

