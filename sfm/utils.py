import yaml
import numpy as np
from pathlib import Path


def load_camera(camera_path: Path) -> dict:
    with open(camera_path, 'r') as f:
        camera = yaml.safe_load(f)
    return camera


def load_pose_priors(pose_prior_path: Path) -> dict:
    """
    File format for GPS data is:
    imagename1 lat1 lon1 alt1 qw1 qx1 qy1 qz1
    imagename2 lat2 lon2 alt2
    ...
    File format for cartesian data is:
    imagename1 X1 Y1 Z1 qw1 qx1 qy1 qz1
    imagename2 X2 Y2 Z2
    ...
    where qw, qx, qy, qz are optional camera-to-world quaternions.
    :param pose_prior_path: path to prior file.
    :return: priors
    """
    pose_priors = {}
    with open(pose_prior_path, 'r') as f:
        for line in f:
            line = line.split(' ')
            assert len(line) == 4 or len(line) == 8, f'File {pose_prior_path} format is invalid.'
            image_name = line[0]
            pose_priors[image_name] = {}
            pose_priors[image_name]['prior_t'] = np.array(list(map(float, line[1:4])))
            if len(line) == 8:
                pose_priors[image_name]['prior_q'] = np.array(list(map(float, line[4:8])))
    return pose_priors


def angle_between_quaternions(q, r):
    """
    Returns the angle in radians between quaternion `q` and quaternions `r`.
    :param q: quaternion.
    :param r: batch of quaternions.
    :return: angular distance between `q` and `r`.
    """
    return 2 * np.arccos(np.abs(np.sum(q * r, axis=1)).clip(0, 1))
