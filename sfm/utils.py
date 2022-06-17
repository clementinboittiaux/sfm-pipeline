import yaml
import numpy as np


def load_camera(camera_file):
    with open(camera_file, 'r') as f:
        camera = yaml.safe_load(f)
    return camera


def load_pose_priors(pose_prior_file):
    """
    File format for GPS data is:
    filename1 lat1 lon1 alt1 qw1 qx1 qy1 qz1
    filename2 lat2 lon2 alt2
    ...
    File format for cartesian data is:
    filename1 X1 Y1 Z1 qw1 qx1 qy1 qz1
    filename2 X2 Y2 Z2
    ...
    where qw, qx, qy, qz are optional quaternions.
    :param pose_prior_file: path to prior file.
    :return: priors
    """
    if pose_prior_file is not None:
        pose_priors = {}
        with open(pose_prior_file, 'r') as f:
            for line in f:
                line = line.split(' ')
                prior_t = np.array(list(map(float, line[1:4])))
                prior_q = np.full(4, np.NaN)
                if len(line) == 8:
                    prior_q = np.array(list(map(float, line[4:8])))
                pose_priors[line[0]] = {'prior_t': prior_t, 'prior_q': prior_q}
        return pose_priors
