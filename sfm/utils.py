import yaml
import numpy as np
from pathlib import Path
from colmap.scripts.python import read_write_model


def load_camera(camera_path: Path) -> dict:
    if camera_path.suffix == '.txt':
        with open(camera_path, 'r') as f:
            camera = yaml.safe_load(f)
        return camera
    elif camera_path.suffix == '.bin':
        cameras = list(read_write_model.read_cameras_binary(camera_path))
        assert len(cameras) == 1, f'Invalid camera file: {camera_path} has {len(cameras)} cameras.'
        return cameras[0]._asdict()
    else:
        raise Exception(f'Invalid suffix for {camera_path}, must be `.txt` or `.bin`.')


def load_pose_priors_txt(pose_prior_path: Path) -> dict:
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


def load_pose_priors_colmap(pose_prior_path: Path) -> dict:
    images = read_write_model.read_images_binary(pose_prior_path)
    pose_priors = {}
    for image in images.values():
        prior_q = image.qvec.copy()
        if prior_q[0] < 0:
            prior_q[0] = -prior_q[0]
        else:
            prior_q[1:] = -prior_q[1:]
        pose_priors[image.name] = {
            'prior_t': -image.qvec2rotmat().T @ image.tvec,
            'prior_q': prior_q
        }
    return pose_priors


def load_pose_priors(pose_prior_path: Path) -> dict:
    if pose_prior_path.suffix == '.txt':
        return load_pose_priors_txt(pose_prior_path)
    elif pose_prior_path.suffix == '.bin':
        return load_pose_priors_colmap(pose_prior_path)
    else:
        raise Exception(f'Invalid suffix for {pose_prior_path}, must be `.txt` or `.bin`.')


def angle_between_quaternions(q, r):
    """
    Returns the angle in radians between quaternion `q` and quaternions `r`.
    :param q: quaternion.
    :param r: batch of quaternions.
    :return: angular distance between `q` and `r`.
    """
    return 2 * np.arccos(np.abs(np.sum(q * r, axis=1)).clip(0, 1))


def symlink_images(image_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_dir.iterdir():
        (output_dir / image_path.name).symlink_to(image_path.resolve())
