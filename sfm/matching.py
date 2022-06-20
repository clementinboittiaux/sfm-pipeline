import numpy as np
from utils import angle_between_quaternions
from pathlib import Path
from database import load_database_images
from navigation import gps_to_enu


def pairs_from_poses(
        database_path: Path,
        output_file: Path,
        max_pairs: int = 20,
        max_dist: float = 3,
        max_angle: float = 30,
        is_gps: bool = False
):
    _, image_names, _, prior_qs, prior_ts = load_database_images(database_path)
    if is_gps:
        prior_ts = gps_to_enu(prior_ts)
    with open(output_file, 'w') as f:
        for image_name, prior_q, prior_t in zip(image_names, prior_qs, prior_ts):
            t_dist = np.linalg.norm(prior_ts - prior_t, axis=1)
            q_dist = np.rad2deg(angle_between_quaternions(prior_q, prior_qs))
            indices = (t_dist <= max_dist) & (q_dist <= max_angle) & (image_name != image_names)
            valid_t_dist, valid_image_names = t_dist[indices], image_names[indices]
            pairs = valid_image_names[np.argsort(valid_t_dist)]
            for pair in pairs[:max_pairs]:
                f.write(f'{image_name} {pair}\n')


if __name__ == '__main__':
    pairs_from_poses(Path('/home/server/Dev/sfm-pipeline/output/database.db'), Path('/home/server/Dev/sfm-pipeline/output/pairs.txt'), max_pairs=10, is_gps=True)
