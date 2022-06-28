import numpy as np
from utils import angle_between_quaternions
from pathlib import Path
from database import load_database_images
from hloc.hloc import extract_features, match_features
from navigation import gps_to_enu


def pairs_from_poses(
        database_path: Path,
        output_path: Path,
        max_pairs: int = 20,
        max_dist: float = 3,
        max_angle: float = 30,
        is_gps: bool = False
):
    print('Computing image pairs...')
    _, image_names, _, prior_qs, prior_ts = load_database_images(database_path)
    if is_gps:
        prior_ts = gps_to_enu(prior_ts)
    num_best_pairs = int(max_pairs * 2 / 3)
    num_stratified_pairs = max_pairs - num_best_pairs
    num_pairs = []
    with open(output_path, 'w') as f:
        for image_name, prior_q, prior_t in zip(image_names, prior_qs, prior_ts):
            t_dist = np.linalg.norm(prior_ts - prior_t, axis=1)
            q_dist = np.rad2deg(angle_between_quaternions(prior_q, prior_qs))
            indices = (t_dist <= max_dist) & (q_dist <= max_angle) & (image_name != image_names)
            valid_t_dist, valid_image_names = t_dist[indices], image_names[indices]
            candidates = valid_image_names[np.argsort(valid_t_dist)]
            if len(candidates) == 0:
                print(f'No pairs found for {image_name}')
            elif len(candidates) <= max_pairs:
                pairs = candidates[:max_pairs]
            else:
                best_pairs = candidates[:num_best_pairs]
                stratified_pairs = candidates[
                    np.linspace(num_best_pairs, len(candidates) - 1, num_stratified_pairs, dtype=np.int32)
                ]
                pairs = np.hstack([best_pairs, stratified_pairs])
            num_pairs.append(min(len(pairs), max_pairs))
            for pair in pairs:
                f.write(f'{image_name} {pair}\n')
    print(f'Finished pairs computing (average number of pairs per image: {sum(num_pairs) / len(num_pairs)}).')


def superpoint(image_dir: Path, features_path: Path):
    extract_features.main(
        extract_features.confs['superpoint_aachen'],
        image_dir=image_dir,
        feature_path=features_path
    )


def superglue(pairs_path: Path, features_path: Path, matches_path: Path):
    match_features.main(
        match_features.confs['superglue'],
        pairs_path,
        features_path,
        matches=matches_path
    )
