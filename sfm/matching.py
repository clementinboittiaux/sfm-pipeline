import numpy as np
import pandas as pd
from utils import angle_between_quaternions
from pathlib import Path
from database import load_database_images
from hloc.hloc import extract_features, match_features, pairs_from_retrieval
from hloc.hloc.utils.io import list_h5_names
from navigation import gps_to_enu


def read_pairs(pairs_path: Path) -> dict[str, list[str]]:
    pairs = {}
    with open(pairs_path, 'r') as f:
        for line in f:
            im0, im1 = line.split(' ')
            if im0 not in pairs:
                pairs[im0] = []
            pairs[im0].append(im1.strip())
    return pairs


def write_pairs(pairs_path: Path, pairs: dict[str, list[str]]):
    with open(pairs_path, 'w') as f:
        for image, image_pairs in pairs.items():
            for image_pair in image_pairs:
                f.write(f'{image} {image_pair}\n')


def slice_pairs_file(pairs_path: Path, new_pairs_path: Path, image_list: list[str]):
    with open(pairs_path, 'r') as r, open(new_pairs_path, 'w') as w:
        for line in r:
            im0, im1 = line.split(' ')
            if im0 in image_list:
                w.write(line)


def list_paired_images(pairs_path: Path) -> list[str]:
    return list(read_pairs(pairs_path).keys())


def pairs_from_poses(
        database_path: Path,
        pairs_path: Path,
        max_pairs: int = 20,
        max_dist: float = 3,
        max_angle: float = 30,
        is_gps: bool = False,
        best_pairs_ratio: float = 1.0,
        num_force_already_paired: int = 0,
):
    print('Computing image pairs...')
    already_paired = list_paired_images(pairs_path)
    if not pairs_path.exists():
        pairs_path.touch()
    _, image_names, _, prior_qs, prior_ts = load_database_images(database_path)
    if is_gps:
        prior_ts = gps_to_enu(prior_ts)
    num_best_pairs = int(max_pairs * best_pairs_ratio)
    num_stratified_pairs = max_pairs - num_best_pairs
    num_pairs = []
    with open(pairs_path, 'a') as f:
        for image_name, prior_q, prior_t in zip(image_names, prior_qs, prior_ts):
            if image_name not in already_paired:
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
                if num_force_already_paired != 0:
                    already_paired_candidates = candidates[np.isin(candidates, already_paired)]
                    pairs = np.hstack([pairs, already_paired_candidates[:num_force_already_paired]])
                    pairs = pd.unique(pairs)
                num_pairs.append(len(pairs))
                for pair in pairs:
                    f.write(f'{image_name} {pair}\n')
    print(f'Finished pairs computing (average number of pairs per image: {sum(num_pairs) / len(num_pairs)}).')


def pairs_from_netvlad(
        netvlad_path: Path,
        pairs_path: Path,
        max_pairs: int = 20,
        num_force_already_paired: int = 0
):
    if not pairs_path.exists():
        pairs_from_retrieval.main(netvlad_path, pairs_path, num_matched=max_pairs)

    else:
        all_names = list_h5_names(netvlad_path)
        already_paired = list_paired_images(pairs_path)
        not_paired = np.setdiff1d(all_names, already_paired)  # all images in H5 file that have no pairs in pairs.txt
        new_pairs_path = pairs_path.parent / 'new_pairs.txt'
        pairs_from_retrieval.main(netvlad_path, new_pairs_path, num_matched=max_pairs, query_list=not_paired)
        new_pairs = read_pairs(new_pairs_path)
        new_pairs_path.unlink()

        if num_force_already_paired != 0:
            new_pairs_db_path = pairs_path.parent / 'new_pairs_db.txt'
            pairs_from_retrieval.main(
                netvlad_path,
                new_pairs_db_path,
                num_matched=num_force_already_paired,
                query_list=not_paired,
                db_list=already_paired
            )
            new_pairs_db = read_pairs(new_pairs_db_path)
            new_pairs_db_path.unlink()
            for image, image_pairs_db in new_pairs_db.items():
                image_pairs = new_pairs[image]
                merged_pairs = pd.unique(np.hstack([image_pairs, image_pairs_db])).tolist()
                new_pairs[image] = merged_pairs

        pairs = read_pairs(pairs_path) | new_pairs
        write_pairs(pairs_path, pairs)


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


def netvlad(image_dir: Path, netvlad_path: Path):
    extract_features.main(
        extract_features.confs['netvlad'],
        image_dir=image_dir,
        feature_path=netvlad_path
    )
