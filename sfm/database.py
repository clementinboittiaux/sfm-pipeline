import numpy as np
from utils import load_camera, load_pose_priors
from pathlib import Path
from hloc.hloc import triangulation
from hloc.hloc.reconstruction import get_image_ids, geometric_verification
from colmap.scripts.python.database import COLMAPDatabase
from colmap.scripts.python.read_write_model import CAMERA_MODEL_NAMES


def create_database(database_path: Path):
    print('Creating database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()
    print('Finished creating database.')


def import_images(database_path: Path, image_dir: Path, camera_path: Path, pose_prior_path: Path = None):
    print('Importing images...')
    db = COLMAPDatabase.connect(database_path)
    camera = load_camera(camera_path)
    camera_id = db.add_camera(
        CAMERA_MODEL_NAMES[camera['model']].model_id,
        camera['width'],
        camera['height'],
        camera['params'],
        prior_focal_length=True
    )
    pose_priors = load_pose_priors(pose_prior_path) if pose_prior_path is not None else {}
    for image_path in image_dir.iterdir():
        if image_path.name in pose_priors:
            prior_t = pose_priors[image_path.name]['prior_t']
            if 'prior_q' in pose_priors[image_path.name]:
                prior_q = pose_priors[image_path.name]['prior_q']
                db.add_image(image_path.name, camera_id, prior_t=prior_t, prior_q=prior_q)
            else:
                print(f'No rotation prior for {image_path.name}.')
                db.add_image(image_path.name, camera_id, prior_t=prior_t)
        else:
            print(f'No pose prior for {image_path.name}.')
            db.add_image(image_path.name, camera_id)
    db.commit()
    db.close()
    print('Finished importing images.')


def load_database_images(database_path: Path):
    db = COLMAPDatabase.connect(database_path)
    cur = db.cursor()
    cur.execute('SELECT * from images')
    results = cur.fetchall()
    image_ids, image_names, camera_ids, prior_qs, prior_ts = [], [], [], [], []
    for row in results:
        image_ids.append(row[0])
        image_names.append(row[1])
        camera_ids.append(row[2])
        if row[3] is not None:
            prior_qs.append(row[3:7])
        else:
            prior_qs.append((np.nan, np.nan, np.nan, np.nan))
        if row[7] is not None:
            prior_ts.append(row[7:10])
        else:
            prior_ts.append((np.nan, np.nan, np.nan))
    db.close()
    return np.array(image_ids), np.array(image_names), np.array(camera_ids), np.array(prior_qs), np.array(prior_ts)


def import_features(database_path: Path, features_path: Path):
    triangulation.import_features(
        get_image_ids(database_path),
        database_path,
        features_path
    )


def import_matches(
        database_path: Path,
        pairs_path: Path,
        matches_path: Path,
        min_match_score=None
):
    triangulation.import_matches(
        get_image_ids(database_path),
        database_path,
        pairs_path,
        matches_path,
        min_match_score=min_match_score,
        skip_geometric_verification=False
    )
    geometric_verification(database_path, pairs_path)
