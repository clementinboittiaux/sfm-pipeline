import numpy as np
from utils import load_camera, load_pose_priors
from pathlib import Path
from colmap.scripts.python.database import COLMAPDatabase
from colmap.scripts.python.read_write_model import CAMERA_MODEL_NAMES


def create_database(database_path: Path):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(database_path: Path, image_path: Path, camera_file: Path, pose_prior_file: Path = None):
    db = COLMAPDatabase.connect(database_path)
    camera = load_camera(camera_file)
    camera_id = db.add_camera(
        CAMERA_MODEL_NAMES[camera['model']].model_id,
        camera['width'],
        camera['height'],
        camera['params'],
        prior_focal_length=True
    )
    pose_priors = load_pose_priors(pose_prior_file) if pose_prior_file is not None else {}
    for image_file in image_path.iterdir():
        if image_file.name in pose_priors:
            prior_t = pose_priors[image_file.name]['prior_t']
            if 'prior_q' in pose_priors[image_file.name]:
                prior_q = pose_priors[image_file.name]['prior_q']
                db.add_image(image_file.name, camera_id, prior_t=prior_t, prior_q=prior_q)
            else:
                print(f'No rotation prior for {image_file.name}.')
                db.add_image(image_file.name, camera_id, prior_t=prior_t)
        else:
            print(f'No pose prior for {image_file.name}.')
            db.add_image(image_file.name, camera_id)
    db.commit()
    db.close()


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
        prior_qs.append(row[3:7])
        prior_ts.append(row[7:10])
    db.close()
    return np.array(image_ids), np.array(image_names), np.array(camera_ids), np.array(prior_qs), np.array(prior_ts)
