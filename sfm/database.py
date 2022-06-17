import numpy as np
from pathlib import Path
from utils import load_camera, load_pose_priors
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
    pose_priors = load_pose_priors(pose_prior_file)
    for image_file in image_path.iterdir():
        prior_t = np.full(3, np.NaN)
        prior_q = np.full(4, np.NaN)
        if pose_priors is not None:
            if image_file.name in pose_priors:
                prior_t = pose_priors[image_file.name]['prior_t']
                prior_q = pose_priors[image_file.name]['prior_q']
            else:
                print(f'No pose prior for {image_file.name}.')
        db.add_image(image_file.name, camera_id, prior_t=prior_t, prior_q=prior_q)
    db.commit()
    db.close()