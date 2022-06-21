from pathlib import Path
from database import create_database, import_images
from matching import pairs_from_poses, superpoint, superglue


def run_sfm(image_dir: Path, camera_path: Path, pose_prior_path: Path, output_dir: Path):
    """
    Run SfM pipeline.
    :param image_dir: path of image directory.
    :param camera_path: path to camera file.
    :param pose_prior_path: path to pose prior file.
    :param output_dir: path to output directory.
    """
    assert not output_dir.exists(), f'Output directory {output_dir} already exists.'

    database_path = output_dir / 'database.db'
    pairs_path = output_dir / 'pairs.txt'
    feature_path = output_dir / 'features.h5'

    output_dir.mkdir(parents=True)

    create_database(database_path)
    import_images(database_path, image_dir, camera_path, pose_prior_path)
    pairs_from_poses(database_path, pairs_path, max_pairs=20, max_dist=3, max_angle=30, is_gps=True)
    superpoint(image_dir, feature_path)
    superglue()


if __name__ == '__main__':
    run_sfm(
        Path('/home/server/Dev/sfm-pipeline/video/images2016'),
        Path('/home/server/Dev/sfm-pipeline/cameras/VictorHD.yaml'),
        Path('/home/server/Dev/sfm-pipeline/priors2016.txt'),
        Path('/home/server/Dev/sfm-pipeline/output')
    )
