import subprocess
from pathlib import Path
from database import create_database, import_images, import_features, import_matches
from matching import pairs_from_poses, superpoint, superglue


def run_sfm(
        colmap_path: Path,
        image_dir: Path,
        camera_path: Path,
        pose_prior_path: Path,
        output_dir: Path,
        is_gps: bool = False,
        use_priors: bool = False,
        max_pairs: int = 20,
        max_pair_dist: float = 3,
        max_pair_angle: float = 30
):
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
    features_path = output_dir / 'features.h5'
    matches_path = output_dir / 'matches.h5'
    model_dir = output_dir / 'model'

    model_dir.mkdir(parents=True)

    create_database(database_path)
    import_images(database_path, image_dir, camera_path, pose_prior_path)
    pairs_from_poses(
        database_path,
        pairs_path,
        max_pairs=max_pairs,
        max_dist=max_pair_dist,
        max_angle=max_pair_angle,
        is_gps=is_gps
    )
    superpoint(image_dir, features_path)
    superglue(pairs_path, features_path, matches_path)
    import_features(database_path, features_path)
    import_matches(database_path, pairs_path, matches_path)

    mapper_options = {
        '--database_path': database_path,
        '--image_path': image_dir,
        '--output_path': model_dir
    }
    if use_priors:
        mapper_options['--Mapper.use_prior_motion'] = 1
        if is_gps:
            mapper_options['--Mapper.prior_is_gps'] = 1
            mapper_options['--Mapper.use_enu_coords'] = 1
    subprocess.run([colmap_path, 'mapper', *[str(x) for kv in mapper_options.items() for x in kv]])


if __name__ == '__main__':
    run_sfm(
        Path('/home/server/softwares/colmap_maxime/build/src/exe/colmap'),
        Path('/home/server/Dev/sfm-pipeline/video/images2016'),
        Path('/home/server/Dev/sfm-pipeline/cameras/VictorHD.yaml'),
        Path('/home/server/Dev/sfm-pipeline/priors2016.txt'),
        Path('/home/server/Dev/sfm-pipeline/output'),
        is_gps=True,
        use_priors=True,
        max_pairs=20,
        max_pair_dist=3,
        max_pair_angle=30
    )
