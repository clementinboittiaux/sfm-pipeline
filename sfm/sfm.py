import subprocess
from pathlib import Path
from database import create_database, import_images, import_features, import_matches
from matching import pairs_from_poses, pairs_from_netvlad, superpoint, superglue, netvlad


def run_sfm(
        colmap_path: Path,
        image_dir: Path,
        camera_path: Path,
        pose_prior_path: Path,
        output_dir: Path,
        spatial_retrieval: bool = True,
        is_gps: bool = False,
        use_priors_for_ba: bool = False,
        align_model_to_priors: bool = False,
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
    adjusted_model_dir = model_dir / 'adjusted'

    adjusted_model_dir.mkdir(parents=True)

    create_database(database_path)
    import_images(database_path, image_dir, camera_path, pose_prior_path)

    if spatial_retrieval:
        pairs_from_poses(
            database_path,
            pairs_path,
            max_pairs=max_pairs,
            max_dist=max_pair_dist,
            max_angle=max_pair_angle,
            is_gps=is_gps
        )
    else:
        netvlad_path = output_dir / 'netvlad.h5'
        netvlad(image_dir, netvlad_path)
        pairs_from_netvlad(netvlad_path, pairs_path, max_pairs=max_pairs)

    superpoint(image_dir, features_path)
    superglue(pairs_path, features_path, matches_path)
    import_features(database_path, features_path)
    import_matches(database_path, pairs_path, matches_path)

    mapper_options = {
        '--database_path': database_path,
        '--image_path': image_dir,
        '--output_path': model_dir
    }
    if use_priors_for_ba:
        mapper_options['--Mapper.use_prior_motion'] = 1
        if is_gps:
            mapper_options['--Mapper.prior_is_gps'] = 1
            mapper_options['--Mapper.use_enu_coords'] = 1
    subprocess.run([colmap_path, 'mapper', *[str(x) for kv in mapper_options.items() for x in kv]])

    bundle_options = {
        '--input_path': model_dir / '0',
        '--output_path': model_dir / 'adjusted',
        '--BundleAdjustment.refine_principal_point': 1,
        '--BundleAdjustment.max_num_iterations': 200,
        '--BundleAdjustment.function_tolerance': 0.00000001
    }
    subprocess.run([colmap_path, 'bundle_adjuster', *[str(x) for kv in bundle_options.items() for x in kv]])

    if align_model_to_priors:
        aligned_model_dir = model_dir / 'aligned'
        aligned_model_dir.mkdir()
        align_options = {
            '--input_path': adjusted_model_dir,
            '--output_path': aligned_model_dir,
            '--database_path': database_path,
            '--robust_alignment': 0
        }
        if is_gps:
            align_options['--ref_is_gps'] = 1
            align_options['--alignment_type'] = 'enu'
        subprocess.run([colmap_path, 'model_aligner', *[str(x) for kv in align_options.items() for x in kv]])


if __name__ == '__main__':
    run_sfm(
        Path('/home/server/softwares/colmap_maxime/build/src/exe/colmap'),
        Path('/home/server/Dev/sfm-pipeline/video/test2020'),
        Path('/home/server/Dev/sfm-pipeline/cameras/Victor4K.yaml'),
        Path('/home/server/Dev/sfm-pipeline/priors2020.txt'),
        Path('/home/server/Dev/sfm-pipeline/test2020'),
        spatial_retrieval=True,
        is_gps=True,
        use_priors_for_ba=True,
        align_model_to_priors=True,
        max_pairs=20,
        max_pair_dist=3,
        max_pair_angle=30
    )
