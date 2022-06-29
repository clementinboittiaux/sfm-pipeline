import subprocess
from utils import symlink_images
from pathlib import Path
from database import create_database, import_images, import_features, import_matches
from matching import pairs_from_poses, pairs_from_netvlad, superpoint, superglue, netvlad


def mapper(
        colmap_path: Path,
        database_path: Path,
        image_dir: Path,
        model_dir: Path,
        use_priors: bool = False,
        is_gps: bool = False,
        refine_camera: bool = True
):
    model_dir.mkdir(parents=True, exist_ok=True)
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
    if not refine_camera:
        mapper_options['--Mapper.ba_refine_focal_length'] = 0
        mapper_options['--Mapper.ba_refine_principal_point'] = 0
        mapper_options['--Mapper.ba_refine_extra_params'] = 0
    subprocess.run([colmap_path, 'mapper', *[str(x) for kv in mapper_options.items() for x in kv]])


def bundle(colmap_path: Path, model_dir: Path, bundle_dir: Path):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_options = {
        '--input_path': model_dir,
        '--output_path': bundle_dir,
        '--BundleAdjustment.refine_principal_point': 1,
        '--BundleAdjustment.max_num_iterations': 200,
        '--BundleAdjustment.function_tolerance': 0.0000000001
    }
    subprocess.run([colmap_path, 'bundle_adjuster', *[str(x) for kv in bundle_options.items() for x in kv]])


def align(colmap_path: Path, model_dir: Path, database_path: Path, align_dir: Path, is_gps: bool = False):
    align_dir.mkdir(parents=True)
    align_options = {
        '--input_path': model_dir,
        '--output_path': align_dir,
        '--database_path': database_path,
        '--robust_alignment': 0
    }
    if is_gps:
        align_options['--ref_is_gps'] = 1
        align_options['--alignment_type'] = 'enu'
    subprocess.run([colmap_path, 'model_aligner', *[str(x) for kv in align_options.items() for x in kv]])



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
    output_dir.mkdir(parents=True, exist_ok=False)
    symlink_images(image_dir, output_dir / 'images')

    database_path = output_dir / 'database.db'
    pairs_path = output_dir / 'pairs.txt'
    features_path = output_dir / 'features.h5'
    matches_path = output_dir / 'matches.h5'
    model_dir = output_dir / 'model'
    bundle_dir = model_dir / 'bundle'

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

    mapper(colmap_path, database_path, image_dir, model_dir, use_priors=use_priors_for_ba, is_gps=is_gps)
    bundle(colmap_path, model_dir / '0', bundle_dir)
    if align_model_to_priors:
        align(colmap_path, bundle_dir, database_path, model_dir / 'align', is_gps=is_gps)


def merge_sfms(colmap_path: Path, sfm_dirs: list[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)

    image_dir = output_dir / 'images'
    database_path = output_dir / 'database.db'
    pairs_path = output_dir / 'pairs.txt'
    features_path = output_dir / 'features.h5'
    matches_path = output_dir / 'matches.h5'
    model_dir = output_dir / 'model'

    create_database(database_path)

    for sfm_dir in sfm_dirs:
        # Registration saved in model/registrated
        import_images(
            database_path,
            sfm_dir / 'images',
            sfm_dir / 'registrated' / 'cameras.bin',
            sfm_dir / 'registrated' / 'images.bin'
        )
        symlink_images(sfm_dir / 'images', image_dir)

    pairs_from_poses(
        database_path,
        pairs_path,
        max_pairs=50,
        max_dist=3,
        max_angle=45
    )

    superpoint(image_dir, features_path)
    superglue(pairs_path, features_path, matches_path)
    import_features(database_path, features_path)
    import_matches(database_path, pairs_path, matches_path)

    mapper(colmap_path, database_path, image_dir, model_dir, refine_camera=False)


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
