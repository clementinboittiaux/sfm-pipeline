import argparse
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


def bundle(colmap_path: Path, model_dir: Path, bundle_dir: Path, refine_camera: bool = True):
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_options = {
        '--input_path': model_dir,
        '--output_path': bundle_dir,
        '--BundleAdjustment.max_num_iterations': 200,
        '--BundleAdjustment.function_tolerance': 0.0000000001
    }
    if refine_camera:
        bundle_options['--BundleAdjustment.refine_principal_point'] = 1
    else:
        bundle_options['--BundleAdjustment.refine_focal_length'] = 0
        bundle_options['--BundleAdjustment.refine_principal_point'] = 0
        bundle_options['--BundleAdjustment.refine_extra_params'] = 0
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
    if output_dir.exists():
        print(f'Warning: {output_dir} already exists.')
    output_dir.mkdir(parents=True, exist_ok=True)
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
    bundle(colmap_path, model_dir / '0', bundle_dir, refine_camera=True)
    if align_model_to_priors:
        align(colmap_path, bundle_dir, database_path, model_dir / 'align', is_gps=is_gps)


def merge_sfms(
        colmap_path: Path,
        sfm_dirs: list[Path],
        output_dir: Path,
        max_pairs: int = 50,
        max_pair_dist: float = 3,
        max_pair_angle: float = 45
):
    if output_dir.exists():
        print(f'Warning: {output_dir} already exists.')
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / 'images'
    database_path = output_dir / 'database.db'
    pairs_path = output_dir / 'pairs.txt'
    features_path = output_dir / 'features.h5'
    matches_path = output_dir / 'matches.h5'
    model_dir = output_dir / 'model'
    bundle_dir = model_dir / 'bundle'

    create_database(database_path)

    for sfm_dir in sfm_dirs:
        import_images(
            database_path,
            sfm_dir / 'images',
            sfm_dir / 'sfm' / 'model' / 'register' / 'cameras.bin',
            sfm_dir / 'sfm' / 'model' / 'register' / 'images.bin'
        )
        symlink_images(sfm_dir / 'images', image_dir)

    pairs_from_poses(
        database_path,
        pairs_path,
        max_pairs=max_pairs,
        max_dist=max_pair_dist,
        max_angle=max_pair_angle
    )

    superpoint(image_dir, features_path)
    superglue(pairs_path, features_path, matches_path)
    import_features(database_path, features_path)
    import_matches(database_path, pairs_path, matches_path)

    mapper(colmap_path, database_path, image_dir, model_dir, refine_camera=False)
    bundle(colmap_path, model_dir / '0', bundle_dir, refine_camera=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structure-from-Motion.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    parser_sfm = subparsers.add_parser('sfm', help='Single scene Strucure-from-Motion',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_merge = subparsers.add_parser('merge', help='Merge SfM outputs.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_sfm.add_argument('--colmap-path', type=Path, help='path to COLMAP executable.',
                            default='/home/server/softwares/colmap_maxime/build/src/exe/colmap')
    parser_sfm.add_argument('--image-dir', required=True, type=Path, help='path to image directory.')
    parser_sfm.add_argument('--camera-path', required=True, type=Path,
                            help='path to camera file (either `.yaml` file or COLMAP `cameras.bin`).')
    parser_sfm.add_argument('--pose-priors-path', required=True, type=Path,
                            help='path to pose priors file (either `.txt` file or COLMAP `images.bin`).')
    parser_sfm.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser_sfm.add_argument('--spatial-retrieval', type=int, default=1,
                            help='use spatial retrieval instead of NetVLAD.')
    parser_sfm.add_argument('--is-gps', type=int, default=0, help='whether or not pose priors are GPS coordinates.')
    parser_sfm.add_argument('--use-priors-for-ba', type=int, default=0,
                            help='use pose priors during bundle adjustments.')
    parser_sfm.add_argument('--align-model-to-priors', type=int, default=0, help='align model to pose priors.')
    parser_sfm.add_argument('--max-pairs', type=int, default=20, help='maximum number of pairs during retrieval.')
    parser_sfm.add_argument('--max-pair-dist', type=float, default=5,
                            help='max distance between pairs during spatial retrieval.')
    parser_sfm.add_argument('--max-pair-angle', type=float, default=45,
                            help='max angular distance between pairs in degrees during spatial retrieval.')
    parser_merge.add_argument('--colmap-path', type=Path, help='path to COLMAP executable.',
                              default='/home/server/softwares/colmap_maxime/build/src/exe/colmap')
    parser_merge.add_argument('--sfm-dirs', required=True, nargs='+', type=Path,
                              help='paths to SfM output directories.')
    parser_merge.add_argument('--output-dir', required=True, type=Path, help='path to output directory.')
    parser_merge.add_argument('--max-pairs', type=int, default=50, help='maximum number of pairs during retrieval.')
    parser_merge.add_argument('--max-pair-dist', type=float, default=5,
                              help='max distance between pairs during spatial retrieval.')
    parser_merge.add_argument('--max-pair-angle', type=float, default=45,
                              help='max angular distance between pairs in degrees during spatial retrieval.')
    args = parser.parse_args()

    if args.command == 'sfm':
        run_sfm(
            args.colmap_path,
            args.image_dir,
            args.camera_path,
            args.pose_priors_path,
            args.output_dir,
            spatial_retrieval=args.spatial_retrieval,
            is_gps=args.is_gps,
            use_priors_for_ba=args.use_priors_for_ba,
            align_model_to_priors=args.align_model_to_priors,
            max_pairs=args.max_pairs,
            max_pair_dist=args.max_pair_dist,
            max_pair_angle=args.max_pair_angle
        )
    elif args.command == 'merge':
        merge_sfms(
            args.colmap_path,
            args.sfm_dirs,
            args.output_dir,
            max_pairs=args.max_pairs,
            max_pair_dist=args.max_pair_dist,
            max_pair_angle=args.max_pair_angle
        )
