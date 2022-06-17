from pathlib import Path
from database import create_database, import_images


def run_sfm(
        output_path: Path,
        image_paths: list[Path],
        camera_files: list[Path],
        pose_prior_files: list[Path] = None
):
    """
    Run SfM pipeline.
    :param output_path: path to output directory.
    :param image_paths: list of image paths.
    :param camera_files: list of camera files.
    :param pose_prior_files: list of pose prior files.
    :return: None.
    """
    assert not output_path.exists(), f'Output directory {output_path} already exists.'
    if pose_prior_files is None:
        pose_prior_files = [None] * len(image_paths)
    assert len(image_paths) == len(camera_files) == len(pose_prior_files), \
        '`image_paths`, `camera_files` and `pose_prior_files` must have the same length.'

    output_path.mkdir(parents=True)

    db_path = output_path / 'database.db'
    create_database(db_path)

    for image_path, camera_file, pose_prior_file in zip(image_paths, camera_files, pose_prior_files):
        import_images(db_path, image_path, camera_file, pose_prior_file)

    create_pairs()



if __name__ == '__main__':
    run_sfm(Path('nice'), [Path('ok')])
