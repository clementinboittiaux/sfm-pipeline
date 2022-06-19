from pathlib import Path
from colmap.scripts.python.database import COLMAPDatabase


def pairs_from_poses(database_path: Path, output_file: Path, is_gps: bool = False):
    db = COLMAPDatabase.connect(database_path)

    db.close()
