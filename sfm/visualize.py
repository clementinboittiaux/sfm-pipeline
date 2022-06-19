import numpy as np
import open3d as o3d
from pathlib import Path
from utils import load_pose_priors
from datetime import datetime
from navigation import gps_to_enu, load_navigation
from scipy.spatial.transform import Rotation


def show_cameras(Rs, ts, scale=0.2):
    camera_mesh = o3d.io.read_triangle_mesh('camera.obj')
    camera_mesh.paint_uniform_color([0, 0.709, 0])

    cameras = []
    for R, t in zip(Rs, ts):
        camera = o3d.geometry.TriangleMesh(camera_mesh)
        camera.transform(np.vstack([
            np.hstack([scale * R, t.reshape(3, 1)]),
            [0, 0, 0, 1]
        ]))
        cameras.append(camera)

    o3d.visualization.draw_geometries(cameras)


def visualize_pose_priors(pose_prior_file: Path, scale: float = 0.2):
    pose_priors = load_pose_priors(pose_prior_file)
    ts = gps_to_enu([image['prior_t'] for image in pose_priors.values()])
    qs = np.array([image['prior_q'] for image in pose_priors.values()])
    Rs = Rotation.from_quat(np.roll(qs, -1, axis=1)).as_matrix()
    show_cameras(Rs, ts, scale)


def visualize_navigation(navigation_file: Path, min_date: datetime, max_date: datetime, scale: float = 0.2):
    dates, gps, rots = load_navigation(navigation_file)
    indices = (min_date <= dates) & (dates <= max_date)
    gps, rots = gps[indices], rots[indices]
    gps, rots = gps[::3], rots[::3]  # TODO: incorporate slicing in arguments
    ts = gps_to_enu(gps)
    Rs = rots.as_matrix()
    show_cameras(Rs, ts, scale)


if __name__ == '__main__':
    visualize_pose_priors(Path('/home/clementin/Dev/sfm-pipeline/priors.txt'))
