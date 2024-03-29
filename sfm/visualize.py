import numpy as np
import open3d as o3d
from pathlib import Path
from utils import load_pose_priors
from datetime import datetime
from navigation import gps_to_enu, load_navigation
from registration import pcl_from_colmap
from scipy.spatial.transform import Rotation


def show_cameras(Rs, ts, scale=0.2, colors=None):
    camera_mesh = o3d.io.read_triangle_mesh('camera.obj')
    cameras = []
    for R, t, color in zip(Rs, ts, colors):
        camera = o3d.geometry.TriangleMesh(camera_mesh)
        if color is None:
            camera_mesh.paint_uniform_color([0, 0.709, 0])
        else:
            camera_mesh.paint_uniform_color(color)
        camera.transform(np.vstack([
            np.hstack([scale * R, t.reshape(3, 1)]),
            [0, 0, 0, 1]
        ]))
        cameras.append(camera)

    o3d.visualization.draw_geometries(cameras)


def visualize_pose_priors(pose_prior_paths: list[Path], scale: float = 0.2):
    pose_priors = {}
    for pose_prior_path in pose_prior_paths:
        color = np.random.rand(3)
        current_pose_priors = load_pose_priors(pose_prior_path)
        for image in current_pose_priors.values():
            image['color'] = color
        pose_priors |= current_pose_priors
    ts = gps_to_enu([image['prior_t'] for image in pose_priors.values()])
    qs = np.array([image['prior_q'] for image in pose_priors.values()])
    Rs = Rotation.from_quat(np.roll(qs, -1, axis=1)).as_matrix()
    colors = [image['color'] for image in pose_priors.values()]
    show_cameras(Rs, ts, scale, colors)


def visualize_navigation(navigation_path: Path, min_date: datetime, max_date: datetime, scale: float = 0.2):
    dates, gps, rots = load_navigation(navigation_path)
    indices = (min_date <= dates) & (dates <= max_date)
    gps, rots = gps[indices], rots[indices]
    gps, rots = gps[::3], rots[::3]  # TODO: incorporate slicing in arguments
    ts = gps_to_enu(gps)
    Rs = rots.as_matrix()
    show_cameras(Rs, ts, scale)


def visualize_models(colmap_dirs: list[Path]):
    pcls = []
    for colmap_dir in colmap_dirs:
        pcl = pcl_from_colmap(colmap_dir / 'points3D.bin')
        pcl.paint_uniform_color(np.random.rand(3))
        pcls.append(pcl)
    o3d.visualization.draw_geometries(pcls)


if __name__ == '__main__':
    # visualize_models([
    #     Path('/workspace/TourEiffelClean/2015/sfm/model/register'),
    #     Path('/workspace/TourEiffelClean/2016/sfm/model/register'),
    #     Path('/workspace/TourEiffelClean/2018/sfm/model/register'),
    #     Path('/workspace/TourEiffelClean/2020/sfm/model/register')
    # ])
    visualize_pose_priors(
        [
            Path('/workspace/TourEiffelClean/2015/sfm/priors.txt'),
            Path('/workspace/TourEiffelClean/2016/sfm/priors.txt'),
            Path('/workspace/TourEiffelClean/2018/sfm/priors.txt'),
            Path('/workspace/TourEiffelClean/2020/sfm/priors.txt')
        ],
        scale=0.2
    )
