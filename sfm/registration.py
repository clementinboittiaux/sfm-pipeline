import teaserpp_python
import numpy as np
import open3d as o3d
from pathlib import Path
from colmap.scripts.python import read_write_model
from teaserpp.examples.teaser_python_fpfh_icp.helpers import find_correspondences


def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def pcl_from_colmap(points_path: Path) -> o3d.geometry.PointCloud:
    points = read_write_model.read_points3D_binary(points_path)
    points = np.array([p.xyz for p in points.values()])
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    return pcl


def teaserpp(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud, vox_size: float):
    src_vox = src.voxel_down_sample(vox_size)
    dst_vox = dst.voxel_down_sample(vox_size)

    src_feats = extract_fpfh(src_vox, vox_size)
    dst_feats = extract_fpfh(dst_vox, vox_size)

    src_corrs, dst_corrs = find_correspondences(src_feats, dst_feats, mutual_filter=True)
    src_corr = np.array(src_vox.points)[src_corrs].T
    dst_corr = np.array(dst_vox.points)[dst_corrs].T

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = vox_size
    solver_params.estimate_scaling = True
    solver_params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(src_corr, dst_corr)
    solution = solver.getSolution()

    print('Teaser++ solution:')
    print(f'Scale: {solution.scale}')
    print(f'Rotation: {solution.rotation}')
    print(f'Translation: {solution.translation}')

    transformation = np.vstack([
        np.hstack([solution.scale * solution.rotation, solution.translation.reshape(3, 1)]),
        [0, 0, 0, 1]
    ])
    return transformation


def icp(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud, max_dist: float, initialization: np.array):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, dst, max_correspondence_distance=max_dist, init=initialization,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    transformation = reg_p2p.transformation
    scale = np.cbrt(np.linalg.det(transformation[:3, :3]))
    rotation = transformation[:3, :3] / scale
    print('ICP solution:')
    print(f'Scale: {scale}')
    print(f'Rotation: {rotation}')
    print(f'Translation: {transformation[:3, 3]}')
    return transformation


def save_transformation(transformation: np.array, output_path: Path):
    with open(output_path, 'w') as f:
        f.write('\n'.join([' '.join(map(str, row)) for row in transformation]) + '\n')


def register(src_dir: Path, dst_dir: Path, output_dir: Path, vox_size: float):
    print(f'Registering {src_dir} on {dst_dir}.')
    output_dir.mkdir(parents=True, exist_ok=True)
    if src_dir.resolve() == dst_dir.resolve():
        (output_dir / 'cameras.bin').symlink_to(src_dir / 'cameras.bin')
        (output_dir / 'images.bin').symlink_to(src_dir / 'images.bin')
        (output_dir / 'points3D.bin').symlink_to(src_dir / 'points3D.bin')
        save_transformation(np.eye(4), output_dir / 'transformation.txt')
    else:
        src_pcl = pcl_from_colmap(src_dir / 'points3D.bin')
        dst_pcl = pcl_from_colmap(dst_dir / 'points3D.bin')
        src_pcl.paint_uniform_color([1, 0.709, 0.0])
        dst_pcl.paint_uniform_color([0.0, 0.709, 1.0])
        initialization = teaserpp(src_pcl, dst_pcl, vox_size)
        transformation = icp(src_pcl, dst_pcl, vox_size, initialization)
        save_transformation(transformation, output_dir / 'transformation.txt')
        src_pcl.transform(transformation)
        o3d.visualization.draw_geometries([src_pcl, dst_pcl])
    print('Registration done.')


if __name__ == '__main__':
    register(
        Path('/workspace/TourEiffelClean/2016/sfm/model/align'),
        Path('/workspace/TourEiffelClean/2020/sfm/model/align'),
        Path('registration'),
        0.15
    )
