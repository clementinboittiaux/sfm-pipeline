import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation, Slerp


def load_navigation(navigation_file: Path, camera_to_vehicle: Rotation = Rotation.from_euler('x', -np.pi / 2)):
    df = pd.read_csv(navigation_file, sep=' ')
    dates = np.array([datetime.strptime(date, '%Y/%m/%d-%H:%M:%S.%f') for date in df['date']])
    gps = df[['lat', 'lon', 'alt']].values
    world_to_vehicle = Rotation.from_euler('zyx', df[['yaw', 'pitch', 'roll']].values, degrees=True)
    rots = world_to_vehicle.inv() * camera_to_vehicle  # camera-to-world
    args = np.argsort(dates)
    return dates[args], gps[args], rots[args]


def interpolate_gps(date, date1, lat1, lon1, alt1, date2, lat2, lon2, alt2):
    line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)
    ratio = (date - date1).total_seconds() / (date2 - date1).total_seconds()
    position = line.Position(line.s13 * ratio)
    lat, lon = position['lat2'], position['lon2']
    alt = (1 - ratio) * alt1 + ratio * alt2
    return lat, lon, alt


def interpolate_rot(date, date1, date2, rots):
    ratio = (date - date1).total_seconds() / (date2 - date1).total_seconds()
    slerp = Slerp([0, 1], rots)
    return slerp(ratio)


def interpolate_image_pose(image_file, dates, gps, rots, max_gap_time=3):
    image_date = datetime.strptime(image_file.with_suffix('').name, '%Y%m%dT%H%M%S.%fZ')

    assert dates[0] <= image_date <= dates[-1], \
        f'Image {image_file.name} is out of navigation range ({dates[0]}, {dates[-1]}).'

    image_date_arg = np.searchsorted(dates, image_date, side='left')
    nav_date2 = dates[image_date_arg]
    lat2, lon2, alt2 = gps[image_date_arg]

    if image_date == nav_date2:
        lat, lon, alt, rot = lat2, lon2, alt2, rots[image_date_arg]

    else:
        nav_date1 = dates[image_date_arg - 1]
        lat1, lon1, alt1 = gps[image_date_arg - 1]

        assert nav_date2 - nav_date1 <= timedelta(seconds=max_gap_time), \
            f'Max gap time exceeded for image {image_file.name} ({nav_date1}, {nav_date2}).'

        lat, lon, alt = interpolate_gps(image_date, nav_date1, lat1, lon1, alt1, nav_date2, lat2, lon2, alt2)
        rot = interpolate_rot(image_date, nav_date1, nav_date2, rots[image_date_arg - 1:image_date_arg + 1])

    q = rot.as_quat()  # scalar-last quaternion
    if q[3] < 0:  # keep the quaternion on the top hypersphere
        q = -q

    return lat, lon, alt, q


def navigation_to_pose_priors(navigation_file: Path, image_path: Path, output_file: Path):
    dates, gps, rots = load_navigation(navigation_file)
    with open(output_file, 'w') as f:
        for image_file in image_path.iterdir():
            try:
                lat, lon, alt, q = interpolate_image_pose(image_file, dates, gps, rots)
                f.write(f'{image_file.name} {lat} {lon} {alt} {q[3]} {q[0]} {q[1]} {q[2]}\n')
            except AssertionError as err:
                print(f'{err} Skipping...')


if __name__ == '__main__':
    navigation_to_pose_priors(
        Path('/home/clementin/Dev/sfm-pipeline/test.txt'),
        Path('/home/clementin/Dev/sfm-pipeline/images'),
        Path('output.txt')
    )
    pass
