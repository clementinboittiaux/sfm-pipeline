import argparse
import numpy as np
import pandas as pd
import pymap3d as pm
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation, Slerp


class CameraToVehicle(Enum):
    """
    Camera-to-vehicle rotation transformation.
    """

    VICTORHD = Rotation.from_euler('x', -np.pi / 2)

    def __str__(self):
        return self.name


def convert_old_nav(old_nav_path: Path, new_nav_path: Path):
    """
    Converts old `.txt` navigation file to new navigation format.
    Old format is:
    Date Heure Latitude Longitude Immersion Cap Pitch Roll
    separated by \\t character.
    :param old_nav_path: path to old navigation file.
    :param new_nav_path: path to new navigation file.
    """
    with open(old_nav_path, 'r') as r, open(new_nav_path, 'w') as w:
        r.readline()
        w.write('date lat lon alt yaw pitch roll\n')
        for line in r:
            line = line.split('\t')
            date = datetime.strptime(line[0]+line[1], '%d/%m/%Y%H:%M:%S.%f')
            w.write(date.strftime('%Y%m%dT%H%M%S.%f') + ' ' + ' '.join(line[2:8]) + '\n')


def load_navigation(navigation_path: Path, camera_to_vehicle: CameraToVehicle = CameraToVehicle.VICTORHD):
    df = pd.read_csv(navigation_path, sep=' ')
    dates = np.array([datetime.strptime(date, '%Y%m%dT%H%M%S.%f') for date in df['date']])
    gps = df[['lat', 'lon', 'alt']].values
    world_to_vehicle = Rotation.from_euler('ZYX', df[['yaw', 'pitch', 'roll']].values, degrees=True)
    rots = world_to_vehicle.inv() * camera_to_vehicle.value  # camera-to-world
    indices = np.argsort(dates)
    return dates[indices], gps[indices], rots[indices]


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


def interpolate_image_pose(image_path, dates, gps, rots, max_gap_time: float = 3):
    image_date = datetime.strptime(image_path.with_suffix('').name, '%Y%m%dT%H%M%S.%fZ')

    assert dates[0] <= image_date <= dates[-1], \
        f'Image {image_path.name} is out of navigation range ({dates[0]}, {dates[-1]}).'

    image_date_arg = np.searchsorted(dates, image_date, side='left')
    nav_date2 = dates[image_date_arg]
    lat2, lon2, alt2 = gps[image_date_arg]

    if image_date == nav_date2:
        lat, lon, alt, rot = lat2, lon2, alt2, rots[image_date_arg]

    else:
        nav_date1 = dates[image_date_arg - 1]
        lat1, lon1, alt1 = gps[image_date_arg - 1]

        assert nav_date2 - nav_date1 <= timedelta(seconds=max_gap_time), \
            f'Max gap time exceeded for image {image_path.name} ({nav_date1}, {nav_date2}).'

        lat, lon, alt = interpolate_gps(image_date, nav_date1, lat1, lon1, alt1, nav_date2, lat2, lon2, alt2)
        rot = interpolate_rot(image_date, nav_date1, nav_date2, rots[image_date_arg - 1:image_date_arg + 1])

    q = rot.as_quat()  # scalar-last quaternion
    if q[3] < 0:  # keep the quaternion on the top hypersphere
        q = -q

    return lat, lon, alt, q


def navigation_to_pose_priors(
        navigation_path: Path,
        image_dir: Path,
        output_path: Path,
        max_gap_time: float = 3,
        camera_to_vehicle: CameraToVehicle = CameraToVehicle.VICTORHD
):
    dates, gps, rots = load_navigation(navigation_path, camera_to_vehicle)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w') as f:
        for image_path in image_dir.iterdir():
            try:
                lat, lon, alt, q = interpolate_image_pose(image_path, dates, gps, rots, max_gap_time)
                f.write(f'{image_path.name} {lat} {lon} {alt} {q[3]} {q[0]} {q[1]} {q[2]}\n')
            except AssertionError as err:
                print(f'{err} Skipping...')


def gps_to_enu(gps: np.array):
    lat0, lon0, alt0 = gps[0]
    enu = np.zeros_like(gps, dtype=np.float64)
    for i, (lat, lon, alt) in enumerate(gps):
        enu[i] = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    return enu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process navigation files.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')

    parser_priors = subparsers.add_parser(
        'nav-to-priors',
        help='Convert navigation file into pose priors file.\n'
        'Navigation data is interpolated at the images dates.\n'
        'Navigation file format is:\n'
        'date lat lon alt yaw pitch roll\n'
        'date1 lat1 lon1 alt1 yaw1 pitch1 roll1\n'
        'date2 lat2 lon2 alt2 yaw2 pitch2 roll2\n'
        '...\n'
        'where dates are in format `YYYYmmddTHHMMSS.ffffff`.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser_priors.add_argument('--navigation-path', required=True, type=Path, help='path to navigation file.')
    parser_priors.add_argument('--image-dir', required=True, type=Path, help='path to images directory.')
    parser_priors.add_argument('--output-path', required=True, type=Path, help='path to output pose priors file.')
    parser_priors.add_argument('--max-gap-time', type=float, default=3,
                               help='maximum time gap between two interpolation points in seconds. '
                                    '(default: %(default)s)')
    parser_priors.add_argument('--camera-to-vehicle', type=lambda x: CameraToVehicle[x],
                               default='VICTORHD', choices=list(CameraToVehicle),
                               help='camera-to-vehicle rotation transformation. (default: %(default)s)')

    parser_convert = subparsers.add_parser(
        'convert-old-nav', help='Convert old navigation file to new format.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser_convert.add_argument('--old-nav-path', required=True, type=Path, help='path to old navigation file.')
    parser_convert.add_argument('--new-nav-path', required=True, type=Path, help='path to new navigation file.')

    args = parser.parse_args()

    if args.command == 'nav-to-priors':
        navigation_to_pose_priors(
            args.navigation_path,
            args.image_dir,
            args.output_path,
            max_gap_time=args.max_gap_time,
            camera_to_vehicle=args.camera_to_vehicle
        )
    elif args.command == 'convert-old-nav':
        convert_old_nav(args.old_nav_path, args.new_nav_path)
