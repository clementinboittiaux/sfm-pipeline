import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation, Slerp


class Navigation:
    """
    File format:
    date(YYYY/mm/dd-HH:MM:SS.ffffff) lat lon alt yaw pitch roll
    date1 lat1 lon1 alt1 yaw1 pitch1 roll1
    date2 lat2 lon2 alt2 yaw2 pitch2 roll2
    ...
    Dates must be sorted.
    """

    def __init__(self, navigation_file: Path):
        """
        Store navigation data in the object.
        :param navigation_file: path to the navigation file.
        """
        df = pd.read_csv(navigation_file, sep=' ')
        df['date'] = df['date'].map(lambda x: datetime.strptime(x, '%Y/%m/%d-%H:%M:%S.%f'))
        self.df = df
        self.min_date = self[0][0]
        self.max_date = self[-1][0]
        self.v_R_c = Rotation.from_euler('x', -np.pi / 2)  # Victor HD camera-to-vehicle transformation

    def __getitem__(self, item):
        return self.df.iloc[item]

    def yaw_pitch_roll_to_rotation(self, yaw, pitch, roll):
        v_R_w = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)  # world-to-vehicle
        w_R_c = v_R_w.inv() * self.v_R_c  # camera-to-world
        return w_R_c

    def interpolate(self, date: datetime, time_limit: float = 3):
        """
        Interpolate latitude, longitude and altitude at a given date.
        :param date: date to interpolate.
        :param time_limit: limit between two dates in seconds.
        :return: lat, lon, alt, qw, qx, qy, qz
        """
        if not self.min_date <= date <= self.max_date:
            raise Exception(f'Requested date ({date}) is out of the date range '
                            f'(from {self.min_date} to {self.max_date})')

        if date == self.min_date:
            _, lat, lon, alt, yaw, pitch, roll = self[0]
            w_R_c = self.yaw_pitch_roll_to_rotation(yaw, pitch, roll)

        else:
            date_arg = self.df['date'].searchsorted(date, side='left')
            date1, lat1, lon1, alt1, yaw1, pitch1, roll1 = self[date_arg - 1]
            date2, lat2, lon2, alt2, yaw2, pitch2, roll2 = self[date_arg]
            w_R1_c = self.yaw_pitch_roll_to_rotation(yaw1, pitch1, roll1)
            w_R2_c = self.yaw_pitch_roll_to_rotation(yaw2, pitch2, roll2)

            if date == date2:
                lat, lon, alt, w_R_c = lat2, lon2, alt2, w_R2_c

            else:
                if date2 - date1 > timedelta(seconds=time_limit):
                    raise Exception(f'Too much time between samples (from {date1} to {date2})')

                line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)
                ratio = (date - date1).total_seconds() / (date2 - date1).total_seconds()
                position = line.Position(line.s13 * ratio)
                lat, lon = position['lat2'], position['lon2']
                alt = (1 - ratio) * alt1 + ratio * alt2
                slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([w_R1_c.as_matrix(), w_R2_c.as_matrix()])))
                w_R_c = slerp(ratio)

        w_q_c = w_R_c.as_quat()  # scalar-last quaternion
        w_q_c = np.roll(w_q_c, 1)  # scalar-first quaternion
        if w_q_c[0] < 0:  # keep the quaterion on the top hypersphere
            w_q_c = -w_q_c

        return lat, lon, alt, w_q_c

    def to_pose_priors(self, image_path: Path, output_file: Path):
        with open(output_file, 'w') as f:
            for image_file in image_path.iterdir():
                image_date = datetime.strptime(image_file.with_suffix('').name, '%Y%m%dT%H%M%S.%fZ')
                lat, lon, alt, q = self.interpolate(image_date)
                f.write(f'{image_file.name} {lat} {lon} {alt} {q[0]} {q[1]} {q[2]} {q[3]}\n')


if __name__ == '__main__':
    nav = Navigation(Path('/home/server/Dev/sfm-pipeline/test_nav.txt'))
    nav.to_pose_priors(Path('/home/server/Dev/sfm-pipeline/video/images2016'), Path('output.txt'))
    pass
