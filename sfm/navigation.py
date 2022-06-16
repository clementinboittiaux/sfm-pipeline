import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic


class Navigation:
    """
    Process Ifremer navigation files.
    """

    def __init__(self, navigation_file: Path, sep: str = ','):
        """
        Store navigation data in the object.
        :param navigation_file: path to the navigation file.
        :param sep: separator used in the navigation file.
        """
        self.df = pd.read_csv(navigation_file, sep=sep)
        dates = self.df.iloc[:, 0] + self.df.iloc[:, 1]
        self.df['date'] = dates.map(lambda x: datetime.strptime(x, '%d/%m/%Y%H:%M:%S.%f'))
        self.min_date = self.df['date'].iloc[0]
        self.max_date = self.df['date'].iloc[-1]

    def interpolate(self, date: datetime, time_limit: float = 3):
        """
        Interpolate latitude, longitude and altitude at a given date.
        :param date: date to interpolate.
        :param time_limit: limit between two dates in seconds.
        :return: lat, lon, alt.
        """
        if not self.min_date <= date <= self.max_date:
            raise Exception(f'Requested date ({date}) is out of the date range '
                            f'(from {self.min_date} to {self.max_date})')

        if date == self.min_date:
            lat, lon, alt = self.df.iloc[0, 2:5]

        else:
            date_arg = self.df['date'].searchsorted(date, side='left')
            date1 = self.df['date'].iloc[date_arg - 1]
            date2 = self.df['date'].iloc[date_arg]

            if date == date2:
                lat, lon, alt = self.df.iloc[date_arg, 2:5]

            else:
                if date2 - date1 > timedelta(seconds=time_limit):
                    raise Exception(f'Too much time between samples (from {date1} to {date2})')

                lat1, lon1, alt1 = self.df.iloc[date_arg - 1, 2:5]
                lat2, lon2, alt2 = self.df.iloc[date_arg, 2:5]
                line = Geodesic.WGS84.InverseLine(lat1, lon1, lat2, lon2)
                ratio = (date - date1).total_seconds() / (date2 - date1).total_seconds()
                position = line.Position(line.s13 * ratio)
                lat, lon = position['lat2'], position['lon2']
                alt = (1 - ratio) * alt1 + ratio * alt2

        alt = -abs(alt)
        return lat, lon, alt


if __name__ == '__main__':
    nav = Navigation(Path('/media/server/Transcend/DATA/TourEiffel RAW/2016/MOMARSAT1613.txt'), sep='\t')
    nav.interpolate(datetime.strptime('10/09/201609:08:21.912', '%d/%m/%Y%H:%M:%S.%f'))
