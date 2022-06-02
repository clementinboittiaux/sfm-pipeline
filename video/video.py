import cv2
import math
import tqdm
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def frame_extraction(
        video_file: Path,
        output_path: Path,
        frame_interval: float = 0.0,
        video_time: datetime = None
):
    output_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open {video_file}')
    capture_frequency = int(frame_interval * cap.get(cv2.CAP_PROP_FPS))
    for frame_index in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        _, frame = cap.read()
        if frame_index % capture_frequency == 0:
            if video_time is None:
                image_name = f'frame{frame_index:06d}.png'
            else:
                frame_date = video_time + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
                image_name = f'{frame_date.strftime("%Y%m%dT%H%M%S.%f")[:-3]}Z.png'
            cv2.imwrite(str(output_path / image_name), frame)


if __name__ == '__main__':
    #frame_extraction(Path('/home/server/Dev/underwater_reloc_benchmark/homography-loss-function/datasets/Cambridge/ShopFacade/videos/seq1.mp4'))
    frame_extraction(
        Path('/media/server/Transcend/DATA/TourEiffel RAW/2020/Momarsat20_Momarsat2005_200918042922_15.mp4'),
        Path('test'),
        frame_interval=3,
        video_time=datetime(year=2020, month=9, day=18, hour=4, minute=29, second=22)
    )
