import cv2
import math
import tqdm
from pathlib import Path
from datetime import datetime, timedelta


def frame_extraction(
        video_file: Path,
        output_path: Path,
        video_time: datetime = None,
        frame_interval: float = 0.0,
        start_time: float = 0.0,
        end_time: float = math.inf
):
    """Extracts frames from video
    frame_interval, start_time and end_time are in seconds
    """

    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open {video_file}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    capture_frequency = int(frame_interval * fps)
    start_frame = math.ceil(start_time * fps)

    for _ in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        _, frame = cap.read()
        frame_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        if (
                frame_index >= start_frame and
                frame_milliseconds / 1000 <= end_time and
                (
                    capture_frequency == 0 or
                    (frame_index - start_frame) % capture_frequency == 0
                )
        ):
            if video_time is None:
                image_name = f'frame{frame_index:06d}.png'
            else:
                frame_date = video_time + timedelta(milliseconds=frame_milliseconds)
                image_name = f'{frame_date.strftime("%Y%m%dT%H%M%S.%f")[:-3]}Z.png'
            cv2.imwrite(str(output_path / image_name), frame)


if __name__ == '__main__':
    #frame_extraction(Path('/home/server/Dev/underwater_reloc_benchmark/homography-loss-function/datasets/Cambridge/ShopFacade/videos/seq1.mp4'))
    frame_extraction(
        Path('/media/server/Transcend/DATA/TourEiffel RAW/2020/Momarsat20_Momarsat2005_200918042922_15.mp4'),
        Path('test'),
        frame_interval=3,
        video_time=datetime(year=2020, month=9, day=18, hour=4, minute=29, second=22),
    )
