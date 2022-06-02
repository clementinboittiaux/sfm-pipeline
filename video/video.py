import cv2
import math
import tqdm
import ffmpeg
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def deinterlace(input_file: Path, output_file: Path, invert_lines: bool = False):
    """Deinterlace a video and save it with a lossless codec.
    Output extension is usually '.mkv'.
    If `invert_lines` is True, inverts odd and even lines before deinterlacing.
    FFV1 help: https://trac.ffmpeg.org/wiki/Encode/FFV1
    Line inverting help: http://underpop.online.fr/f/ffmpeg/help/il.htm.gz
    """
    stream = ffmpeg.input(str(input_file))

    if invert_lines:
        stream = stream.filter('il', ls=1, cs=1)

    stream = stream.filter('yadif', 0)
    stream = stream.output(str(output_file), vcodec='ffv1', level=3)
    stream.run()


def frame_extraction(
        input_file: Path,
        output_path: Path,
        video_time: datetime = None,
        frame_interval: float = 0.0,
        start_time: float = 0.0,
        end_time: float = math.inf
):
    """Extracts frames from video.
    `frame_interval`, `start_time` and `end_time` are in seconds.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_file))

    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open {input_file}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    capture_frequency = int(frame_interval * fps)
    start_frame = math.ceil(start_time * fps)

    for _ in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        _, frame = cap.read()
        frame_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)

        if (
                frame_index >= start_frame and frame_milliseconds / 1000 <= end_time and
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
    parser = argparse.ArgumentParser(description='Video processing.')
    parser.add_argument('command', choices=['deinterlace', 'frame_extraction'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--video-time', help='in format YYYY/mm/dd-HH:MM:SS.',
                        type=lambda x: datetime.strptime(x, '%Y/%m/%d-%H:%M:%S'))
    parser.add_argument('--frame-interval', type=float, default=0.0, help='time between frames in seconds.')
    parser.add_argument('--start-time', type=float, default=0.0, help='extraction start time in seconds.')
    parser.add_argument('--end-time', type=float, default=math.inf, help='extraction end time in seconds')
    parser.add_argument('--invert-lines', action='store_true')
    args = parser.parse_args()

    if args.command == 'deinterlace':
        for arg in ['input', 'output']:
            assert args.__dict__[arg] is not None, f'Argument --{arg} is required for {args.command}.'
        deinterlace(args.input, args.output, args.invert_lines)

    elif args.command == 'frame_extraction':
        for arg in ['input', 'output']:
            assert args.__dict__[arg] is not None, f'Argument --{arg} is required for {args.command}.'
        frame_extraction(
            args.input,
            args.output,
            args.video_time,
            args.frame_interval,
            args.start_time,
            args.end_time
        )
