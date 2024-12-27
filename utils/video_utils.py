import cv2
from typing import List
import numpy as np
from numpy.typing import NDArray


def read_video(video_path: str) -> List[NDArray[np.uint8]]:
    """
    Read a video file and return a list of frames.

    Args:
        video_path: Path to the video file

    Returns:
        List of frames as numpy arrays
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    frames: List[NDArray[np.uint8]] = []

    while True:
        ret: bool
        frame: NDArray[np.uint8]
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def save_video(
    output_video_frames: List[NDArray[np.uint8]], output_video_path: str
) -> None:
    """
    Save a list of frames as a video file.

    Args:
        output_video_frames: List of frames to save
        output_video_path: Path where to save the video
    """
    height: int = output_video_frames[0].shape[0]
    width: int = output_video_frames[0].shape[1]
    fps: int = 24

    fourcc: int = cv2.VideoWriter_fourcc(*"XVID")
    out: cv2.VideoWriter = cv2.VideoWriter(
        output_video_path, fourcc, fps, (width, height)
    )

    for frame in output_video_frames:
        out.write(frame)

    out.release()
