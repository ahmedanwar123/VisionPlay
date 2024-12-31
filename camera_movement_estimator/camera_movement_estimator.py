import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append("../")
from utils import measure_distance, measure_xy_distance
from typing import List, Tuple, Dict, Any, Optional  # noqa


class CameraMovementEstimator:
    """
    A class to estimate and adjust camera movement in a sequence of video frames.

    Attributes:
        minimum_distance (float): The minimum distance threshold for considering camera movement.
        lk_params (dict): Parameters for the Lucas-Kanade optical flow algorithm.
        features (dict): Parameters for detecting good features to track in the frames.
    """

    def __init__(self, frame: np.ndarray) -> None:
        """
        Initializes the CameraMovementEstimator with the first frame of the video.

        Args:
            frame (np.ndarray): The first frame of the video.
        """
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(
        self,
        tracks: Dict[str, List[Dict[int, Dict[str, Any]]]],
        camera_movement_per_frame: List[List[float]],
    ) -> None:
        """
        Adjusts the positions of objects in the tracks based on the estimated camera movement.

        Args:
            tracks (Dict[str, List[Dict[int, Dict[str, Any]]]]): A dictionary containing the tracks of objects.
            camera_movement_per_frame (List[List[float]]): A list of camera movements for each frame.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1],
                    )
                    tracks[object][frame_num][track_id]["position_adjusted"] = (
                        position_adjusted
                    )

    def get_camera_movement(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Estimates the camera movement for each frame in the video.

        Args:
            frames (List[np.ndarray]): A list of video frames.
            read_from_stub (bool): Whether to read the camera movement from a stub file.
            stub_path (Optional[str]): The path to the stub file.

        Returns:
            List[List[float]]: A list of camera movements for each frame.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self, frames: List[np.ndarray], camera_movement_per_frame: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Draws the camera movement information on the frames.

        Args:
            frames (List[np.ndarray]): A list of video frames.
            camera_movement_per_frame (List[List[float]]): A list of camera movements for each frame.

        Returns:
            List[np.ndarray]: A list of frames with camera movement information drawn on them.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            output_frames.append(frame)

        return output_frames
