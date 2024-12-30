import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import sys

sys.path.append("../")
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame: np.ndarray) -> None:
        """
        Initialize the CameraMovementEstimator with the first frame of the video.

        Args:
            frame (np.ndarray): The first frame of the video (in BGR format).
        """
        # Minimum distance threshold to consider camera movement
        self.minimum_distance: float = 5

        # Parameters for Lucas-Kanade optical flow algorithm
        self.lk_params: Dict = dict(
            winSize=(15, 15),  # Size of the search window
            maxLevel=2,  # Maximum pyramid level
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),  # Termination criteria
        )

        # Convert the first frame to grayscale
        first_frame_grayscale: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask to restrict feature detection to specific regions
        mask_features: np.ndarray = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # Left region
        mask_features[:, 900:1050] = 1  # Right region

        # Parameters for Shi-Tomasi corner detection
        self.features: Dict = dict(
            maxCorners=100,  # Maximum number of corners to detect
            qualityLevel=0.3,  # Minimum quality of corners
            minDistance=3,  # Minimum distance between corners
            blockSize=7,  # Size of the neighborhood for corner detection
            mask=mask_features,  # Mask to restrict detection to specific regions
        )

    def add_adjust_positions_to_tracks(
        self, tracks: Dict, camera_movement_per_frame: List[List[float]]
    ) -> None:
        """
        Adjust the positions of tracked objects based on camera movement.

        Args:
            tracks (Dict): Dictionary containing object tracks.
            camera_movement_per_frame (List[List[float]]): List of camera movements for each frame.
        """
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position: Tuple[float, float] = track_info["position"]
                    camera_movement: List[float] = camera_movement_per_frame[frame_num]
                    # Adjust the position by subtracting camera movement
                    position_adjusted: Tuple[float, float] = (
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
        stub_path: Optional[Path] = None,
    ) -> List[List[float]]:
        """
        Estimate camera movement for each frame in the video.

        Args:
            frames (List[np.ndarray]): List of video frames (in BGR format).
            read_from_stub (bool): Whether to read camera movement from a saved stub file.
            stub_path (Optional[Path]): Path to the stub file for saving/loading camera movement data.

        Returns:
            List[List[float]]: List of camera movements for each frame.
        """
        # Load camera movement from stub file if requested and the file exists
        if read_from_stub and stub_path is not None and stub_path.exists():
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        # Initialize camera movement list with zeros
        camera_movement: List[List[float]] = [[0.0, 0.0] for _ in range(len(frames))]

        # Convert the first frame to grayscale and detect features
        old_gray: np.ndarray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features: np.ndarray = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Iterate over frames to estimate camera movement
        for frame_num, frame in enumerate(frames[1:], start=1):
            frame_gray: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow between the previous and current frame
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance: float = 0.0
            camera_movement_x: float = 0.0
            camera_movement_y: float = 0.0

            # Find the maximum movement among all features
            for new, old in zip(new_features, old_features):
                new_features_point: np.ndarray = new.ravel()
                old_features_point: np.ndarray = old.ravel()

                distance: float = measure_distance(
                    new_features_point, old_features_point
                )
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    )

            # If the maximum movement exceeds the threshold, update camera movement
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Re-detect features in the current frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update the previous frame's grayscale image
            old_gray = frame_gray

        # Save camera movement to stub file if a path is provided
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self, frames: List[np.ndarray], camera_movement_per_frame: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Draw camera movement information on each frame.

        Args:
            frames (List[np.ndarray]): List of video frames (in BGR format).
            camera_movement_per_frame (List[List[float]]): List of camera movements for each frame.

        Returns:
            List[np.ndarray]: List of frames with camera movement information overlaid.
        """
        output_frames: List[np.ndarray] = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create an overlay to display camera movement information
            overlay: np.ndarray = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha: float = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get camera movement for the current frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]

            # Draw camera movement text on the frame
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
