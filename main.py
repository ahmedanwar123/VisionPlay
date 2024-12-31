"""
Video Processing Pipeline for Football Tracking and Analysis.

This module processes a video to track objects (players, referees and a ball), estimate camera movement,
transform views, calculate speed and distance, assign teams, and determine ball control. The final
output is an annotated video with the analyzed data visualized.

Functions:
    main: The main function that orchestrates the video processing pipeline.
"""

from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from color_assigner import ColorAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    """
    Execute the video processing pipeline.

    This function performs the following steps:
    1. Reads the input video.
    2. Initializes the object tracker and extracts object tracks.
    3. Estimates camera movement and adjusts object positions.
    4. Transforms object positions to a different view.
    5. Interpolates ball positions for smooth tracking.
    6. Estimates speed and distance for tracked objects.
    7. Assigns teams to players based on team colors.
    8. Assigns ball possession to players in each frame.
    9. Draws annotations (tracks, camera movement, speed, distance, ball control) on the video.
    10. Saves the annotated video as an output file.
    """
    # Read Video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker("models/best.pt")

    # Extract object tracks from the video
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )
    # Add positions to the tracks
    tracker.add_position_to_tracks(tracks)

    # Estimate camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # Transform object positions to a different view
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions for smooth tracking
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Estimate speed and distance for tracked objects
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    color_assigner = ColorAssigner()
    color_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = color_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                color_assigner.team_colors[team]
            )

    # Assign ball possession to players in each frame
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw annotations on the video
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the annotated video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
