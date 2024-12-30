from utils import read_video, save_video
from tracker import Tracker
import cv2
import numpy as np
from color_assigner import ColorAssigner


def main():
    video_frames = read_video("video_samples/test_video_1.mp4")

    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

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

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
