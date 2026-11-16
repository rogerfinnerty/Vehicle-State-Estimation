"""Assign persistent vehicle IDs to detections across video frames.

The tracker predicts each vehicle's position with a Kalman filter, associates
frame detections to existing tracks with Hungarian assignment and a distance
threshold, and starts new tracks for unmatched detections. The command-line
entry point saves the ID-annotated detection JSON and an annotated video.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman import KMfilter
from utils import load_obj_each_frame, object_center, draw_objects_in_video


def new_track(track_id, measurement):
    """Create a track initialized from a two-dimensional position.

    Args:
        track_id (int): Unique identifier assigned to the new track.
        measurement (numpy.ndarray): Position measurement with shape ``(2, 1)``
            in ``[x, y]`` order.

    Returns:
        dict: Track state containing its ``id``, constant-velocity state vector
        in ``[x, vx, y, vy]`` order, and identity covariance matrix.
    """
    x, y = measurement[:, 0]
    return {
        "id": track_id,
        "state": np.array([[x], [0.0], [y], [0.0]], dtype=float),
        "covariance": np.identity(4),
    }


def assign_detections(detections, tracks, kalman_filter, threshold):
    """Match detections to predicted tracks using gated Hungarian assignment.

    Euclidean center-to-center distances form the assignment costs. Dummy rows
    and columns allow detections and tracks to remain unmatched; real matches
    are accepted only when their distance is strictly less than ``threshold``.
    ``kalman_filter`` is accepted for call compatibility but is not used here.

    Args:
        detections (sequence of dict): Current-frame detections containing
            bounding-box fields accepted by :func:`object_center`.
        tracks (sequence of dict): Predicted track states, each containing an
            ``id`` and a state vector with position at indices 0 and 2.
        kalman_filter (KMfilter): Filter instance; currently unused by the
            assignment calculation.
        threshold (float): Maximum accepted Euclidean distance between a
            detection center and predicted track center.

    Returns:
        tuple: A three-item tuple containing a list of ``(detection_index,
        track_index)`` matches, a set of unmatched detection indices, and a
        set of unmatched track indices.
    """
    detection_count = len(detections)
    track_count = len(tracks)
    if detection_count == 0 or track_count == 0:
        return [], set(range(detection_count)), set(range(track_count))

    distances = np.zeros((detection_count, track_count), dtype=float)
    for detection_index, detection in enumerate(detections):
        center = object_center(detection)
        for track_index, track in enumerate(tracks):
            predicted_center = track["state"][[0, 2], :]
            distances[detection_index, track_index] = np.linalg.norm(
                center - predicted_center
            )

    # Dummy columns/rows let detections and tracks remain unmatched instead of
    # forcing a distant detection onto an existing identity.
    size = detection_count + track_count
    unmatched_cost = threshold / 2
    costs = np.full((size, size), threshold * 1_000_000, dtype=float)
    costs[:detection_count, :track_count] = distances

    for detection_index in range(detection_count):
        costs[detection_index, track_count + detection_index] = unmatched_cost
    for track_index in range(track_count):
        costs[detection_count + track_index, track_index] = unmatched_cost
    costs[detection_count:, track_count:] = 0

    rows, columns = linear_sum_assignment(costs)
    matches = []
    matched_detections = set()
    matched_tracks = set()
    for row, column in zip(rows, columns):
        if row < detection_count and column < track_count:
            if distances[row, column] < threshold:
                matches.append((row, column))
                matched_detections.add(row)
                matched_tracks.add(column)

    unmatched_detections = set(range(detection_count)) - matched_detections
    unmatched_tracks = set(range(track_count)) - matched_tracks
    return matches, unmatched_detections, unmatched_tracks


def track_objects(frame_dict, threshold=35):
    """Assign track IDs to detections and record track positions per frame.

    The first frame initializes one track for each detection. On subsequent
    frames, existing tracks are Kalman-predicted, detections are associated
    using :func:`assign_detections`, matched states are corrected, unmatched
    tracks retain their predictions, and unmatched detections start new
    tracks. Track IDs are added directly to the detection dictionaries.

    Args:
        frame_dict (dict): Mapping of consecutive stringified frame indices
            (``"0"``, ``"1"``, ...) to lists of detection dictionaries. Each
            detection must provide ``x_min``, ``y_min``, ``width``, and
            ``height`` values. This mapping is modified in place by adding
            ``id`` to each detection.
        threshold (float, optional): Maximum center distance for an existing
            track match. Defaults to ``35``.

    Returns:
        tuple: The mutated ``frame_dict`` and a list containing one track
        snapshot per frame. Each snapshot is a list of ``[track_id, x, y]``
        entries using the filtered or predicted positions.

    Raises:
        ValueError: If a consecutive numeric frame key is missing.
        KeyError: If a detection lacks a required bounding-box field.
        numpy.linalg.LinAlgError: If the Kalman innovation covariance cannot
            be inverted.
    """
    kalman_filter = KMfilter()
    tracks = []
    frame_track_data = []
    next_track_id = 0

    for frame_index in range(len(frame_dict)):
        frame_key = str(frame_index)
        if frame_key not in frame_dict:
            raise ValueError(f"Missing frame key {frame_key!r} in tracking data")
        detections = frame_dict[frame_key]

        if frame_index == 0:
            for detection in detections:
                measurement = object_center(detection)
                detection["id"] = next_track_id
                tracks.append(new_track(next_track_id, measurement))
                next_track_id += 1
        else:
            for track in tracks:
                track["state"] = kalman_filter.state_predict(track["state"])
                track["covariance"] = kalman_filter.cov_predict(track["covariance"])
                track["gain"] = kalman_filter.kalman_gain(track["covariance"])

            matches, unmatched_detections, unmatched_tracks = assign_detections(
                detections, tracks, kalman_filter, threshold
            )

            for detection_index, track_index in matches:
                detection = detections[detection_index]
                track = tracks[track_index]
                measurement = object_center(detection)
                track["state"] = kalman_filter.state_update(
                    track["state"], track["gain"], measurement
                )
                track["covariance"] = kalman_filter.cov_update(
                    track["covariance"], track["gain"]
                )
                detection["id"] = track["id"]

            for track_index in unmatched_tracks:
                track = tracks[track_index]
                predicted_measurement = kalman_filter.H @ track["state"]
                track["state"] = kalman_filter.state_update(
                    track["state"], track["gain"], predicted_measurement
                )
                track["covariance"] = kalman_filter.cov_update(
                    track["covariance"], track["gain"]
                )

            for detection_index in unmatched_detections:
                detection = detections[detection_index]
                measurement = object_center(detection)
                detection["id"] = next_track_id
                tracks.append(new_track(next_track_id, measurement))
                next_track_id += 1

        frame_track_data.append(
            [
                [
                    track["id"],
                    float(track["state"][0, 0]),
                    float(track["state"][2, 0]),
                ]
                for track in tracks
            ]
        )

    return frame_dict, frame_track_data


def main():
    """Parse command-line options and run tracking, JSON, and video outputs.

    Supported options select the detection JSON, source video, annotated
    output video, tracked JSON output, and association-distance threshold.
    Parent directories for the output paths are created when needed.

    Returns:
        None.

    Raises:
        OSError: If the source video or output video cannot be opened.
        ValueError: If the tracking JSON is missing a required frame key.
        KeyError: If an input detection is missing a required field.
    """
    parser = argparse.ArgumentParser(description="Multi-object vehicle tracking")
    parser.add_argument(
        "--tracks_path",
        default="data/frame_dict.json",
        help="JSON file containing frame-indexed object detections",
    )
    parser.add_argument(
        "--video_path",
        default="data/commonwealth.mp4",
        help="Source video to annotate",
    )
    parser.add_argument(
        "--output_video",
        default="results/multi_vehicle_tracking.mp4",
        help="Path for the annotated output video",
    )
    parser.add_argument(
        "--output_tracks",
        default="results/multi_vehicle_tracks.json",
        help="Path for the JSON output with assigned track IDs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=35,
        help="Maximum center distance for associating a detection to a track",
    )
    args = parser.parse_args()

    frame_dict = load_obj_each_frame(args.tracks_path)
    tracked_frame_dict, frame_track_data = track_objects(
        frame_dict, threshold=args.threshold
    )

    output_tracks = Path(args.output_tracks)
    output_tracks.parent.mkdir(parents=True, exist_ok=True)
    with output_tracks.open("w", encoding="utf-8") as json_file:
        json.dump(tracked_frame_dict, json_file)

    draw_objects_in_video(
        args.video_path,
        tracked_frame_dict,
        frame_track_data,
        args.output_video,
    )
    logging.info("Tracked detections saved to %s", output_tracks)
    logging.info("Annotated video saved to %s", args.output_video)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()