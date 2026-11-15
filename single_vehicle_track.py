"""Estimate and visualize a single vehicle's 2D track from measurements.

The module provides alpha-beta and Kalman-filter tracking functions and a
command-line entry point that loads measurements from JSON, writes the
estimated track to ``data/single_vehicle_track.json``, and generates an
annotated video. The command-line filter can be selected with ``--filter-type``;
input JSON is expected to contain an ``obj`` key with per-frame ``[x, y]``
measurements, where ``[-1, -1]`` represents a missing observation.

Originally written for CS585 Assignment 3, Part 1 (February 26, 2024).
"""

import json
import numpy as np
import argparse
import logging

from utils import load_obj_each_frame, draw_smooth_track
from kalman import KMfilter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Alpha Beta Function
def run_Alpha_Beta(measurements):
    """Estimate a vehicle track with a fixed-parameter alpha-beta filter.

    The filter starts from ``measurements[1]`` with zero velocity, then
    processes measurements from index 2 onward using ``alpha=0.85`` and
    ``beta=0.05`` with a unit time step. A missing measurement marked by
    ``[-1, -1]`` is replaced by the predicted position, resulting in no
    position residual for that update. Output coordinates are rounded to
    integers.

    Args:
        measurements (sequence of sequence of numbers): Per-frame ``[x, y]``
            measurements. At least two entries are required; the first entry
            is not used to initialize the track.

    Returns:
        list of list of int: Estimated ``[x, y]`` positions, beginning at
        input index 1 and followed by one estimate for each input measurement
        from index 2.

    Raises:
        IndexError: If fewer than two measurements are provided.

    Note:
        The nested prediction and update helpers operate on position and
        velocity vectors; this function does not modify ``measurements``.
    """
    def predict_step(prev_pos, prev_vel, step=1):
        """Predict position and velocity for one time step.

        Args:
            prev_pos (numpy.ndarray): Previous 2D position estimate.
            prev_vel (numpy.ndarray): Previous 2D velocity estimate.
            step (number): Time interval for the prediction. Defaults to 1.

        Returns:
            tuple of numpy.ndarray: Predicted position and unchanged velocity.
        """
        pos_prior = prev_pos + step * prev_vel
        vel_prior = prev_vel
        return pos_prior, vel_prior

    def update_step(pos_prior, vel_prior, pos_observed, alpha, beta, step=1):
        """Correct position and velocity estimates with an observation.

        Args:
            pos_prior (numpy.ndarray): Predicted 2D position.
            vel_prior (numpy.ndarray): Predicted 2D velocity.
            pos_observed (numpy.ndarray): Observed 2D position used to
                calculate the residual.
            alpha (number): Position correction coefficient.
            beta (number): Velocity correction coefficient.
            step (number): Time interval used to scale the velocity
                correction. Defaults to 1.

        Returns:
            tuple of numpy.ndarray: Corrected position and velocity estimates.
        """
        residual = pos_observed - pos_prior
        pos_posterior = pos_prior + alpha * residual
        vel_posterior = vel_prior + (beta / step) * residual
        return pos_posterior, vel_posterior

    # Initialization
    prev_pos = np.array(measurements[1])
    prev_vel = np.array([0, 0])
    step = 1
    alpha = 0.85
    beta = 0.05

    # Save first position to track
    track = [[int(prev_pos[0]), int(prev_pos[1])]]

    # Measurements start at second time step
    for measurement in measurements[2:]:
        # Prediction step
        pos_prior, vel_prior = predict_step(prev_pos, prev_vel, step)

        # Update step
        if measurement == [-1, -1]:
            # Use prior estimate for position as measurement
            pos_post, vel_post = update_step(
                pos_prior, vel_prior, pos_prior, alpha, beta, step
            )
        else:
            pos_post, vel_post = update_step(
                pos_prior, vel_prior, np.array(measurement), alpha, beta, step
            )

        track.append([int(round(pos_post[0])), int(round(pos_post[1]))])
        prev_pos = pos_post
        prev_vel = vel_post

    return track


def kalman_track(measurements):
    """Estimate a vehicle track with the configured :class:`KMfilter`.

    The initial position comes from ``measurements[1]`` and initial velocity
    is the difference between ``measurements[2]`` and ``measurements[1]``.
    The filter starts with a 4-by-4 identity covariance. Missing observations
    marked by ``[-1, -1]`` are replaced by the predicted position. The final
    output point is duplicated once after filtering.

    Args:
        measurements (sequence of sequence of numbers): Per-frame ``[x, y]``
            measurements. At least three entries are required for
            initialization. A missing observation is represented by
            ``[-1, -1]``.

    Returns:
        list of list of int: Estimated ``[x, y]`` positions, converted to
        integers, including one duplicate of the last filtered point.

    Raises:
        IndexError: If fewer than three measurements are provided.

    Note:
        This function does not modify ``measurements``. Its covariance
        behavior is determined by the formulas implemented by :class:`KMfilter`.
    """
    # Initialization
    prev_pos = np.array(measurements[1])
    x_pos_init = prev_pos[0]
    y_pos_init = prev_pos[1]

    prev_pos2 = np.array(measurements[2])
    x_pos_init2 = prev_pos2[0]
    y_pos_init2 = prev_pos2[1]

    x_vel_init = x_pos_init2 - x_pos_init
    y_vel_init = y_pos_init2 - y_pos_init

    prev_state = np.array([[x_pos_init], [x_vel_init], [y_pos_init], [y_vel_init]])

    # Initialize Kalman Filter
    km = KMfilter()

    # Initialize covariance matrix as identity
    mse_post = np.identity(4)

    # Save first position to track
    track = [[int(prev_pos[0]), int(prev_pos[1])]]

    # Measurements start at second time step
    for measurement in measurements[1:]:
        # Prediction step
        state_prior = km.state_predict(prev_state)
        mse_prior = km.cov_predict(mse_post)
        k = km.kalman_gain(mse_prior)

        # Update step
        if measurement == [-1, -1]:
            # Use prior estimate for position as measurement
            sub_measurement = np.array([state_prior[0], state_prior[2]])
            state_post = km.state_update(state_prior, k, sub_measurement)
        else:
            state_post = km.state_update(
                state_prior, k, np.array(measurement).reshape(2, 1)
            )

        mse_post = km.cov_update(mse_prior, k)
        track.append([int(state_post[0]), int(state_post[2])])
        prev_state = state_post

    # Duplicate last point to account for not having the first point
    track.append(track[-1])

    return track


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Single vehicle state estimation")
    argparser.add_argument(
        "--filter-type",
        help="Filter type: alpha_beta or kalman",
        choices=["alpha_beta", "kalman"],
        default="alpha_beta",
    )
    argparser.add_argument(
        "--track_path",
        help="Path to json file containing object measurements",
        default="data/object_to_track.json",
    )
    argparser.add_argument(
        "--video_path", help="Path to video file", default="data/commonwealth.mp4"
    )
    argparser.add_argument("--output_video", help="Path to output video file")

    args = argparser.parse_args()
    track_path = args.track_path
    frame_dict = load_obj_each_frame(track_path)
    measurements = frame_dict["obj"]

    output_video = (
        args.output_video
        if args.output_video
        else f"results/{args.filter_type}_single_veh_track.mp4"
    )

    logging.info(f"Using {args.filter_type} filter to track vehicle...")
    logging.info(f"Measurements loaded from {track_path}")

    # Run filter and generate
    if args.filter_type == "alpha_beta":
        track = run_Alpha_Beta(measurements)
    elif args.filter_type == "kalman":
        track = kalman_track(measurements)
    else:
        raise ValueError("Invalid filter type. Must be 'alpha_beta' or 'kalman'.")

    # Save track to json
    data = {"obj": track}
    with open("data/single_vehicle_track.json", "w") as json_file:
        json.dump(data, json_file)

    # Generate video
    draw_smooth_track(
        args.video_path, track, measurements, output_video, draw_original=True
    )
