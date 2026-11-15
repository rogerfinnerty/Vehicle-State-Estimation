# Single Vehicle Tracking

A lightweight Python project for single-object tracking using an Alpha-Beta filter and a Kalman filter.

This repository focuses on estimating a tracked object's center over time from noisy measurements and visualizing the filtered trajectory on video.

## Overview

The project includes:

- a simple Alpha-Beta tracker
- a 2D Kalman filter implementation
- JSON loading for measurement data
- video annotation for the estimated track
- example output generation for single-vehicle tracking

### Kalman Filter

![Kalman filter tracking animation](results/kalman_single_veh_track.gif)

### Alpha-Beta Filter

![Alpha-Beta filter tracking animation](results/alpha_beta_single_veh_track.gif)

## Included components

- `single_vehicle_track.py` — main tracking script
- `kalman.py` — Kalman filter implementation
- `utils.py` — file loading and video rendering helpers
- `data/` — input tracking and video data
- `results/` — generated output files

## Usage

Run the tracker with either filter:

```bash
python single_vehicle_track.py --filter-type kalman
```

or

```bash
python single_vehicle_track.py --filter-type alpha_beta
```

Optional arguments:

```bash
python single_vehicle_track.py \
  --filter-type kalman \
  --track_path data/object_to_track.json \
  --video_path data/commonwealth.mp4 \
  --output_video results/single_vehicle_track.mp4
```

## Data

The project expects tracking JSON and video inputs in the `data/` directory.

Example placeholder paths:

- `data/object_to_track.json`
- `data/commonwealth.mp4`

## Example outputs

Generated outputs are written to the `results/` directory, for example:

- `results/kalman_track.mp4`
- `results/alpha_beta_track.mp4`
- `results/single_vehicle_track.mp4`

## Placeholder media

Add your actual example videos here:

- `data/commonwealth.mp4`
- `results/demo_video.mp4`

## Requirements

See `requirements.txt` for the project dependencies.

## Notes

This project is intentionally scoped to the single-vehicle tracking problem. It demonstrates the difference between a simple motion model filter and a state-estimation Kalman filter approach.
