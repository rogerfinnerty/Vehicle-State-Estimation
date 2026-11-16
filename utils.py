"""Utilities for loading tracking data and visualizing vehicle tracks.

The video helpers use OpenCV and assume input coordinates correspond to frames
resized to 700 by 500 pixels. Drawing functions update image arrays in place
through OpenCV and write generated videos to the requested or fixed output path.
"""

import json
import logging
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_obj_each_frame(data_file):
    """Load a JSON tracking-data file.

    Args:
        data_file (str or os.PathLike): Path to a JSON file containing
            frame-based object data.

    Returns:
        dict: The decoded JSON value, typically a dictionary keyed by frame.

    Raises:
        OSError: If the file cannot be opened.
        json.JSONDecodeError: If the file contents are not valid JSON.
    """
    with open(data_file, 'r') as file:
        frame_dict = json.load(file)
    return frame_dict


def draw_target_object_center(video_file, obj_centers, vid_name):
    """Write a video showing the target object's center over time.

    Frames are resized to 700 by 500 pixels. Earlier centers are drawn in
    blue, and the center for the current frame is drawn in red. The output is
    encoded at 30 frames per second using the ``MP4V`` codec.
    The function creates or overwrites ``vid_name``, releases the video
    writer, and logs a success message.

    Args:
        video_file (str or os.PathLike): Path to the source video.
        obj_centers (sequence of sequence of numbers): One ``(x, y)`` center
            coordinate per video frame, in resized-frame coordinates.
        vid_name (str or os.PathLike): Path for the generated video.

    Returns:
        None.

    Note:
        The center sequence must contain an entry for every frame read from
        the source video. Coordinates are converted to integers before
        drawing.
    """
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter(vid_name, cv.VideoWriter.fourcc(*'MP4V'), 30, (700,500))
    while ok:
        image = cv.resize(image, (700, 500))

        for past in obj_centers[:count]:
            pos_x,pos_y = past
            image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (255,0,0), 2)

        pos_x,pos_y = obj_centers[count]
        image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)

        vidwrite.write(image)
        ok, image = cap.read()

        count+=1
    vidwrite.release()

    logging.info(f"Video saved to {vid_name}")


def draw_smooth_track(video_file, obj_centers, measurements, vid_name, draw_original=False):
    """Write a video overlaying an estimated track and optional measurements.

    Frames are resized to 700 by 500 pixels. The estimated track is drawn as
    a blue polyline. When ``draw_original`` is true, past valid measurements
    are drawn in red and the current valid measurement in green; ``[-1, -1]``
    is treated as a missing-measurement marker. A legend is included on every
    frame.
    The function creates or overwrites ``vid_name`` and logs a success
    message. The output uses the ``mp4v`` codec at 30 frames per second.

    Args:
        video_file (str or os.PathLike): Path to the source video.
        obj_centers (sequence of sequence of numbers): Estimated ``(x, y)``
            positions, one per video frame, in resized-frame coordinates.
        measurements (sequence of sequence of numbers): Measured ``(x, y)``
            positions. Used for drawing only when ``draw_original`` is true;
            entries must align with video frames.
        vid_name (str or os.PathLike): Path for the generated video.
        draw_original (bool): Whether to include the measurements and
            measurement legend markers. Defaults to ``False``.

    Returns:
        None.

    Note:
        The coordinate sequences must cover all frames read from the source
        video. OpenCV drawing operations modify each frame before it is
        written.
    """
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter(vid_name, cv.VideoWriter.fourcc(*'mp4v'), 30, (700,500))

    # Legend box settings
    legend_x, legend_y = 450, 20
    legend_w, legend_h = 230, 100
    legend_color = (30, 30, 30)
    text_color = (255, 255, 255)

    while ok:
        image = cv.resize(image, (700, 500))

        track = np.array(obj_centers[:count], dtype=np.int32)
        image = cv.polylines(image, [track], isClosed=False, color=(255,0,0), thickness=3)

        # Draw original measurements
        if draw_original:
            for measurement in measurements[:count]:
                pos_x,pos_y = measurement
                if measurement != [-1, -1]:
                    image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)

            current = measurements[count]
            if current != [-1,-1]:
                image = cv.circle(image, (int(current[0]),int(current[1])), 1, (0,255,0), 2)

        # Add legend
        image = cv.rectangle(image, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), legend_color, -1)
        image = cv.rectangle(image, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), (200, 200, 200), 1)

        image = cv.line(image, (legend_x + 15, legend_y + 25), (legend_x + 90, legend_y + 25), (255, 0, 0), 3)
        image = cv.putText(image, 'Estimated track', (legend_x + 105, legend_y + 31), cv.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        image = cv.circle(image, (legend_x + 52, legend_y + 55), 4, (0, 0, 255), -1)
        image = cv.putText(image, 'Measurements', (legend_x + 105, legend_y + 60), cv.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        image = cv.circle(image, (legend_x + 52, legend_y + 85), 4, (0, 255, 0), -1)
        image = cv.putText(image, 'Current', (legend_x + 105, legend_y + 91), cv.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        vidwrite.write(image)
        ok, image = cap.read()

        count += 1
    vidwrite.release()

    logging.info(f"Video saved to {vid_name}")


def plot_track_and_measurements(track, measurements):
    """Display estimated track positions and measured positions in a plot.

    Estimated positions are plotted as blue asterisks and measurements as red
    asterisks. Both inputs are expected to contain ``(x, y)`` coordinate
    pairs. A Matplotlib figure is displayed with
    :func:`matplotlib.pyplot.show`.

    Args:
        track (iterable of sequence of numbers): Estimated positions to plot.
        measurements (iterable of sequence of numbers): Measured positions to
            plot.

    Returns:
        None.

    """
    for pos in track:
        plt.plot(pos[0], pos[1], 'b*')
    for pos in measurements:
        plt.plot(pos[0], pos[1], 'r*')

    plt.show()


def draw_car(car, image):
    """Draw a known car's identifier and circular marker on an image.

    The input record is interpreted as ``[identifier, x, y]``. The identifier
    is rendered at ``(x, y)`` and a circle with radius 35 is drawn around that
    point. Coordinates are converted to integers.

    Args:
        car (sequence): Car record containing an identifier followed by its x
            and y coordinates.
        image (numpy.ndarray): Image to draw on. OpenCV modifies this array in
            place.

    Returns:
        numpy.ndarray: The modified image (the same image object supplied by
        the caller).
    """
    x = int(car[1])
    y = int(car[2])

    id = car[0]

    # Initialize cv.putText
    text = str(id)
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (x, y)
    font_scale = 0.5
    color = (0, 0, 255)  # blue in BGR format
    thickness = 2

    # Add the text to the image
    image = cv.putText(image, text, position, font, font_scale, color, thickness)
    return cv.circle(image, (x,y), radius=35, color=(0, 0, 255), thickness=1)


def draw_object(object_dict,image,color = (0, 255, 0), thickness = 2,c_color= (255, 0, 0)):
    """Draw a labeled bounding box for one detected object.

    The rectangle is positioned from the object's top-left corner and its
    width and height. The object identifier is then drawn at the top-left
    corner. OpenCV interprets colors as BGR tuples.

    Args:
        object_dict (mapping): Object record with ``x_min``, ``y_min``,
            ``width``, ``height``, and ``id`` keys.
        image (numpy.ndarray): Image to draw on. OpenCV modifies this array in
            place.
        color (tuple of int): BGR color for the bounding rectangle. Defaults
            to ``(0, 255, 0)``.
        thickness (int): Bounding-rectangle line thickness in pixels. Defaults
            to ``2``.
        c_color (tuple of int): Retained for API compatibility; it is
            currently unused. The label color is set internally. Defaults to
            ``(255, 0, 0)``.

    Returns:
        numpy.ndarray: The modified image (the same image object supplied by
        the caller).
    """
    # draw box
    x = object_dict['x_min']
    y = object_dict['y_min']
    width = object_dict['width']
    height = object_dict['height']
    image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    # Add id to image
    id = object_dict['id']

    # Add the text to the image
    text = str(id)
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (x, y)
    font_scale = 1
    color = (255, 0, 0)  # blue in BGR format
    thickness = 2
    cv.putText(image, text, position, font, font_scale, color, thickness)

    return image


def draw_objects_in_video(video_file, frame_dict, frame_track_data, output_video):
    """Write a video with detection boxes and tracked vehicle markers.

    Source frames are resized to 700 by 500 pixels. Detection bounding boxes
    and IDs are drawn from ``frame_dict``; filtered or predicted track centers
    and IDs are drawn from the matching entry in ``frame_track_data``.

    Args:
        video_file (str or os.PathLike): Path to the source video.
        frame_dict (mapping): Frame-indexed detection records. Keys are
            stringified zero-based frame numbers and values are sequences of
            objects accepted by :func:`draw_object`.
        frame_track_data (sequence of sequence): Per-frame track snapshots.
            Each track entry is ``[track_id, x, y]`` and is accepted by
            :func:`draw_car`.
        output_video (str or os.PathLike): Path for the annotated video. Its
            parent directory is created if needed.

    Returns:
        None.

    Raises:
        OSError: If the input video cannot be opened or the output video
            writer cannot be created.

    Side effects:
        Creates or overwrites ``output_video`` using the ``mp4v`` codec at
        30 frames per second. Frames are modified by OpenCV drawing operations
        before they are written.
    """
    capture = cv.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise OSError(f"Could not open input video: {video_file}")

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter.fourcc(*"mp4v"),
        30,
        (700, 500),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not open output video: {output_video}")

    frame_index = 0
    try:
        ok, image = capture.read()
        while ok:
            image = cv.resize(image, (700, 500))
            for detection in frame_dict.get(str(frame_index), []):
                image = draw_object(detection, image)
            if frame_index < len(frame_track_data):
                for car in frame_track_data[frame_index]:
                    image = draw_car(car, image)
            writer.write(image)
            frame_index += 1
            ok, image = capture.read()
    finally:
        capture.release()
        writer.release()


def object_center(detection):
    """Calculate the center of an object's bounding box.

    Args:
        detection (mapping): Object record containing ``x_min``, ``y_min``,
            ``width``, and ``height`` values.

    Returns:
        numpy.ndarray: Center coordinates in ``[x, y]`` order, shaped as a
        column array with shape ``(2, 1)``.

    Raises:
        KeyError: If a required bounding-box field is missing.
        TypeError: If a bounding-box value cannot be used in arithmetic.
    """
    x = detection["x_min"] + detection["width"] / 2
    y = detection["y_min"] + detection["height"] / 2
    return np.array([[x], [y]], dtype=float)
