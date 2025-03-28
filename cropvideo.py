import cv2
import os
import numpy as np


import os
import cv2
import numpy as np

def crop(input_video_path, output_dir, crop_size=(640, 640)):
    """
    Crop a video into segments with padding and store top-left coordinates for each crop.

    Args:
    - input_video_path: Path to the input video.
    - output_dir: Directory where cropped segments and coordinate data are saved.
    - crop_size: Size of each crop (width, height).

    Returns:
    - crop_coordinates: A dictionary mapping (segment name) to its top-left (x, y) coordinate.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the number of crops needed along width and height
    num_crops_x = -(-width // crop_size[0])  # Ceiling division
    num_crops_y = -(-height // crop_size[1])  # Ceiling division

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video writers for each segment and store crop coordinates
    video_writers = {}
    crop_coordinates = {}  # Dictionary to store the top-left coordinates

    for i in range(num_crops_x):
        for j in range(num_crops_y):
            # Create a unique filename for each cropped segment video
            segment_filename = os.path.join(output_dir, f"crop_{i}_{j}.mp4")
            video_writers[(i, j)] = cv2.VideoWriter(
                segment_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),  # Codec
                fps,
                crop_size  # Output size (640x640)
            )

            # Store the top-left coordinate for this crop
            crop_coordinates[f"crop_{i}_{j}"] = (i * crop_size[0], j * crop_size[1])

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Loop over each segment and crop/pad the frame
        for i in range(num_crops_x):
            for j in range(num_crops_y):
                # Define the coordinates of the crop
                x_start = i * crop_size[0]
                y_start = j * crop_size[1]
                x_end = min(x_start + crop_size[0], width)
                y_end = min(y_start + crop_size[1], height)

                # Crop the frame and create a blank canvas for padding
                crop = frame[y_start:y_end, x_start:x_end]
                padded_crop = np.zeros((crop_size[1], crop_size[0], 3), dtype=np.uint8)

                # Place the cropped frame onto the blank canvas
                padded_crop[0:crop.shape[0], 0:crop.shape[1]] = crop

                # Write the padded crop to the appropriate video writer
                video_writers[(i, j)].write(padded_crop)

    # Release the video capture and all video writers
    cap.release()
    for writer in video_writers.values():
        writer.release()

    # Save crop coordinates to a file
    coord_file = os.path.join(output_dir, "crop_coordinates.txt")
    with open(coord_file, "w") as f:
        for crop_name, coord in crop_coordinates.items():
            f.write(f"{crop_name}: {coord}\n")

    print("Cropping with padding and saving as videos completed.")
    print(f"Crop coordinates saved to {coord_file}.")

    return crop_coordinates



# Usage example
input_video_path = "data.mp4"
output_dir = "video_segments/"
crop(input_video_path, output_dir)
