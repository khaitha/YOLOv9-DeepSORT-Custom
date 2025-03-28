import cv2
import os
import numpy as np
import glob

def stitch_segments_to_video(input_dir, output_video_path, original_resolution, crop_size=(640, 640)):
    # Calculate the number of crops needed along width and height
    num_crops_x = -(-original_resolution[0] // crop_size[0])  # Ceiling division
    num_crops_y = -(-original_resolution[1] // crop_size[1])  # Ceiling division

    # Get the FPS from one of the cropped videos (assuming they all have the same FPS)
    first_segment = cv2.VideoCapture(glob.glob(os.path.join(input_dir, "crop_0_0.mp4"))[0])
    fps = first_segment.get(cv2.CAP_PROP_FPS)
    first_segment.release()

    # Initialize the final video writer with the original resolution
    stitched_video_path = os.path.join(output_video_path, "stitched_video.mp4")
    out = cv2.VideoWriter(
        stitched_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        original_resolution
    )

    # Create a dictionary to hold VideoCapture objects for each segment
    video_readers = {}
    for i in range(num_crops_x):
        for j in range(num_crops_y):
            segment_filename = os.path.join(input_dir, f"crop_{i}_{j}.mp4")
            video_readers[(i, j)] = cv2.VideoCapture(segment_filename)

    # Process frames until one of the segments runs out of frames
    while True:
        # Create a blank frame for stitching
        stitched_frame = np.zeros((original_resolution[1], original_resolution[0], 3), dtype=np.uint8)
        frame_complete = True

        # Loop through each crop position
        for i in range(num_crops_x):
            for j in range(num_crops_y):
                # Read the next frame from the segment video
                ret, crop_frame = video_readers[(i, j)].read()
                if not ret:
                    frame_complete = False
                    break

                # Calculate the coordinates for placing this crop
                x_start = i * crop_size[0]
                y_start = j * crop_size[1]

                # Calculate effective width and height for this crop (to handle edges)
                effective_width = min(crop_size[0], original_resolution[0] - x_start)
                effective_height = min(crop_size[1], original_resolution[1] - y_start)

                # Resize crop frame if needed to fit into effective dimensions
                crop_frame_resized = cv2.resize(crop_frame, (effective_width, effective_height))

                # Place the resized crop in the stitched frame
                stitched_frame[y_start:y_start + effective_height, x_start:x_start + effective_width] = crop_frame_resized

            if not frame_complete:
                break

        # If all frames for this time step are complete, write it to the output video
        if frame_complete:
            out.write(stitched_frame)
        else:
            break

    # Release all resources
    out.release()
    for reader in video_readers.values():
        reader.release()

    print("Stitching completed and saved to:", stitched_video_path)


# Usage example
input_dir = "video_segments/"
output_video_path = "./"
original_resolution = (2048, 2048)  # Replace with the original resolution of your video
stitch_segments_to_video(input_dir, output_video_path, original_resolution)
