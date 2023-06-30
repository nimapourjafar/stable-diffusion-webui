import os
import cv2
import re

def natural_sort_key(s):
    """
    Key function for natural sorting of strings
    """
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', s)]

def create_video_from_images(number):
    output_folder = f"{number}/output"
    output_video = f"{number}/output.mp4"

    # Get all image filenames from the folder and sort them by name
    image_files = sorted([os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith('.png')], key=natural_sort_key)

    # Determine the size of the first image
    sample_image = cv2.imread(image_files[0])
    height, width, _ = sample_image.shape

    # Calculate the frame rate based on the desired frame duration
    frame_duration = 0.05  # seconds
    frame_rate = int(1.0 / frame_duration)

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Iterate through each image and add it to the video
    for image_file in image_files:
        image = cv2.imread(image_file)
        # Repeat each frame for the desired frame duration
        for _ in range(int(frame_rate * frame_duration)):
            video.write(image)

    # Release the VideoWriter and print the path of the output video
    video.release()
    print(f"Video created successfully at: {output_video}")

# Usage example
number = input("Enter the number: ")
create_video_from_images(number)
