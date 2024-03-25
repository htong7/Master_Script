# import necessary package
import cv2
import pytesseract
import pandas as pd
from pytesseract import Output
import re
import argparse
import os
import ffmpeg
import json
from tqdm import tqdm  # Import tqdm for the progress bar
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# the average movemenet/second is going to depend on FPS of the video file. so let's first calculate that
def get_fps(video_file_path):
    # Open the video file.
    video = cv2.VideoCapture(video_file_path)
    
    # Get FPS using video.get(cv2.CAP_PROP_FPS)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Release the video file.
    video.release()
    
    return round(fps)

# captures an image of a given frame
def extract_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Move to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # Read the specified frame
    ret, frame = cap.read()

    # Close the video file
    cap.release()

    if ret:
        return frame
    else:
        return None

def ocr_from_image(image, type):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Manually adjust the threshold if necessary
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    kernel = np.ones((1,1), np.uint8)
    processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Sharpen image
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(processed_image, -1, sharpen_kernel)

    # Crop to region of interest if the timestamp location is consistent
    h, w = sharpened_image.shape
    if type == 'bwc':
        cropped_image = sharpened_image[:int(h * .1), int(w * 0.35):]
    elif type == 'bwc2':
        cropped_image = sharpened_image[int(h*.9):, int(w * 0.65):]
    else:
        cropped_image = sharpened_image


    # # Display the grayscale image
    # plt.imshow(cropped_image, cmap='gray')
    # plt.title('Grayscale Image')
    # plt.show()

    # Use Tesseract to do OCR on the cropped image
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cropped_image, config=custom_config)

    return text

# conducts regex to pull timestamp based on if the video is dashcam or bodycam
def extract_timestamp_from_text(text):
    pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{2}/[0-9]{2}/[0-9]{4}'
    match = re.search(pattern, text)
    
    if match is None:
        pattern = r'[0-9]{2}/[0-9]{2}/[0-9]{4}.{1}[0-9]{2}:[0-9]{2}:[0-9]{2}'
        match = re.search(pattern, text) 

    if match is None:
        pattern = r'[0-9]{4}-[0-9]{2}-[0-9]{2}.{1}[0-9]{2}:[0-9]{2}:[0-9]{2}'
        match = re.search(pattern, text)
        
    if match:
        return match.group()
    return None

# function to extract relevant metadata
def get_video_metadata(file_path):
    try:
        # Probe the video file
        probe = ffmpeg.probe(file_path)
        
        # Convert the probe result to a JSON string for pretty printing
        probe_json = json.dumps(probe, indent=4)
        # Convert the string to a dictionary
        data = json.loads(probe_json)

        # Accessing the creation time from the video stream's tags
        video_stream = next((stream for stream in data['streams'] if stream['codec_type'] == 'video'), None)
        creation_time = None
        if video_stream and 'tags' in video_stream:
            creation_time = video_stream['tags'].get('creation_time')

        # Accessing the duration from the format section
        duration = data['format'].get('duration')
        
        return creation_time, duration
    except ffmpeg.Error as e:
        print("Error:", e)
    except Exception as e:
        print("An error occurred:", e)

def format_text_for_timestamp(text):
    # Replace '-8' with '-0'
    text = text.replace('-8', '-0')

    # Replace '/8' with '/0'
    text = text.replace('/8', '/0')

    # Replace '282' with '202'
    text = text.replace('282', '202')

    # Replace '002' with '202'
    text = text.replace('002', '202')

    # Correct the timestamp pattern
    # Find all matches of the pattern and replace them
    pattern = re.compile(r'\d{2}/\d{2}/\d{4}.{1}\d{2}:\d{2}:\d{2}')
    matches = pattern.findall(text)
    for match in matches:
        corrected = match.replace('-', ' ')
        text = text.replace(match, corrected)

    return text

# master function to extract timestamps and metadata
def extract_timestamp(video_path):

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get frame count using OpenCV's property
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_iterations = 0
    end_iterations = 0
    start_timestamp = None
    end_timestamp = None
    iteration = 1
    last_iteration = frame_count - 1

    fps = get_fps(video_path)

    while start_timestamp is None or end_timestamp is None:
        # Checking for start timestamp
        if start_timestamp is None:
            frame_start = iteration
            screenshot_start = extract_frame(video_path, frame_start)
            if screenshot_start is not None:
                start_text = ocr_from_image(screenshot_start, type = 'dash')
                if 'MPH' in start_text:
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)
                elif '_REDACTED_' in video_path:
                    start_text = ocr_from_image(screenshot_start, type = 'bwc2')
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)
                else:
                    start_text = ocr_from_image(screenshot_start, type = 'bwc')
                    start_text = format_text_for_timestamp(start_text)
                    start_timestamp = extract_timestamp_from_text(start_text)

                if start_timestamp is None:
                    iteration += fps
                    start_iterations += 1
                    #print('Could not find first timestamp, trying next frame.')

        # Checking for end timestamp
        if end_timestamp is None:
            frame_end = last_iteration
            screenshot_end = extract_frame(video_path, frame_end)
            if screenshot_end is not None:
                end_text = ocr_from_image(screenshot_end, type = 'dash')
                # Perform OCR on the end frame
                if '_REDACTED_' in video_path:
                    end_text = ocr_from_image(screenshot_end, type='bwc2')
                    end_text = format_text_for_timestamp(end_text)
                elif 'MPH' in end_text:  # Assuming you check for 'MPH' to identify dashcam type
                    end_text = ocr_from_image(screenshot_end, type='dash')
                    end_text = format_text_for_timestamp(end_text)
                else:
                    end_text = ocr_from_image(screenshot_end, type='bwc')
                    end_text = format_text_for_timestamp(end_text)       


                # Extract timestamp from OCR text
                end_timestamp = extract_timestamp_from_text(end_text)

                if end_timestamp is None:
                    end_iterations += 1
                    last_iteration = last_iteration - fps
                    #print('Could not find last timestamp, trying preceding frame.')

        last_iteration = int(last_iteration - fps)

        if iteration >= last_iteration:  # Prevent infinite loop
            print("NO VIABLE FRAMES")
            break

    creation_time, duration = get_video_metadata(video_path)

    return start_timestamp, end_timestamp, creation_time, duration, start_iterations, end_iterations

def standardize_timestamp(timestamp):
    if timestamp is None:
        return None
    # Define the different possible formats
    formats = [
        '%H:%M:%S %m/%d/%Y', # Format: 'HH:MM:SS MM/DD/YYYY'
        '%m/%d/%Y %H:%M:%S', # Format: 'MM/DD/YYYY HH:MM:SS'
        '%Y-%m-%d %H:%M:%S'  # Format: 'YYYY-MM-DD HH:MM:SS'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None

def process_videos(directory):
    # List to store the results
    results = []

    # Prepare a list of all eligible video files
    video_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(subdir, file))

    # Initialize the progress bar
    with tqdm(total=len(video_files), desc="Processing Videos") as pbar:
        for file_path in video_files:
            print(file_path)
            start_timestamp, end_timestamp, creation_time, duration, start_iterations, end_iterations = extract_timestamp(file_path)
            print(f'Video Start Timestamp:\n{start_timestamp}\n')
            print(f'Video End Timestamp:\n{end_timestamp}\n')
            print(f'Creation Time:\n{creation_time}\n')
            print(f'Duration:\n{duration}\n')

            # Standardize timestamps
            standardized_start_timestamp = standardize_timestamp(start_timestamp)
            print(standardized_start_timestamp)
            standardized_end_timestamp = standardize_timestamp(end_timestamp)
            print(standardized_end_timestamp)

            if standardized_start_timestamp and start_iterations is not None:
                adjusted_start_timestamp = standardized_start_timestamp - timedelta(seconds=start_iterations)
                #print(adjusted_start_timestamp)
                start_date, start_time = adjusted_start_timestamp.date(), adjusted_start_timestamp.time()
            else:
                start_date, start_time = None, None

            if standardized_end_timestamp and end_iterations is not None:
                adjusted_end_timestamp = standardized_end_timestamp + timedelta(seconds=end_iterations) + timedelta(seconds=end_iterations)
                #print(adjusted_end_timestamp)
                end_date, end_time = adjusted_end_timestamp.date(), adjusted_end_timestamp.time()
            else:
                end_date, end_time = None, None

            results.append({'file_name': os.path.basename(file_path),
                            'start_date': start_date, 'start_time': start_time,
                            'end_date': end_date, 'end_time': end_time,
                            'creation_time': creation_time, 'duration': duration})

            # Update the progress bar
            pbar.update(1)

    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv('video_timestamps.csv', index=False)
    print("CSV file created: video_timestamps.csv")


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Process video files for timestamp extraction.')
    parser.add_argument('directory', type=str, help='Directory containing video files')

    # Parse command line arguments
    args = parser.parse_args()

    # Call the process_videos function with the provided directory
    process_videos(args.directory)