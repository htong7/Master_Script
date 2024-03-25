import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# the average movemenet/second is going to depend on FPS of the video file. so let's first calculate that
def get_fps(video_file_path):
    # Open the video file.
    video = cv2.VideoCapture(video_file_path)
    
    # Get FPS using video.get(cv2.CAP_PROP_FPS)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Release the video file.
    video.release()
    
    return fps

# process movement using frame differencing and your previously defined get_fps() function
def process_video_movement_frame_differencing(video_path):
    #frame differencing (simplest)
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    movement_magnitude_list_differencing = []

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        movement = np.sum(gray_diff)
        movement_magnitude_list_differencing.append(movement)

        prev_frame = current_frame

    # Calculate average magnitude of movement between each fps frames
    fps = int(get_fps(video_path))
    movement_per_second_differencing = [sum(movement_magnitude_list_differencing[i:i+fps])/fps for i in range(0, len(movement_magnitude_list_differencing), fps)]

    # Cool! We now have the average amount of movement between frames for each second. Let's organize these in tuples (second, movement)
    second = 1
    movement_per_second_differencing_with_timestamps = []
    for element in movement_per_second_differencing:
        movement_per_second_differencing_with_timestamps.append((second,element))
        second += 1

    return movement_per_second_differencing_with_timestamps


# create graph from output of process_video_movement_frame_differencing(video_path)
def create_timeseries_graph(movement_per_second_differencing_with_timestamps):   
    #lets quickly visualize movement/second to get a rough idea of how frame differencing performs
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4))
    X = [element[0] for element in movement_per_second_differencing_with_timestamps]
    Y = [element[1] for element in movement_per_second_differencing_with_timestamps]
    ax.plot(X, Y, color='#d2691e')

    ax.set_xlabel('Second')
    ax.set_ylabel('Movement')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title("Movement Analysis Using Frame Differencing",c='grey')
    plt.show()

# generate moving average of time series tuple list
def moving_average(series, window):
    return series.rolling(window = window, center = True).mean()


#detect faces in each frame of video
def detect_faces_in_video(video_path):

    # Initialize the video capture object
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return None

    # Get video FPS (frames per second) to calculate the number of frames per second
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the number of frames to skip to check one frame per second
    frames_to_skip = math.ceil(fps)

    # Initialize the list to store results for each second
    results_per_second = []

    # Initialize frame counter
    frame_counter = 0

    # Process the video frame by frame
    while True:
        # Read a single frame
        ret, frame = video_capture.read()

        # Break the loop if we've reached the end of the video
        if not ret:
            break

        # Print the current frame number
#         print(f"Processing frame {frame_counter}")

        # Check for a face only on specific frames to save on processing
        if frame_counter % frames_to_skip == 0:
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            rgb_frame = frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame, model="cnn")

            # Determine if a face was detected
            face_detected = len(face_locations) > 0

            # Append 1 if face is detected, else 0
            results_per_second.append(int(face_detected))

        # Increment the frame counter
        frame_counter += 1

    # Release the video capture object
    video_capture.release()

    # Close all the frames
    cv2.destroyAllWindows()

    return results_per_second



