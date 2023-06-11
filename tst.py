import cv2
import numpy as np
from scipy.signal import convolve2d
from itertools import product
import matplotlib.pyplot as plt
import csv

# Define the kernel
# The kernel is a 5x5 matrix with 1/25 at indices 1, 4, 8, 12, and 16, and zeros elsewhere.
# This kernel will be used for convolution with the video frames.
K = np.ones((5, 5), dtype=np.float32) / 25


# Define the Number of Non-Matching Points (NNMP) function
# This function compares macroblocks in the current and reference frames and calculates the number of non-matching points.
def NNMP(Bt, Bt1, M, s):
    H, W = Bt.shape
    best_score = np.inf
    best_mv = (0, 0)

    # Iterate over all possible motion vectors within the search window (defined by s)
    for p, q in product(range(-s, s), repeat=2):
        score = 0
        # For each possible motion vector, compute the score
        for i in range(M):
            for j in range(M):
                if 0 <= i + p < H and 0 <= j + q < W:
                    # The score is the number of non-matching points between the current and reference macroblocks
                    score += Bt[i, j] != Bt1[i + p, j + q]
        # If this motion vector results in a better score, update the best motion vector
        if score < best_score:
            best_score = score
            best_mv = (p, q)

    # Return the best motion vector
    return best_mv

# Open the video
cap = cv2.VideoCapture('videos/_import_6140455b6c6fa0.31477371_preview.mp4')
print("Frame width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames per second: ", cap.get(cv2.CAP_PROP_FPS))
print("Total frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []

# Read the frames from the video and convert them to grayscale
while (cap.isOpened()):
    ret, frame = cap.read()
    # print("Read frame: ", ret)
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

cap.release()

# Initialize counters for the number of frames with motion in each direction
right_motion = 0
left_motion = 0
up_motion = 0
down_motion = 0
still_frames = 0
total_motion = 0

# Initialize a list to store the magnitude of the motion vector for each frame
magnitudes = []

# Process the frames
for i in range(len(frames) - 1):

    # Convolve the frames with the kernel
    Rf1 = convolve2d(frames[i], K, mode='same')
    Rf2 = convolve2d(frames[i + 1], K, mode='same')

    # Display the frames
    # Only display every 10th frame
    # if i % 10 == 0:
    #     cv2.imshow('Frame', frames[i])
    #     cv2.waitKey(500)  # Wait for 500 ms

    # Obtain one-bit frames by comparing the original frames to the convolved frames
    G1 = np.where(frames[i] >= Rf1, 1, 0)
    G2 = np.where(frames[i + 1] >= Rf2, 1, 0)

    # Apply NNMP to the pair of frames to estimate the motion vector
    mv = NNMP(G1, G2, M=32, s=4)  # Adjust M and s as needed

    # Print the motion vector for each frame
    # print(f"Frame {i}: Motion vector = {mv}")

    # Count the number of frames with motion in each direction
    if mv[0] > 0:
        down_motion += 1
    elif mv[0] < 0:
        up_motion += 1
    if mv[1] > 0:
        right_motion += 1
    elif mv[1] < 0:
        left_motion += 1
    if mv[0] == 0 and mv[1] == 0:
        still_frames += 1

    # Calculate the magnitude of the motion vector
    magnitude = (mv[0] ** 2 + mv[1] ** 2) ** 0.5
    magnitudes.append(magnitude)
    total_motion += magnitude

    # Print the magnitude of motion for each frame
    # print(f"Frame {i}: Magnitude of motion = {magnitude}")

# Create a bar chart for motion directions
# This bar chart shows the number of frames with motion in each direction.
directions = ['Right', 'Left', 'Up', 'Down', 'Still']
counts = [right_motion, left_motion, up_motion, down_motion, still_frames]
plt.bar(directions, counts)
plt.title('Direction of Motion')
plt.xlabel('Direction')
plt.ylabel('Number of Frames')
plt.show()

# Create a line plot for motion magnitudes
# This plot shows the magnitude of the motion vector over time.
plt.plot(range(len(magnitudes)), magnitudes)
plt.title('Magnitude of Motion Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Magnitude of Motion')
plt.show()

# Print the average magnitude of motion
# This is the average magnitude of the motion vectors for all frames.
print(f"Average magnitude of motion: {total_motion / (len(frames) - 1)}")
# Print the total and average magnitude of motion
print(f"Total magnitude of motion: {total_motion}")


# Specify the file path where you want to save the CSV file
file_path = 'magnitudes.csv'

# Open the file in write mode
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(magnitudes)

print("List saved to CSV file successfully.")

# When done, destroy the windows
cv2.destroyAllWindows()

