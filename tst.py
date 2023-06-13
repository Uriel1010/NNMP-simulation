import cv2
import numpy as np
from scipy.signal import convolve2d
from itertools import product
import matplotlib.pyplot as plt
import csv
import time
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr


# Function to save list to CSV
def csvSave(file_path, listName):
    # Open the file in write mode
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(listName)

    print(f"List {file_path} saved to CSV file successfully.")


# Define the kernel
# The kernel is a 5x5 matrix with 1/25 at indices 1, 4, 8, 12, and 16, and zeros elsewhere.
# This kernel will be used for convolution with the video frames.
K = np.ones((5, 5), dtype=np.float32) / 25

def full_search(block, search_area):

    best_match = (0, 0)
    min_difference = float('inf')

    for y in range(search_area.shape[0] - block.shape[0]):
        for x in range(search_area.shape[1] - block.shape[1]):
            candidate_block = search_area[y:y+block.shape[0], x:x+block.shape[1]]
            difference = np.sum(np.abs(block - candidate_block))

            if difference < min_difference:
                min_difference = difference
                best_match = (y, x)


    return best_match


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
cap = cv2.VideoCapture('videos/_import_6140455b6c6fa0.31477371.mov')
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
NNMP_times = []
full_search_time = []

# Initialize lists to store motion vectors
motion_vectors_NNMP = []
motion_vectors_full_search = []

# Process the frames
for i in range(len(frames) - 1):

    # Convolve the frames with the kernel
    Rf1 = convolve2d(frames[i], K, mode='same')
    Rf2 = convolve2d(frames[i + 1], K, mode='same')

    # Obtain one-bit frames by comparing the original frames to the convolved frames
    G1 = np.where(frames[i] >= Rf1, 1, 0)
    G2 = np.where(frames[i + 1] >= Rf2, 1, 0)

    start_time = time.time()  # Start time measurement

    # Apply NNMP to the pair of frames to estimate the motion vector
    mv_NNMP = NNMP(G1, G2, M=32, s=4)  # Adjust M and s as needed
    motion_vectors_NNMP.append(mv_NNMP)

    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time  # Calculate elapsed time
    NNMP_times.append(elapsed_time)
    print("Time taken for NNMP: ", elapsed_time, "seconds")


    # Print the motion vector for each frame
    # print(f"Frame {i}: Motion vector = {mv_NNMP}")
    try:
        with open('full_search_time.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            full_search_time = []
            for x in reader:
                full_search_time = [float(num) for num in x]
    except FileNotFoundError:
        # If the file does not exist, initialize full_search_time to an empty list
        # Define block size
        M = 16
        # Define search area size
        s = 8
        start_time = time.time()  # Start time measurement

        # Split frames into blocks and find motion vectors for each
        for y in range(0, G1.shape[0], M):
            for x in range(0, G1.shape[1], M):
                block = G1[y:y + M, x:x + M]
                search_area = G2[max(0, y - s):min(G2.shape[0], y + M + s), max(0, x - s):min(G2.shape[1], x + M + s)]
                mv_full_search = full_search(block, search_area)
                motion_vectors_full_search.append(mv_full_search)


        end_time = time.time()  # End time measurement
        elapsed_time = end_time - start_time  # Calculate elapsed time
        full_search_time.append(elapsed_time)
        print("Time taken for full_search: ", elapsed_time, "seconds")

    # Count the number of frames with motion in each direction
    if mv_NNMP[0] > 0:
        down_motion += 1
    elif mv_NNMP[0] < 0:
        up_motion += 1
    if mv_NNMP[1] > 0:
        right_motion += 1
    elif mv_NNMP[1] < 0:
        left_motion += 1
    if mv_NNMP[0] == 0 and mv_NNMP[1] == 0:
        still_frames += 1

    # Calculate the magnitude of the motion vector
    magnitude = (mv_NNMP[0] ** 2 + mv_NNMP[1] ** 2) ** 0.5
    magnitudes.append(magnitude)
    total_motion += magnitude

    # Print the magnitude of motion for each frame
    # print(f"Frame {i}: Magnitude of motion = {magnitude}")
csvSave('motion_vectors_full_search.csv',motion_vectors_full_search)
csvSave('full_search_time.csv',full_search_time)
csvSave('NNMP_times.csv',NNMP_times)

# Create a bar chart for motion directions
# This bar chart shows the number of frames with motion in each direction.
directions = ['Right', 'Left', 'Up', 'Down', 'Still']
counts = [right_motion, left_motion, up_motion, down_motion, still_frames]
plt.bar(directions, counts)
plt.title('Direction of Motion')
plt.xlabel('Direction')
plt.ylabel('Number of Frames')
plt.savefig('DirectionOfMotion.png')
plt.show()


# Create a line plot for motion magnitudes
# This plot shows the magnitude of the motion vector over time.
plt.plot(range(len(magnitudes)), magnitudes)
plt.title('Magnitude of Motion Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Magnitude of Motion')
plt.savefig('MagnitudeOfMotionOverTime.png')
plt.show()


# Print the average magnitude of motion
# This is the average magnitude of the motion vectors for all frames.
print(f"Average magnitude of motion: {total_motion / (len(frames) - 1)}")
# Print the total and average magnitude of motion
print(f"Total magnitude of motion: {total_motion}")

MSE_list = []
for mv_NNMP, mv_full_search in zip(motion_vectors_NNMP, motion_vectors_full_search):
    mse = mean_squared_error(mv_NNMP, mv_full_search)
    MSE_list.append(mse)
csvSave('MSE_list.csv',MSE_list)
time_diff = np.array(full_search_time) - np.array(NNMP_times)
plt.plot(range(len(time_diff)), time_diff)
plt.title('Time Difference Over Frames')
plt.xlabel('Frame Number')
plt.ylabel('Time Difference (seconds)')
plt.savefig('TimeDifferenceOverFrames.png')
plt.show()

plt.plot(range(len(MSE_list)), MSE_list)
plt.title('MSE Over Frames')
plt.xlabel('Frame Number')
plt.ylabel('Mean Squared Error')
plt.savefig('MSEOverFrames.png')
plt.show()

# Assume that each motion vector is a pixel value in our generated images
mv_image_NNMP = np.array(motion_vectors_NNMP, dtype=np.float64)
mv_image_FS = np.array(motion_vectors_full_search, dtype=np.float64)
original_image = np.array(frames[:-1], dtype=np.float64)

mv_image_NNMP_resized = cv2.resize(mv_image_NNMP, (original_image.shape[2], original_image.shape[1]), interpolation = cv2.INTER_AREA)
mv_image_FS_resized = cv2.resize(mv_image_FS, (original_image.shape[2], original_image.shape[1]), interpolation = cv2.INTER_AREA)

psnr_NNMP_list = []
psnr_FS_list = []

for i in range(original_image.shape[0]):
    psnr_NNMP_list.append(psnr(original_image[i], mv_image_NNMP_resized, data_range=255))
    psnr_FS_list.append(psnr(original_image[i], mv_image_FS_resized, data_range=255))

# Then you can calculate the average PSNR for each method
psnr_NNMP = np.mean(psnr_NNMP_list)
psnr_FS = np.mean(psnr_FS_list)

print(f"PSNR for NNMP: {psnr_NNMP}")
print(f"PSNR for Full Search: {psnr_FS}")

# Plotting the comparison
labels = ['NNMP', 'Full Search']
psnr_values = [psnr_NNMP, psnr_FS]

plt.bar(labels, psnr_values)
plt.title('PSNR Comparison')
plt.xlabel('Algorithm')
plt.ylabel('PSNR')
plt.savefig('PSNRComparison.png')
plt.show()

csvSave('magnitudes.csv',magnitudes)

if __name__=="__main__":
    VideoProcessor