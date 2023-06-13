import json

import cv2
import numpy as np
from scipy.signal import convolve2d
from itertools import product
import matplotlib.pyplot as plt
import csv
import time
import os
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr

class VideoProcessor:
    def __init__(self, video_file):
        self.video_file = video_file
        self.width = 0
        self.height = 0
        self.fps = 0
        self.numFrames = 0
        self.filename = os.path.splitext(video_file)[0].split("/")[-1]
        self.frames = []
        self.magnitudes = []
        self.motion_vectors_NNMP = []
        self.motion_vectors_full_search = []
        self.NNMP_times = []
        self.full_search_times = []
        self.motions = {dir:0 for dir in ['Right', 'Left', 'Up', 'Down', 'Still']}
        self.total_motion = 0
        self.K = np.ones((5, 5), dtype=np.float32) / 25

    def snapshot_first_frame(self):
        cap = cv2.VideoCapture(self.video_file)
        ret, frame = cap.read()
        if ret == True:
            # resize the frame
            frame = cv2.resize(frame, (640, 480))
            # save the frame to an image file
            cv2.imwrite('video_first_frame.png', frame)
        cap.release()

    def capture_video(self):
        cap = cv2.VideoCapture(self.video_file)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        update_json_file("res.json", {"width": self.width, "height": self.height, "fps":self.fps, "numFrames":self.numFrames})
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.append(gray)

        cap.release()

    @staticmethod
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

    @staticmethod
    def full_search(block, search_area):

        best_match = (0, 0)
        min_difference = float('inf')

        for y in range(search_area.shape[0] - block.shape[0]):
            for x in range(search_area.shape[1] - block.shape[1]):
                candidate_block = search_area[y:y + block.shape[0], x:x + block.shape[1]]
                difference = np.sum(np.abs(block - candidate_block))

                if difference < min_difference:
                    min_difference = difference
                    best_match = (y, x)

        return best_match

    @staticmethod
    def full_searching(G1, G2, M, s):
        best_match = (0, 0)
        min_difference = float('inf')
        motion_vectors_full_search = []
        for y in range(0, G1.shape[0], M):
            for x in range(0, G1.shape[1], M):
                block = G1[y:y + M, x:x + M]
                search_area = G2[max(0, y - s):min(G2.shape[0], y + M + s), max(0, x - s):min(G2.shape[1], x + M + s)]
                mv_full_search = VideoProcessor.full_search(block, search_area)
                motion_vectors_full_search.append(mv_full_search)

        return motion_vectors_full_search

    def convolve_frames(self, i):
        Rf1 = convolve2d(self.frames[i], self.K, mode='same')
        Rf2 = convolve2d(self.frames[i + 1], self.K, mode='same')
        return Rf1, Rf2

    def apply_threshold(self, Rf1, Rf2, i):
        G1 = np.where(self.frames[i] >= Rf1, 1, 0)
        G2 = np.where(self.frames[i + 1] >= Rf2, 1, 0)
        return G1, G2

    def time_motion_vector_calculation(self, func, *args):
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return result, elapsed_time

    def process_frames(self):
        # Initialize counters for the number of frames with motion in each direction
        right_motion = 0
        left_motion = 0
        up_motion = 0
        down_motion = 0
        still_frames = 0
        for i in range(len(self.frames)//20 - 1):
            Rf1, Rf2 = self.convolve_frames(i)
            G1, G2 = self.apply_threshold(Rf1, Rf2, i)

            mv_NNMP, NNMP_time = self.time_motion_vector_calculation(self.NNMP, G1, G2, 32, 4)
            self.motion_vectors_NNMP.append(mv_NNMP)
            self.NNMP_times.append(NNMP_time)

            mv_full_search, full_search_time = self.time_motion_vector_calculation(self.full_searching, G1, G2, 16, 8)
            self.motion_vectors_full_search += mv_full_search
            self.full_search_times.append(full_search_time)

            # Count the number of frames with motion in each direction
            if mv_NNMP[0] > 0:
                self.motions['Down'] += 1
            elif mv_NNMP[0] < 0:
                self.motions['Up'] += 1
            if mv_NNMP[1] > 0:
                self.motions['Right'] += 1
            elif mv_NNMP[1] < 0:
                self.motions['Left'] += 1
            if mv_NNMP[0] == 0 and mv_NNMP[1] == 0:
                self.motions['Still'] += 1

            # Calculate the magnitude of the motion vector
            magnitude = (mv_NNMP[0] ** 2 + mv_NNMP[1] ** 2) ** 0.5
            self.magnitudes.append(magnitude)
            self.total_motion += magnitude
        # Get the average magnitude of motion
        # This is the average magnitude of the motion vectors for all frames.
        average_motion = self.total_motion / (len(self.frames) - 1)

        # Print the average magnitude of motion
        print(f"Average magnitude of motion: {average_motion}")

        # Print the total and average magnitude of motion
        print(f"Total magnitude of motion: {self.total_motion}")

        # Save the average and total magnitude of motion to a JSON file
        update_json_file("res.json", {"average_motion": average_motion, "total_motion": self.total_motion})

    def plots(self):
        self.motion_plot()
        self.motion_magnitude_plot()
        self.mse_plot()
        self.psnr_comp()

    def motion_plot(self):
        # Create a bar chart for motion directions
        # This bar chart shows the number of frames with motion in each direction.
        directions = ['Right', 'Left', 'Up', 'Down', 'Still']
        counts = [self.motions[dir] for dir in directions]
        plt.bar(directions, counts)
        plt.title('Direction of Motion')
        plt.xlabel('Direction')
        plt.ylabel('Number of Frames')
        plt.savefig('DirectionOfMotion.png')
        plt.show()

    def motion_magnitude_plot(self):
        # Create a line plot for motion magnitudes
        # This plot shows the magnitude of the motion vector over time.
        plt.plot(range(len(self.magnitudes)), self.magnitudes)
        plt.title('Magnitude of Motion Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Magnitude of Motion [dB]')
        plt.savefig('MagnitudeOfMotionOverTime.png')
        plt.show()

    def mse_plot(self):
        MSE_list = []
        for mv_NNMP, mv_full_search in zip(self.motion_vectors_NNMP, self.motion_vectors_full_search):
            mse = mean_squared_error(mv_NNMP, mv_full_search)
            MSE_list.append(mse)
        time_diff = np.array(self.full_search_times) - np.array(self.NNMP_times)
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

    def psnr_comp(self):
        # Assume that each motion vector is a pixel value in our generated images
        mv_image_NNMP = np.array(self.motion_vectors_NNMP, dtype=np.float64)
        mv_image_FS = np.array(self.motion_vectors_full_search, dtype=np.float64)
        original_image = np.array(self.frames[:-1], dtype=np.float64)

        mv_image_NNMP_resized = cv2.resize(mv_image_NNMP, (original_image.shape[2], original_image.shape[1]),
                                           interpolation=cv2.INTER_AREA)
        mv_image_FS_resized = cv2.resize(mv_image_FS, (original_image.shape[2], original_image.shape[1]),
                                         interpolation=cv2.INTER_AREA)

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
        # Save the average and total magnitude of motion to a JSON file
        update_json_file("res.json", {"PSNR_for_NNMP": psnr_NNMP, "PSNR_for_Full_Search": psnr_FS})

        # Plotting the comparison
        labels = ['NNMP', 'Full Search']
        psnr_values = [psnr_NNMP, psnr_FS]

        plt.bar(labels, psnr_values)
        plt.title('PSNR Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('PSNR')
        plt.savefig('PSNRComparison.png')
        plt.show()



def update_json_file(filename, new_data):
    if not os.path.isfile(filename) or os.stat(filename).st_size == 0:  # if file doesn't exist or is empty
        with open(filename, "w") as f:
            json.dump(new_data, f)
    else:
        with open(filename, "r+") as f:  # opening for reading and writing (updating)
            data = json.load(f)  # load existing data
            data.update(new_data)  # merge new data with existing
            f.seek(0)  # reset file position to the beginning
            json.dump(data, f)  # write updated data back to the file
            f.truncate()  # remove remaining part of original file, if new data is shorter

def csv_save(file_path, list_name):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_name)

    print(f"List {file_path} saved to CSV file successfully.")

# Using the class:
video1 = VideoProcessor('videos/_import_6140455b6c6fa0.31477371_preview.mp4')
video1.snapshot_first_frame()
video1.capture_video()
video1.process_frames()
video1.plots()
import valuesUpdateFromJSON
