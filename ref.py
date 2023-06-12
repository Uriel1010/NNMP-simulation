import cv2
import numpy as np
from scipy.signal import convolve2d
from itertools import product
import matplotlib.pyplot as plt
import csv
import time
from sklearn.metrics import mean_squared_error

K = np.ones((5, 5), dtype=np.float32) / 25

def csv_save(file_path, list_name):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_name)

def full_search(block, search_area):
    y_range = search_area.shape[0] - block.shape[0]
    x_range = search_area.shape[1] - block.shape[1]
    best_match = min(((y, x) for y in range(y_range) for x in range(x_range)),
                    key=lambda yx: np.sum(np.abs(block - search_area[yx[0]:yx[0]+block.shape[0], yx[1]:yx[1]+block.shape[1]])))

    return best_match

def NNMP(Bt, Bt1, M, s):
    H, W = Bt.shape
    best_mv = min(product(range(-s, s), repeat=2),
                key=lambda pq: sum(Bt[i, j] != Bt1[i + pq[0], j + pq[1]]
                                    for i in range(M)
                                    for j in range(M)
                                    if 0 <= i + pq[0] < H and 0 <= j + pq[1] < W))

    return best_mv

def calculate_magnitude(mv_NNMP):
    return (mv_NNMP[0] ** 2 + mv_NNMP[1] ** 2) ** 0.5

def plot_chart(title, xlabel, ylabel, data):
    plt.plot(range(len(data)), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def process_frame(G1, G2):
    start_time = time.time()
    mv_NNMP = NNMP(G1, G2, M=32, s=4)
    end_time = time.time()

    NNMP_time = end_time - start_time
    full_search_time = calculate_full_search_time(G1, G2)
    motion_magnitude = calculate_magnitude(mv_NNMP)

    return mv_NNMP, NNMP_time, full_search_time, motion_magnitude

def calculate_full_search_time(G1, G2):
    try:
        with open('full_search_time.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            return [float(num) for num in next(reader)]
    except FileNotFoundError:
        M = 16
        s = 8
        start_time = time.time()

        for y in range(0, G1.shape[0], M):
            for x in range(0, G1.shape[1], M):
                block = G1[y:y + M, x:x + M]
                search_area = G2[max(0, y - s):min(G2.shape[0], y + M + s), max(0, x - s):min(G2.shape[1], x + M + s)]
                mv_full_search = full_search(block, search_area)
                motion_vectors_full_search.append(mv_full_search)

        end_time = time.time()
        return end_time - start_time

def process_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    magnitudes = []
    NNMP_times = []
    full_search_time = []
    motion_vectors_NNMP = []
    motion_vectors_full_search = []

    for i in range(len(frames)//20 - 1):
        Rf1 = convolve2d(frames[i], K, mode='same')
        Rf2 = convolve2d(frames[i + 1], K, mode='same')
        G1 = np.where(frames[i] >= Rf1, 1, 0)
        G2 = np.where(frames[i + 1] >= Rf2, 1, 0)

        mv_NNMP, NNMP_time, full_search_time_frame, magnitude = process_frame(G1, G2)

        NNMP_times.append(NNMP_time)
        full_search_time.append(full_search_time_frame)
        magnitudes.append(magnitude)
        motion_vectors_NNMP.append(mv_NNMP)

    csv_save('motion_vectors_full_search.csv',motion_vectors_full_search)
    csv_save('full_search_time.csv',full_search_time)
    csv_save('NNMP_times.csv',NNMP_times)

    plot_chart('Magnitude of Motion Over Time', 'Frame Number', 'Magnitude of Motion', magnitudes)

    print(f"Average magnitude of motion: {sum(magnitudes) / len(magnitudes)}")
    print(f"Total magnitude of motion: {sum(magnitudes)}")

    MSE_list = [mean_squared_error(mv_NNMP, mv_full_search) for mv_NNMP, mv_full_search in zip(motion_vectors_NNMP, motion_vectors_full_search)]
    csv_save('MSE_list.csv',MSE_list)

    time_diff = np.array(full_search_time) - np.array(NNMP_times)
    plot_chart('Time Difference Over Frames', 'Frame Number', 'Time Difference (seconds)', time_diff)
    plot_chart('MSE Over Frames', 'Frame Number', 'Mean Squared Error', MSE_list)

    csv_save('magnitudes.csv',magnitudes)

process_video('videos/_import_6140455b6c6fa0.31477371_preview.mp4')

cv2.destroyAllWindows()
