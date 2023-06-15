## Comparison of Full Search and Number of Non-Matching Points Algorithms for Motion Estimation in Video Processing
<!-- TOC -->
  * [Comparison of Full Search and Number of Non-Matching Points Algorithms for Motion Estimation in Video Processing](#comparison-of-full-search-and-number-of-non-matching-points-algorithms-for-motion-estimation-in-video-processing)
    * [Abstract](#abstract)
    * [Introduction](#introduction)
    * [Test Video Details](#test-video-details)
    * [Methodology](#methodology)
      * [Simulation Setup](#simulation-setup)
      * [Evaluation Metrics](#evaluation-metrics)
    * [Results](#results)
      * [Time Performance](#time-performance)
      * [Accuracy of Motion Estimation](#accuracy-of-motion-estimation)
      * [Magnitude of Motion](#magnitude-of-motion)
      * [Direction of Motion](#direction-of-motion)
    * [Image Quality](#image-quality)
    * [Discussion](#discussion)
    * [Conclusion](#conclusion)
    * [References](#references)
<!-- TOC -->
### Abstract
In the field of video processing, accurate motion estimation is a pivotal component. This document presents a comparative study of two algorithms used for this purpose - the Full Search algorithm and the Number of Non-Matching Points (NNMP) algorithm. The study utilizes a self-designed simulation where both algorithms are implemented and then analyzed based on Mean Squared Error (MSE), time efficiency, and magnitude of motion. The results reveal a substantial time advantage of NNMP over Full Search, albeit with a variance in MSE across the two algorithms.

### Introduction
Motion estimation forms a cornerstone of various video processing applications such as video compression, video stabilization, object tracking, and computer vision. It involves the determination of motion vectors that describe the transformation from one 2D image to another; usually from adjacent frames in a video sequence. This paper focuses on two prevalent algorithms, Full Search and Number of Non-Matching Points (NNMP), assessing their performance under a controlled simulation. Our work is largely based on the research conducted by T.Muralidhar Reddy, P.Muralidhar, and Dr.C B.Rama Rao titled "New Fast Search Block Matching Motion Estimation Algorithm for H.264 /AVC."

### Test Video Details

The video file selected for this study has the following characteristics:

- Frame width: 1280.0
- Frame height: 720.0
- Frames per second: 60.0
- Total frames: 391.0

![frame 1](video_first_frame.png)

This high resolution and frame rate allowed for a thorough analysis of the performance of the two algorithms under test.

### Methodology

#### Simulation Setup
The simulation was structured around this specific video file, with frames captured and converted to grayscale for simplicity. Full Search and NNMP algorithms were subsequently applied to the frames to estimate motion vectors. Key metrics such as the time taken by each algorithm, the motion vectors, and the magnitude of motion were meticulously recorded.

#### Evaluation Metrics
The primary evaluation metrics for the study included:
- **Mean Squared Error (MSE):** Utilized as an accuracy measurement for comparing the results of the two algorithms.
- **Time Performance:** The time taken by each algorithm to estimate motion vectors.
- **Magnitude of Motion:** This helped in understanding the amount and direction of motion detected by the algorithms.

### Results

#### Time Performance
The results pointed to a significant time advantage for the NNMP algorithm over Full Search, with NNMP consistently performing faster across all frames.

Figure 1: Time Difference Over Frames
![Time Difference Over Frames](TimeDifferenceOverFrames.png)

The NNMP utilize the time by: 22,210.58% .

#### Accuracy of Motion Estimation
On comparing the motion vectors produced by both methods, a variance was observed as indicated by the MSE values. The differences in accuracy between the two methods varied per frame.

Figure 2: MSE Over Frames
![MSE Over Frames](MSEOverFrames.png)

#### Magnitude of Motion
Analysis of the magnitudes of motion indicated a varied distribution of motion estimation between both algorithms. This graph provides insights into how much motion is happening in the video over time. Each point in the graph represents the overall magnitude of motion at a specific frame of the video.

Figure 3: Magnitude of Motion Over Time
![Magnitude of Motion Over Time](MagnitudeOfMotionOverTime.png)

Average magnitude of motion: 0.70

Total magnitude of motion: 274.11

#### Direction of Motion
Further analysis on the direction of motion was conducted. The direction of motion graph illustrates the prevalent direction of movement in each frame of the video. The direction is usually depicted in degrees (from 

0 to 360) or in terms of cardinal directions if the motion is projected onto a plane.

Figure 4: Direction of Motion
![Direction of Motion](DirectionOfMotion.png)

### Image Quality
To compare the image quality produced by both methods, the Peak Signal-to-Noise Ratio (PSNR) was calculated.

Figure 5: PSNR Comparison
![PSNR Comparison](PSNRComparison.png)

PSNR for NNMP: 6.22

PSNR for Full Search: 6.75

The marginal loss between NNMP and Full Search is 7.84% in Peak-Signal-to-Noise ratio (PSNR).

This figure represents the comparison of PSNR values for both Full Search and NNMP algorithms. The higher the PSNR, the better the quality of the compressed or reconstructed image.

### Discussion
The observed time efficiency of NNMP can be attributed to its approach of computing a score for each possible motion vector and updating the best motion vector, compared to Full Search's exhaustive search method. Despite the time advantage of NNMP, the varying MSE values suggest a potential trade-off between time efficiency and accuracy of motion estimation.

### Conclusion
The findings underscore the importance of understanding the specific requirements of any given task. While NNMP offers significant time efficiency, the selection between Full Search and NNMP should consider the acceptable trade-offs in terms of motion estimation accuracy.

### References
1. Reddy, T.M., Muralidhar, P., & Rao, C.B.R. (Year). New Fast Search Block Matching Motion Estimation Algorithm for H.264 /AVC. NIT Warangal, India. Email: muralidhar417@gmail.com, pmurali@nitw.ac.in, cbrr@nitw.ac.in. 

Further references will include other relevant studies that have been referred to in this document.
