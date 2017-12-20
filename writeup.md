
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib_source]: ./camera_cal/calibration2.jpg "Camera Calibration"
[calib_points]: ./output_images/calib_calibration2.jpg "Calibration Points"
[calib_undist]: ./output_images/undist_calibration2.jpg "Undistorted Calibration"
[orig]: ./output_images/orig.png
[undist]: ./output_images/undistorted.png
[grayscale]: ./output_images/gray.png
[hls_l_binary]: ./output_images/hls_l_binary.png
[hls_s_binary]: ./output_images/hls_s_binary.png
[hsv_h_binary]: ./output_images/hsv_h_binary.png
[hsv_v_binary]: ./output_images/hsv_v_binary.png
[sobelx]: ./output_images/sobelx.png
[mag_grad]: ./output_images/mag_grad.png
[dir_grad]: ./output_images/dir_grad.png
[params_utils_1]: ./output_images/params_utils_1.png
[param_utils_2]: ./output_images/param_utils_2.png
[params_utils_4]: ./output_images/params_utils_4.png
[combined_grad]: ./output_images/combined_grad.png
[transform_grad]: ./output_images/transform_grad.png
[final]: ./output_images/final.png
[detect]: ./output_images/detect.png

[video1]: ./output_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Main Application

The application to detect lanes from images and videos is lane_find.py.

```
usage: lane_find.py [-h] [-c CAM_CAL] [-x NX] [-y NY] [-p PARAMS]
                    [-d DUMP_DIR] [-t TEST_IMAGE] [-v TEST_VIDEO]
                    [-o OUTPUT_VIDEO] [-s SUBCLIP] [-i]

Advanced Lane Finding

optional arguments:
  -h, --help            show this help message and exit
  -c CAM_CAL, --camera_calib CAM_CAL
                        Camera calibration parameters file
  -x NX, --camera_calib_nx NX
                        Camera calibration chessboard number of x inner
                        corners
  -y NY, --camera_calib_ny NY
                        Camera calibration chessboard number of y inner
                        corners
  -p PARAMS, --parameters PARAMS
                        Image processing parameters
  -d DUMP_DIR, --dump_dir DUMP_DIR
                        Directory to dump images
  -t TEST_IMAGE, --test_image TEST_IMAGE
                        test image to use
  -v TEST_VIDEO, --test_video TEST_VIDEO
                        test video file to use
  -o OUTPUT_VIDEO, --output_video OUTPUT_VIDEO
                        name of output video file
  -s SUBCLIP, --subclip SUBCLIP
                        process up to this point
  -i, --debug           display debug info into output
```

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the calibrate_camera() function in utils.py[lines 9-53].  

The function is called with a list of calibration image filenames, and the number of inner corners for the x and y axes. For this calibration, the number of x inners corners is 9, and the number of y inner corners is 6.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I loop over each image in the list of filenames passed into this function. Each image is opened, converted to grayscale, and the chessboard detection is accomplished using cv2.findChessboardCorners().  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Undistortion of images is done with the undistort_image() function in utils.py[lines 55-61]. This distortion correction is applied to the requested image using the `cv2.undistort()` function.

The original image:
![alt text][calib_source]

Detected points:
![alt text][calib_points]

Undistorted image:
![alt text][calib_undist]

When the camera is calibrated in the lane_find.py application, the camera calibration parameters are pickled and written to *camera_params.p*. These saved calibration parameters can be used in subsequent runs to skip the calibration step.

### Pipeline (single images)

To process a single image and dump the results, run:

```
% python lane_find.py -c camera_params.p -p parameters.p -t test_images/test_image1.jpg -d output_images
```

#### 1. Provide an example of a distortion-corrected image.

Each process image is first run through the undistort_image() method:
Original Image:
![alt text][orig]

Results after undistortion:
![alt text][undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Each image is first undistorted using the calibration parameters then processed for color and gradient thresholds using the lane_image class in lane_image.py. Each image is converted to grayscale, HSV and HLS colorspaces. The HSV and HLS colorspace images are each split into their constituent parts. The sobel-X, sobel-Y, maginitude and direction gradients are also calculated.

Once the colorspace conversion is complete, the individual channels are passed through a binary filter using threshold values passed into the class initialization. These parameters are given to the lane_find.py application via a pickled data file, and set using the param_utils.py GUI application. Using this application, I was able to experiment with different combinations of images and different threshold/kernel values. This application was then used to save out the parameters as a pickled data file to be used with the lane_find.py application.

![alt text][params_utils_1]
![alt text][param_utils_2]

The controls for the GUI application allowed me to step through each frame of a video file, choose which processed image to view, and set each of the parameters via sliders.

![alt text][params_utils_4]

Experimenting with this application, I decided to use a combination of processed binary images to create a composite image used to identify the lanes. The processed binary images used were the H-channel of the HSV, the V-channel of the HSV, the L-channel of the HLS, the S-channel of the HLS, the Sobel-X binary image, the Magnitude and Direction gradients.

HSV H-Channel:
![alt text][hsv_h_binary]

HSV V-Channel:
![alt text][hsv_v_binary]

HLS L-Channel:
![alt text][hls_l_binary]

HLS S-Channel:
![alt text][hls_s_binary]

Sobel-X:
![alt text][sobelx]

Magnitude Gradient:
![alt text][mag_grad]

Direction Gradient:
![alt text][dir_grad]

The gradient images were AND'd together, then that result was OR'd together with the binary colorspace images to create the composite image:

![alt text][combined_grad]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for the perspective transform is contained in the transform_image method of the lane_image class found in lane_image.py [lines 179-185]. This method takes in the image to transform, and a parameter to determine if this is a forward transform or a reverse transform. The parameters used for the transform are members of the lane_image class found in lane_find.py [lines 79-87].

```
        self.transform_params = {}
        self.transform_params['src'] = np.float32([ [675, 444], 
                                                    [1120, 719],
                                                    [190, 719],
                                                    [600, 444]])
        self.transform_params['dst'] = np.float32([ [840, 50], 
                                                    [840, 719],
                                                    [440, 719],
                                                    [440, 50]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 675, 444      | 840, 50       | 
| 1120, 719     | 840, 719      |
| 190, 719      | 440, 719      |
| 600, 444      | 440, 50       |

The transform was then performed on the combined gradient image:

![alt text][transform_grad]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

With the combined gradient image generated and contained in the lane_image object, the process of detecting the lane is begun. The lane detection is performed in the process_image() method of the lane class contained in lane.py [lines 50-116].  

Persistent information for each lane line is kept in a lane_line object, which is defined in lane_line.py. Here previous fit information is stored, and best fits are averaged over the previous 6 fitted lines, based on weight averages, favoring the latest first, and reducing weights of previous fit lines.

First, lines are detected in the current image using the fit_line() method of the lane class in lane.py [lines 119-145]. In this method, we first check to see if a line has been previously detected. If there is no current line detected (either we have not found a line yet, or the previous line was decayed out), a sliding window method is used to detect lines. This is done using the sliding_window() method of the lane class foudn in lane.py [lines 164-213]. This method starts by getting a histogram of the image at the point closest to the car. Windows are then created around the peaks of the histogram. If the windows detects more than 50 pixels, the next window is re-centered. This is done for 9 windows across the image. The resulting detection is visualized:

![alt text][detect]

If a line had been previously detected, we only look for valid pixels within a margin around the previously fit line. This is performed by the sliding_window_pretrack() method of the lane object in lane.py [lines 215-219].

Once both lines of the lane have been detected, some basic sanity checks are performed to determine if this line should be save. If this is the first detection, we calculated the radius of curvature of both lines. If the root mean squared logarithmic error of the two radius of curvatures are within an acceptable margin, we save this detection. Otherwise this detection is rejected.

If we have previous detection of lines, the sanity check is performed for each line independently. If the position of the first point of the detected line closest to the car is within an acceptable margin, that line detection is added to the detected lines. In this stage, each line can be saved independently.

Once, the lines are through the detection process, we run a decay method on both lines found using the decay method of the lane_line object in lane_line.py [lines 73-79]. This will pop off the oldest lane detection and increment the counter which tracks how many frames since the last update.

If a lane line was detected, we then update the lane_line object to save this particular detection, generate the averaged best fit, and clear the update counter. This averaged best fit is what is used as the actual lane detection.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature of the lane is calculated using the calculate_curvature() method of the lane_line object in lane_line.py [lines 57-65]. The curvature is calculated for each lane on the current detection along with each lane's best fit. The current fit curvature is used for sanity checks when the lane is first being detected. The best fit curvature results are then averaged together to determine the curvature of the lane. The lane curvature is then overlayed on the resulting image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The final image is rendered in the draw_final_image() method of the lane object in lane.py [lines 223-272]. The resulting image is visualized:

![alt text][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I began with creating the colorspace and gradient transforms of the source image. With this in hand, I then created the param_utils.py GUI application to aid in the analysis of these images. With a suitable combination in hand, I then used the techniques shown in the lecture material to detect the lane lines. I experimented with different averaging techniques, and sanity checks, including using the root mean squared log error for individual lines before declaring them as valid. My finding was that using the rmsle between both lane lines was useful for the intial detection, but for subsequent detections the variation within a single lane line was too great for this to be useful. The position of the root of the line was much more useful for this purpose. Using these techniques, the detection of the project_video was working well.

There are quite a few limitations with this approach. The primary limitation with using colorspace conversions and sobel gradients is that the detection is very sensitive to lighting changes, exposure differences and shadows. This creates a system that is somewhat unreliable. 

It occurs to me that a more robust technique for identifying a lane would be to use a convolutional neural network that was used in the previous project. This system inherently detected the lane and was also able to make decisions based on this detection with far more robust results. I believe a modified CNN could be used to output the perimiter of a detected lane, and training data could be created synthetically using a similar simulator which could input lane images 'labelled' with the polynomial fit numbers or fit pixels of the lane.

