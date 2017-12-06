from utils import calibrate_camera, undistort_image
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse

if __name__ == "__main__":
    ## Handle command-line arguments
    parser = argparse.ArgumentParser(description='Advanced Lane Finding')
    parser.add_argument('-c', '--camera_calib', help='Camera calibration parameters file', dest='cam_cal', type=str, default=None)
    args = parser.parse_args()

    ## Calibrate camera
    # get calibration images
    cam_params = None

    if args.cam_cal == None:
        print("Getting list of calibration images...")
        cal_images = glob.glob('./camera_cal/calibration*.jpg')

        # get camera calibration parameters
        print("Calibrating camera...")
        cam_params = calibrate_camera(cal_images, 9, 6)

        pickle.dump(cam_params, open('camera_params.p', 'wb'))
    else:
        cam_params = pickle.load( open(args.cam_cal, 'rb'))

    # test undistortion of a calibration image
    print("Undistorting test image...")
    test_img = mpimg.imread('./camera_cal/calibration1.jpg')

    undist_img = undistort_image(test_img, cam_params)

    fig, ax = plt.subplots(ncols=2, nrows=1)

    ax[0].imshow(test_img)
    ax[1].imshow(undist_img)

    plt.tight_layout()
    plt.show()


