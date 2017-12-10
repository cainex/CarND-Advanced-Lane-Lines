from utils import calibrate_camera, undistort_image
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import argparse
from lane_image import lane_image
from moviepy import editor
from lane import lane


if __name__ == "__main__":
    ## Handle command-line arguments
    parser = argparse.ArgumentParser(description='Advanced Lane Finding')
    parser.add_argument('-c', '--camera_calib', help='Camera calibration parameters file', dest='cam_cal', type=str, default=None)
    parser.add_argument('-x', '--camera_calib_nx', help='Camera calibration chessboard number of x inner corners', dest='nx', type=int, default=9)
    parser.add_argument('-y', '--camera_calib_ny', help='Camera calibration chessboard number of y inner corners', dest='ny', type=int, default=6)
    parser.add_argument('-d', '--dump_dir', help="Directory to dump images", dest='dump_dir', type=str, default=None)
    parser.add_argument('-t', '--test_image', help="test image to use", dest='test_image', type=str, default=None)
    parser.add_argument('-v', '--test_video', help='test video file to use', dest='test_video', type=str, default='project_video.mp4')
    parser.add_argument('-o', '--output_video', help='name of output video file', dest='output_video', type=str, default='video_out.mp4')
    args = parser.parse_args()

    ## Calibrate camera
    # get calibration images
    cam_params = None

    if args.cam_cal == None:
        print("Getting list of calibration images...")
        cal_images = glob.glob('./camera_cal/calibration*.jpg')

        # get camera calibration parameters
        print("Calibrating camera...")
        cam_params = calibrate_camera(cal_images, args.nx, args.ny)

        pickle.dump(cam_params, open('camera_params.p', 'wb'))
    else:
        cam_params = pickle.load( open(args.cam_cal, 'rb'))


    if (args.test_image == None):
        print("Processing video file:{}".format(args.test_video))
        current_lane = lane(cam_params)
        # TODO : add code to handle processing of a video 
        clip1 = VideoFileClip(args.test_video)
        vid_clip = clip1.fl_image(current_lane.process_image)
        vid_clip.write_videofile(args.output_video, audio=False)
    else:
        # test undistortion of a calibration image
        print("processing test image...")
        test_image = lane_image(cam_params, args.test_image)

        if (args.dump_dir == None):
            test_image.plot_images()
        else:
            test_image.dump_images(args.dump_dir)


