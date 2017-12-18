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
    parser.add_argument('-p', '--parameters', help='Image processing parameters', dest='params', type=str, default=None)
    parser.add_argument('-d', '--dump_dir', help="Directory to dump images", dest='dump_dir', type=str, default=None)
    parser.add_argument('-t', '--test_image', help="test image to use", dest='test_image', type=str, default=None)
    parser.add_argument('-v', '--test_video', help='test video file to use', dest='test_video', type=str, default='project_video.mp4')
    parser.add_argument('-o', '--output_video', help='name of output video file', dest='output_video', type=str, default='video_out.mp4')
    parser.add_argument('-s', '--subclip', help='process up to this point', dest='subclip', type=int, default=None)
    parser.add_argument('-i', '--debug', help='display debug info into output', dest='debug', action='store_true', default=False)
    args = parser.parse_args()

    ## Calibrate camera
    # get calibration images
    cam_params = None

    if args.cam_cal is None:
        print("Getting list of calibration images...")
        cal_images = glob.glob('./camera_cal/calibration*.jpg')

        # get camera calibration parameters
        print("Calibrating camera...")
        cam_params = calibrate_camera(cal_images, args.nx, args.ny)

        pickle.dump(cam_params, open('camera_params.p', 'wb'))
    else:
        cam_params = pickle.load( open(args.cam_cal, 'rb'))

    params = None
    if args.params is not None:
        params = pickle.load(open(args.params, 'rb'))
        
    current_lane = lane(cam_params, params)
    current_lane.debug_output = args.debug

    if (args.test_image == None):
        print("Processing video file:{}".format(args.test_video))
        # TODO : add code to handle processing of a video 
        if args.subclip is None:
            clip1 = editor.VideoFileClip(args.test_video)
        else:
            clip1 = editor.VideoFileClip(args.test_video).subclip(0, args.subclip)
        vid_clip = clip1.fl_image(current_lane.process_image)
        vid_clip.write_videofile(args.output_video, audio=False)
    else:
        # test undistortion of a calibration image
        print("processing test image...")

        current_lane.sanity_check = False
        test_image = mpimg.imread(args.test_image)

        final_image = current_lane.process_image(test_image)
        current_lane.get_img().set_final_image(final_image)

        if (args.dump_dir == None):
            current_lane.get_img().plot_images()
        else:
            current_lane.get_img().dump_images(args.dump_dir)


