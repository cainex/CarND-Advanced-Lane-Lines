from utils import calibrate_camera, undistort_image
import argparse
import cv2
from os.path import basename
import matplotlib.image as mpimg
import glob
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Calibration')
    parser.add_argument('-x', '--camera_calib_nx', help='Camera calibration chessboard number of x inner corners', dest='nx', type=int, default=9)
    parser.add_argument('-y', '--camera_calib_ny', help='Camera calibration chessboard number of y inner corners', dest='ny', type=int, default=6)
    parser.add_argument('-o', '--output_dir', help="output directory", dest='out_dir', type=str, default='output_images')
    args = parser.parse_args()

    print("Getting calibration images...")
    cal_images = glob.glob('./camera_cal/calibration*.jpg')

    # get camera calibration parameters
    print("Calibrating camera...")
    cam_params = calibrate_camera(cal_images, args.nx, args.ny)

    for fname in tqdm(cal_images):
        image = mpimg.imread(fname)

        undist_image = undistort_image(image, cam_params)

        cv2.imwrite('{}/undist_{}'.format(args.out_dir, basename(fname)), undist_image)
