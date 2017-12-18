import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import basename


def calibrate_camera(images, nx, ny):
    '''
    Calibrate the camera.

    Input is a list of calibration images
    '''

    board_shape = (nx, ny)
    objpoints = []
    imgpoints = []

    # Create object reference points
    objp = np.zeros((board_shape[0]*board_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)

    image_size = None
    
#    for fname in tqdm(images):
    for fname in tqdm(images):
        # Open next calibration image
        img = mpimg.imread(fname)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # set image_size for later use if we haven't already set it
        if image_size == None:
            image_size = gray.shape[1::-1]
            
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_shape, None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            cv2.drawChessboardCorners(img, board_shape, corners, ret)
            cv2.imwrite('output_images/calib_{}'.format(basename(fname)), img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    cam_params = {'mtx' : mtx, 'dist' : dist}

    return cam_params

def undistort_image(img, cam_params):
    '''
    Undistorts an image given the image and camera parameters
    '''
    dst = cv2.undistort(img, cam_params['mtx'], cam_params['dist'], None, cam_params['mtx'])

    return dst

