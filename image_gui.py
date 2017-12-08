import cv2
from lane_image import lane_image
import argparse
import pickle
import numpy as np

class image_gui:
    def __init__(self, name, camera_params, img_fname):
        self.img = lane_image(camera_params, img_fname)
        self.window_name = name

        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.binary2gray(self.img.images['gray_binary']))
        cv2.createTrackbar("gray threshold min", self.window_name,
                           self.img.color_params['gray']['thresh'][0],
                           255,
                           self.update_grayscale_min)
        cv2.createTrackbar("gray threshold max", self.window_name,
                           self.img.color_params['gray']['thresh'][1],
                           255,
                           self.update_grayscale_max)
        # cv2.createTrackbar("S threshold", self.window_name,
        #                    self.img.color_params['s_channel']['thresh'],
        #                    self.update_s_channel)
        # cv2.createTrackbar("H threshold", self.window_name,
        #                    self.img.color_params['h_channel']['thresh'],
        #                    self.update_h_channel)
        

    def update_grayscale_min(self, value):
        self.img.color_params['gray']['thresh'] = (value, self.img.color_params['gray']['thresh'][1])
        self.redraw()

    def update_grayscale_max(self, value):
        self.img.color_params['gray']['thresh'] = (self.img.color_params['gray']['thresh'][0], value)
        self.redraw()

    def redraw(self):
        self.img.images['gray_binary'] = self.img.binary_image(self.img.images['gray'], self.img.color_params['gray']['thresh'], True)
        display_image = self.binary2gray(self.img.images['gray_binary'])
        cv2.imshow(self.window_name, display_image)

    def binary2gray(self, img):
        out_img = np.zeros_like(img)
        out_img[img == 1] = 255
        return out_img



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Lane Finding')
    parser.add_argument('-c', '--camera_calib', help='Camera calibration parameters file', dest='cam_cal', type=str, default='./camera_params.p')
    parser.add_argument('-t', '--test_image', help="test image to use", dest='test_image', type=str, default='./test_images/straight_lines1.jpg')
    args = parser.parse_args()

    cam_params = pickle.load( open(args.cam_cal, 'rb'))

    gui = image_gui("GRAYSCALE", cam_params, args.test_image)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
    