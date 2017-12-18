import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

class lane_image:
    """
    Class to hold an image and provide image processing routines
    """
    def __init__(self, cam_params, input_image, params=None):
        self.cam_params = cam_params
#        self.filename = filename
        self.images = {}
#        self.images['orig'] = mpimg.imread(self.filename)
        self.images['orig'] = input_image

        self.image_size = (self.images['orig'].shape[1], self.images['orig'].shape[0])

        # region of interest mask parameters
        self.roi_params = {}
        self.roi_params['verts'] = np.array([[ (int(0+self.image_size[0]*.1), int(self.image_size[1] - self.image_size[1]*.1)),
                                              (int(self.image_size[0]/2 - (self.image_size[0]/2)*0.1), int(self.image_size[1]*.6)),
                                              (int(self.image_size[0]/2 + (self.image_size[0]/2)*0.1), int(self.image_size[1]*.6)),
                                              (int(self.image_size[0] - self.image_size[0]*.1), int(self.image_size[1] - self.image_size[1]*.1)) ]], dtype=np.int32)

        # print(self.roi_params['verts'])
        # undistort image
        self.images['undistorted'] = self.undistort_image()

        # params
        if params == None:
            self.params = {}
            self.params['color']['gray'] = {'thresh' : (25,100)}
            self.params['color']['hls_h_channel'] = {'thresh' : (15, 100)}
            self.params['color']['hls_l_channel'] = {'thresh' : (15, 100)}
            self.params['color']['hls_s_channel'] = {'thresh' : (90, 255)}
            self.params['color']['hsv_h_channel'] = {'thresh' : (15, 100)}
            self.params['color']['hsv_s_channel'] = {'thresh' : (15, 100)}
            self.params['color']['hsv_v_channel'] = {'thresh' : (90, 255)}
            self.params['thresh'] = {}
            self.params['thresh']['abs_sobel'] = {"kernel" : 3, "thresh" : (20,100)}
            self.params['thresh']['mag_grad'] = {"kernel" : 3, "thresh" : (30, 100)}
            self.params['thresh']['dir_grad'] = {"kernel" : 15, "thresh" : (0.7, 1.0)}
        else:
            self.params = params

        # colorspace conversions
        self.images['gray'] = cv2.cvtColor(self.images['undistorted'], cv2.COLOR_RGB2GRAY)
        self.images['hls'] = cv2.cvtColor(self.images['undistorted'], cv2.COLOR_RGB2HLS)
        self.images['hsv'] = cv2.cvtColor(self.images['undistorted'], cv2.COLOR_RGB2HSV)

        # binary colorspace conversions
        self.images['gray_binary'] = self.binary_image(self.images['gray'], self.params['color']['gray']['thresh'])
        self.images['hls_h'] = self.images['hls'][:,:,0]
        self.images['hls_l'] = self.images['hls'][:,:,1]
        self.images['hls_s'] = self.images['hls'][:,:,2]

        self.images['hsv_h'] = self.images['hsv'][:,:,0]
        self.images['hsv_s'] = self.images['hsv'][:,:,1]
        self.images['hsv_v'] = self.images['hsv'][:,:,2]

        self.images['hls_h_binary'] = self.binary_image(self.images['hls'][:,:,0], self.params['color']['hls_h_channel']['thresh'])
        self.images['hls_l_binary'] = self.binary_image(self.images['hls'][:,:,1], self.params['color']['hls_l_channel']['thresh'])
        self.images['hls_s_binary'] = self.binary_image(self.images['hls'][:,:,2], self.params['color']['hls_s_channel']['thresh'])

        self.images['hsv_h_binary'] = self.binary_image(self.images['hsv'][:,:,0], self.params['color']['hsv_h_channel']['thresh'])
        self.images['hsv_s_binary'] = self.binary_image(self.images['hsv'][:,:,1], self.params['color']['hsv_s_channel']['thresh'])
        self.images['hsv_v_binary'] = self.binary_image(self.images['hsv'][:,:,2], self.params['color']['hsv_v_channel']['thresh'])

        # Get gradient images
        self.images['sobelx'] = self.abs_sobel_thresh('x')
        self.images['sobely'] = self.abs_sobel_thresh('y')
        self.images['mag_grad'] = self.mag_thresh()
        self.images['dir_grad'] = self.dir_thresh()
        self.images['mag_and_dir'] = self.and_img(self.images['mag_grad'], self.images['dir_grad'])
        self.images['combined_grad'] = self.combine_gradients()

        # Transform parameters
        self.transform_params = {}
        self.transform_params['src'] = np.float32([ [675, 444], 
                                                    [1120, 719],
                                                    [190, 719],
                                                    [600, 444]])
        self.transform_params['dst'] = np.float32([ [840, 50], 
                                                    [840, 719],
                                                    [440, 719],
                                                    [440, 50]])

        # Transformed image
        #self.images['masked'] = self.mask_image(self.images['undistorted'])
        self.images['transform_grad'] = self.transform_image(self.mask_image(self.images['combined_grad']))

    def set_parameters(self, params):
        self.params = params

    def get_images(self):
        return self.images

    def set_final_image(self, img):
        self.images['final'] = img
        
    def undistort_image(self):
        '''
        Undistorts an image given the image and camera parameters
        '''
        dst = cv2.undistort(self.images['orig'], self.cam_params['mtx'], self.cam_params['dist'], None, self.cam_params['mtx'])

        return dst

    def abs_sobel_thresh(self, orient='x'):
        # 1) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize = self.params['thresh']['abs_sobel']['kernel'])
        if orient == 'y':
            sobel = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize = self.params['thresh']['abs_sobel']['kernel'])
            
        # 2) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 4) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        binary_output = self.binary_image(scaled_sobel, self.params['thresh']['abs_sobel']['thresh'])
        # 5) Return this mask as your binary_output image
        return binary_output

    def mag_thresh(self):
        # 1) Take the gradient in x and y separately
        sobelx = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize=self.params['thresh']['mag_grad']['kernel'])
        sobely = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize=self.params['thresh']['mag_grad']['kernel'])
        # 2) Calculate the magnitude 
        mag = np.sqrt(sobelx**2+sobely**2)
        # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*mag/np.max(mag))
        # 4) Create a binary mask where mag thresholds are met
        binary_output = self.binary_image(scaled_sobel, self.params['thresh']['mag_grad']['thresh'])
        # 5) Return this mask as your binary_output image
        return binary_output
        
    def dir_thresh(self):
        # 1) Take the gradient in x and y separately
        sobelx = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize=self.params['thresh']['dir_grad']['kernel'])
        sobely = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize=self.params['thresh']['dir_grad']['kernel'])
        # 2) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        dirgrad = np.arctan2(abs_sobely,abs_sobelx)
        # 4) Create a binary mask where direction thresholds are met
        binary_output = self.binary_image(dirgrad, self.params['thresh']['dir_grad']['thresh'])
        # 5) Return this mask as your binary_output image
        return binary_output

    def mask_image(self, img):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, np.int32(self.roi_params['verts']), ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def transform_image(self, img, inverse=False):
        # print(img.shape)
        if inverse:
            M = cv2.getPerspectiveTransform(self.transform_params['dst'], self.transform_params['src'])
        else:
            M = cv2.getPerspectiveTransform(self.transform_params['src'], self.transform_params['dst'])
        return cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))

    def and_img(self, left, right):
        combined = np.zeros_like(left)
        combined[ (left == 1) & (right == 1)] = 1
        return combined

    def or_img(self, left, right):
        combined = np.zeros_like(left)
        combined[ (left == 1) | (right == 1)] = 1
        return combined
        
    def combine_gradients(self):
        grad_img = self.and_img(self.images['sobelx'], self.and_img(self.images['mag_grad'], self.images['dir_grad']))
        color_img = self.or_img(self.or_img(self.images['hls_l_binary'], self.images['hsv_h_binary']), self.or_img(self.images['hsv_s_binary'], self.images['hsv_v_binary']))
        return self.or_img(grad_img, color_img)
        
        #grad_img = self.or_img(self.and_img(self.images['sobelx'], self.images['sobely']), self.and_img(self.images['mag_grad'], self.images['dir_grad']))
        #return self.and_img(grad_img, self.or_img(self.images['hls_s_binary'], self.images['hls_h_binary']))
        #return self.and_img(grad_img, self.or_img(self.images['s_binary'], self.images['h_binary']))
        
    def binary_image(self, src, thresh, invert=False):
        if invert == True:
            binary = np.ones_like(src)
            binary[(src > thresh[0]) & (src <= thresh[1])] = 0
        if invert == False:
            binary = np.zeros_like(src)
            binary[(src > thresh[0]) & (src <= thresh[1])] = 1
        return binary

    def plot_images(self):
        fig, ax = plt.subplots(ncols=4, nrows=3)

        ax[0][0].imshow(self.images['orig'])
        ax[0][0].set_title('original')
        ax[0][1].imshow(self.images['undistorted'])
        ax[0][1].set_title('undistorted')
        ax[0][2].imshow(self.images['gray'], cmap='gray')
        ax[0][2].set_title('gray')
        ax[0][3].imshow(self.images['gray_binary'], cmap='gray')
        ax[0][3].set_title('gray_binary')

        ax[1][0].imshow(self.images['hls'])
        ax[1][0].set_title('hls')
        ax[1][1].imshow(self.images['hls_s_binary'], cmap='gray')
        ax[1][1].set_title('hls_s_channel_binary')
        ax[1][2].imshow(self.images['hls_h_binary'], cmap='gray')
        ax[1][2].set_title('hls_h_channel_binary')

        ax[2][0].imshow(self.images['sobelx'], cmap='gray')
        ax[2][0].set_title('sobelx')
        ax[2][1].imshow(self.images['sobely'], cmap='gray')
        ax[2][1].set_title('sobely')
        ax[2][2].imshow(self.images['mag_grad'], cmap='gray')
        ax[2][2].set_title('mag_grad')
        ax[2][3].imshow(self.images['dir_grad'], cmap='gray')
        ax[2][3].set_title('dir_grad')

        plt.tight_layout()
        plt.show()

    def dump_images(self, dir):
        for key, value in self.images.items():
            if len(value.shape) == 2:
                cmap = 'gray'
            else:
                cmap = None
            print("Saving {}...".format(key))
            mpimg.imsave("{}/{}.png".format(dir, key), value, cmap=cmap)

