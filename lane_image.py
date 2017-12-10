import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

class lane_image:
    """
    Class to hold an image and provide image processing routines
    """
    def __init__(self, cam_params, filename):
        self.cam_params = cam_params
        self.filename = filename
        self.images = {}
        self.images['orig'] = mpimg.imread(self.filename)

        self.image_size = (self.images['orig'].shape[1], self.images['orig'].shape[0])

        # region of interest mask parameters
        self.roi_params = {}
        self.roi_params['verts'] = np.array([[ (int(0+self.image_size[0]*.1), int(self.image_size[1] - self.image_size[1]*.1)),
                                              (int(self.image_size[0]/2 - (self.image_size[0]/2)*0.1), int(self.image_size[1]*.6)),
                                              (int(self.image_size[0]/2 + (self.image_size[0]/2)*0.1), int(self.image_size[1]*.6)),
                                              (int(self.image_size[0] - self.image_size[0]*.1), int(self.image_size[1] - self.image_size[1]*.1)) ]], dtype=np.int32)

        print(self.roi_params['verts'])
        # undistort image
        self.images['undistorted'] = self.undistort_image()

        # colorspace params
        self.color_params = {}
        self.color_params['gray'] = {'thresh' : (25,100)}
        self.color_params['s_channel'] = {'thresh' : (90, 255)}
        self.color_params['h_channel'] = {'thresh' : (15, 100)}

        # colorspace conversions
        self.images['gray'] = cv2.cvtColor(self.images['undistorted'], cv2.COLOR_RGB2GRAY)
        self.images['hls'] = cv2.cvtColor(self.images['undistorted'], cv2.COLOR_RGB2HLS)

        # binary colorspace conversions
        self.images['gray_binary'] = self.binary_image(self.images['gray'], self.color_params['gray']['thresh'], True)
        self.images['s_binary'] = self.binary_image(self.images['hls'][:,:,2], self.color_params['s_channel']['thresh'])
        self.images['h_binary'] = self.binary_image(self.images['hls'][:,:,1], self.color_params['h_channel']['thresh'], True)

        # gradient parameters
        self.thresh_params = {}
        self.thresh_params['abs_sobel'] = {"kernel" : 3, "thresh" : (20,100)}
        self.thresh_params['mag_grad'] = {"kernel" : 3, "thresh" : (30, 100)}
        self.thresh_params['dir_grad'] = {"kernel" : 15, "thresh" : (0.7, 1.0)}

        # Get gradient images
        self.images['sobelx'] = self.abs_sobel_thresh('x')
        self.images['sobely'] = self.abs_sobel_thresh('y')
        self.images['mag_grad'] = self.mag_thresh()
        self.images['dir_grad'] = self.dir_thresh()
        self.images['combined_grad'] = self.combine_gradients()

        # Transform parameters
        self.transform_params = {}
        self.transform_params['src'] = np.float32([ [675, 444], 
                                                    [1120, 719],
                                                    [190, 719],
                                                    [600, 444]])
        self.transform_params['dst'] = np.float32([ [950, 50], 
                                                    [950, 719],
                                                    [375, 719],
                                                    [375, 50]])

        # Transformed image
        self.images['masked'] = self.mask_image(self.images['undistorted'])
        self.images['transform_grad'] = self.transform_image(self.mask_image(self.images['combined_grad']))
        self.left_fitx, self.right_fitx, self.ploty = self.sliding_windows(self.images['transform_grad'])
        self.images['final'] = self.draw_final_image()
        
    def undistort_image(self):
        '''
        Undistorts an image given the image and camera parameters
        '''
        dst = cv2.undistort(self.images['orig'], self.cam_params['mtx'], self.cam_params['dist'], None, self.cam_params['mtx'])

        return dst

    def abs_sobel_thresh(self, orient='x'):
        # 1) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize = self.thresh_params['abs_sobel']['kernel'])
        if orient == 'y':
            sobel = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize = self.thresh_params['abs_sobel']['kernel'])
            
        # 2) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 4) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        binary_output = self.binary_image(scaled_sobel, self.thresh_params['abs_sobel']['thresh'])
        # 5) Return this mask as your binary_output image
        return binary_output

    def mag_thresh(self):
        # 1) Take the gradient in x and y separately
        sobelx = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize=self.thresh_params['mag_grad']['kernel'])
        sobely = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize=self.thresh_params['mag_grad']['kernel'])
        # 2) Calculate the magnitude 
        mag = np.sqrt(sobelx**2+sobely**2)
        # 3) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*mag/np.max(mag))
        # 4) Create a binary mask where mag thresholds are met
        binary_output = self.binary_image(scaled_sobel, self.thresh_params['mag_grad']['thresh'])
        # 5) Return this mask as your binary_output image
        return binary_output
        
    def dir_thresh(self):
        # 1) Take the gradient in x and y separately
        sobelx = cv2.Sobel(self.images['gray'], cv2.CV_64F, 1, 0, ksize=self.thresh_params['dir_grad']['kernel'])
        sobely = cv2.Sobel(self.images['gray'], cv2.CV_64F, 0, 1, ksize=self.thresh_params['dir_grad']['kernel'])
        # 2) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 3) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        dirgrad = np.arctan2(abs_sobely,abs_sobelx)
        # 4) Create a binary mask where direction thresholds are met
        binary_output = self.binary_image(dirgrad, self.thresh_params['dir_grad']['thresh'])
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
        print(img.shape)
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
        grad_img = self.or_img(self.and_img(self.images['sobelx'], self.images['sobely']), self.and_img(self.images['mag_grad'], self.images['dir_grad']))
        return self.or_img(grad_img, self.images['s_binary'])
#        grad_s_img = self.or_img(grad_img, self.images['s_binary'])
#        return self.or_img(grad_s_img, self.images['gray_binary'])
        
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
        ax[1][1].imshow(self.images['s_binary'], cmap='gray')
        ax[1][1].set_title('s_channel_binary')
        ax[1][2].imshow(self.images['h_binary'], cmap='gray')
        ax[1][2].set_title('h_channel_binary')

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

    def sliding_windows(self, image):
        # Assuming you have created a warped binary image called "images"
        # Take a histogram of the bottom half of the image
        print(image.shape)
        print(image.shape[0]/2)
        histogram = np.sum(image[np.uint32(image.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

    def draw_final_image(self):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.images['transform_grad']).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.transform_image(color_warp, True)
        # newwarp = cv2.warpPerspective(color_warp, Minv, (self.images['undistorted'].shape[1], self.images['undistorted'].shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(self.images['undistorted'], 1, newwarp, 0.3, 0)
        return result
