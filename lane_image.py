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
        self.images['centroids'] = self.add_centroids_to_image(self.images['transform_grad'], self.find_window_centroids(self.images['transform_grad'], 50, 80, 100))

        
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


    def transform_image(self, img):
        print(img.shape)
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
            mpimg.imsave("{}/{}.png".format(dir, key), value, cmap=cmap)

    def window_mask(self, width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, image, window_width, window_height, margin):
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    def add_centroids_to_image(self, img, window_centroids):
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(img)
            r_points = np.zeros_like(img)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(50,80,img,window_centroids[level][0],level)
                r_mask = self.window_mask(50,80,img,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage= np.dstack((img, img, img))*255 # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
        
        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((img,img,img)),np.uint8)

        return output
