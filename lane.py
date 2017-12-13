from lane_image import lane_image
import numpy as np
import cv2

class lane:
    def __init__(self, camera_params):
        self.left_fit = None
        self.right_fit = None
        self.camera_params = camera_params
        self.img = None

    def get_img(self):
        return self.img
    
    def process_image(self, image):
        self.img = lane_image(self.camera_params, image)
        if self.left_fit == None or self.right_fit == None:
            left_fitx, right_fitx, ploty = self.sliding_windows(self.img)
        else:
            left_fitx, right_fitx, ploty = self.sliding_windows_pretrack(self.img)
            
        return self.draw_final_image(self.img, left_fitx, right_fitx, ploty)

    def sliding_windows(self, img):
        # Take a histogram of the bottom half of the image
        # print(image.shape)
        # print(image.shape[0]/2)
        image = img.get_images()['transform_grad']
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
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        return left_fitx, right_fitx, ploty

    def sliding_windows_pretrack(self, img):
        image = img.get_images()['transform_grad']

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]        

        return left_fitx, right_fitx, ploty

    def draw_final_image(self, img, left_fitx, right_fitx, ploty):

        image = img.get_images()['transform_grad']
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = img.transform_image(color_warp, True)
        # newwarp = cv2.warpPerspective(color_warp, Minv, (self.images['undistorted'].shape[1], self.images['undistorted'].shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img.get_images()['undistorted'], 1, newwarp, 0.3, 0)
        return result

    # def window_mask(self, width, height, img_ref, center,level):
    #     output = np.zeros_like(img_ref)
    #     output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1

    #     y_coord = np.uint32(((img_ref.shape[0]-level*height)+(img_ref.shape[0]-(level+1)*height))/2)

    #     return output, [ np.uint32(center), y_coord]

    # def find_window_centroids(self, image, window_width, window_height, margin):
        
    #     window_centroids = [] # Store the (left,right) window centroid positions per level
    #     window = np.ones(window_width) # Create our window template that we will use for convolutions
        
    #     # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    #     # and then np.convolve the vertical image slice with the window template 
        
    #     # Sum quarter bottom of image to get slice, could use a different ratio
    #     l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    #     l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    #     r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    #     r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
        
    #     # Add what we found for the first layer
    #     window_centroids.append((l_center,r_center))
        
    #     # Go through each layer looking for max pixel locations
    #     for level in range(1,(int)(image.shape[0]/window_height)):
    #         # convolve the window into the vertical slice of the image
    #         image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
    #         conv_signal = np.convolve(window, image_layer)
    #         # Find the best left centroid by using past left center as a reference
    #         # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    #         offset = window_width/2
    #         l_min_index = int(max(l_center+offset-margin,0))
    #         l_max_index = int(min(l_center+offset+margin,image.shape[1]))
    #         l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
    #         # Find the best right centroid by using past right center as a reference
    #         r_min_index = int(max(r_center+offset-margin,0))
    #         r_max_index = int(min(r_center+offset+margin,image.shape[1]))
    #         r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
    #         # Add what we found for that layer
    #         window_centroids.append((l_center,r_center))

    #     return window_centroids

    # def add_centroids_to_image(self, img, window_centroids, width, height):
    #     # If we found any window centers
    #     if len(window_centroids) > 0:

    #         # Points used to draw all the left and right windows
    #         l_points = np.zeros_like(img)
    #         r_points = np.zeros_like(img)

    #         l_center_points = []
    #         r_center_points = []

    #         # Go through each level and draw the windows 	
    #         for level in range(0,len(window_centroids)):
    #             # Window_mask is a function to draw window areas
    #             l_mask, l_center = self.window_mask(width,height,img,window_centroids[level][0],level)
    #             r_mask, r_center = self.window_mask(width,height,img,window_centroids[level][1],level)

    #             print(l_center)
    #             l_center_points.append(l_center)
    #             r_center_points.append(r_center)

    #             # Add graphic points from window mask here to total pixels found 
    #             l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    #             r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    #         # Draw the results
    #         template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    #         zero_channel = np.zeros_like(template) # create a zero color channel
    #         template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    #         warpage= np.dstack((img, img, img))*255 # making the original road pixels 3 color channels
    #         output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    #         for center_point in l_center_points:
    #             cv2.circle(output, (center_point[0], center_point[1]), 5, (128,128,255), -1)

    #         for center_point in r_center_points:
    #             cv2.circle(output, (center_point[0], center_point[1]), 5, (255,128,128), -1)
        
    #         left_fit = np.polyfit(np.array(l_center_points)[:,0], np.array(l_center_points)[:,1], 2)
    #         right_fit = np.polyfit(np.array(r_center_points)[:,0], np.array(r_center_points)[:,1], 2)

    #         ploty = np.linspace(0, output.shape[0]-1, output.shape[0] )
    #         left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #         right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #         fig, ax = plt.subplots(ncols=1, nrows=1)
    #         ax.imshow(output)
    #         ax.plot(left_fitx, ploty, color='yellow')
    #         ax.plot(right_fitx, ploty, color='yellow')

    #     #    plt.show()

    #     # If no window centers found, just display orginal road image
    #     else:
    #         output = np.array(cv2.merge((img,img,img)),np.uint8)

    #     return output
        