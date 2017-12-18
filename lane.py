from lane_image import lane_image
from lane_line import lane_line
import numpy as np
import cv2
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

class lane:
    def __init__(self, camera_params, params):
        self.left_fit = None
        self.right_fit = None
        self.camera_params = camera_params
        self.params = params
        self.img = None
        self.ym_per_pix = 60/720
        self.xm_per_pix = 3.7/600

        self.left_curverad = 0
        self.right_curverad = 0

        self.left_slope = 0
        self.right_slope = 0

        self.left_info = ''
        self.right_info = ''

        self.left_lane = lane_line()
        self.right_lane = lane_line()

        self.rms = 0.0
        self.mse = 0.0
        self.left_mse = 0.0
        self.right_mse = 0.0
        self.good_left_lane = True
        self.good_right_lane = True
        self.left_lane_pos = 0.0
        self.right_lane_pos = 0.0
        self.left_lane_pos_rmse = 0.0
        self.right_lane_pos_rmse = 0.0

    def get_img(self):
        return self.img

    def process_image(self, image):
        self.img = lane_image(self.camera_params, image, self.params)
        image = self.img.get_images()['transform_grad']
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])

        self.fit_line(image, self.left_lane, True)
        self.fit_line(image, self.right_lane, False)

        ### Make some decisions about detected lines
        self.good_left_lane = True
        self.good_right_lane = True

        self.left_curverad = self.left_lane.calculate_curvature(self.left_lane.currentx)
        self.right_curverad = self.right_lane.calculate_curvature(self.right_lane.currentx)
        self.rms = np.sqrt(np.mean(np.array([self.left_curverad,self.right_curverad])**2))
#        self.mse = np.sqrt(mean_squared_error([self.left_curverad], [self.right_curverad]))
        self.mse = mean_squared_log_error([self.left_curverad], [self.right_curverad])

        if len(self.left_lane.recent_xfitted) >= self.left_lane.history_depth and len(self.right_lane.recent_xfitted) >= self.right_lane.history_depth :
            ### Both lane lines are locked, make some determinations based on 
            ### previous values. 
            self.left_mse = mean_squared_log_error([self.left_curverad], [self.left_lane.radius_of_curvature])
            self.right_mse = mean_squared_log_error([self.right_curverad], [self.right_lane.radius_of_curvature])
            if self.left_mse > 3.0:
                self.good_left_lane = False
                # self.good_left_lane = True
            if self.right_mse > 3.0:
                self.good_right_lane = False
                # self.good_right_lane = True

            self.left_lane_pos = self.left_lane.get_pos()
            self.right_lane_pos = self.right_lane.get_pos()
            self.left_lane_pos_rmse = np.sqrt(mean_squared_error([self.left_lane_pos],[self.left_lane.line_base_pos]))
            self.right_lane_pos_rmse = np.sqrt(mean_squared_error([self.right_lane_pos],[self.right_lane.line_base_pos]))

            if self.left_lane_pos_rmse > 0.15:
                self.good_left_lane = False
            if self.right_lane_pos_rmse > 0.15:
                self.good_right_lane = False
            
        else:
            if self.left_lane.detected == True and self.right_lane.detected == True:
                ### Both lanes were detected in this run
                # 1) Check that curvature is similar

                if self.mse > 1.000:
                    self.good_left_lane = False
                    self.good_right_lane = False
            else:
                self.good_left_lane = False
                self.good_right_lane = False

        if self.good_left_lane:        
            self.update_lane(self.left_lane)
        if self.good_right_lane:        
            self.update_lane(self.right_lane)

        self.left_lane.decay()
        self.right_lane.decay()

        return_image = self.draw_final_image(self.img, self.left_lane.bestx, self.right_lane.bestx, ploty)

        return return_image

    def fit_line(self, image, lane, left):
        # print(image.shape)
        # print(image.shape[0]/2)

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = []

        if lane.best_fit is None:
            lane_inds = self.sliding_windows(image, nonzerox, nonzeroy, left)
        else:
            lane_inds = self.sliding_windows_pretrack(image, nonzerox, nonzeroy, lane.best_fit)

        if len(lane_inds) > 1000 :
            lane.detected = True
        else:
            lane.detected = False
        
        # Extract left and right line pixel positions
        lane.allx = nonzerox[lane_inds]
        lane.ally = nonzeroy[lane_inds] 

        # Fit a second order polynomial to each
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        lane.fit_polynomial(ploty)

        #######################################################################################################

    def update_lane(self, lane):
        lane.frames_since_update = 0
        lane.add_new_fit(lane.currentx)

        lane.average_xfit()

        # Fit new polynomials to x,y in world space
        lane.radius_of_curvature = lane.calculate_curvature(lane.bestx)

        #find slope at ymax
        lane.calculate_slope()

        lane.line_base_pos = lane.get_best_pos()
       
    def sliding_windows(self, image, nonzerox, nonzeroy, left):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[np.uint32(image.shape[0]/2):,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        if left == True:
            base = np.argmax(histogram[:midpoint])
        else:
            base = np.argmax(histogram[midpoint:]) + midpoint

        ## Get Lane indices
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        # Current positions to be updated for each window
        current = base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window+1)*window_height
            win_y_high = image.shape[0] - window*window_height
            win_x_low = current - margin
            win_x_high = current + margin
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        return lane_inds

    def sliding_windows_pretrack(self, image, nonzerox, nonzeroy, fit):
        margin = 75
        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + 
        fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + 
        fit[1]*nonzeroy + fit[2] + margin))) 

        return lane_inds

    def draw_final_image(self, img, left_fitx, right_fitx, ploty):
        result = img.get_images()['undistorted']

        if left_fitx is not None and right_fitx is not None:
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
            # Combine the result with the original image
            result = cv2.addWeighted(img.get_images()['undistorted'], 1, newwarp, 0.3, 0)

        # out_img_gray = np.zeros_like(img.get_images()['combined_grad'])
        # out_img_gray[img.get_images()['combined_grad'] == 1] = 255
        # out_img = cv2.cvtColor(out_img_gray, cv2.COLOR_GRAY2RGB)
        # result = cv2.addWeighted(out_img, 1, newwarp, 0.3, 0)
        cv2.putText(result,'left:{:.2f}m curr:{:.2f}m'.format(self.left_lane.radius_of_curvature, self.left_curverad), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'right:{:.2f}m curr{:.2f}m'.format(self.right_lane.radius_of_curvature, self.right_curverad), (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'lslope:{:.2f}'.format(self.left_lane.slope), (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'rslope:{:.2f}'.format(self.right_lane.slope), (1050, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'lrmse:{:.4f}'.format(self.left_mse), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'rrmse:{:.4f}'.format(self.right_mse), (1050, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'lfsu:{}'.format(self.left_lane.frames_since_update), (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'rfsu:{}'.format(self.right_lane.frames_since_update), (1050, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'lpos:{:.2f}-{:.2f}[{:.2f}]'.format(self.left_lane_pos, self.left_lane.line_base_pos, self.left_lane_pos_rmse), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'rpos:{:.2f}-{:.2f}[{:.2f}]'.format(self.right_lane_pos, self.right_lane.line_base_pos, self.right_lane_pos_rmse), (1050, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        

        cv2.putText(result,'rms:{:.2f}m'.format(self.rms), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'rmse:{:.4f}'.format(self.mse), (600, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(result,'lupdate:{}'.format(self.good_left_lane), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.good_left_lane else (255, 0, 0), 2)
        cv2.putText(result,'rupdate:{}'.format(self.good_right_lane), (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.good_right_lane else (255, 0, 0), 2)
        
        cv2.putText(result, 'ldet:{}[{}]'.format(self.left_lane.detected, len(self.left_lane.allx)), (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.left_lane.detected == True else (255,0,0), 2)
        cv2.putText(result, 'rdet:{}[{}]'.format(self.right_lane.detected, len(self.right_lane.allx)), (1050, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if self.right_lane.detected == True else (255,0,0), 2)
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
        