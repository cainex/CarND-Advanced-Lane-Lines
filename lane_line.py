import numpy as np

class lane_line():
    def __init__(self):
        self.history_depth = 6
        # was the line detected in the last iteration?
        self.detected = False
        # current fitx
        self.currentx = None  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        self.currentx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
#        self.current_fit = [np.array([False])]  
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0.0 
        self.slope = 0.0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0.0
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        self.frames_since_update = 0
        self.update_threshold = 15

        self.ym_per_pix = 60/720
        self.xm_per_pix = 3.7/600
    

    def fit_polynomial(self, ploty):
        self.current_fit = np.polyfit(self.ally, self.allx, 2)           
        self.currentx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]

    def add_new_fit(self, fit):
        self.recent_xfitted.append(fit)
        if len(self.recent_xfitted) > self.history_depth :
            self.recent_xfitted.pop(0)
        
    def average_xfit(self):
        weights = None
        if len(self.recent_xfitted) == self.history_depth:
            #weights=[1,2,2,3,3,3,3,4]
            weights=[1,2,2,3,3,4]
        self.bestx = np.average(np.array(self.recent_xfitted), axis=0, weights=weights)
        ploty = np.linspace(0, len(self.bestx)-1, len(self.bestx))
        self.best_fit = np.polyfit(ploty, self.bestx, 2)

    def calculate_curvature(self, fitx):
        ploty = np.linspace(0, len(fitx)-1, len(fitx))
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty*self.ym_per_pix, fitx*self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*self.ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        return curverad
        
    def calculate_slope(self):
        ploty = np.linspace(0, len(self.bestx)-1, len(self.bestx))
        y_eval = np.max(ploty)
        polyder = np.polyder(self.current_fit)
        self.slope = polyder[0]*y_eval + polyder[1]

    def decay(self):
        self.frames_since_update = self.frames_since_update + 1
        if self.frames_since_update > self.update_threshold and len(self.recent_xfitted) > 0:
            self.recent_xfitted.pop(0)
        if len(self.recent_xfitted) == 0:
            self.bestx = None
            self.best_fit = None
    
    def get_pos(self):
        y_eval = len(self.currentx)-1
        x = self.current_fit[0]*y_eval**2 + self.current_fit[1]*y_eval + self.current_fit[2]
        return np.abs(640-x)*self.xm_per_pix

    def get_best_pos(self):
        y_eval = len(self.bestx)-1
        x = self.best_fit[0]*y_eval**2 + self.best_fit[1]*y_eval + self.best_fit[2]
        return np.abs(640-x)*self.xm_per_pix