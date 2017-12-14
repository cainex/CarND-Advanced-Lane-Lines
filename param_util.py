import cv2
import sys
from PyQt4 import QtGui
from PyQt4 import QtCore
import argparse
from moviepy import editor
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from lane_image import lane_image
import pickle

class kernel_layout(QtGui.QHBoxLayout):
    def __init__(self, parent, initial_params):
        super(kernel_layout, self).__init__()

        self.params = initial_params
        self.parent = parent

        self.initUI()

    def initUI(self):
        self.num_lbl = QtGui.QLabel()
        self.num_lbl.setText('Num')
        self.addWidget(self.num_lbl)
        self.num_value_lbl = QtGui.QLabel()
        self.num_value_lbl.setText('{}'.format(self.params))
        self.num_value_lbl.setFixedWidth(25)
        self.addWidget(self.num_value_lbl)
        self.num_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.num_value.setMinimum(0)
        self.num_value.setMaximum(15)
        self.num_value.setValue((self.params-1)/2)
        self.num_value.valueChanged.connect(self.update_kernels)
        self.num_value.sliderReleased.connect(self.update_image)
        self.addWidget(self.num_value)

    def update_kernels(self):
        self.params = (self.num_value.value() * 2 + 1)
        self.num_value_lbl.setText('{}'.format(self.params))
    
    def update_image(self):
        self.parent.get_vid_image()

    def set_params(self, params):
        self.params = params
        self.num_value_lbl.setText('{}'.format(self.params))
        self.num_value.setValue((self.params-1)/2)

    def get_params(self):
        return self.params
       
class threshold_layout(QtGui.QHBoxLayout):
    def __init__(self, parent, initial_params, scale_factor=None, maximum=255):
        super(threshold_layout, self).__init__()

        self.params = initial_params
        self.parent = parent
        self.max = maximum
        self.scale_factor = scale_factor

        self.initUI()

    def initUI(self):
        self.min_lbl = QtGui.QLabel()
        self.min_lbl.setText('Min')
        self.addWidget(self.min_lbl)
        self.min_value_lbl = QtGui.QLabel()
        self.min_value_lbl.setText('{}'.format(self.params[0]))
        self.min_value_lbl.setFixedWidth(25)
        self.addWidget(self.min_value_lbl)
        self.min_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        if self.scale_factor == None:
            self.min_value.setMaximum(self.max)
            self.min_value.setValue(self.params[0])
        else:
            self.min_value.setMaximum(int(self.max * self.scale_factor))
            self.min_value.setValue(int(self.params[0]*self.scale_factor))
        self.min_value.valueChanged.connect(self.update_min)
        self.min_value.sliderReleased.connect(self.update_image)
        self.addWidget(self.min_value)
        self.max_lbl = QtGui.QLabel()
        self.max_lbl.setText('Max')
        self.addWidget(self.max_lbl)
        self.max_value_lbl = QtGui.QLabel()
        self.max_value_lbl.setText('{}'.format(self.params[1]))
        self.max_value_lbl.setFixedWidth(25)
        self.addWidget(self.max_value_lbl)
        self.max_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        if self.scale_factor == None:
            self.max_value.setMaximum(self.max)
            self.max_value.setValue(self.params[1])
        else:
            self.max_value.setMaximum(int(self.max * self.scale_factor))
            self.max_value.setValue(int(self.params[1]*self.scale_factor))
        self.max_value.valueChanged.connect(self.update_max)
        self.max_value.sliderReleased.connect(self.update_image)
        self.addWidget(self.max_value)

    def update_min(self):
        if self.scale_factor is None:
            self.params = (self.min_value.value(), self.params[1])
        else:
            self.params = (float(self.min_value.value()/self.scale_factor), self.params[1])
        self.min_value_lbl.setText('{}'.format(self.params[0]))
 
    def update_max(self):
        if self.scale_factor is None:
            self.params = (self.params[0], self.max_value.value())
        else:
            self.params = (self.params[0], float(self.max_value.value()/self.scale_factor))
        self.max_value_lbl.setText('{}'.format(self.params[1]))

    def update_image(self):
        self.parent.get_vid_image()

    def set_params(self, params):
        self.params = params
        self.min_value_lbl.setText('{}'.format(self.params[0]))
        if self.scale_factor == None:
            self.min_value.setValue(self.params[0])
        else:
            self.min_value.setValue(int(self.params[0]*self.scale_factor))
        self.max_value_lbl.setText('{}'.format(self.params[1]))
        if self.scale_factor == None:
            self.max_value.setValue(self.params[1])
        else:
            self.max_value.setValue(int(self.params[1]*self.scale_factor))

    def get_params(self):
        return self.params

class layout_widget(QtGui.QWidget):
    def __init__(self, video_file):
        super(layout_widget, self).__init__()

        self.cam_params = pickle.load( open('./camera_params.p', 'rb'))

        self.vid_clip = editor.VideoFileClip(video_file)
        print(self.vid_clip.duration)

        self.frame_value = 0.0
        self.frame_select_value = 'orig'

        self.params = {}
        self.params['color'] = {}
        self.params['color']['gray'] = {'thresh' : (25,100)}
        self.params['color']['s_channel'] = {'thresh' : (90, 255)}
        self.params['color']['h_channel'] = {'thresh' : (15, 100)}
        self.params['thresh'] = {}
        self.params['thresh']['abs_sobel'] = {"kernel" : 3, "thresh" : (20,100)}
        self.params['thresh']['mag_grad'] = {"kernel" : 3, "thresh" : (30, 100)}
        self.params['thresh']['dir_grad'] = {"kernel" : 15, "thresh" : (0.7, 1.0)}

        self.initUI()

    def initUI(self):
        # image = self.vid_clip.get_frame(1.0)
        # # image = mpimg.imread('./test_images/test2.jpg')
        # print('got image {}x{}'.format(image.shape[0], image.shape[1]))
        # qimage = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)

        self.layout = QtGui.QHBoxLayout()

        #########################################################################
        ## Controls view        
        self.controls_layout = QtGui.QVBoxLayout()

        self.gray_bin_lbl = QtGui.QLabel()
        self.gray_bin_lbl.setText('Gray-Binary Threshold')
        self.gray_bin_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.gray_bin_lbl)
        self.gray_bin_controls = threshold_layout(self, self.params['color']['gray']['thresh'])
        self.controls_layout.addLayout(self.gray_bin_controls)
       
        self.s_bin_lbl = QtGui.QLabel()
        self.s_bin_lbl.setText('S-Binary Threshold')
        self.s_bin_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.s_bin_lbl)
        self.s_bin_controls = threshold_layout(self, self.params['color']['s_channel']['thresh'])
        self.controls_layout.addLayout(self.s_bin_controls)
       
        self.h_bin_lbl = QtGui.QLabel()
        self.h_bin_lbl.setText('H-Binary Threshold')
        self.h_bin_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.h_bin_lbl)
        self.h_bin_controls = threshold_layout(self, self.params['color']['h_channel']['thresh'])
        self.controls_layout.addLayout(self.h_bin_controls)
       
        self.abs_sobel_k_lbl = QtGui.QLabel()
        self.abs_sobel_k_lbl.setText('ABS Sobel Kernels')
        self.abs_sobel_k_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.abs_sobel_k_lbl)
        self.abs_sobel_k_controls = kernel_layout(self, self.params['thresh']['abs_sobel']['kernel'])
        self.controls_layout.addLayout(self.abs_sobel_k_controls)

        self.abs_sobel_lbl = QtGui.QLabel()
        self.abs_sobel_lbl.setText('ABS Sobel Threshold')
        self.abs_sobel_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.abs_sobel_lbl)
        self.abs_sobel_controls = threshold_layout(self, self.params['thresh']['abs_sobel']['thresh'])
        self.controls_layout.addLayout(self.abs_sobel_controls)

        self.mag_grad_k_lbl = QtGui.QLabel()
        self.mag_grad_k_lbl.setText('Magnitude Gradient Kernels')
        self.mag_grad_k_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.mag_grad_k_lbl)
        self.mag_grad_k_controls = kernel_layout(self, self.params['thresh']['mag_grad']['kernel'])
        self.controls_layout.addLayout(self.mag_grad_k_controls)

        self.mag_grad_lbl = QtGui.QLabel()
        self.mag_grad_lbl.setText('Magnitude Gradient Threshold')
        self.mag_grad_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.mag_grad_lbl)
        self.mag_grad_controls = threshold_layout(self, self.params['thresh']['mag_grad']['thresh'])
        self.controls_layout.addLayout(self.mag_grad_controls)

        self.dir_grad_k_lbl = QtGui.QLabel()
        self.dir_grad_k_lbl.setText('Direction Gradient Kernels')
        self.dir_grad_k_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.dir_grad_k_lbl)
        self.dir_grad_k_controls = kernel_layout(self, self.params['thresh']['dir_grad']['kernel'])
        self.controls_layout.addLayout(self.dir_grad_k_controls)

        self.dir_grad_lbl = QtGui.QLabel()
        self.dir_grad_lbl.setText('Direction Gradient Threshold')
        self.dir_grad_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.dir_grad_lbl)
        self.dir_grad_controls = threshold_layout(self, self.params['thresh']['dir_grad']['thresh'], scale_factor=10.0, maximum=10)
        self.controls_layout.addLayout(self.dir_grad_controls)

        self.controls_layout.addStretch(1)
        ##
        #########################################################################
        #########################################################################
        ## Image view
        self.image_layout = QtGui.QVBoxLayout()

        self.lbl1 = QtGui.QLabel()
        # self.lbl1.setPixmap(QtGui.QPixmap(qimage))
        self.get_vid_image()
        self.image_layout.addWidget(self.lbl1)

        self.frame_layout = QtGui.QHBoxLayout()

        self.frame_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMaximum(int(self.vid_clip.duration * 10))
        self.frame_slider.valueChanged.connect(self.frame_value_change)
        
        self.frame_label = QtGui.QLabel()
        self.frame_label.setText('{}'.format(self.frame_value))
        
        self.frame_select = QtGui.QComboBox(self)
        self.frame_select.addItem("orig")
        self.frame_select.addItem("undistorted")
        self.frame_select.addItem("hls")
        self.frame_select.addItem("gray")
        self.frame_select.addItem("gray_binary")
        self.frame_select.addItem("s_binary")
        self.frame_select.addItem("h_binary")
        self.frame_select.addItem("sobelx")
        self.frame_select.addItem("sobely")
        self.frame_select.addItem("mag_grad")
        self.frame_select.addItem("dir_grad")
        self.frame_select.addItem("combined_grad")
        self.frame_select.addItem("transform_grad")
        self.frame_select.currentIndexChanged.connect(self.frame_select_change)
        
        
        self.frame_layout.addWidget(self.frame_label)
        self.frame_layout.addWidget(self.frame_slider)
        self.frame_layout.addWidget(self.frame_select)
        self.image_layout.addLayout(self.frame_layout)
        ##
        #########################################################################

        self.layout.addLayout(self.image_layout)
        self.layout.addLayout(self.controls_layout)
        
        self.setLayout(self.layout)
    
    def get_parameters(self):
        self.params['color']['gray'] = {'thresh' : self.gray_bin_controls.get_params()}
        self.params['color']['s_channel'] = {'thresh' : self.s_bin_controls.get_params()}
        self.params['color']['h_channel'] = {'thresh' : self.h_bin_controls.get_params()}
        self.params['thresh']['abs_sobel'] = {"kernel" : self.abs_sobel_k_controls.get_params(), "thresh" : self.abs_sobel_controls.get_params()}
        self.params['thresh']['mag_grad'] = {"kernel" : self.mag_grad_k_controls.get_params(), "thresh" : self.mag_grad_controls.get_params()}
        self.params['thresh']['dir_grad'] = {"kernel" : self.dir_grad_k_controls.get_params(), "thresh" : self.dir_grad_controls.get_params()}

        return self.params

    def set_parameters(self, params):
        self.params = params
        self.gray_bin_controls.set_params(self.params['color']['gray']['thresh'])
        self.s_bin_controls.set_params(self.params['color']['s_channel']['thresh'])
        self.h_bin_controls.set_params(self.params['color']['h_channel']['thresh'])
        self.abs_sobel_k_controls.set_params(self.params['thresh']['abs_sobel']['kernel'])
        self.abs_sobel_controls.set_params(self.params['thresh']['abs_sobel']['thresh'])
        self.mag_grad_k_controls.set_params(self.params['thresh']['mag_grad']['kernel'])
        self.mag_grad_controls.set_params(self.params['thresh']['mag_grad']['thresh'])
        self.dir_grad_k_controls.set_params(self.params['thresh']['dir_grad']['kernel'])
        self.dir_grad_controls.set_params(self.params['thresh']['dir_grad']['thresh'])
        self.get_vid_image()

    def frame_value_change(self):
        self.frame_value = float(self.frame_slider.value()) / 10.0
        self.frame_label.setText('{}'.format(self.frame_value))
        self.get_vid_image()

    def frame_select_change(self, i):
        self.frame_select_value = self.frame_select.currentText()
        self.get_vid_image()

    def get_vid_image(self):
        image = self.vid_clip.get_frame(self.frame_value)
        # image = mpimg.imread('./test_images/test2.jpg')
        params = self.get_parameters()
        proc_image = lane_image(self.cam_params, image, params)

        disp_image = None
        if self.frame_select_value in ['gray_binary', 's_binary', 'h_binary', 'sobelx', 'sobely', 'mag_grad', 'dir_grad', 'combined_grad', 'transform_grad']:
            disp_image = np.zeros_like(proc_image.get_images()[self.frame_select_value])
            disp_image[proc_image.get_images()[self.frame_select_value] == 1] = 255
            
        else:
            disp_image = proc_image.get_images()[self.frame_select_value]

        if len(disp_image.shape) == 2:
            disp_image = cv2.cvtColor(disp_image, cv2.COLOR_GRAY2RGB)
            qimage = QtGui.QImage(disp_image, disp_image.shape[1], disp_image.shape[0], QtGui.QImage.Format_RGB888)
        else:
            qimage = QtGui.QImage(disp_image, disp_image.shape[1], disp_image.shape[0], QtGui.QImage.Format_RGB888)
        self.lbl1.setPixmap(QtGui.QPixmap(qimage))        
        

class gui_app(QtGui.QMainWindow):
    def __init__(self, video_file):
        super(gui_app, self).__init__()

        self.layout = layout_widget(video_file)
        self.initUI()
    
    def initUI(self):

        exitAct = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(QtGui.qApp.quit)

        paramLoadAct = QtGui.QAction(QtGui.QIcon('open.png'), '&Load', self)        
        paramLoadAct.setShortcut('Ctrl+L')
        paramLoadAct.setStatusTip('Load Parameters')
        paramLoadAct.triggered.connect(self.load_parameters)

        paramSaveAct = QtGui.QAction(QtGui.QIcon('save.png'), '&Save', self)        
        paramSaveAct.setShortcut('Ctrl+S')
        paramSaveAct.setStatusTip('Save Parameters')
        paramSaveAct.triggered.connect(self.save_parameters)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(paramLoadAct)
        fileMenu.addAction(paramSaveAct)
        fileMenu.addAction(exitAct)
        
        self.setCentralWidget(self.layout)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Simple menu')    
        self.show()

    def load_parameters(self):
        w = QtGui.QWidget()
        w.resize(320,240)
        w.setWindowTitle("Load Parameters")

        filename = QtGui.QFileDialog.getOpenFileName(w, 'Open File', '.')
        print(filename)
        params = pickle.load( open(filename, 'rb'))
        print(params)
        self.layout.set_parameters(params)

        w.show()
            
    def save_parameters(self):
        w = QtGui.QWidget()
        w.resize(320,240)
        w.setWindowTitle("Save Parameters")

        filename = QtGui.QFileDialog.getOpenFileName(w, 'Save File', '.')
        print(filename)
        pickle.dump(self.layout.get_parameters(), open(filename, 'wb'))

        w.show()
            
       
def main(argv):
    parser = argparse.ArgumentParser('Parameter Utility')
    parser.add_argument('-v', '--test_video', help='video file to use', dest='test_video', type=str, default='./challenge_video.mp4')
    parser.add_argument('-o', '--output_file', help='filename for pickled parameteres', dest='outfile', type=str, default='parameters.p')
    args, unknown_args = parser.parse_known_args(argv)

    app = QtGui.QApplication(argv[:1]+unknown_args)
    print('Using test_video:{} output:{}'.format(args.test_video, args.outfile))

    gui = gui_app(args.test_video)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main(sys.argv)



