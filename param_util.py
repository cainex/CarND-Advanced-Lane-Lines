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

class threshold_layout(QtGui.QHBoxLayout):
    def __init__(self, parent, initial_params):
        super(threshold_layout, self).__init__()

        self.params = initial_params
        self.parent = parent

        self.initUI()

    def initUI(self):
        self.min_lbl = QtGui.QLabel()
        self.min_lbl.setText('Min')
        self.addWidget(self.min_lbl)
        self.min_value_lbl = QtGui.QLabel()
        self.min_value_lbl.setText('{}'.format(self.params[0]))
        self.addWidget(self.min_value_lbl)
        self.min_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.min_value.setMaximum(255)
        self.min_value.setValue(self.params[0])
        self.addWidget(self.min_value)
        self.max_lbl = QtGui.QLabel()
        self.max_lbl.setText('Max')
        self.addWidget(self.max_lbl)
        self.max_value_lbl = QtGui.QLabel()
        self.max_value_lbl.setText('{}'.format(self.params[1]))
        self.addWidget(self.max_value_lbl)
        self.max_value = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.max_value.setMaximum(255)
        self.max_value.setValue(self.params[1])
        self.addWidget(self.max_value)

    def update_min()
        

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
        self.params['color']['gray_binary'] = {'thresh' : (25,100)}
        self.params['color']['s_binary'] = {'thresh' : (25,100)}
        self.params['color']['h_binary'] = {'thresh' : (25,100)}

        self.initUI()

    def initUI(self):
        # image = self.vid_clip.get_frame(1.0)
        # # image = mpimg.imread('./test_images/test2.jpg')
        # print('got image {}x{}'.format(image.shape[0], image.shape[1]))
        # qimage = QtGui.QImage(image, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)

        self.layout = QtGui.QHBoxLayout()

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

        #########################################################################
        ## Controls view        
        self.controls_layout = QtGui.QVBoxLayout()
        self.gray_bin_lbl = QtGui.QLabel()
        self.gray_bin_lbl.setText('Gray-Binary Threshold')
        self.gray_bin_lbl.setFixedWidth(400)
        self.controls_layout.addWidget(self.gray_bin_lbl)

        self.gray_bin_controls = threshold_layout(self, (25,100))
        self.controls_layout.addLayout(self.gray_bin_controls)
        self.controls_layout.addStretch(1)
        ##
        #########################################################################
        self.layout.addLayout(self.controls_layout)
        
        self.setLayout(self.layout)
    
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
        proc_image = lane_image(self.cam_params, image)

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

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        
        self.setCentralWidget(self.layout)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Simple menu')    
        self.show()
            
       
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



