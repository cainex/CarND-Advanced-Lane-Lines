import cv2
import sys
from PyQt5.QtWidgets import QApplication, QWidget
import argparse

class gui_app(QWidget):
    def __init__(self):
        super.__init__()

        self.initUI()
    
    def initUI(self):
        
        QToolTip.setFont(QFont('SansSerif', 10))
        
        self.setToolTip('This is a <b>QWidget</b> widget')
        
        btn = QPushButton('Button', self)
        btn.setToolTip('This is a <b>QPushButton</b> widget')
        btn.resize(btn.sizeHint())
        btn.move(50, 50)       
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Tooltips')    
        self.show()
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parameter Utility')
    parser.add_argument('-t', '--test_image', help='test image to use', dest='test_image', type=str, default='test_images/test1.jpg')
    parser.add_argument('-o', '--output_file', help='filename for pickled parameteres', dest='outfile', type=str, default='parameters.p')
    args, unknown_args = parser.parse_known_args()

    app = QApplication(sys.argv[:1]+unknown_args)

    print('using test_iamge:{} output:{}'.format(args.test_image, args.outfile))
    w = QWidget()
    w.resize(250,150)
    w.move(300,300)
    w.setWindowTitle('Simple')
    
    w.show()

    sys.exit(app.exec_())


