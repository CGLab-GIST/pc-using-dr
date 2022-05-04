import socket
import sys
import time
import threading
import socket
import time

import cv2

import numpy as np
import imageio

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, pyqtSignal

import qimage2ndarray

import pyexr

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Float, Thread, Bitmap, Struct

TAIL_UTF8 = "__TAIL_TAIL_TAIL__".encode('utf-8')

CAM_WIDTH = 640
CAM_HEIGHT = 480

PROJ_WIDTH = 800
PROJ_HEIGHT = 600

scene_path = "data/curved"

def read_exr(path):
    # Remove alpha channel
    rgb_np = pyexr.read(path)[:,:,:3]
    rgb_ek = Float(np.reshape(rgb_np, (CAM_WIDTH*CAM_HEIGHT*3)))
    return rgb_ek


def read_png(filename, width=CAM_WIDTH, height=CAM_HEIGHT):

    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("cannot read ", filename)
        exit(-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Float(np.reshape(img, (width*height*3))) / 255.0
    img = np.clip(img, 0.0, 1.0)

    return Float(img)


class iPadCamera():
    def __init__(self, ip='192.168.10.19', port=12345):
        # iPad device's IP address
        ADDR = (ip, port)
        
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect(ADDR)

        print("Connecting...")
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect(ADDR)
        print("Done")

        self.sample_num = 1

    def __del__(self):
        self.clientSocket.close()


    # Request signal to the RCDCamera application and return received data
    def request(self, signal):
        self.clientSocket.send(signal.encode())
        data = self.clientSocket.recv(50000)

        while TAIL_UTF8 not in data:
            data += self.clientSocket.recv(50000)

        # if not sleep app may crashes
        time.sleep(0.1)

        return data

    # PNG formatted image itself is returned from the RCDCamera.
    def get_rgb_image(self, filename):
        img_data = self.request('rgb')
        img_data = img_data[:-len(TAIL_UTF8)]
        path = filename + '.png'

        f = open(path, 'wb')
        f.write(img_data)
        f.close()

    # Depth array binary is returned from the RCDCamera.
    # You have to convert it to exr or other image format manually.
    
    def get_depth_image(self, filename):
        path = filename + '.exr'

        array = np.zeros(192 * 256, dtype=np.float32)

        for i in range(self.sample_num):
            img_data = self.request('depth')
            array += np.frombuffer(img_data[:-len(TAIL_UTF8)], dtype=np.float32)

        array /= self.sample_num

        # LiDAR scene depth resolution is 256*192 in iPad 12.9 4th-gen.
        array = np.reshape(array, (192, 256))
        print(array.shape)

        imageio.imwrite(path, array)


# Reference : https://stackoverflow.com/a/59539843
class myImageDisplayApp (QObject):

    # Define the custom signal
    # https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#the-pyqtslot-decorator
    signal_update_image = pyqtSignal(str)
    signal_update_image_arr = pyqtSignal(Float)

    def __init__ (self):

        super().__init__()

        # Setup the seperate thread 
        # https://stackoverflow.com/a/37694109/4988010
        self.thread = threading.Thread(target=self.run_img_widget_in_background) 
        self.thread.daemon = True
        self.thread.start()

    def run_img_widget_in_background(self):
        self.app = QApplication(sys.argv)
        self.my_bg_qt_app = qtAppWidget(main_thread_object=self)
        self.app.exec_()

    def emit_image_update(self, pattern_file=None):
        self.signal_update_image.emit(pattern_file)

    def emit_image_update_arr(self, arr):
        self.signal_update_image_arr.emit(arr)

class qtAppWidget (QLabel):

    def __init__ (self, main_thread_object):

        super().__init__()

        # Connect the singal to slot
        main_thread_object.signal_update_image.connect(self.updateImageByPath)
        main_thread_object.signal_update_image_arr.connect(self.updateImageByArr)

        self.setupGUI()

    def setupGUI(self):

        self.app = QApplication.instance()

        # Get avaliable screens/monitors
        # https://doc.qt.io/qt-5/qscreen.html
        # Get info on selected screen 
        self.selected_screen = 1            # Select the desired monitor/screen

        self.screens_available = self.app.screens()
        self.screen = self.screens_available[self.selected_screen]
        self.screen_width = self.screen.size().width()
        self.screen_height = self.screen.size().height()

        # Create a black image for init 
        self.pixmap = QPixmap(self.screen_width, self.screen_height)
        self.pixmap.fill(QColor('black'))

        # Create QLabel object
        self.img_widget = QLabel()

        # Varioius flags that can be applied to make displayed window frameless, fullscreen, etc...
        # https://doc.qt.io/qt-5/qt.html#WindowType-enum
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum
        self.img_widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowStaysOnTopHint)
        
        # Hide mouse cursor 
        self.img_widget.setCursor(Qt.BlankCursor)       

        self.img_widget.setStyleSheet("background-color: black;") 

        self.img_widget.setGeometry(0, 0, self.screen_width, self.screen_height)            # Set the size of Qlabel to size of the screen
        self.img_widget.setWindowTitle('myImageDisplayApp')
        self.img_widget.setAlignment(Qt.AlignCenter | Qt.AlignTop) #https://doc.qt.io/qt-5/qt.html#AlignmentFlag-enum                         
        self.img_widget.setPixmap(self.pixmap)
        self.img_widget.show()

        # Set the screen on which widget is on
        self.img_widget.windowHandle().setScreen(self.screen)
        # Make full screen 
        self.img_widget.showFullScreen()
        

    def updateImageByPath(self, pattern_file=None):
        pixmap = QPixmap(pattern_file).scaled(self.screen_width,self.screen_height, Qt.KeepAspectRatio)         # Update pixmap with desired image
        self.pixmap = pixmap

        self.img_widget.setPixmap(self.pixmap)      # Show desired image on Qlabel

    def updateImageByArr(self, arr):
        # Convert to png data
        arr = np.array(arr.numpy()).reshape(600, 600, -1)
        bitmap = Bitmap(arr).convert(Bitmap.PixelFormat.RGB, Struct.Type.UInt8, True)
        arr = np.array(bitmap)

        qimage_var = qimage2ndarray.array2qimage(arr, normalize=False)
        pixmap = QPixmap.fromImage(qimage_var).scaled(self.screen_width,self.screen_height, Qt.KeepAspectRatio)         # Update pixmap with desired image
        self.pixmap = pixmap

        self.img_widget.setPixmap(self.pixmap)      # Show desired image on Qlabel

