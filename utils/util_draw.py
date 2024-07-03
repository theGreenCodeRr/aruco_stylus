# --- OpenCV
import cv2
import numpy as np

# --- Pyqtgraph
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
from collections import defaultdict

# Global variable
MyKey = ''

# Get key mappings from Qt namespace
# https://stackoverflow.com/questions/40423999/pyqtgraph-where-to-find-signal-for-key-preses
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)

class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

key_code = ''
def get_key_pressed(input_key):
    # print(input_key)
    global key_code
    key_code= input_key
    return key_code


# Call below first
app=QtWidgets.QApplication([])

# 2D plot
# win2d = pg.GraphicsLayoutWidget(title="Webcam")
win2d= KeyPressWindow(show=True)
win2d.setWindowTitle("Dodecahedron")
win2d.sigKeyPress.connect(lambda event: get_key_pressed(keys_mapping[event.key()]))
win2d.resize(1500,750)
# win2d.setBackground('k')
# win2d.showFullScreen()
win2d.showNormal()
p1 = pg.PlotWidget()

# Fix widget aspect ratio
p1.setAspectLocked(True)

# 3d plot
win = gl.GLViewWidget()
win.setBackgroundColor('k')

# get a layout
layoutgb = QtWidgets.QGridLayout()
win2d.setLayout(layoutgb)
layoutgb.addWidget(win, 0, 0)
layoutgb.addWidget(p1, 0, 1)
p1.sizeHint = lambda: pg.QtCore.QSize(100, 100)
win.sizeHint = lambda: pg.QtCore.QSize(100, 100)
win.setSizePolicy(p1.sizePolicy())

# Initialize the webcam camera
img = pg.ImageItem(border='w')
p1.addItem(img)

def plot_dodecahedron(image, points, point_size):
    # plot using pyqtgraph
    # Draw grid ------------
    win.clear()
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    win.addItem(gx)
    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    win.addItem(gy)
    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    win.addItem(gz)
    # Draw the webcam image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # img_flip = cv2.flip(img_rotate_90_clockwise, 0)
    img.setImage(image)
    sp2 = gl.GLScatterPlotItem(pos=points, color=(1, 1, 1, 1), size=point_size)
    win.addItem(sp2)
    win.show()
    pg.QtWidgets.QApplication.processEvents()

global_pose = True
plot_pen_tip = False

def draw_image(image):
    # plot using pyqtgraph
    # Draw grid ------------
    win.clear()
    gx = gl.GLGridItem()
    gx.rotate(90, 0, 1, 0)
    gx.translate(-10, 0, 0)
    win.addItem(gx)
    gy = gl.GLGridItem()
    gy.rotate(90, 1, 0, 0)
    gy.translate(0, -10, 0)
    win.addItem(gy)
    gz = gl.GLGridItem()
    gz.translate(0, 0, -10)
    win.addItem(gz)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Draw the webcam image
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # img_flip = cv2.flip(img_rotate_90_clockwise, 0)
    img.setImage(image)
    # ---
    global key_code
    global global_pose
    global plot_pen_tip
    if key_code=='Space':
        if global_pose:
            global_pose = False
            key_code = ''
        else:
            global_pose = True
            key_code = ''
    if key_code=='P':
        if plot_pen_tip:
            plot_pen_tip = False
            key_code = ''
        else:
            plot_pen_tip = True
            key_code = ''

    win.show()
    pg.QtWidgets.QApplication.processEvents()
    return  global_pose, plot_pen_tip


