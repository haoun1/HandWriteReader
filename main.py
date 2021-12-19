import mnist_train
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# 순서정리
# 1. 트레인셋을 작성한다 (x_train,y_train)
# 2. 모델의 형태(layer)를 설정한다.
# 3. 모델의 loss함수, optimizer를 설정한다 (컴파일작업) 
# 4. model.fit를 통해서 트레이닝을 시작한다. verbose는 상세 로그 출력여부를 결정한다. validation_split은 비율만큼을 testset으로 설정하며 이 set에 대해서는 가중치를 변경하지 않는다. loss는 val_loss로 따로 저장한다.#

class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(400,400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 5
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.initUI()
    
    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        save_action = QAction('save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)

        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.setWindowTitle('Simple Painter')
        self.setGeometry(300,300,400,400)
        self.show()
    
    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if(e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False
    
    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(self, ' Save Image', '', "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")


        if fpath:
            self.image.save(fpath)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

def main():
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()