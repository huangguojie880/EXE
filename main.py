from PyQt5.QtWidgets import *
import fileinput
import cv2
import os
import numpy as np
from keras.layers import Conv2D,Activation,Add, Input,Layer
from keras.models import Model
import tensorflow as tf

lines = ''
for line in fileinput.input('process.py'):
    lines += line
exec(lines)

class Img_Proc_Gui(QWidget):
    def __init__(self, parent=None):
        super(Img_Proc_Gui, self).__init__(parent)

        self.process_result = None
        self.mm = my_model()
        print('正在加载模型......')
        try:
            self.mm.load_weights('./my_model.h5')
            print('模型加载成功！')
        except Exception as e:
            print('模型加载失败！')
            print(e)

        hbox_address1 = QHBoxLayout()
        self.address1 = QLineEdit()
        hbox_address1.addWidget(self.address1)
        btn_img_explorer1 = QPushButton('原始图片')
        hbox_address1.addWidget(btn_img_explorer1)
        btn_img_explorer1.clicked.connect(self.open1)

        hbox_address2 = QHBoxLayout()
        self.address2 = QLineEdit()
        hbox_address2.addWidget(self.address2)
        btn_img_explorer2 = QPushButton('保存路径')
        hbox_address2.addWidget(btn_img_explorer2)
        btn_img_explorer2.clicked.connect(self.open2)


        btn_process_img3 = QPushButton("处理图片")
        btn_process_img3.clicked.connect(self.getInput)
        btn_process_img4 = QPushButton("保存结果")
        btn_process_img4.clicked.connect(self.save)
        btn_quit = QPushButton("退出程序")
        btn_quit.clicked.connect(self.quit_clicked)
        hbox_btn = QHBoxLayout()
        hbox_btn.addWidget(btn_process_img3)
        hbox_btn.addWidget(btn_process_img4)
        hbox_btn.addWidget(btn_quit)


        vbox = QVBoxLayout()
        vbox.addLayout(hbox_address1)
        vbox.addLayout(hbox_address2)
        vbox.addLayout(hbox_btn)

        self.setGeometry(400,300,400,200)
        self.setWindowTitle('AITool')
        self.setLayout(vbox)

    #@pyqtSlot()
    def quit_clicked(self):
        cv2.destroyAllWindows()
        self.close()

    def open1(self):
        fileName = QFileDialog.getOpenFileName(self,'openFile')
        self.address1.setText(fileName[0])
        self.img = cv2.imread(self.address1.text())
        cv2.imshow('Original Image', self.img)
        cv2.waitKey()

    def open2(self):
        fileName = QFileDialog.getExistingDirectory()
        self.address2.setText(fileName)

    def save(self):
        name = self.address1.text().split('/')[-1].split('.')[0]
        my_save(self.address2.text(), name, self.process_result)

    def getInput(self):
        self.process_result = my_imgProcess(self.img, self.mm)
        my_show(self.process_result)


if __name__ == '__main__':
    import sys

    try:
        app = QApplication(sys.argv)
        screen = Img_Proc_Gui()
        screen.show()
        sys.exit(app.exec_())
    except:
        pass

