# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser
from tab1 import Tab1
from tab2 import Tab2
from tab3 import Tab3

class MainWindow(QTabWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tab1 = Tab1()
        self.tab2 = Tab2()
        self.tab3 = Tab3()

        self.addTab(self.tab1, "TAB1")
        self.addTab(self.tab2, "TAB2")
        self.addTab(self.tab3, "TAB3")

        self.resize(self.tab1.size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
