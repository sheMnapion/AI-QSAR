# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser
from tab0 import Tab0
from tab1 import Tab1
from tab2 import Tab2
from tab3 import Tab3

class MainWindow(QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow.ui", self)

        self.tab0 = Tab0()
        self.tab1 = Tab1()
        self.tab2 = Tab2()
        self.tab3 = Tab3()

        self.tabWidget.addTab(self.tab0, "Data Processing")
        self.tabWidget.addTab(self.tab1, "Model Training")
        self.tabWidget.addTab(self.tab2, "TAB2")
        self.tabWidget.addTab(self.tab3, "TAB3")

        self._bind()

    def _bind(self):

        self.openAction = self.toolBar.addAction(QIcon("images/fileopen.png"), "Open Project(&O)")
        self.openAction.triggered.connect(self.projectBrowseSlot)
        self.projectBrowseBtn.released.connect(self.projectBrowseSlot)

        self.saveModelAction = self.toolBar.addAction(QIcon("images/gtk-save.png"), "Save Model(&S)")
        self.saveASModelAction = self.toolBar.addAction(QIcon("images/gtk-save-as.png"), "Save As Model")

        self.loadModelAction = self.toolBar.addAction(QIcon("images/add.png"), "Load Model(&O)")
        self.loadModelAction.triggered.connect(self.tab1.modelBrowseSlot)

    def projectBrowseSlot(self):
        folder = self.tab1._getFolder()
        if folder:
            self.tab1._debugPrint("setting project folder: " + folder)
            self.projectLabel.setText(folder)
            self._currentProjectFolder = folder
            self.tab1._resetFolderList(self.projectList, folder)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
