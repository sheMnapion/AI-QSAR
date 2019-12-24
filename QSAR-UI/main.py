# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from PyQt5.QtCore import QCoreApplication
from os.path import expanduser
from tab0 import Tab0
from tab1 import Tab1
from tab2 import Tab2
from tab3 import Tab3

from types import MethodType
from utils import resetFolderList, getFolder, mousePressEvent


class MainWindow(QMainWindow):

    def __init__(self):
        # Load Tabs into MainWindow.tabWidget
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

        self.setWindowIcon(QIcon("molPredict.ico"))

        self.projectList.mousePressEvent = MethodType(mousePressEvent, self.projectList)
        self._currentProjectFolder = None
        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals & Add Buttons to ToolBar.
        """
        self.projectList.itemDoubleClicked.connect(self.projectDoubleClickedSlot)

        self.openAction = self.toolBar.addAction(QIcon("images/fileopen.png"), "Open Project(&O)")
        self.openAction.triggered.connect(self.projectBrowseSlot)
        self.projectBrowseBtn.released.connect(self.projectBrowseSlot)
        self.projectBrowseAction.triggered.connect(self.projectBrowseSlot)

        self.projectLineEdit.textChanged.connect(lambda folder: self.tab0.dataSetSlot(folder))
        self.projectLineEdit.textChanged.connect(lambda folder: self.tab1.dataSetSlot(folder))
        self.tab0.dataLineEdit.textChanged.connect(lambda folder: self.tab1.dataSetSlot(folder))
        self.tab1.dataLineEdit.textChanged.connect(lambda folder: self.tab0.dataSetSlot(folder))

        self.saveModelAction = self.toolBar.addAction(QIcon("images/gtk-save.png"), "Save Model(&S)")
        self.saveASModelAction = self.toolBar.addAction(QIcon("images/gtk-save-as.png"), "Save As Model")

        self.loadModelAction = self.toolBar.addAction(QIcon("images/add.png"), "Load Model(&O)")
        self.loadModelAction.triggered.connect(self.tab1.modelBrowseSlot)

        exitAction = self.actionExit_E
        exitAction.triggered.connect(QCoreApplication.instance().quit)

    def projectSetSlot(self, folder):
        """
        Slot Function of Setting Project Folder without Browsing
        """
        resetFolderList(self.projectList, folder)
        self.projectLineEdit.setText(folder)
        self._currentProjectFolder = folder

    def projectDoubleClickedSlot(self, item):
        selectedFile = os.path.join(self._currentProjectFolder, item.text())
        if os.path.isfile(selectedFile):
            self.projectSelectBtn.click()
        elif os.path.isdir(selectedFile):
            self.projectSetSlot(selectedFile)

    def projectBrowseSlot(self):
        """
        Slot Function of Opening the Project Folder
        """
        folder = getFolder()
        if folder:
            self._currentProjectFolder = folder
            self.projectLineEdit.setText(folder)
            resetFolderList(self.projectList, folder)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
