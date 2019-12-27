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
        self.tabWidget.addTab(self.tab2, "Result Analysis")
        self.tabWidget.addTab(self.tab3, "Activity Prediction")

        self.setWindowIcon(QIcon("molPredict.ico"))

        self.projectList.mousePressEvent = MethodType(mousePressEvent, self.projectList)
        self._currentProjectFolder = None
        self._bind()

        self.showMaximized()

    def _bind(self):
        """
        Bind Slots and Signals & Add Buttons to ToolBar.
        """
        self.projectList.itemDoubleClicked.connect(self.projectDoubleClickedSlot)
        self.projectBrowseBtn.released.connect(self.projectBrowseSlot)

        self.projectLineEdit.textChanged.connect(lambda folder: self.tab0.dataSetSlot(folder))
        self.projectLineEdit.textChanged.connect(lambda folder: self.tab1.dataSetSlot(folder))
        self.projectLineEdit.textChanged.connect(lambda folder: self.tab3.dataSetSlot(folder))

        self.tab0.dataLineEdit.textChanged.connect(lambda folder: self.tab1.dataSetSlot(folder))
        self.tab0.dataLineEdit.textChanged.connect(lambda folder: self.tab3.dataSetSlot(folder))
        self.tab0.syncBtn.clicked.connect(lambda: resetFolderList(self.tab0.dataList, self.tab0._currentDataFolder))
        self.tab0.syncBtn.clicked.connect(lambda: resetFolderList(self.tab1.dataList, self.tab1._currentDataFolder))
        self.tab0.syncBtn.clicked.connect(lambda: resetFolderList(self.tab3.dataList, self.tab3._currentDataFolder))

        self.tab1.dataLineEdit.textChanged.connect(lambda folder: self.tab0.dataSetSlot(folder))
        self.tab1.dataLineEdit.textChanged.connect(lambda folder: self.tab3.dataSetSlot(folder))

        self.tab3.dataLineEdit.textChanged.connect(lambda folder: self.tab0.dataSetSlot(folder))
        self.tab3.dataLineEdit.textChanged.connect(lambda folder: self.tab1.dataSetSlot(folder))

        self.tab1.trainingReturnLineEdit.textChanged.connect(self.tab2.refreshTrainingList)

        self.actionOpen_ProjectFolder_P.triggered.connect(self.projectBrowseSlot)
        self.actionOpen_DataFolder_O.triggered.connect(self.tab1.dataBrowseSlot)
        self.actionLoad_Model.triggered.connect(lambda: self.commonSlot('modelBrowseSlot'))
        self.actionSelect_Data_D.triggered.connect(lambda: self.commonSlot('dataSelectSlot'))
        self.actionSelect_Model.triggered.connect(lambda: self.commonSlot('modelSelectSlot'))
        self.actionSave_Model_S.triggered.connect(lambda: self.commonSlot('modelSaveSlot'))
        self.actionAnalyze.triggered.connect(lambda: self.commonSlot('analyzeSlot'))

        self.actionExit_E.triggered.connect(QCoreApplication.instance().quit)

    def projectSetSlot(self, folder):
        """
        Slot Function of Setting Project Folder without Browsing
        """
        resetFolderList(self.projectList, folder)
        self.projectLineEdit.setText(folder)
        self._currentProjectFolder = folder

    def projectDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.projectList
        """
        selectedFile = os.path.join(self._currentProjectFolder, item.text())
        if os.path.isdir(selectedFile):
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

    def commonSlot(self, funcName):
        if getattr(self.tabWidget.currentWidget(), funcName, None) is not None:
            method = getattr(self.tabWidget.currentWidget(), funcName)
            method()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
