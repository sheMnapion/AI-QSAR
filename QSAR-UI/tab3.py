# This Python file uses the following encoding: utf-8
import re
import os
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QListWidget

from types import MethodType

from utils import resetFolderList, getFolder, getFile, getIcon, saveModel, mousePressEvent
from utils import DNN_PATH, CACHE_PATH

sys.path.append(DNN_PATH)
from QSAR_DNN import QSARDNN


class Tab3(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow3.ui", self)

        self.dataList.mousePressEvent = MethodType(mousePressEvent, self.dataList)
        self.modelList.mousePressEvent = MethodType(mousePressEvent, self.modelList)

        self.data = None

        # Currently Opened Folder
        self._currentDataFolder = None
        self._currentDataFile = None

        self.DNN = QSARDNN()

        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals
        """
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)
        self.modelSelectBtn.released.connect(self.modelSelectSlot)
        self.dataList.itemDoubleClicked.connect(self.dataDoubleClickedSlot)
        self.modelList.itemDoubleClicked.connect(self.modelDoubleClickedSlot)

        # Ensure Scroll to Bottom in Realtime
        self.infoList.model().rowsInserted.connect(self.infoList.scrollToBottom)

    def modelBrowseSlot(self):
        """
        Slot Function of Loading the Model File
        """
        file = getFile()
        if file:
            self._debugPrint("openning model file: " + file)
            icon = getIcon(os.path.join(os.getcwd(), file))
            self.modelList.addItem(QListWidgetItem(icon, file))
            self.modelList.repaint()

    def modelDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.modelList
        """
        if self.modelSelectBtn.isEnabled():
            selectedFile = item.text()
            if os.path.isfile(selectedFile):
                self.modelSelectBtn.click()

    def modelSelectSlot(self):
        """
        Slot Function of Selecting Model File
        """
        try:
            model = self.modelList.currentItem().text()
        except:
            self._debugPrint("Current Model File Not Found")
            return

        if re.match(".+.pxl$", model):
            try:
                # If not set, loading will fail without a correct propertyNum
                self.DNN.setPropertyNum(self.numericData.shape[1] - 1)
                self.DNN.load(model)
                self._debugPrint("Model Loaded: {}".format(model))
            except:
                self._debugPrint("Load Model {} Error!".format(model))
                return
        else:
            self._debugPrint("Not a .pxl pytorch model!")

    def dataBrowseSlot(self):
        """
        Slot Function of Opening Data Folder
        """
        folder = getFolder()
        if folder:
            resetFolderList(self.dataList, folder)
            self.dataLineEdit.setText(folder)
            self._currentDataFolder = folder

    def dataSetSlot(self, folder):
        """
        Slot Function of Setting Data Folder without Browsing
        """
        resetFolderList(self.dataList, folder)
        self.dataLineEdit.setText(folder)
        self._currentDataFolder = folder

    def dataDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.dataList
        """
        selectedFile = os.path.join(self._currentDataFolder, item.text())
        if os.path.isfile(selectedFile):
            self.dataSelectBtn.click()
        elif os.path.isdir(selectedFile):
            self.dataSetSlot(selectedFile)

    def dataSelectSlot(self):
        """
        Slot Function of Selecting Data File in the Opened Folder
        """
        try:
            file = self.dataList.currentItem().text()
        except:
            self._debugPrint("Current Data File Not Found")
            return

        selectedFile = os.path.join(self._currentDataFolder, file)

        if re.match(".+.csv$", file):
            try:
                self.data = pd.read_csv(selectedFile, index_col = False,
                                            header = (0 if (self.headerCheckBox.isChecked()) else None))
                self.numericData = self.data.select_dtypes(include = np.number)
            except:
                self.data = None
                self.numericData = None
                self._debugPrint("Load Data Error!")
                return

            self._debugPrint("csv file {} loaded".format(file))
            self._debugPrint(str(self.data.head()))
        else:
            self._debugPrint("Not a csv file!")

        self.modelSelectBtn.setEnabled(True)
        self.modelSelectBtn.repaint()
        self._currentDataFile = file

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
        self.infoList.repaint()
