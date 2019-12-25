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
#        self.dataSelectBtn.released.connect(self.dataSelectSlot)
#        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
#        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)
#        self.modelSelectBtn.released.connect(self.modelSelectSlot)
#        self.dataList.itemDoubleClicked.connect(self.dataDoubleClickedSlot)
#        self.modelList.itemDoubleClicked.connect(self.modelDoubleClickedSlot)

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
        self.infoList.repaint()
