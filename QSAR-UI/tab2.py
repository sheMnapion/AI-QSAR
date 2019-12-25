# This Python file uses the following encoding: utf-8
import os
import re
import numpy as np
import pandas as pd
from types import MethodType

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow

from utils import resetFolderList, mousePressEvent
from utils import DNN_PATH, CACHE_PATH

class Tab2(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow2.ui", self)

        self.result = None
        self._currentTrainingHistoryFile = None

        self.trainingHistoryList.mousePressEvent = MethodType(mousePressEvent, self.trainingHistoryList)
        self.refreshTrainingList()

        self._bind()

    def _bind(self):
        self.trainingHistorySelectBtn.released.connect(self.trainingHistorySelectSlot)
        self.trainingHistoryList.itemDoubleClicked.connect(self.trainingHistoryDoubleClickedSlot)


    def refreshTrainingList(self):
        """
        Refresh QListWidget of training history after each training process
        """
        if not os.path.exists(CACHE_PATH):
            os.mkdir(CACHE_PATH)

        resetFolderList(self.trainingHistoryList, CACHE_PATH)

    def trainingHistorySelectSlot(self):
        """
        Slot Function of Selecting training history in the training history list
        """
        try:
            file = self.trainingHistoryList.currentItem().text()
        except:
            self._debugPrint("Current Training History Not Found")
            return

        selectedFile = os.path.join(CACHE_PATH, file)

        if re.match(".+.npy$", file):
            try:
                self.result = np.load(selectedFile, allow_pickle = 'TRUE').item()
            except:
                self._debugPrint("Load Training History Error!")
                return

            self._debugPrint("Training History {} loaded".format(selectedFile))
            self._debugPrint("(n, len(loss), len(mse), len(pred)) = ( {}, {}, {}, {} )".format(
                        self.result["numEpochs"],
                        len(self.result["lossList"]),
                        len(self.result["mseList"]),
                        len(self.result["testPred"])))

        else:
            self._debugPrint("Not a Training History(*.npy)!")
            return

        self._currentTrainingHistoryFile = file
        self.analyzeBtn.setEnabled(True)

    def trainingHistoryDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.modelList
        """
        selectedFile = os.path.join(CACHE_PATH, item.text())
        if os.path.isfile(selectedFile):
            self.trainingHistorySelectBtn.click()

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.resultList.addItem(msg)
