# This Python file uses the following encoding: utf-8
import os
import re
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser
import numpy as np
import pandas as pd

from utils import resetFolderList, getFolder, getFile, getIcon


class Tab1(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow1.ui", self)

        # Currently Loaded Data (pandas.DataFrame)
        self.data = None
        # Training Params
        self.trainingParams = {}

        # Currently Opened Folder
        self._currentDataFolder = None
        self._currentProjectFolder = None

        # object name of QtWidgets -> key of self.trainingParams
        self._trainingParamsMap = {
            "batchSizeSpinBox": "batchSize",
            "dropoutDoubleSpinBox": "dropout",
            "epochsSpinBox": "epochs",
            "learningRateDoubleSpinBox": "learningRate",
            "biasInitializerComboBox": "biasInitializer",
            "weightInitializerComboBox": "weightInitializer"
        }

        self._bind()

    def _bind(self):
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.enterParamsBtn.released.connect(self.updateTrainingParamsSlot)
        self.trainParamsBtn.released.connect(self.startTrainingSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)

    # Modify Training Methods Here
    def startTrainingSlot(self):
        self._debugPrint("Start Training")
        pass

    def updateTrainingParamsSlot(self):
        for (objName, key) in self._trainingParamsMap.items():
            obj = getattr(self, objName)
            if getattr(obj, "value", None):
                self.trainingParams[key] = obj.value()
            if getattr(obj, "currentText", None):
                self.trainingParams[key] = obj.currentText()

        self._debugPrint(str(self.trainingParams.items()))

    def modelBrowseSlot(self):
        file = getFile()
        if file:
            self._debugPrint("openning model file: " + file)
            icon = getIcon(os.path.join(os.getcwd(), file))
            self.modelList.addItem(QListWidgetItem(icon, file))

    def dataBrowseSlot(self):
        folder = getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            self.dataLabel.setText(folder)
            resetFolderList(self.dataList, folder)
            self._currentDataFolder = folder

    def dataSelectSlot(self):
        try:
            file = self.dataList.currentItem().text()
        except:
            self._debugPrint("Current Data File Not Found")
            return

        selectedFile = os.path.join(self._currentDataFolder, file)

        self._debugPrint(selectedFile)

        if re.match(".+.csv$", file):
            self.data = pd.read_csv(selectedFile)
            self._debugPrint("csv file {} loaded".format(file))
            self._debugPrint(str(self.data.head()))
        else:
            self._debugPrint("Not a csv file")

    def _debugPrint(self, msg):
        self.trainingList.addItem(msg)
