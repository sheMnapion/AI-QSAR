# This Python file uses the following encoding: utf-8
import os
import re
import sys
import torch

from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from PyQt5.QtCore import QThreadPool, QRunnable
from os.path import expanduser
from types import MethodType

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import resetFolderList, getFolder, getFile, saveModel, getIcon, mousePressEvent, Worker

DNN_PATH = os.path.abspath('../QSAR-DNN')
sys.path.append(DNN_PATH)
from QSAR_DNN import QSARDNN

class Tab1(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow1.ui", self)

        # Currently Loaded Data (pandas.DataFrame)
        self.data = None
        self.numericData = None
        # Training Params
        self.trainingParams = {}

        # Currently Opened Folder
        self._currentDataFolder = None
        self._currentProjectFolder = None
        self._currentOutputPath = None

        # object name of QtWidgets -> key of self.trainingParams
        self._trainingParamsMap = {
            "batchSizeSpinBox": "batchSize",
            "epochsSpinBox": "epochs",
            "learningRateDoubleSpinBox": "learningRate",
            "earlyStopCheckBox": "earlyStop",
            "earlyStopEpochsSpinBox": "earlyStopEpochs",
            "targetTypeComboBox": "targetType",
            "columnSelectComboBox": "targetColumn",
            "fromLoadedModelCheckBox": "fromModel"
        }

        self.dataList.mousePressEvent = MethodType(mousePressEvent, self.dataList)
        self.modelList.mousePressEvent = MethodType(mousePressEvent, self.modelList)
        self.trainingList.mousePressEvent = MethodType(mousePressEvent, self.trainingList)

        self.threadPool = QThreadPool()
        self.trainer = None
        self.trainData = None
        self.trainLabel = None
        self.testData = None
        self.testLabel = None
        self.DNN = QSARDNN()
        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals
        """
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
#        self.enterParamsBtn.released.connect(self.updateTrainingParamsSlot)
        self.trainParamsBtn.released.connect(self.startTrainingSlot)
        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)
        self.modelSelectBtn.released.connect(self.modelSelectSlot)
        self.modelSaveBtn.released.connect(self.modelSaveSlot)
        self.dataList.itemDoubleClicked.connect(self.dataDoubleClickedSlot)
        self.modelList.itemDoubleClicked.connect(self.modelDoubleClickedSlot)

        # Ensure Scroll to Bottom in Realtime
        self.trainingList.model().rowsInserted.connect(self.trainingList.scrollToBottom)

    def startTrainingSlot(self):
        """
        The Training Function Given Data and Training Parameters
        """

        self._updateTrainingParams()

        try:
            self.trainer = Worker(fn = self.DNN.train,
                             train_set = self.trainData,
                             train_label = self.trainLabel,
                             batch_size = int(self.trainingParams["batchSize"]),
                             learning_rate = float(self.trainingParams["learningRate"]),
                             num_epoches = int(self.trainingParams["epochs"]),
                             early_stop = bool(self.trainingParams["earlyStop"]),
                             max_tolerance = int(self.trainingParams["earlyStopEpochs"])
                       )       
        except:
            self._debugPrint("DNN Fail to Start")
            return

        self.trainer.sig.progress.connect(self._appendDebugInfoSlot)
        self.threadPool.start(self.trainer)

    def _appendDebugInfoSlot(self, info):
        self._debugPrint(info)

    def _updateTrainingParams(self):
        """
        Slot Function of Updating Training Parameters
        """

        for (objName, key) in self._trainingParamsMap.items():
            obj = getattr(self, objName)
            if getattr(obj, "value", None):
                self.trainingParams[key] = obj.value()
            if getattr(obj, "checkState", None):
                self.trainingParams[key] = bool(obj.checkState())
            if getattr(obj, "currentText", None):
                self.trainingParams[key] = obj.currentText()

        # If not from model, then restart the DNN instance.
        if self.trainingParams["fromModel"] is False:
            self._debugPrint("Remove loaded model(if any)")
            self.DNN = QSARDNN()

        self._debugPrint(str(self.trainingParams.items()))

        try:
            self.trainData, self.testData = train_test_split(self.numericData,
                    test_size = 0.2, shuffle = False)
            labelColumn = self.trainingParams["targetColumn"]

            self.trainLabel = self.trainData[labelColumn].values.reshape(-1,1)
            self.trainData = self.trainData.loc[:, self.trainData.columns != labelColumn].values

            self.testLabel = self.testData[labelColumn].values.reshape(-1,1)
            self.testData = self.testData.loc[:, self.testData.columns != labelColumn].values

            self.DNN.setPropertyNum(self.trainData.shape[1])

            self._debugPrint("DNN's Set up")
        except:
            self.DNN = self.trainData = self.testData = self.trainLabel = self.testLabel = None
            self._debugPrint("Fail to Set up DNN")

    def modelBrowseSlot(self):
        """
        Slot Function of Loading the Model File
        """
        file = getFile()
        if file:
            self._debugPrint("openning model file: " + file)
            icon = getIcon(os.path.join(os.getcwd(), file))
            self.modelList.addItem(QListWidgetItem(icon, file))

    def dataBrowseSlot(self):
        """
        Slot Function of Opening Data Folder
        """
        folder = getFolder()
        if folder:
#            self._debugPrint("setting data folder: " + folder)
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

    def modelDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.modelList
        """
        if self.modelSelectBtn.isEnabled():
            selectedFile = os.path.join(self._currentDataFolder, item.text())
            if os.path.isfile(selectedFile):
                self.modelSelectBtn.click()

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

        self._debugPrint(selectedFile)

        if re.match(".+.csv$", file):
            try:
                self.data = pd.read_csv(selectedFile, index_col = False,
                                            header = (0 if (self.headerCheckBox.isChecked()) else None))
                self.numericData = self.data.select_dtypes(include = np.number)
                self.columnSelectComboBox.clear()
                self.columnSelectComboBox.addItems(self.numericData.columns)
            except:
                self.data = None
                self.numericData = None
                self._debugPrint("Load Data Error!")
                return

            self._debugPrint("csv file {} loaded".format(file))
            self._debugPrint(str(self.data.head()))
        else:
            self._debugPrint("Not a csv file!")

        self.trainParamsBtn.setEnabled(True)
        self.modelSelectBtn.setEnabled(True)

    def modelSaveSlot(self):
        """
        Slot Function of Saving Model
        """
        path = saveModel()
        if path is not None:
            self._currentOutputPath = path
            try:
                self.DNN.save(path)
                self._debugPrint('File {} saved'.format(path))
            except:
                self._debugPrint('DNN not Available yet, or Path Invalid!')

    def modelSelectSlot(self):
        try:
            model = self.modelList.currentItem().text()
        except:
            self._debugPrint("Current Model File Not Found")
            return

        self._debugPrint(model)
        if re.match(".+.pxl$", model):
            try:
                # If not set, loading will fail without a correct propertyNum
                self.DNN.setPropertyNum(self.numericData.shape[1] - 1)
                self.DNN.load(model)
                self._debugPrint("Model Loaded")
            except:
                self._debugPrint("Load Model Error!")
                return
        else:
            self._debugPrint("Not a .pxl pytorch model!")

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.trainingList.addItem(msg)
