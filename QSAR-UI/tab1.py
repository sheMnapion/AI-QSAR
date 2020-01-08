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
from datetime import datetime
from sklearn.model_selection import train_test_split

from utils import resetFolderList, getFolder, getFile, saveModel, getIcon, mousePressEvent, Worker
from utils import DNN_PATH, RNN_PATH, CACHE_PATH

sys.path.append(DNN_PATH)
sys.path.append(RNN_PATH)

from QSAR_DNN import QSARDNN
from smilesRNN import SmilesRNNPredictor as QSARRNN

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
        self._currentDataFile = None
        self._currentModelFile = None

        # object name of QtWidgets -> key of self.trainingParams
        self._trainingParamsMap = {
            "batchSizeSpinBox": "batchSize",
            "epochsSpinBox": "epochs",
            "learningRateDoubleSpinBox": "learningRate",
            "earlyStopCheckBox": "earlyStop",
            "earlyStopEpochsSpinBox": "earlyStopEpochs",
            "targetTypeComboBox": "targetType",
            "columnSelectComboBox": "targetColumn",
            "fromLoadedModelCheckBox": "fromModel",
            "modelTypeComboBox": "modelType"
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
#        self.RNN = QSARRNN()
        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals
        """
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
        self.trainParamsBtn.released.connect(self.startTrainingSlot)
        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)
        self.modelSelectBtn.released.connect(self.modelSelectSlot)
        self.modelSaveBtn.released.connect(self.modelSaveSlot)
        self.dataList.itemDoubleClicked.connect(self.dataDoubleClickedSlot)
        self.modelList.itemDoubleClicked.connect(self.modelDoubleClickedSlot)

        self.fromLoadedModelCheckBox.stateChanged.connect(self._resetTrainParamsBtn)
        # Ensure Scroll to Bottom in Realtime
        self.trainingList.model().rowsInserted.connect(self.trainingList.scrollToBottom)

    def _resetTrainParamsBtn(self):
        if self.fromLoadedModelCheckBox.isChecked():
            if self._currentDataFile and self._currentModelFile \
                    and self.DNN.model.state_dict() is not None \
                    and self.numericData is not None \
                    and len(self.DNN.model.state_dict()["layer1.0.bias"]) == self.numericData.shape[1] - 1:
                self.trainParamsBtn.setEnabled(True)
                self.trainParamsBtn.repaint()
            else:
                self.trainParamsBtn.setEnabled(False)
                self.trainParamsBtn.repaint()
        else:
            if self._currentDataFile and self.numericData is not None:
                self.trainParamsBtn.setEnabled(True)
                self.trainParamsBtn.repaint()

    def _setTrainingReturnsSlot(self, result):
        """
        Slot Function of Processing Return Value of Training Thread
        """
        outputName = '{}_{}.npy'.format(self._currentDataFile.rsplit('.', 1)[0],
                                            datetime.now().strftime("%Y%m%d_%H_%M_%S"))

        if not os.path.exists(CACHE_PATH):
            os.mkdir(CACHE_PATH)

        np.save(os.path.join(CACHE_PATH, outputName), result)

        self.trainingReturnLineEdit.setText(outputName)
        self.progressBar.setValue(self.progressBar.maximum())
        self._debugPrint("{} saved to {}".format(outputName, CACHE_PATH))

    def startTrainingSlot(self):
        """
        The Training Function Given Data and Training Parameters
        """
        if not self.trainParamsBtn.isEnabled():
            return

        self._updateTrainingParams()

        modelName=self.trainingParams['ModelType']
        if modelName=='DNN':
            try:
                self.trainer = Worker(fn = self.DNN.train_and_test,
                                 train_set = self.trainData,
                                 train_label = self.trainLabel,
                                 test_set = self.testData,
                                 test_label = self.testLabel,
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
            self.trainer.sig.result.connect(lambda result: self._setTrainingReturnsSlot(result))
            self.threadPool.start(self.trainer)
            self.progressBar.setMinimum(0)
            numEpochs=int(self.trainingParams['epochs'])
            self.progressBar.setMaximum(int(numEpochs))
            self.progressBar.setValue(0)
        else:
            self._debugPrint('STILL DOING!')

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
            if self.trainingParams['fromModel']==True:
                try:
                    self.DNN.load(self.modelList.currentItem().text())
                    print('LOADED AGAIN FOR SAFETY')
                except:
                    self._debugPrint("BAD MODEL! Please check whether the model matches the input csv file.")

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
            self.modelList.repaint()

    def modelDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.modelList
        """
        if self.modelSelectBtn.isEnabled():
            selectedFile = item.text()
            if os.path.isfile(selectedFile):
                self.modelSelectBtn.click()

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
        """
        Slot Function of Selecting Model File
        """
        if not self.modelSelectBtn.isEnabled():
            return

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
                self._debugPrint("Load Model Error!")
                return
        else:
            self._debugPrint("Not a .pxl pytorch model!")
            return

        self._currentModelFile = model

        self._resetTrainParamsBtn()

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
            return

        self.trainParamsBtn.setEnabled(True)
        self.modelSelectBtn.setEnabled(True)
        self.trainParamsBtn.repaint()
        self.modelSelectBtn.repaint()
        self._currentDataFile = file

        self._resetTrainParamsBtn()

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        NOW ADD ONE MORE FUNCTION:
            if the msg is in the format of %d/%d, then interpret it as the information of progress
        """
        try:
            tempMsgs=msg.split('/')
            # print(tempMsgs)
            progress=int(tempMsgs[0])
            total=int(tempMsgs[1])
            self.progressBar.setValue(progress)
        except:
            self.trainingList.addItem(msg)
            self.trainingList.repaint()
