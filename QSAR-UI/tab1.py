# This Python file uses the following encoding: utf-8
import os
import re
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser
import numpy as np
import pandas as pd

DEFAULT_ICON = "images/stock_media-play.png"

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

#        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
#        self.projectBrowseBtn.released.connect(self._projectBrowseSlot)

#        self.openAction = self.toolBar.addAction(QIcon("images/fileopen.png"), "Open Project(&O)")
#        self.openAction.triggered.connect(self._projectBrowseSlot)

#        self.saveModelAction = self.toolBar.addAction(QIcon("images/gtk-save.png"), "Save Model(&S)")
#        self.saveASModelAction = self.toolBar.addAction(QIcon("images/gtk-save-as.png"), "Save As Model")
#        self.modelBrowseBtn.released.connect(self.modelBrowseSlot)

#        self.loadModelAction = self.toolBar.addAction(QIcon("images/add.png"), "Load Model(&O)")
#        self.loadModelAction.triggered.connect(self.modelBrowseSlot)

#        self.dataSelectBtn.released.connect(self.dataSelectSlot)

#        self.enterParamsBtn.released.connect(self.updateTrainingParamsSlot)
#        self.trainParamsBtn.released.connect(self.startTrainingSlot)

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
        file = self._getFile()
        if file:
            self._debugPrint("openning model file: " + file)
            icon = self._getIcon(os.path.join(os.getcwd(), file))
            self.modelList.addItem(QListWidgetItem(icon, file))

    def dataBrowseSlot(self):
        folder = self._getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            self.dataLabel.setText(folder)
            self._resetFolderList(self.dataList, folder)
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

    def _resetFolderList(self, List, folder):
        fileInfo = QtCore.QFileInfo(folder)
        List.clear()
        List.setUpdatesEnabled(False)
        for file in fileInfo.dir():
            if file in ['.', '..']:
                continue
            icon = self._getIcon(os.path.join(folder, file))
            List.addItem(QListWidgetItem(icon, file))
        List.setUpdatesEnabled(True)

    def _getFolder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self,
                                                 "Open Directory",
                                                 expanduser("~"),
                                                 QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        folder += '/'
        return folder

    def _getFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file = QFileDialog.getOpenFileName(self,
                                         "Open File",
                                         expanduser("~"),
                                         "All Files (*)")
        return file[0]

    def _getIcon(self, path):

        iconProvider = QFileIconProvider()
        icon = iconProvider.icon(QtCore.QFileInfo(path))
        if icon.isNull():
            return QtGui.QIcon(DEFAULT_ICON)
        else:
            return icon
