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

class Tab0(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow0.ui", self)

        self._currentDataFolder = None
        self._currentOutputFolder = None
        self._currentDataFile = None
        self.originalData = None
        self.transformedData = None
        self._bind()

    def _bind(self):
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
        self.outputBrowseBtn.released.connect(self.outputBrowseSlot)
        self.outputSaveBtn.released.connect(self.outputSaveSlot)

    def outputBrowseSlot(self):
        folder = getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            self.outputLineEdit.setText(folder)
            self._currentOutputFolder = folder

    def processData(self):
        if self.originalData is None:
            return
        outLierPolicy = self.outlierOperationComboBox.currentText()
        self.transformedData = {
            'Delete Data': self.originalData.dropna(),
            'Zero Padding': self.originalData.fillna(0.0),
            'Average Padding': self.originalData.fillna(self.originalData.mean()),
            'Keep Data': self.originalData
        }.get(outLierPolicy, None)

    def outputSaveSlot(self):
        self.processData()
        self.transformedDataList.addItem(str(self.transformedData.head()))

        selectedFile = os.path.join(self._currentOutputFolder, self._currentDataFile)
        selectedFileNoSuffix = selectedFile.rsplit('.', 1)[0]

        outputFile = '{}_transformed.csv'.format(selectedFileNoSuffix)
        self.transformedData.to_csv(outputFile)

        self._debugPrint("csv file {} saved: {shape[0]} lines, {shape[1]} columns".format(
                            outputFile, shape=self.transformedData.shape))

    def dataBrowseSlot(self):
        folder = getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            resetFolderList(self.dataList, folder)
            self._currentDataFolder = folder

    def dataSelectSlot(self):
        try:
            file = self.dataList.currentItem().text()
        except:
            self._debugPrint("Current Data File Not Found")
            return

        self._currentDataFile = file
        selectedFile = os.path.join(self._currentDataFolder, file)  

        self._debugPrint(selectedFile)

        if re.match(".+.csv$", file):
            self.originalData = pd.read_csv(selectedFile)
            self.originalDatatList.addItem(str(self.originalData.head()))
            self._debugPrint("csv file {} loaded: {shape[0]} lines, {shape[1]} columns".format(
                                file, shape=self.originalData.shape))
        else:
            self._debugPrint("Not a csv file")

    def _debugPrint(self, msg):
        self.infoList.addItem(msg)
