# This Python file uses the following encoding: utf-8
import os
import re
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QListWidget, QFileIconProvider
from os.path import expanduser
import numpy as np
import pandas as pd
from types import MethodType

from utils import resetFolderList, getFolder, getFile, getIcon, mousePressEvent

class Tab0(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow0.ui", self)

        self._currentDataFolder = None
        self._currentOutputFolder = None
        self._currentDataFile = None
        self.originalData = None
        self.transformedData = None
        self.header = False

        # Allowing deselection in a QListWidget by clicking off an item
        self.originalDatatList.mousePressEvent = MethodType(mousePressEvent, self.originalDatatList)
        self.transformedDataList.mousePressEvent = MethodType(mousePressEvent, self.transformedDataList)
        self.dataList.mousePressEvent = MethodType(mousePressEvent, self.dataList)
        self.infoList.mousePressEvent = MethodType(mousePressEvent, self.infoList)

        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals
        """
        self.dataSelectBtn.released.connect(self.dataSelectSlot)
        self.dataBrowseBtn.released.connect(self.dataBrowseSlot)
        self.outputBrowseBtn.released.connect(self.outputBrowseSlot)
        self.outputSaveBtn.released.connect(self.outputSaveSlot)
        self.columnSelectBtn.released.connect(self.columnSelectSlot)

    def outputBrowseSlot(self):
        """
        Slot Function of Browsing Output Folder
        """
        folder = getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            self.outputLineEdit.setText(folder)
            self._currentOutputFolder = folder

    def outputSaveSlot(self):
        """
        Slot Function of Saving Output CSV
        """
        try:
            self._processData()
        except:
            self._debugPrint("Data Processing Throw Error!")
            return

        self.transformedDataList.addItem(str(self.transformedData.head()))

        selectedFile = os.path.join(self._currentOutputFolder, self._currentDataFile)
        selectedFileNoSuffix = selectedFile.rsplit('.', 1)[0]

        outputFile = '{}_transformed.csv'.format(selectedFileNoSuffix)
        self.transformedData.to_csv(outputFile)

        self._debugPrint("csv file {} saved: {shape[0]} lines, {shape[1]} columns".format(
                            outputFile, shape=self.transformedData.shape))

    def dataBrowseSlot(self):
        """
        Slot Function of Opening Data Folder
        """
        folder = getFolder()
        if folder:
            self._debugPrint("setting data folder: " + folder)
            resetFolderList(self.dataList, folder)
            self._currentDataFolder = folder

    def dataSelectSlot(self):
        """
        Slot Function of Selecting Data File in the Opened Folder
        """
        try:
            file = self.dataList.currentItem().text()
        except:
            self._debugPrint("Current Data File Not Found")
            return

        self._currentDataFile = file
        selectedFile = os.path.join(self._currentDataFolder, file)  

        self._debugPrint(selectedFile)

        if re.match(".+.csv$", file):
            self.originalData = pd.read_csv(selectedFile, header = (0 if (self.headerCheckBox.isChecked()) else None))
            self.originalDatatList.addItem(str(self.originalData.head()))
            self._debugPrint("csv file {} loaded: {shape[0]} lines, {shape[1]} columns".format(
                                file, shape=self.originalData.shape))
        else:
            self._debugPrint("Not a csv file")

    def columnSelectSlot(self):
        """
        Slot Function of Selecting Data Column for Display
        """
        column = self.columnSelectComboBox.currentText()
        self._debugPrint("Column {} selected for display".format(column))
        pass

    def _processData(self):
        """
        Process and Transform CSV Data
        """
        if self.originalData is None:
            return
        outLierPolicy = self.outlierOperationComboBox.currentText()
        self.transformedData = {
            'Delete Data': self.originalData.dropna(),
            'Zero Padding': self.originalData.fillna(0.0),
            'Average Padding': self.originalData.fillna(self.originalData.mean()),
            'Keep Data': self.originalData
        }.get(outLierPolicy, None)

        if self.transformedData.columns is not None:
            self.columnSelectComboBox.clear()
            self.columnSelectComboBox.addItems(self.transformedData.columns)

    def _updatePlot(self):
        pass

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
