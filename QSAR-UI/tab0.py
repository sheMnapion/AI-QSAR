# This Python file uses the following encoding: utf-8
import os
import re
import matplotlib
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QListWidget, QFileIconProvider
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from os.path import expanduser
import numpy as np
import pandas as pd
from types import MethodType
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

from utils import resetFolderList, getFolder, getFile, getIcon, mousePressEvent, clearLayout

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
        # self.originalDataTable.mousePressEvent = MethodType(mousePressEvent, self.originalDataTable)
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
        self.dataList.itemDoubleClicked.connect(self.dataDoubleClickedSlot)     
        self.dataLineEdit.textChanged.connect(lambda folder: self.outputSetSlot(folder))

        # Ensure Scroll to Bottom in Realtime
        self.infoList.model().rowsInserted.connect(self.infoList.scrollToBottom)

    def _addmpl(self, fig):
        """
        Add matplotlib Canvas
        """
        clearLayout(self.plotLayout)

        self.canvas = FigureCanvas(fig)
        self.plotLayout.addWidget(self.canvas)
        self.canvas.draw()

        self.plotToolBar = NavigationToolbar(self.canvas, self.plotWidget, coordinates=True)
        self.plotLayout.addWidget(self.plotToolBar)

    def outputBrowseSlot(self):
        """
        Slot Function of Browsing Output Folder
        """
        folder = getFolder()
        if folder:
#            self._debugPrint("setting data folder: " + folder)
            self.outputLineEdit.setText(folder)
            self._currentOutputFolder = folder

    def outputSetSlot(self, folder):
        """
        Slot Function of Setting Output Folder
        """
#        self._debugPrint("setting data folder: " + folder)
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

        if os.path.exists(self._currentOutputFolder) is not True:
            self._debugPrint("Invalid Save Folder")
            return

        selectedFile = os.path.join(self._currentOutputFolder, self._currentDataFile)
        selectedFileNoSuffix = selectedFile.rsplit('.', 1)[0]

        outputFile = '{}_transformed.csv'.format(selectedFileNoSuffix)       

        self.transformedData.to_csv(outputFile, index = None)

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

        self._currentDataFile = file
        selectedFile = os.path.join(self._currentDataFolder, file)  

        self._debugPrint(selectedFile)

        if re.match(".+.csv$", file):
            self.originalData = pd.read_csv(selectedFile, index_col = False,
                                                header = (0 if (self.headerCheckBox.isChecked()) else None))
            # self.originalDatatList.addItem(str(self.originalData.head()))
            npOriginalData=np.array(self.originalData)[:100] # show only top 100 terms
            w,h=npOriginalData.shape[:2]
            self.originalDataTable.setRowCount(w)
            self.originalDataTable.setColumnCount(h)
            self.originalDataTable.setHorizontalHeaderLabels(self.originalData.columns)
            for i in range(w):
                for j in range(h):
                    tempItem=QTableWidgetItem()
                    tempItem.setText(str(npOriginalData[i][j]))
                    self.originalDataTable.setItem(i,j,tempItem)
            self._debugPrint("csv file {} loaded: {shape[0]} lines, {shape[1]} columns".format(
                                file, shape=self.originalData.shape))
        else:
            self._debugPrint("Not a csv file")

        self.outputSaveBtn.setEnabled(True)

    def columnSelectSlot(self):
        """
        Slot Function of Selecting Data Column for Display
        """
        column = self.columnSelectComboBox.currentText()
        try:
            selectedColumn = self.transformedData[column]
        except:
            self._debugPrint("Column Selection Error. Column not Found!")
            return

        self.transformedDataList.addItem(str(selectedColumn.describe()))
        self._debugPrint("Column {} selected for display".format(column))

        self._updatePlot(selectedColumn)

    def _processData(self):
        """
        Process and Transform CSV Data
        """
        if self.originalData is None or self.originalData.columns is None:
            raise Exception("Original data is invalid")

        outLierPolicy = self.outlierOperationComboBox.currentText()
        self.transformedData = {
            'Delete Data': self.originalData.dropna(),
            'Zero Padding': self.originalData.fillna(0.0),
            'Average Padding': self.originalData.fillna(self.originalData.mean()),
            'Keep Data': self.originalData
        }.get(outLierPolicy, None)

        if self.transformedData is None or self.transformedData.columns is None:
            raise Exception("Transformed data is invalid")

        self.columnSelectComboBox.clear()
        self.columnSelectComboBox.addItems(self.transformedData.columns)

    def _updatePlot(self, selectedColumn):
        name = selectedColumn.name
        if len(name) >= 50:
            name = name[:50] + '...'

        if np.issubdtype(selectedColumn.dtype, np.number):
            fig = Figure()

            ax1f1 = fig.add_subplot(111)
            selectedColumn.plot.kde(ax=ax1f1, legend=False, title='Histogram')
            selectedColumn.plot.hist(density=True, ax=ax1f1, color = '#0504aa', alpha = 0.7, rwidth = 0.85)
            ax1f1.set_title(name)
        else:
            fig = Figure()
            ax1f1 = fig.add_subplot(111)
            selectedColumn.value_counts().plot(ax=ax1f1, kind='pie', startangle=90)
            ax1f1.set_title(name)
            ax1f1.set_ylabel('')
            ax1f1.set_xlabel('')

        self._addmpl(fig)

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
