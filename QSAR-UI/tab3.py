# This Python file uses the following encoding: utf-8
import re
import os
import sys
import pandas as pd
import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QListWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from types import MethodType

from utils import resetFolderList, getFolder, getFile, getIcon, saveModel, mousePressEvent, clearLayout
from utils import DNN_PATH, CACHE_PATH

from rdkit import Chem
from rdkit.Chem import Draw
from PIL.ImageQt import ImageQt

sys.path.append(DNN_PATH)
from QSAR_DNN import QSARDNN


class Tab3(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow3.ui", self)

        self.dataList.mousePressEvent = MethodType(mousePressEvent, self.dataList)
        self.modelList.mousePressEvent = MethodType(mousePressEvent, self.modelList)

        # Original test data and test data with only numeric columns
        self.data = None
        self.numericData = None

        # Splitted test data and label
        self.testData = None
        self.testLabel = None
        self.testPred = None

        # Currently Opened Folder
        self._currentDataFolder = None
        self._currentDataFile = None
        self._currentModelFile = None

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

        self.analyzeBtn.released.connect(self.analyzeSlot)

        # Ensure Scroll to Bottom in Realtime
        self.infoList.model().rowsInserted.connect(self.infoList.scrollToBottom)

    def _addmpl(self, layout, fig):
        """
        Add matplotlib Canvas
        """
        clearLayout(layout)

        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.canvas.draw()

    def _resetAnalyzeBtn(self):
        if self._currentDataFile and self._currentModelFile \
                and self.DNN.model.state_dict() is not None \
                and self.numericData is not None \
                and len(self.DNN.model.state_dict()["layer1.0.bias"]) == self.numericData.shape[1] - 1:
            self.analyzeBtn.setEnabled(True)
            self.analyzeBtn.repaint()
        else:
            self.analyzeBtn.setEnabled(False)
            self.analyzeBtn.repaint()

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

        self._resetAnalyzeBtn()

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

        self.modelSelectBtn.setEnabled(True)
        self.modelSelectBtn.repaint()
        self._currentDataFile = file

        self._resetAnalyzeBtn()

    def analyzeSlot(self):
        """
        Slot Function of Updating Prediction & Plots
        """
        if not self.analyzeBtn.isEnabled():
            return

        labelColumn = self.columnSelectComboBox.currentText()
        predColumn = 'predict'

        self.testLabel = self.numericData[labelColumn].values.reshape(-1, 1)
        self.testData = self.numericData.loc[:, self.numericData.columns != labelColumn]

        self.testPred = self.DNN.test(self.testData.values, self.testLabel)
        self.testPred = pd.DataFrame(data = { predColumn : self.testPred.reshape(-1) })

        # Include non numeric columns to testDataWithPred
        testDataWithPred = pd.concat([self.testPred, self.data], axis = 1)
        sortedTestDataWithPred = testDataWithPred.sort_values(by = [predColumn], ascending=False)

        # Prediction Info
        npTestDataWithPred = np.array(testDataWithPred)[:100] # show only top 100 terms
        w, h = npTestDataWithPred.shape[:2]
        self.predTable.setRowCount(w)
        self.predTable.setColumnCount(h)
        self.predTable.setHorizontalHeaderLabels(testDataWithPred.columns)
        for i in range(w):
            for j in range(h):
                tempItem = QTableWidgetItem()
                tempItem.setText(str(npTestDataWithPred[i][j]))
                self.predTable.setItem(i,j,tempItem)
        self.predTable.repaint()

        # Sorted Prediction Info
        npSortedTestDataWithPred = np.array(sortedTestDataWithPred)[:100] # show only top 100 terms
        w, h = npSortedTestDataWithPred.shape[:2]
        self.sortedPredTable.setRowCount(w)
        self.sortedPredTable.setColumnCount(h)
        self.sortedPredTable.setHorizontalHeaderLabels(sortedTestDataWithPred.columns)
        for i in range(w):
            for j in range(h):
                tempItem = QTableWidgetItem()
                tempItem.setText(str(npSortedTestDataWithPred[i][j]))
                self.sortedPredTable.setItem(i,j,tempItem)
        self.sortedPredTable.repaint()

        # Molecule Plot

        if 'smiles' in sortedTestDataWithPred.columns:
            shortLabelColumn = labelColumn if len(labelColumn) < 20 else labelColumn[:20] + '...'
            for i in range( min(5, len(sortedTestDataWithPred)) ):
                smiles = sortedTestDataWithPred.loc[i, 'smiles']
                mol = Chem.MolFromSmiles(smiles)

                im = Draw.MolToImage(mol)
                qim = ImageQt(im)
                pixmap = QPixmap.fromImage(qim)

                widget = self.molPlotLayout.itemAt(i).widget()
                label = self.molLabelLayout.itemAt(i).widget()
                pixmap = pixmap.scaled(widget.size())
                widget.setPixmap(pixmap)              
                label.setText('{} : {:.3f}'.format(shortLabelColumn, sortedTestDataWithPred.loc[i, predColumn]))
                label.repaint()
        else:
            for i in range(5):
                widget = self.molPlotLayout.itemAt(i).widget()
                label = self.molLabelLayout.itemAt(i).widget()
                widget.clear()
                label.setText("SMILES Column not Found")

        # Fitting Plot
        fig = Figure()
        ax1f1 = fig.add_subplot(111)
        x1 = self.testPred.values.reshape(-1)
        y1 = self.testLabel.reshape(-1)

        ax1f1.scatter(x1, y1)
        ax1f1.set_title('Fitting Curve')
        ax1f1.set_xlabel('Predict Value')
        ax1f1.set_ylabel('Real Value')

        lims = [
            np.min([ax1f1.get_xlim(), ax1f1.get_ylim()]),  # min of both axes
            np.max([ax1f1.get_xlim(), ax1f1.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax1f1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

        self._addmpl(self.fittingPlotLayout, fig)

#        self._debugPrint(str(self.DNN.model.state_dict()))

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
        self.infoList.repaint()
