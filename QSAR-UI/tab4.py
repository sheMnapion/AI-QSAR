# This Python file uses the following encoding: utf-8
import re
import os
import sys
import pandas as pd
import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QListWidgetItem, QListWidget, QTableWidgetItem, QLabel
from PyQt5.QtGui import QPixmap
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from types import MethodType
import time
from utils import resetFolderList, getFolder, getFile, getIcon, saveModel, mousePressEvent, clearLayout
from utils import DNN_PATH, CACHE_PATH, SMILE_REGEX

from rdkit import Chem
from rdkit.Chem import Draw
from PIL.ImageQt import ImageQt

sys.path.append(DNN_PATH)
from QSAR_DNN import QSARDNN
from smilesPredictor import SmilesDesigner

class SmilesDesignerTrainThread(QThread):
    """wrapper class for carrying out smiles designing procedures"""
    _signal=pyqtSignal(str)

    def __init__(self, moleculeDesigner):
        super(SmilesDesignerTrainThread, self).__init__()
        self.moleculeDesigner=moleculeDesigner

    def __del__(self):
        self.wait()

    def run(self):
        """run training process"""
        self._signal.emit('---------------------------------Start Training------------------------------------------------')
        self.moleculeDesigner.trainVAE(nRounds=200,lr=3e-4,batchSize=30,signal=self._signal)
        self._signal.emit('Training finished.')
        self.moleculeDesigner.encodeDataset()
        self._signal.emit('Dataset encoded into latent space.')
        self.moleculeDesigner.trainLatentModel(lr=3e-4,batchSize=20,nRounds=1000,earlyStopEpoch=20)
        tempDict=self.moleculeDesigner.decodeDict
        tempVAE=torch.load('/tmp/tmpBestModel.pt')
        tempLatentModel=torch.load('/tmp/latentModel.pt')
        torch.save([tempVAE,tempLatentModel,tempDict],'/tmp/totalVAEModel.pt')
        self._signal.emit('Regression model on latent space trained.')

class SmilesDesignerDesignThread(QThread):
    """wrapper class for carrying out smiles designing procedures"""
    _signal=pyqtSignal(str)
    _finishSignal=pyqtSignal(bool)

    def __init__(self, moleculeDesigner):
        super(SmilesDesignerDesignThread, self).__init__()
        self.moleculeDesigner=moleculeDesigner

    def __del__(self):
        self.wait()

    def run(self):
        """run training process"""
        self._signal.emit('---------------------------------Start Designing------------------------------------------------')
        designed=self.moleculeDesigner.molecularRandomDesign(aimNumber=10,batchSize=20,signal=self._signal)
        np.save('/tmp/designed',designed)
        self._finishSignal.emit(True)
        self._signal.emit('---------------------------------End Designing--------------------------------------------------')

class Tab4(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow4.ui", self)

        self.dataList.mousePressEvent = MethodType(mousePressEvent, self.dataList)
        self.modelList.mousePressEvent = MethodType(mousePressEvent, self.modelList)

        # Original test data and test data with only numeric columns
        self.data = None
        self.numericData = None
        self.nonNumericData = None

        # Currently Opened Folder
        self._currentDataFolder = None
        self._currentDataFile = None
        self._currentModelFile = None

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

        self.designBtn.released.connect(self.designSlot)
        self.trainBtn.released.connect(self.startTrainingSlot)

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

    def _resetTrainBtn(self):
        """
        When Should We Enable trainBtn, and Where to Call This Method
        """
        if self._currentDataFile:
            self.trainBtn.setEnabled(True)
            self.trainBtn.repaint()
        else:
            self.trainBtn.setEnabled(False)
            self.trainBtn.repaint()

    def _resetDesignBtn(self):
        """
        When Should We Enable designBtn, and Where to Call This Method
        """
        if self._currentModelFile:
            self.designBtn.setEnabled(True)
            self.designBtn.repaint()
        else:
            self.designBtn.setEnabled(False)
            self.designBtn.repaint()

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
        try:
            model = self.modelList.currentItem().text()
        except:
            self._debugPrint("Current Model File Not Found")
            return

        if re.match(".+.pxl$", model) or re.match(".+.pt", model):
            try:
                print('Path:',model)
                self.designer=SmilesDesigner()
                self.designer.initFromModel(model)
                self._debugPrint("Model Loaded: {}".format(model))
            except:
                self._debugPrint("Load Model Error!")
                return
        else:
            self._debugPrint("Not a pytorch model!")
            return

        self._currentModelFile = model

        self._resetDesignBtn()

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
                if self.numericData is not None:
                    self.columnSelectComboBox.clear()
                    self.columnSelectComboBox.addItems(self.numericData.columns)

                self.nonNumericData = self.data.select_dtypes(exclude = np.number)
                if self.nonNumericData is not None:
                    self.smilesSelectComboBox.clear()
                    self.smilesSelectComboBox.addItems(self.nonNumericData.columns)

                    for i in range(len(self.nonNumericData.columns)):
                        if re.search(SMILE_REGEX, self.nonNumericData.columns[i]):
                            self.smilesSelectComboBox.setCurrentIndex(i)
                            self.smilesSelectComboBox.repaint()
                            break
            except:
                self.data = self.numericData = self.nonNumericData = None
                self._debugPrint("Load Data Error!")
                return

            self._debugPrint("csv file {} loaded".format(file))
            self._debugPrint(str(self.data.head()))
        else:
            self._debugPrint("Not a csv file!")
            return

        self._currentDataFile = file

        self._resetTrainBtn()

    def designSlot(self):
        """
        Slot Function of Design Molecules after Training
        """
        if not self.designBtn.isEnabled():
            return
        self.designerDesignThread=SmilesDesignerDesignThread(self.designer)
        self.designerDesignThread._signal.connect(self.getTrainResults) # use same for showing debug information
        self.designerDesignThread._finishSignal.connect(self.getDesignedMolecules)
        self.designerDesignThread.start()

    def analyzeSlot(self):
        self.designSlot()

    def startTrainingSlot(self):
        """
        Slot Function of Training After Loading Data and Model
        """
        if not self.trainBtn.isEnabled():
            return
        propName=self.columnSelectComboBox.currentText()
        smilesName=self.smilesSelectComboBox.currentText()
        propColumn=np.array(self.data[propName])
        smilesColumn=np.array(self.data[smilesName])
        try:
            self.designer=SmilesDesigner()
            self.designer.initFromSmilesAndProps(smilesColumn,propColumn)
        except ValueError as e:
            self._debugPrint("Error loading the model! Please check the columns selected.")
            return
        self.designerThread = SmilesDesignerTrainThread(self.designer)
        self.designerThread._signal.connect(self.getTrainResults)
        self.designerThread.start()

    def getTrainResults(self, msg):
        """receiving results of training"""
        self._debugPrint(msg)

    def getDesignedMolecules(self, state):
        """receive bool state for identifying success or not"""
        if state==True:
            print("DESIGNING FINISHED!")
            molecules=np.load('/tmp/designed.npy')
            nMol=molecules.shape[0]
            nColumns=6
            nRows=int(np.ceil(nMol/nColumns))
            self.designTable.setRowCount(nRows*2); self.designTable.setColumnCount(nColumns)
            for i in range(nRows):
                for j in range(nColumns):
                    index=i*nColumns+j
                    if index>=nMol: break
                    molValue=molecules[index][0]
                    molSmiles=molecules[index][1]
                    tempItem = QTableWidgetItem()
                    tempItem.setText(molValue)
                    self.designTable.setItem(2*i+1, j, tempItem)
                    tempMol=Chem.MolFromSmiles(molSmiles)
                    pixmap=Draw.MolToQPixmap(tempMol)
                    tempLabel=QLabel()
                    tempLabel.setPixmap(pixmap)
                    self.designTable.setCellWidget(2*i,j,tempLabel)

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.infoList.addItem(msg)
        self.infoList.repaint()
