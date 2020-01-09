# This Python file uses the following encoding: utf-8
import os
import re
import numpy as np
import pandas as pd
from types import MethodType

from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QErrorMessage
from PyQt5.QtGui import QPixmap
from utils import resetFolderList, mousePressEvent, clearLayout
from utils import DNN_PATH, CACHE_PATH, RNN_PATH

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


class Tab2(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow2.ui", self)

        self.result = None
        self._currentTrainingHistoryFile = None

        self.trainingHistoryList.mousePressEvent = MethodType(mousePressEvent, self.trainingHistoryList)
        self.resultList.mousePressEvent = MethodType(mousePressEvent, self.resultList)
        self.refreshTrainingList()

        self._bind()

    def _bind(self):
        """
        Bind Slots and Signals
        """
        self.trainingHistorySelectBtn.released.connect(self.trainingHistorySelectSlot)
        self.trainingHistoryList.itemDoubleClicked.connect(self.trainingHistoryDoubleClickedSlot)
        self.analyzeBtn.released.connect(self.analyzeSlot)

        # Ensure Scroll to Bottom in Realtime
        self.resultList.model().rowsInserted.connect(self.resultList.scrollToBottom)

    def _addmpl(self, widget, fig):
        """
        Add matplotlib Canvas
        """
        clearLayout(widget.layout())

        self.canvas = FigureCanvas(fig)
        widget.layout().addWidget(self.canvas)
        self.canvas.draw()
#        self.plotToolBar = NavigationToolbar(self.canvas, self.plotWidget, coordinates=True)
#        layout.addWidget(self.plotToolBar)

    def refreshTrainingList(self):
        """
        Refresh QListWidget of training history after each training process
        """
        if not os.path.exists(CACHE_PATH):
            os.mkdir(CACHE_PATH)

        resetFolderList(self.trainingHistoryList, CACHE_PATH, lastModified=True, showParent=False)

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
            self._debugPrint("(n, len(loss), len(mse), len(testPred), len(testLabel):\n( {}, {}, {}, {}, {} )".format(
                        self.result["numEpochs"],
                        len(self.result["lossList"]),
                        len(self.result["mseList"]),
                        len(self.result["testPred"]),
                        len(self.result["testLabel"])))
            if 'modelName' not in self.result: # make it compatible with former training history
                self.result['modelName']='DNN'
        else:
            self._debugPrint("Not a Training History(*.npy)!")
            return

        self._currentTrainingHistoryFile = file
        self.analyzeBtn.setEnabled(True)
        self.analyzeBtn.repaint()

    def trainingHistoryDoubleClickedSlot(self, item):
        """
        Slot Function of Double Clicking a Folder or a File in self.trainingHistoryList
        """
        selectedFile = os.path.join(CACHE_PATH, item.text())
        if os.path.isfile(selectedFile):
            self.trainingHistorySelectBtn.click()

    def analyzeSlot(self):
        """
        Slot Function of Updating Matplotlib Plots about Selected Training History
        """
        if not self.analyzeBtn.isEnabled():
            return

        try:
            if self.precisionCurveCheckBox.isChecked():
                fig = Figure()
                ax1f1 = fig.add_subplot(111)
                x1 = self.result["testPred"]
                y1 = self.result["testLabel"]
                ax1f1.scatter(x1, y1)
                ax1f1.set_title('Precision Curve')
                ax1f1.set_xlabel('Predict Value')
                ax1f1.set_ylabel('Real Value')

                lims = [
                    np.min([ax1f1.get_xlim(), ax1f1.get_ylim()]),  # min of both axes
                    np.max([ax1f1.get_xlim(), ax1f1.get_ylim()]),  # max of both axes
                ]
                # now plot both limits against eachother
                ax1f1.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

                self._addmpl(self.precisionCurveWidget, fig)
            else:
                clearLayout(self.precisionCurveLayout)

            if self.lossCurveCheckBox.isChecked():
                fig = Figure()
                ax1f1 = fig.add_subplot(111)
                y1 = self.result["mseList"]
                x1 = np.linspace(0, len(y1) - 1, len(y1))
                ax1f1.plot(x1, y1)
                ax1f1.set_title('Training Loss Curve')
                ax1f1.set_xlabel('Epochs')
                ax1f1.set_ylabel('MSE')
                self._addmpl(self.lossCurveWidget, fig)
            else:
                clearLayout(self.lossCurveLayout)

            if self.modelStructureCheckBox.isChecked():
                if self.result['modelName']=='DNN':
                    pixmap = QPixmap(os.path.join(DNN_PATH, 'architecture.png'))
                else:
                    pixmap = QPixmap(os.path.join(RNN_PATH, 'architecture.png'))
                pixmap = pixmap.scaled(self.modelStructureLabel.size()) #,QtCore.Qt.KeepAspectRatio)
                self.modelStructureLabel.setPixmap(pixmap)
            else:
                self.modelStructureLabel.clear()
        except:
            errorMsg=QErrorMessage(self)
            errorMsg.setWindowTitle("Analyzing results")
            errorMsg.showMessage("The training history contains messages that cannot be interpreted correctly. Please check your training process\
                                 for more information.")

    def _debugPrint(self, msg):
        """
        Print Debug Info on the UI
        """
        self.resultList.addItem(msg)
        self.resultList.repaint()
