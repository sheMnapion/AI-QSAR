# This Python file uses the following encoding: utf-8

import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QFileIconProvider, QListWidget
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal
from os.path import expanduser
import numpy as np, pandas as pd
import random as r

from rdkit import Chem
from rdkit.Chem import Draw

DEFAULT_ICON = "images/stock_media-play.png"
DNN_PATH = os.path.abspath('../QSAR-DNN')
RNN_PATH = os.path.abspath('../SMILES_RNN')
CACHE_PATH = os.path.join(DNN_PATH, "__trainingcache__")
SMILE_REGEX = "[s|S][m|M][i|I][l|L]|[e|E][s|S]"

def resetFolderList(List, folder):
    """
    Reset a QListWidget Which List Files in Folders with a New Folder
    """
    if folder is not None and folder[-1] != '/':
        folder += '/'
    fileInfo = QtCore.QFileInfo(folder)
    List.clear()
    List.setUpdatesEnabled(False)
    for file in fileInfo.dir():
        if file in ['.']:
            continue
        icon = getIcon(os.path.join(folder, file))
        List.addItem(QListWidgetItem(icon, file))
    List.setUpdatesEnabled(True)


def getFolder():
    """
    Get Folder Path by File Browser
    """
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    folder = QFileDialog.getExistingDirectory(None,
                                             "Open Directory",
                                             expanduser("."),
                                             QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
    if folder == '':
        return None
    return folder


def getFile():
    """
    Get File Path by File Browser
    """
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file = QFileDialog.getOpenFileName(None,
                                     "Open File",
                                     expanduser("."),
                                     "All Files (*)")
    return file[0]


def saveModel():
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file = QFileDialog.getSaveFileName(None,
                                       'Save File',
                                       expanduser("."),
                                       "Pytorch Models (*.pxl)")
    return file[0]


def getIcon(path):
    """
    Get Default Icon According to File Path
    """
    iconProvider = QFileIconProvider()
    icon = iconProvider.icon(QtCore.QFileInfo(path))
    if icon.isNull():
        return QtGui.QIcon(DEFAULT_ICON)
    else:
        return icon


def mousePressEvent(self, event):
    """
    Change Default Mouse Press Event to deselect in a ListView by clicking off an item
    """
    self.clearSelection()
    QListWidget.mousePressEvent(self, event)


def clearLayout(layout):
    """
    Clear All Contents of QLayout
    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget() is not None:
            child.widget().deleteLater()
        elif child.layout() is not None:
            clearLayout(child.layout())


def getSmilesColumnName(data: pd.DataFrame):
    """try to find column which contains valid smiles strs
        if no valid found, return None
    """
    nRows,nColumns=data.shape[:2]
    columnNames=data.columns
    checkNumber=20
    checkRows=min(checkNumber,nRows)
    randIndexes=r.sample(range(nRows),checkRows)
    for i in range(nColumns):
        properties=data[columnNames[i]].loc[randIndexes]
        validNumber=0
        for prop in properties:
            try:
                tempMol=Chem.MolFromSmiles(prop)
                tempPILImage=Draw.MolToImage(tempMol)
                validNumber+=1
            except:
                continue
        validRatio=validNumber/checkRows
        if validRatio>=0.5:
            return columnNames[i]
    return None

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.sig = WorkerSignals()
        self.kwargs['progress_callback'] = self.sig.progress
        self.kwargs['result_callback'] = self.sig.result

        print(str(self.args))
        print(str(self.kwargs))

    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    '''
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    progress = pyqtSignal(str)
    update = pyqtSignal()
