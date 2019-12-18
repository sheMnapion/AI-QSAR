# This Python file uses the following encoding: utf-8
import os
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser

DEFAULT_ICON = "images/stock_media-play.png"


class Tab1(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow1.ui", self)
        self.bindTab()

    def bindTab(self):
        self.dataBrowseBtn.released.connect(self.browseDataSlot)

        self.openAction = self.toolBar.addAction(QIcon("images/fileopen.png"), "Open Project(&O)")
        self.openAction.triggered.connect(self.browseProjectSlot)

        self.saveModelAction = self.toolBar.addAction(QIcon("images/gtk-save.png"), "Save Model(&S)")
        self.saveASModelAction = self.toolBar.addAction(QIcon("images/gtk-save-as.png"), "Save As Model(")

        self.loadModelAction = self.toolBar.addAction(QIcon("images/add.png"), "Load Model(&O)")
        self.loadModelAction.triggered.connect(self.browseModelSlot)

    def debugPrint(self, msg):
        self.trainingList.addItem(msg)

    def browseProjectSlot(self):
        folder = self._getFolder()
        if folder:
            self.debugPrint("setting project folder: " + folder)
            self._resetFolderList(self.projectList, folder)

    def browseModelSlot(self):
        file = self._getFile()
        if file:
            self.debugPrint("openning model file: " + file)
            icon = self._getIcon(os.path.join(os.getcwd(), file))
            self.modelList.addItem(QListWidgetItem(icon, file))

    def browseDataSlot(self):
        folder = self._getFolder()
        if folder:
            self.debugPrint("setting data folder: " + folder)
            self.dataLabel.setText(folder)
            self._resetFolderList(self.dataList, folder)

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
