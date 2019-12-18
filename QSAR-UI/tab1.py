# This Python file uses the following encoding: utf-8
import os
from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser

DEFAULT_ICON = "images/stock_media-play.png"


class Tab1(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow1.ui", self)
        self.bindTab()

    def bindTab(self):
        self.browseBtn.released.connect(self.browseSlot)

    def debugPrint(self, msg):
        self.trainingList.addItem(msg)

    def browseSlot(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(None,
                                                 "Open Directory",
                                                 expanduser("~"),
                                                 QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        folder += '/'
        if folder:
            self.debugPrint("setting folder: " + folder)
            self.labelFolder.setText(folder)

            fileInfo = QtCore.QFileInfo(folder)
            self.folderList.clear()

            self.folderList.setUpdatesEnabled(False)
            for file in fileInfo.dir():
                icon = self.getIcon(os.path.join(folder, file))
                self.folderList.addItem(QListWidgetItem(icon, file))
            self.folderList.setUpdatesEnabled(True)

    def getIcon(self, path):

        iconProvider = QFileIconProvider()
        icon = iconProvider.icon(QtCore.QFileInfo(path))
        if icon.isNull():
            return QtGui.QIcon(DEFAULT_ICON)
        else:
            return icon
