# This Python file uses the following encoding: utf-8
import sys
import os
from PyQt5 import uic, QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QMainWindow, QFileDialog, QListWidgetItem, QFileIconProvider
from model import Model
from os.path import expanduser

DEFAULT_ICON = "images/stock_media-play.png"


class MainWindow(QTabWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = Model()
        self.mime_database = QtCore.QMimeDatabase()

        self.tab1 = QMainWindow()
        self.tab2 = QMainWindow()
        self.tab3 = QMainWindow()

        self.addTab(self.tab1, "TAB1")
        self.addTab(self.tab2, "TAB2")
        self.addTab(self.tab3, "TAB3")

        uic.loadUi("mainwindow1.ui", self.tab1)
        uic.loadUi("mainwindow2.ui", self.tab2)
        uic.loadUi("mainwindow3.ui", self.tab3)

        self.resize(self.tab1.size())

        self.bindTabs()

    def bindTabs(self):
        self.tab1.browseBtn.released.connect(self.browseSlot)

    def debugPrint(self, msg):
        self.tab1.trainingList.addItem(msg)

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
            self.tab1.labelFolder.setText(folder)

            fileInfo = QtCore.QFileInfo(folder)
            self.tab1.folderList.clear()

            for file in fileInfo.dir():
                icon = self.getIcon(os.path.join(folder, file))
                self.tab1.folderList.addItem(QListWidgetItem(icon, file))

    def getIcon(self, path):

        iconProvider = QFileIconProvider()
        icon = iconProvider.icon(QtCore.QFileInfo(path))
        if icon.isNull():
            return QtGui.QIcon(DEFAULT_ICON)
        else:
            return icon


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec_()
