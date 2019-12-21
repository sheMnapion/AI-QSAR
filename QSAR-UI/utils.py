# This Python file uses the following encoding: utf-8

import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QFileIconProvider, QListWidget
from os.path import expanduser

DEFAULT_ICON = "images/stock_media-play.png"


def resetFolderList(List, folder):
    """
    Reset a QListWidget Which List Files in Folders with a New Folder
    """
    fileInfo = QtCore.QFileInfo(folder)
    List.clear()
    List.setUpdatesEnabled(False)
    for file in fileInfo.dir():
        if file in ['.', '..']:
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
    folder += '/'
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

