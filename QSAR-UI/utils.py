# This Python file uses the following encoding: utf-8

import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QFileIconProvider
from os.path import expanduser

DEFAULT_ICON = "images/stock_media-play.png"


def resetFolderList(List, folder):
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
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    folder = QFileDialog.getExistingDirectory(None,
                                             "Open Directory",
                                             expanduser("~"),
                                             QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
    folder += '/'
    return folder


def getFile():
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file = QFileDialog.getOpenFileName(None,
                                     "Open File",
                                     expanduser("~"),
                                     "All Files (*)")
    return file[0]


def getIcon(path):
    iconProvider = QFileIconProvider()
    icon = iconProvider.icon(QtCore.QFileInfo(path))
    if icon.isNull():
        return QtGui.QIcon(DEFAULT_ICON)
    else:
        return icon
