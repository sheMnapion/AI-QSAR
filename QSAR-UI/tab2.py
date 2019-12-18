# This Python file uses the following encoding: utf-8
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow

class Tab2(QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi("mainwindow2.ui", self)

    def bind(self):
        pass
