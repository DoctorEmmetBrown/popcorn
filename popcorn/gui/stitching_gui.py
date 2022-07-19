import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *

"""
###############
NON FONCTIONNEL
###############
"""



class Stitching(QWidget):
    def __init__(self, father):
        super().__init__()
        self.father = father
        self.layoutStitching = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutStitching)
        
        self.open_stitching=QPushButton("Open Stitching Window")
        self.open_stitching.clicked.connect(self.stitching_para)
        self.layoutStitching.addWidget(self.open_stitching, 0, 0)
        
        self.window=Stitching_window(self)
    def stitching_para(self):
        self.window.hide()
        self.window.show()


class Stitching_window(QWidget):
    def __init__(self, father):
        super().__init__()
        self.father = father
        self.layoutStitching = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutStitching)
        self.liste_folder=[]
        
        self.load_folder=QPushButton("Load Folder")
        self.load_folder.clicked.connect(self.open_folder)
        self.layoutStitching.addWidget(self.load_folder, 0, 0)
        
        self.output_list_folder=QTextEdit()
        self.layoutStitching.addWidget(self.output_list_folder, 1, 0)
        
        self.output_folder=QPushButton("Output Folder")
        self.output_folder.clicked.connect(self.choose_output_folder)
        self.layoutStitching.addWidget(self.output_folder, 0, 1)
        
        self.display_output_folder=QTextEdit()
        self.layoutStitching.addWidget(self.display_output_folder, 1, 1)
        
    def open_folder(self):
        path=QFileDialog.getExistingDirectory(self, str("Choose Exp Directory"))
        if path!="":
            self.liste_folder.append(path+'/')
        self.output_list_folder.setText("\n".join(self.liste_folder))
        
    def choose_output_folder(self):
        path=QFileDialog.getExistingDirectory(self, str("Choose Output Directory"))
        if path!="":
            self.display_output_folder.setText(path+'/')
