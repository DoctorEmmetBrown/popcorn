"""
Ce programme necessite :
PyQt6
qtrangeslider
"""

import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *

from pathlib import Path
path_root = str(Path(__file__).parents[1])
if path_root not in sys.path:
    sys.path.append(path_root)

from popcorn.input_output import *
from visualisation import *
from gray import *
from decomposition import *
from recalage import *
from recup_phase import *
from stitching_gui import *
from paresis import *


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()  # Constructeur parent
        self.setWindowTitle("popcorn")
        self.setWindowIcon(QIcon('../media/popcorn_logo.png'))
        self.mainWidget = QWidget()  # widget principal
        self.layoutMainW = (
            QGridLayout()
        )  # Permet l organisation des differents elements de la fenetre

        self.splitter = QSplitter(self)

        self.setCentralWidget(self.mainWidget)
        self.layoutMainW = (
            QGridLayout()
        )  # Permet l organisation des differents elements de la fenetre
        self.mainWidget.setLayout(self.layoutMainW)

        self.rightWidgetTwo = None
        self.rightWidgetRecalage = None
        """
        Realisation de la partie de gauche (permanente)
        """

        self.leftWidget = Visualisation(
            self
        )  # widget sur lequel on mettra les elements
        # self.layoutMainW.addWidget(self.leftWidget,0,0) #On place ce nouveau widget sur le main widget
        ssFile = "stylesheet.css"
        with open(ssFile, "r") as fh:
            app.setStyleSheet(fh.read())

        ####Fin du widget de gauche
        """
        Realisation de la partie de droite (QTabWidget)
        """
        self.TabW = QTabWidget()

        self.splitter.addWidget(self.leftWidget)
        self.splitter.addWidget(self.TabW)
        self.layoutMainW.addWidget(self.splitter, 0, 0)


        ####Debut du widget de Droite pour nuance de gris et histogramme

        self.rightWidgetOne = Gray(self)  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetOne, "Histogram")

        ####Fin du widget de Droite pour nuance de gris et histogramme

        ####Debut du widget de Droite pour utiliser la fonction de decomposition de Popcorn

        self.rightWidgetTwo = Decomposition(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetTwo, "Material Decomposition")
        ####Fin
        
        ####Debut du widget de Droite pour utiliser la fonction de recalage de Popcorn
        self.rightWidgetRecalage = Recalage(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetRecalage, "Registration")
        ####Fin
        
        
        ###########################
        ####Debut du widget de Droite pour utiliser les fonctions de recup de phase
        self.rightWidgetRecup = Recup_phase(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetRecup, "Phase Retrieval")
        ####Fin
        
        
        ###########################
        ####Debut du widget de Droite pour utiliser la fonction de stitching

        self.rightWidgetStitching = Stitching(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetStitching, "Visual Stitching")
        ####Fin

        ###########################
        ####Debut du widget de Droite pour utiliser la fonction de stitching

        self.rightWidgetParesis = Paresis(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetParesis, "Paresis")
        ####Fin

# ------#
# -Main-#
# ------#



# Initialisation du GUI
app = QApplication(sys.argv)
window = MainWindow()
screen = app.primaryScreen()
screen_width, screen_height = screen.size().width(), screen.size().height()
width = min(int(0.75*screen_width), 1200)
height = min(int(0.75*screen_height), 900)
top = screen_width//2 - width//2
left = screen_height//2 - height//2
window.setGeometry(top, left, width, height)
window.show()
sys.exit(app.exec())
