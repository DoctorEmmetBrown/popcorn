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
        self.showMaximized()  # Redimension de la fenetre

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
        app.setStyleSheet("QRadioButton {color: red }")

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

        self.TabW.addTab(self.rightWidgetOne, "Transformation Image")

        ####Fin du widget de Droite pour nuance de gris et histogramme

        ####Debut du widget de Droite pour utiliser la fonction de decomposition de Popcorn

        self.rightWidgetTwo = Decomposition(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetTwo, "Decomposition")
        ####Fin
        
        ####Debut du widget de Droite pour utiliser la fonction de recalage de Popcorn
        self.rightWidgetRecalage = Recalage(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetRecalage, "Recalage")
        ####Fin
        
        
        ###########################
        ####Debut du widget de Droite pour utiliser les fonctions de recup de phase
        self.rightWidgetRecup = Recup_phase(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetRecup, "Recuperation de phase")
        ####Fin
        
        
        ###########################
        ####Debut du widget de Droite pour utiliser la fonction de stitching

        self.rightWidgetStitching = Stitching(
            self
        )  # widget sur lequel on mettra les elements

        self.TabW.addTab(self.rightWidgetStitching, "Stitching")
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
window.show()
sys.exit(app.exec())
