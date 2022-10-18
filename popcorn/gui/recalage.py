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

from popcorn import input_output
from spectral_imaging.registration import registration_computation
from spectral_imaging.registration import apply_itk_transformation
from datetime import datetime

class Recalage(QWidget):
    my_signal = pyqtSignal(str)
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutRecalage = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutRecalage)
        
        self.movIm = QLabel("Moving Image")
        self.layoutRecalage.addWidget(self.movIm, 0, 0)
        
        self.ComboBoxMov = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxMov, 0, 1)
        
        self.refIm = QLabel("Reference Image")
        self.layoutRecalage.addWidget(self.refIm, 1, 0)
        
        self.ComboBoxRef = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxRef, 1, 1)
        
        self.TransType = QLabel("Transformation type")
        self.layoutRecalage.addWidget(self.TransType, 2, 0)
        
        self.ComboBoxTrans = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxTrans, 2, 1)
        self.ComboBoxTrans.addItem("rotation (default)")
        self.ComboBoxTrans.addItem("transformation")
        
        self.Metric = QLabel("Metric")
        self.layoutRecalage.addWidget(self.Metric, 3, 0)
        
        self.ComboBoxMet = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxMet, 3, 1)
        self.ComboBoxMet.addItem("cross-correlation (default)")
        self.ComboBoxMet.addItem("ANTS-cross-correlation")
        self.ComboBoxMet.addItem("mutual information")
        self.ComboBoxMet.addItem("mean square diff")
        
        self.movMask = QLabel("Moving Mask")
        self.layoutRecalage.addWidget(self.movMask, 4, 0)
        
        self.ComboBoxMaskMov = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxMaskMov, 4, 1)
        self.ComboBoxMaskMov.addItem("None (default)")

        self.refMask = QLabel("Reference Mask")
        self.layoutRecalage.addWidget(self.refMask, 5, 0)
        
        self.ComboBoxMaskRef = QComboBox()
        self.layoutRecalage.addWidget(self.ComboBoxMaskRef, 5, 1)
        self.ComboBoxMaskRef.addItem("None (default)")
        
        
        for name_file in range(self.father.leftWidget.combobox.count()):
                    self.ComboBoxMov.addItem(self.father.leftWidget.combobox.itemText(name_file))
                    self.ComboBoxRef.addItem(self.father.leftWidget.combobox.itemText(name_file))
                    self.ComboBoxMaskMov.addItem(self.father.leftWidget.combobox.itemText(name_file))
                    self.ComboBoxMaskRef.addItem(self.father.leftWidget.combobox.itemText(name_file))
                    
                    
                    
        self.recalage_button=QPushButton("Start Recalage")
        #self.recalage_button.clicked.connect(self.exec_recalage)
        self.a=WObject(self)
        self.recalage_button.clicked.connect(self.a.start)
        self.layoutRecalage.addWidget(self.recalage_button, 6, 0,1,2)
        
        
        self.output=QTextEdit()
        self.layoutRecalage.addWidget(self.output, 7, 0,1,2)
        self.output.setReadOnly(True)
        self.my_signal.connect(self.display_text)
        
    def exec_recalage(self):
        """
        Read the differents data from the GUI and start registration_computation from popcorn
        """
        if (self.father.leftWidget.liste_image[self.ComboBoxMov.currentIndex()].color or 
            self.father.leftWidget.liste_image[self.ComboBoxRef.currentIndex()].color):
            
            message_error=QErrorMessage()
            message_error.showMessage("At lease one image is in color. This fonction take only gray image")
            message_error.exec()
            return
        
        self.arg1=self.father.leftWidget.liste_image[
                    self.ComboBoxMov.currentIndex()
                ].image
        
        self.arg2=self.father.leftWidget.liste_image[
                    self.ComboBoxRef.currentIndex()
                ].image
        
        self.arg3=self.ComboBoxTrans.currentText()

       
        self.arg4=self.ComboBoxMet.currentText()
        
        self.arg5=None
        if self.ComboBoxMaskMov.currentIndex()>0:
            if self.father.leftWidget.liste_image[self.ComboBoxMaskMov.currentIndex()].color:
                
                message_error=QErrorMessage()
                message_error.showMessage("At lease one image is in color. This fonction take only gray image")
                message_error.exec()
                return
            self.arg5=self.father.leftWidget.liste_image[
                    self.ComboBoxMaskMov.currentIndex()-1
                ].image
        
        self.arg6=None
        if self.ComboBoxMaskRef.currentIndex()>0:
            if self.father.leftWidget.liste_image[self.ComboBoxMaskRef.currentIndex()].color:
                
                message_error=QErrorMessage()
                message_error.showMessage("At lease one image is in color. This fonction take only gray image")
                message_error.exec()
                return
            self.arg6=self.father.leftWidget.liste_image[
                    self.ComboBoxMaskRef.currentIndex()-1
                ].image

        transfo=registration_computation(self.arg1,self.arg2,self.arg3,self.arg4,self.arg5,self.arg6,False,self)

        new_im=apply_itk_transformation(self.arg1,transfo,ref_img=self.arg2)

        date=str(datetime.today()).split(".")[0]
        log=date+"\n"+"Image do by Recalage\nTransformation type: "+self.arg3+"\n"+"Metric: "+self.arg4+"\n"

        if self.arg5==None and self.arg6==None:
            log=log+"No Mask\n"
        else:
            log=log+"With Mask\n"
        ok=False

        while ok==False:
            name_image, ok = QInputDialog.getText(self, "integer input dualog", "enter name image")

        self.father.leftWidget.ajouter_image(name_image,new_im,False)

        self.father.leftWidget.liste_log[name_image]=log



    #@pyqtSlot(str)
    def display_text(self, text):
        """
        Display the output of registration_computation on the GUI
        """
        self.output.setText(text)

class WObject(QThread):
    """
    Thread for exec_recalage
    """
    def __init__(self,gui):
        super().__init__()

        self.gui=gui

    def run(self):

        self.gui.exec_recalage()


    

