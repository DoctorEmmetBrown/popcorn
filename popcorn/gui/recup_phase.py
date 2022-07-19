import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *

from popcorn.input_output import *
from datetime import datetime
import xml.etree.cElementTree as ET



import sys
sys.path.insert(1,'/Documents/Paul_ANDRE/Git/popcorn/phase_retrieval/')

from popcorn.phase_retrieval.saveParameters import saveParameters
from popcorn.phase_retrieval.PhaseRetrievalClasses import Phase_Retrieval_Experiment
import time
import datetime
import os


def launchPhaseRetrieval(experiment, do):
    """
    launches the phase retrieval algorithms set to true in do

    Args:
        experiment (Object): contains all experiment parameters and methods.
        do (DICT): boolean associated to each phase retrieval method.

    Returns:
        processing_time (float): Processing time of the different algos.

    """
    processing_time={}
    
    for method, to_do in do.items():
        if to_do:
            time0=time.time()
            experiment.process_method(method)
            processing_time[method]=time.time()-time0
            
    return processing_time

class Recup_phase(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutRecup = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutRecup )
        
        self.number_execution=0
        
        self.button_load_para=QPushButton("Load Parameter")
        self.button_load_para.clicked.connect(self.load_para)
        self.layoutRecup.addWidget(self.button_load_para, 0, 0,1,6)
        
        self.button_save_para=QPushButton("Save Parameter")
        self.button_save_para.clicked.connect(self.write_xml)
        self.layoutRecup.addWidget(self.button_save_para, 2, 0,1,3)
        
        self.choose_method=QPushButton("Choose Method")
        self.choose_method.clicked.connect(self.liste_method)
        self.layoutRecup.addWidget(self.choose_method, 1, 4,1,2)
        
        self.button_exp_para=QPushButton("Experiment Parameters")
        self.button_exp_para.clicked.connect(self.exp_para)
        self.layoutRecup.addWidget(self.button_exp_para, 1, 0,1,2)
        
        self.button_algo_para=QPushButton("Algorithmic Parameters")
        self.button_algo_para.clicked.connect(self.algo_para)
        self.layoutRecup.addWidget(self.button_algo_para, 1, 2,1,2)
        
        self.button_reset=QPushButton("Reset Parameters")
        self.button_reset.clicked.connect(self.reset)
        self.layoutRecup.addWidget(self.button_reset, 2, 3,1,3)
        
        self.button_start=QPushButton("Start")
        self.button_start.clicked.connect(self.start)
        self.layoutRecup.addWidget(self.button_start, 3, 0,1,6)
        
        
        self.widget_liste_method=widget_method(self)
        self.widget_exp_para=widget_experiment(self)
        self.widget_algo_para=widget_algorithm(self)
        
        self.reset()
        
    def liste_method(self):
        """
        displays the window listing the different methods
        """
        self.widget_liste_method.hide()
        self.widget_liste_method.show()
        self.widget_liste_method.activateWindow( )
        self.widget_liste_method.raise_()
    def exp_para(self):
        """
        displays the window listing the experiment parameters
        """
        self.widget_exp_para.hide()
        self.widget_exp_para.show()
        self.widget_exp_para.activateWindow( )
        self.widget_exp_para.raise_()
    def algo_para(self):
        """
        displays the window listing the algorithm parameters
        """
        self.widget_algo_para.hide()
        self.widget_algo_para.show()
        self.widget_algo_para.activateWindow( )
        self.widget_algo_para.raise_()
    def reset(self):
        """
        reset all parameters
        """
        self.widget_liste_method.reset()
        self.widget_exp_para.reset()
        self.widget_algo_para.reset()
    def write_xml(self):
        """
        write the differents xml files
        """
        self.write_exp_xml()
        self.write_algo_xml()
    def write_exp_xml(self):
        """
        write parameters in ExperimentParameters.xml
        """
        f=open("ExperimentParameters.xml",'r')
        texte=f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('experiment'):
            if experience[0].text==self.widget_exp_para.experiment_name.text():
                root.remove(experience)

        doc=ET.SubElement(root, "experiment")
        ET.SubElement(doc,"experiment_name").text=self.widget_exp_para.experiment_name.text()
        ET.SubElement(doc,"tomo").text=str(self.widget_exp_para.tomo_check.isChecked())
        ET.SubElement(doc,"exp_folder").text=self.widget_exp_para.exp_folder.text()
        ET.SubElement(doc,"output_folder").text=self.widget_exp_para.output_folder.text()
        ET.SubElement(doc,"energy",unit="keV").text=self.widget_exp_para.energy_edit.text()
        ET.SubElement(doc,"pixel",unit="m").text=self.widget_exp_para.pixel_edit.text()
        ET.SubElement(doc,"dist_object_detector",unit="m").text=self.widget_exp_para.detector_object_edit.text()
        ET.SubElement(doc,"dist_source_object",unit="m").text=self.widget_exp_para.detector_source_edit.text()
        ET.SubElement(doc,"delta").text=self.widget_exp_para.delta.text()
        ET.SubElement(doc,"beta").text=self.widget_exp_para.beta.text()
        ET.SubElement(doc,"source_size",unit="um").text=self.widget_exp_para.source_size.text()
        ET.SubElement(doc,"detector_PSF",unit="pix").text=self.widget_exp_para.detector_psf.text()
        ET.SubElement(doc,"crop_on").text=str(self.widget_exp_para.crop_check.isChecked())
        ET.SubElement(doc,"cropDebX",unit="pix").text=self.widget_exp_para.crop_checkDx.text()
        ET.SubElement(doc,"cropDebY",unit="pix").text=self.widget_exp_para.crop_checkDy.text()
        ET.SubElement(doc,"cropEndX",unit="pix").text=self.widget_exp_para.crop_checkEx.text()
        ET.SubElement(doc,"cropEndY",unit="pix").text=self.widget_exp_para.crop_checkEy.text()
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("ExperimentParameters.xml",encoding='utf-8')

    def write_algo_xml(self): 
        """
        write parameters in AlgorithmParameter.xml
        """
        f=open("AlgorithmParameter.xml",'r')
        texte=f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('experiment'):
            if experience[0].text==self.widget_exp_para.experiment_name.text():
                root.remove(experience)
                
        doc=ET.SubElement(root, "experiment")
        ET.SubElement(doc,"experiment_name").text=self.widget_exp_para.experiment_name.text()
        ET.SubElement(doc,"nb_of_point").text=self.widget_algo_para.nb_point_edit.text()
        ET.SubElement(doc,"pad_size").text=self.widget_algo_para.pad_size_edit.text()
        ET.SubElement(doc,"pad_type").text=self.widget_algo_para.pad_type_edit.text()
        ET.SubElement(doc,"do_deconvolution").text=str(self.widget_algo_para.do_deconvolution_edit.isChecked())
        ET.SubElement(doc,"deconvolution_type").text=self.widget_algo_para.deconvolution_type_edit.text()
        ET.SubElement(doc,"absorption_correction_sigma").text=self.widget_algo_para.tomo_absorption_correction_sigma.text()
        ET.SubElement(doc,"max_shift").text=self.widget_algo_para.max_shift_edit.text()
        ET.SubElement(doc,"LCS_median_filter").text=self.widget_algo_para.lcs_median_filter_edit.text()
        ET.SubElement(doc,"umpaNw").text=self.widget_algo_para.umpaNw_edit.text()
        ET.SubElement(doc,"XSVT_Nw").text=self.widget_algo_para.XSVT_Nw_edit.text()
        ET.SubElement(doc,"XSVT_median_filter").text=self.widget_algo_para.XSVT_median_filter_edit.text()
        ET.SubElement(doc,"MIST_median_filter").text=self.widget_algo_para.MIST_median_filter_edit.text()
        ET.SubElement(doc,"sigma_regularization").text=self.widget_algo_para.sigma_regularization_edit.text()
        ET.SubElement(doc,"proj_to_treat_start").text=self.widget_algo_para.proj_to_treat_start_edit.text()
        ET.SubElement(doc,"proj_to_treat_end").text=self.widget_algo_para.proj_to_treat_edit.text()

        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("AlgorithmParameter.xml",encoding='utf-8')

    def load_para(self):
        """
        Read a xml and load parameters
        """
        exp_name,booleen=QInputDialog.getText(self, "Load experiment", "enter experiment name")
        if not booleen:
            return
        self.widget_exp_para.reset()
        self.widget_algo_para.reset()
        f=open("ExperimentParameters.xml",'r')
        texte=f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    self.change_value(name,value)
        f.close()
        f=open("AlgorithmParameter.xml",'r')
        texte=f.read()
        f.close()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    self.change_value(name,value)
        
                
    def change_value(self,name,value):
        """
        switch name and change the value of the QWidget correspond
        
        """
        if name=='experiment_name':
            self.widget_exp_para.experiment_name.setText(value)
        elif name=="tomo":
            if value=="True":
                self.widget_exp_para.tomo_check.setCheckState(Qt.CheckState.Checked)
            else:
                self.widget_exp_para.tomo_check.setCheckState(Qt.CheckState.Unchecked)
        elif name=="exp_folder":
            self.widget_exp_para.exp_folder.setText(value)
        elif name=="output_folder":
            self.widget_exp_para.output_folder.setText(value)
        elif name=="energy":
            self.widget_exp_para.energy_edit.setText(value)
        elif name=="pixel":
            self.widget_exp_para.pixel_edit.setText(value)
        elif name=="dist_object_detector":
            self.widget_exp_para.detector_object_edit.setText(value)
        elif name=="dist_source_object":
            self.widget_exp_para.detector_source_edit.setText(value)
        elif name=="delta":
            self.widget_exp_para.delta.setText(value)
        elif name=="beta":
            self.widget_exp_para.beta.setText(value)
        elif name=="source_size":
            self.widget_exp_para.source_size.setText(value)
        elif name=="detector_PSF":
            self.widget_exp_para.detector_psf.setText(value)
        elif name=="crop_on":
            if value=="True":
                self.widget_exp_para.crop_check.setCheckState(Qt.CheckState.Checked)
            else:
                self.widget_exp_para.crop_check.setCheckState(Qt.CheckState.Unchecked)
        elif name=="cropDebX":
            self.widget_exp_para.crop_checkDx.setText(value)
        elif name=="cropDebY":
            self.widget_exp_para.crop_checkDy.setText(value)
        elif name=="cropEndX":
            self.widget_exp_para.crop_checkEx.setText(value)
        elif name=="cropEndY":
            self.widget_exp_para.crop_checkEy.setText(value)
        elif name=="nb_of_point":
            self.widget_algo_para.nb_point_edit.setText(value)
        elif name=="pad_size":
            self.widget_algo_para.pad_size_edit.setText(value)
        elif name=="pad_type":
            self.widget_algo_para.pad_type_edit.setText(value)
        elif name=="do_deconvolution":
            if value=="True":
                self.widget_algo_para.do_deconvolution_edit.setCheckState(Qt.CheckState.Checked)
            else:
                self.widget_algo_para.do_deconvolution_edit.setCheckState(Qt.CheckState.Unchecked)
        elif name=="deconvolution_type":
            self.widget_algo_para.deconvolution_type_edit.setText(value)
        elif name=="absorption_correction_sigma":
            self.widget_algo_para.tomo_absorption_correction_sigma.setText(value)
        elif name=="max_shift":
            self.widget_algo_para.max_shift_edit.setText(value)
        elif name=="LCS_median_filter":
            self.widget_algo_para.lcs_median_filter_edit.setText(value)
        elif name=="umpaNw":
            self.widget_algo_para.umpaNw_edit.setText(value)
        elif name=="XSVT_Nw":
            self.widget_algo_para.XSVT_Nw_edit.setText(value)
        elif name=="XSVT_median_filter":
            self.widget_algo_para.XSVT_median_filter_edit.setText(value)
        elif name=="MIST_median_filter":
            self.widget_algo_para.MIST_median_filter_edit.setText(value)
        elif name=="sigma_regularization":
            self.widget_algo_para.sigma_regularization_edit.setText(value)
        elif name=="proj_to_treat_start":
            self.widget_algo_para.proj_to_treat_start_edit.setText(value)
        elif name=="proj_to_treat_end":
            self.widget_algo_para.proj_to_treat_edit.setText(value)
            

    def start(self):
        """
        start start_exec in a thread
        """
        self.thread=recup_phase_thread(self)
        self.thread.start()
        
        
    def start_exec(self):
        """
        similar to allPhaseRetrievalMethods.py but do is create differently and at the end, images are load on the GUI
        """
        studied_case = self.widget_exp_para.experiment_name.text() # name of the experiment we want to work on
        
        do={}
        do['LCS']=self.widget_liste_method.method_1_check.isChecked()
        do['LCS_DF']=self.widget_liste_method.method_2_check.isChecked()
        do['MISTII_2']=self.widget_liste_method.method_3_check.isChecked()
        do['MISTII_1']=self.widget_liste_method.method_4_check.isChecked()
        do['MISTI']=self.widget_liste_method.method_5_check.isChecked()
        do['UMPA']=self.widget_liste_method.method_6_check.isChecked()
        do['OF']=self.widget_liste_method.method_7_check.isChecked()
        do['Pavlov']=self.widget_liste_method.method_8_check.isChecked()
        do['XSVT']=self.widget_liste_method.method_9_check.isChecked()
        save_parameters=True

        phase_retrieval_experiment=Phase_Retrieval_Experiment(studied_case, do)
        # We create a folder for each retrieval test
        now=datetime.datetime.now()
        phase_retrieval_experiment.expID=now.strftime("%Y%m%d-%H%M%S") #
        phase_retrieval_experiment.output_folder+=phase_retrieval_experiment.expID
        print(phase_retrieval_experiment.output_folder)
        os.mkdir(phase_retrieval_experiment.output_folder)
        print(phase_retrieval_experiment.tomo)
        if not phase_retrieval_experiment.tomo:
            phase_retrieval_experiment.open_Is_Ir()
            phase_retrieval_experiment.preProcessAndPadImages()
            processing_time=launchPhaseRetrieval(phase_retrieval_experiment,do)
            print(processing_time)
        
            if save_parameters:
                saveParameters(phase_retrieval_experiment, processing_time, do)
            
        if phase_retrieval_experiment.tomo:
            outpurFolder0=phase_retrieval_experiment.output_folder
            for iproj in range(phase_retrieval_experiment.proj_to_treat_start,phase_retrieval_experiment.proj_to_treat_end, 1):
                print("\n\n Processing projection:" ,iproj)
                phase_retrieval_experiment.open_Is_Ir_tomo(iproj, phase_retrieval_experiment.number_of_projections)
                phase_retrieval_experiment.preProcessAndPadImages()
                phase_retrieval_experiment.currentProjection=iproj
                processing_time=launchPhaseRetrieval(phase_retrieval_experiment, do)
                
            if save_parameters:
                saveParameters(phase_retrieval_experiment, processing_time, do)
        #On reouvre les images qui ont étaient créé dans le nouveau dossier
        
        liste_of_files=glob.glob(phase_retrieval_experiment.output_folder+"/*")
        liste_of_files.sort()
        for file_name in liste_of_files:
            try:
                image = open_image(file_name)
                Image_load = np.copy(image)
                Image_load=np.expand_dims(Image_load, axis = 0)
                self.father.leftWidget.ajouter_image(file_name.split("/")[-1], Image_load,False)
            except:
                pass
class widget_experiment(QWidget):
    """
    widget that show the Experiments parameters
    """
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutExp = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutExp)
        
        self.label_exp_name=QLabel("Experiment name")
        self.layoutExp.addWidget(self.label_exp_name, 0, 0)
        self.experiment_name=QLineEdit("")
        self.layoutExp.addWidget(self.experiment_name, 0, 1)
        
        self.label_tomo=QLabel("Tomo")
        self.layoutExp.addWidget(self.label_tomo, 1, 0)
        self.tomo_check=QCheckBox()
        self.layoutExp.addWidget(self.tomo_check, 1, 1)
        
        self.label_nbr_proj=QLabel("Number of projections")
        self.layoutExp.addWidget(self.label_nbr_proj, 2, 0)
        self.nbr_proj_edit=QLineEdit("0")
        self.layoutExp.addWidget(self.nbr_proj_edit, 2, 1)
        
        self.button_exp_folder=QPushButton("exp folder")
        self.exp_folder=QLineEdit()
        self.button_exp_folder.clicked.connect(self.choose_exp_folder)
        self.layoutExp.addWidget(self.button_exp_folder, 3, 0,1,2)
        self.layoutExp.addWidget(self.exp_folder, 4, 0,1,2)
        
        self.button_output_folder=QPushButton("ouput folder")
        self.output_folder=QLineEdit()
        self.button_output_folder.clicked.connect(self.choose_output_folder)
        self.layoutExp.addWidget(self.button_output_folder, 5, 0,1,2)
        self.layoutExp.addWidget(self.output_folder, 6, 0,1,2)
       
        self.label_energy=QLabel("Energy (keV)")
        self.layoutExp.addWidget(self.label_energy, 7, 0)
        self.energy_edit=QLineEdit("0")
        self.layoutExp.addWidget(self.energy_edit, 7, 1)
        
        self.label_pixel=QLabel("Pixel (m)")
        self.layoutExp.addWidget(self.label_pixel, 8, 0)
        self.pixel_edit=QLineEdit("0")
        self.layoutExp.addWidget(self.pixel_edit, 8, 1)
        
        self.label_detector=QLabel("dist_object_detector (m)")
        self.layoutExp.addWidget(self.label_detector, 9, 0)
        self.detector_object_edit=QLineEdit("0")
        self.layoutExp.addWidget(self.detector_object_edit, 9, 1)
        
        self.label_source=QLabel("dist_source_object (m)")
        self.layoutExp.addWidget(self.label_source, 10, 0)
        self.detector_source_edit=QLineEdit("0")
        self.layoutExp.addWidget(self.detector_source_edit, 10, 1)
        
        self.label_delta=QLabel("delta")
        self.layoutExp.addWidget(self.label_delta, 11, 0)
        self.delta=QLineEdit("0")
        self.layoutExp.addWidget(self.delta, 11, 1)
        
        self.label_beta=QLabel("beta")
        self.layoutExp.addWidget(self.label_beta,12, 0)
        self.beta=QLineEdit("0")
        self.layoutExp.addWidget(self.beta, 12, 1)
        
        self.label_source_size=QLabel("source_size (um)")
        self.layoutExp.addWidget(self.label_source_size, 13, 0)
        self.source_size=QLineEdit("0")
        self.layoutExp.addWidget(self.source_size, 13, 1)
        
        self.label_psf=QLabel("detector_PSF (pix)")
        self.layoutExp.addWidget(self.label_psf, 14, 0)
        self.detector_psf=QLineEdit("0")
        self.layoutExp.addWidget(self.detector_psf, 14, 1)
        
        self.label_crop=QLabel("Crop")
        self.layoutExp.addWidget(self.label_crop, 15, 0)
        self.crop_check=QCheckBox()
        self.layoutExp.addWidget(self.crop_check, 15, 1)
        
        self.label_cropDx=QLabel("CropDebX (pix)")
        self.layoutExp.addWidget(self.label_cropDx, 16, 0)
        self.crop_checkDx=QLineEdit("0")
        self.layoutExp.addWidget(self.crop_checkDx, 16, 1)
        
        self.label_cropDy=QLabel("CropDebY (pix)")
        self.layoutExp.addWidget(self.label_cropDy, 17, 0)
        self.crop_checkDy=QLineEdit("0")
        self.layoutExp.addWidget(self.crop_checkDy, 17, 1)
        
        self.label_cropEx=QLabel("CropEndX (pix)")
        self.layoutExp.addWidget(self.label_cropEx, 18, 0)
        self.crop_checkEx=QLineEdit("0")
        self.layoutExp.addWidget(self.crop_checkEx, 18, 1)
        
        self.label_cropEy=QLabel("CropEndY (pix)")
        self.layoutExp.addWidget(self.label_cropEy, 19, 0)
        self.crop_checkEy=QLineEdit("0")
        self.layoutExp.addWidget(self.crop_checkEy, 19, 1)
    def reset(self):
        """
        reset parameter for that widget
        """
        self.experiment_name.setText("")
        self.tomo_check.setCheckState(Qt.CheckState.Unchecked)
        self.exp_folder.setText("")
        self.output_folder.setText("")
        self.energy_edit.setText("0")
        self.pixel_edit.setText("0")
        self.detector_object_edit.setText("0")
        self.detector_source_edit.setText("0")
        self.delta.setText("0")
        self.beta.setText("0")
        self.source_size.setText("0")
        self.detector_psf.setText("0")
        self.crop_check.setCheckState(Qt.CheckState.Unchecked)
        self.crop_checkDx .setText("0")
        self.crop_checkDy.setText("0")
        self.crop_checkEx .setText("0")
        self.crop_checkEy.setText("0")
        
       
    def choose_exp_folder(self):
        """
        choose a directory thank to QDialog and write him in a QLineEdit (exp_folder)
        """
        path=QFileDialog.getExistingDirectory(self, str("Choose Exp Directory"))
        if path!="":
            self.exp_folder.setText(path+'/')
            print(self.exp_folder.text())      
    def choose_output_folder(self):
        """
        choose a directory thank to QDialog and write him in a QLineEdit (ouput_folder)
        """
        path=QFileDialog.getExistingDirectory(self, str("Choose Output Directory"))
        if path!="":
            self.output_folder.setText(path+'/')
            print(self.output_folder) 
    
        
class widget_algorithm(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutExp = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutExp)
        
        self.label_nb_point=QLabel("nb_of_point")
        self.layoutExp.addWidget(self.label_nb_point, 0, 0)
        self.nb_point_edit=QLineEdit()
        self.layoutExp.addWidget(self.nb_point_edit, 0, 1)
        
        self.label_pad_size=QLabel("pad_size")
        self.layoutExp.addWidget(self.label_pad_size, 1, 0)
        self.pad_size_edit=QLineEdit()
        self.layoutExp.addWidget(self.pad_size_edit, 1, 1)
        
        self.label_pad_type=QLabel("pad_type")
        self.layoutExp.addWidget(self.label_pad_type, 2, 0)
        self.pad_type_edit=QLineEdit()
        self.layoutExp.addWidget(self.pad_type_edit, 2, 1)
        
        self.label_do_deconvolution=QLabel("do_deconvolution")
        self.layoutExp.addWidget(self.label_do_deconvolution, 3, 0)
        self.do_deconvolution_edit=QCheckBox()
        self.layoutExp.addWidget(self.do_deconvolution_edit, 3, 1)
        
        self.label_deconvolution_type=QLabel("deconvolution_type")
        self.layoutExp.addWidget(self.label_deconvolution_type, 4, 0)
        self.deconvolution_type_edit=QLineEdit()
        self.layoutExp.addWidget(self.deconvolution_type_edit, 4, 1)
        
        self.label_absorption_correction_sigma=QLabel("absorption_correction_sigma")
        self.layoutExp.addWidget(self.label_absorption_correction_sigma, 5, 0)
        self.tomo_absorption_correction_sigma=QLineEdit()
        self.layoutExp.addWidget(self.tomo_absorption_correction_sigma, 5, 1)
        
        self.label_max_shift=QLabel("max_shift")
        self.layoutExp.addWidget(self.label_max_shift, 6, 0)
        self.max_shift_edit=QLineEdit()
        self.layoutExp.addWidget(self.max_shift_edit, 6, 1)
        
        self.label_lcs_median_filter=QLabel("LCS_median_filter")
        self.layoutExp.addWidget(self.label_lcs_median_filter, 7, 0)
        self.lcs_median_filter_edit=QLineEdit()
        self.layoutExp.addWidget(self.lcs_median_filter_edit, 7, 1)
        
        self.label_umpaNw=QLabel("umpaNw")
        self.layoutExp.addWidget(self.label_umpaNw, 8, 0)
        self.umpaNw_edit=QLineEdit()
        self.layoutExp.addWidget(self.umpaNw_edit, 8, 1)
        
        self.label_XSVT_Nw=QLabel("XSVT_Nw")
        self.layoutExp.addWidget(self.label_XSVT_Nw, 9, 0)
        self.XSVT_Nw_edit=QLineEdit()
        self.layoutExp.addWidget(self.XSVT_Nw_edit, 9, 1)
        
        self.label_XSVT_median_filter=QLabel("XSVT_median_filter")
        self.layoutExp.addWidget(self.label_XSVT_median_filter, 10, 0)
        self.XSVT_median_filter_edit=QLineEdit()
        self.layoutExp.addWidget(self.XSVT_median_filter_edit, 10, 1)
        
        self.label_MIST_median_filter=QLabel("MIST_median_filter")
        self.layoutExp.addWidget(self.label_MIST_median_filter, 11, 0)
        self.MIST_median_filter_edit=QLineEdit()
        self.layoutExp.addWidget(self.MIST_median_filter_edit, 11, 1)
        
        self.label_sigma_regularization=QLabel("sigma_regularization")
        self.layoutExp.addWidget(self.label_sigma_regularization, 12, 0)
        self.sigma_regularization_edit=QLineEdit()
        self.layoutExp.addWidget(self.sigma_regularization_edit, 12, 1)
        
        self.label_proj_to_treat_start=QLabel("proj_to_treat_start")
        self.layoutExp.addWidget(self.label_proj_to_treat_start, 13, 0)
        self.proj_to_treat_start_edit=QLineEdit()
        self.layoutExp.addWidget(self.proj_to_treat_start_edit, 13, 1)
        
        self.label_proj_to_treat=QLabel("proj_to_treat_end")
        self.layoutExp.addWidget(self.label_proj_to_treat, 14, 0)
        self.proj_to_treat_edit=QLineEdit()
        self.layoutExp.addWidget(self.proj_to_treat_edit, 14, 1)
    def reset(self):
        """
        Reset algorithms parameters
        """
        self.nb_point_edit.setText("0")
        self.pad_size_edit.setText("0")
        self.pad_type_edit.setText("reflect")
        self.do_deconvolution_edit.setCheckState(Qt.CheckState.Unchecked)
        self.deconvolution_type_edit.setText("")
        self.tomo_absorption_correction_sigma.setText("15")
        self.max_shift_edit.setText("0")
        self.lcs_median_filter_edit.setText("")
        self.umpaNw_edit.setText("")
        self.XSVT_Nw_edit.setText("")
        self.XSVT_median_filter_edit.setText("")
        self.MIST_median_filter_edit.setText("")
        self.sigma_regularization_edit.setText("0")
        self.proj_to_treat_start_edit.setText("0")
        self.proj_to_treat_edit.setText("1")
               
           
class widget_method(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutRecup = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutRecup )
        
        self.method_1_name=QLabel("LCS")
        self.layoutRecup.addWidget(self.method_1_name, 0, 0)
        self.method_1_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_1_check, 0, 1)
        
        self.method_2_name=QLabel("LCS_DF")
        self.layoutRecup.addWidget(self.method_2_name, 1, 0)
        self.method_2_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_2_check, 1, 1)
        
        self.method_3_name=QLabel("MISTII_2")
        self.layoutRecup.addWidget(self.method_3_name, 2, 0)
        self.method_3_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_3_check, 2, 1)
        
        self.method_4_name=QLabel("MISTII_1")
        self.layoutRecup.addWidget(self.method_4_name, 3, 0)
        self.method_4_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_4_check, 3, 1)
        
        self.method_5_name=QLabel("MISTI")
        self.layoutRecup.addWidget(self.method_5_name, 4, 0)
        self.method_5_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_5_check, 4, 1)
        
        self.method_6_name=QLabel("UMPA")
        self.layoutRecup.addWidget(self.method_6_name, 5, 0)
        self.method_6_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_6_check, 5, 1)
        
        self.method_7_name=QLabel("OF")
        self.layoutRecup.addWidget(self.method_7_name, 6, 0)
        self.method_7_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_7_check, 6, 1)
        
        self.method_8_name=QLabel("Pavlov")
        self.layoutRecup.addWidget(self.method_8_name, 7, 0)
        self.method_8_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_8_check, 7, 1)
        
        self.method_9_name=QLabel("XSVT")
        self.layoutRecup.addWidget(self.method_9_name, 8, 0)
        self.method_9_check=QCheckBox()
        self.layoutRecup.addWidget(self.method_9_check, 8, 1)
    def reset(self):
        """
        reset (unchecked) the differents methods
        """
        self.method_1_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_2_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_3_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_4_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_5_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_6_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_7_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_8_check.setCheckState(Qt.CheckState.Unchecked)
        self.method_9_check.setCheckState(Qt.CheckState.Unchecked)
         
         
         
class recup_phase_thread(QThread):
    """
    thread for the start_exec
    """
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

    def run(self):
        
        self.gui.number_execution=self.gui.number_execution+1
        self.load=load_thread(self.gui)
        self.load.start()
        self.gui.start_exec()
        self.gui.number_execution=self.gui.number_execution-1
        
class load_thread(QThread):
    """
    thread for the load bar
    """
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

    def run(self):
        i=0
        while self.gui.number_execution>0:
            i=(i+1)%11
            self.gui.father.leftWidget.prog_bar.setValue(i*10)
            time.sleep(0.1)
        self.gui.father.leftWidget.prog_bar.setValue(0)
