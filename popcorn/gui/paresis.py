import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from PyQt6.QtCharts import *
import os
from pathlib import Path
path_root = Path(__file__).parents[3]
path_root = str(path_root)+"/PARESIS-master/CodePython/"
if str(path_root) not in sys.path :
    print(str(path_root))
    sys.path.append(str(path_root))
import datetime
import xml.etree.cElementTree as ET
import time
import importlib



class Paresis(QWidget):
    """
    class which contains the different elements needed for Paresis
    """
    def __init__(self, father):
        """
        init and create button/window
        Args:
            father: QMainWindow from GUI_popcorn
        """
        super().__init__()
        self.father = father
        self.layoutParesis = QGridLayout()  # layout en grille
        self.setLayout(self.layoutParesis)
        self.number_execution=0

        self.load_button = QPushButton("Load")
        self.layoutParesis.addWidget(self.load_button,0,0,1,4)

        self.name_text = QLabel("Experiment name")
        self.layoutParesis.addWidget(self.name_text, 1, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_exp_name)
        self.layoutParesis.addWidget(self.name_value, 1, 1)

        self.output_text = QPushButton("Output filepath")
        self.output_text.clicked.connect(self.change_output)
        self.layoutParesis.addWidget(self.output_text, 1, 2)

        self.output_value = QLineEdit("")
        self.layoutParesis.addWidget(self.output_value, 1, 3)

        self.overSampling_text = QLabel("overSampling")
        self.layoutParesis.addWidget(self.overSampling_text, 2, 0)

        self.overSampling_value = QLineEdit("")
        self.layoutParesis.addWidget(self.overSampling_value, 2, 1)

        self.nbExpPoints_text = QLabel("nbExpPoints")
        self.layoutParesis.addWidget(self.nbExpPoints_text, 2, 2)

        self.nbExpPoints_value = QLineEdit("")
        self.layoutParesis.addWidget(self.nbExpPoints_value, 2, 3)

        self.format_text = QLabel("Output format")
        self.layoutParesis.addWidget(self.format_text, 3, 0)

        self.format_value = QComboBox()
        self.format_value.addItem("tif")
        self.format_value.addItem("edf")
        self.layoutParesis.addWidget(self.format_value, 3, 1)

        self.simulation_type_text = QLabel("simulation_type")
        self.layoutParesis.addWidget(self.simulation_type_text, 3, 2)

        self.simulation_type_value = QComboBox()
        self.simulation_type_value.addItem("RayT")
        self.simulation_type_value.addItem("Fresnel")
        self.layoutParesis.addWidget(self.simulation_type_value, 3, 3)

        self.experiment_button = QPushButton("Experiment")
        self.layoutParesis.addWidget(self.experiment_button,4,0,1,2)

        self.detector_button = QPushButton("Detector")
        self.layoutParesis.addWidget(self.detector_button,4,2,1,2)

        self.samples_button = QPushButton("Samples")
        self.layoutParesis.addWidget(self.samples_button,5,0,1,2)

        self.membranes_button = QPushButton("Membrane")
        self.layoutParesis.addWidget(self.membranes_button, 5, 2, 1, 2)

        self.sources_button = QPushButton("Sources")
        self.layoutParesis.addWidget(self.sources_button,6,0,1,2)

        self.reset_button = QPushButton("Reset all")
        self.reset_button.clicked.connect(self.reset_all)
        self.layoutParesis.addWidget(self.reset_button,7,0,1,4)

        self.add_list_button = QPushButton("Add list")
        self.add_list_button.clicked.connect(self.add_list)
        self.layoutParesis.addWidget(self.add_list_button,9,0,1,4)

        self.list_exp=QTextEdit()
        self.layoutParesis.addWidget(self.list_exp, 10, 0, 1, 4)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.layoutParesis.addWidget(self.start_button, 11, 0, 1, 4)

        self.experiment_window = experiment(self)
        self.experiment_button.clicked.connect(self.display_experiment)
        self.load_button.clicked.connect(self.experiment_window.load)

        self.detector_window = detector(self)
        self.detector_button.clicked.connect(self.display_detector)

        self.samples_window = samples(self)
        self.samples_button.clicked.connect(self.display_samples)

        self.membranes_window = membranes(self)
        self.membranes_button.clicked.connect(self.display_membranes)

        self.sources_window = sources(self)
        self.sources_button.clicked.connect(self.display_sources)
    def add_list(self):
        """
        add the parameter to the QTextEdit
        Returns:
            void
        """
        self.experiment_window.write_xml()
        self.detector_window.write_xml()
        self.samples_window.write_xml()
        self.membranes_window.write_xml()
        self.sources_window.write_xml()

        t = self.list_exp.toPlainText()
        t = t+self.name_value.text()+','\
            + self.output_value.text()+',' \
            + self.overSampling_value.text() + ',' \
            + self.nbExpPoints_value.text()+','\
            + self.format_value.currentText()+','\
            + self.simulation_type_value.currentText()+'\n'
        self.list_exp.setText(t)
    def change_exp_name(self):
        """
        if experiment is chenged, then change self.experiment_window.name_value wit the same name
        Returns:

        """
        self.experiment_window.name_value.setText(self.name_value.text())

    def change_output(self):
        """
                choose a directory thank to QDialog and write him in a QLineEdit (self.output_value)
        """
        path = QFileDialog.getExistingDirectory(self, str("Choose a Directory"))
        if path != "":
            self.output_value.setText(path + '/')
            print(self.output_value.text())

    def reset_all(self):
        """
        reset all parameters from this class and from all this windows (experiment/detector/sample/membrane/source)
        Returns:
            void
        """
        self.name_value.setText("")
        self.output_value.setText("")
        self.nbExpPoints_value.setText("")
        self.overSampling_value.setText("")
        self.format_value.setCurrentIndex(0)
        self.simulation_type_value.setCurrentIndex(0)
        self.experiment_window.reset()
        self.detector_window.reset()
        self.samples_window.reset()
        self.sources_window.reset()

    def display_experiment(self):
        """
        hide then shw experiment window
        Returns:

        """
        self.experiment_window.hide()
        self.experiment_window.show()

    def display_detector(self):
        """
        hide then shw detector window
        Returns:

        """
        self.detector_window.hide()
        self.detector_window.show()

    def display_samples(self):
        """
        hide then shw sample window
        Returns:

        """
        self.samples_window.hide()
        self.samples_window.show()

    def display_membranes(self):
        """
        hide then shw membrane window
        Returns:

        """
        self.membranes_window.hide()
        self.membranes_window.show()

    def display_sources(self):
        """
        hide then shw source window
        Returns:

        """
        self.sources_window.hide()
        self.sources_window.show()

    def start(self):
        """
        start paresis in a thread
        Returns:

        """
        thread = start_thread(self)
        thread.run()

    def start_exec(self):
        """
        Read parameters from QTextEdit and do paresis
        """




        text = self.list_exp.toPlainText()
        text = text.split("\n")
        for line in text:
            list_para = line.split(",")
            if len(list_para) != 6:
                print("pas assez de para")
                break
            print(list_para)
            self.one_exec_start(list_para)
    def one_exec_start(self,list_para):
        """
        paresis main
        Args:
            list_para: contain [experiment_name,outputfile,oversampling,nbExpPoints,saving_format,simulation_type]

        Returns:

        """

        Experiment = importlib.import_module('Experiment')
        Experiment = Experiment.Experiment
        pagailleIO = importlib.import_module('InputOutput.pagailleIO')
        save_image = pagailleIO.save_image

        time0 = time.time()  # timer for computation
        exp_dict = {}

        ## PARAMETERS TO SET
        # Define experiment
        exp_dict['experimentName'] = list_para[0]
        # Output filepath to store the result images
        exp_dict['filepath'] = list_para[1]
        exp_dict['fileN'] = "nylon"
        # Define algorithm parameters
        exp_dict[
            'overSampling'] = int(list_para[2])  # MUST BE AN INTEGER - >2 for ray-tracing model and even more for Fresnel (cf usefullScripts/getSamplingFactor.py)
        exp_dict[
            'nbExpPoints'] = int(list_para[3])  # number of pair of acquisitions (Ir, Is) simulated with different positions of the membrane
        save = True
        saving_format = list_para[4] # .tif or .edf
        exp_dict['simulation_type'] = list_para[5]  # "Fresnel" or "RayT"

        # ************************************************************************
        # **********START OF CALCULATIONS*****************************************
        # ************************************************************************

        now = datetime.datetime.now()
        exp_dict['expID'] = now.strftime("_")  # define experiment ID

        SampleImage = []
        ReferenceImage = []
        PropagImage = []
        AbsImage = []
        SubImage = []
        Geometry = []

        print("\n\nINITIALIZING EXPERIMENT PARAMETERS AND GEOMETRIES")
        print("*************************")
        experiment = Experiment(exp_dict)

        # save_image(experiment.mySampleofInterest.myGeometry[0],r"C:\Users\clara.magnin\OneDrive - XENOCS\Documents\Simulations\Simulations_Clara\20220601Algo_comparison\nylonGeometry.tif" )
        print("\nImages calculation")
        print("*************************")
        for pointNum in range(exp_dict['nbExpPoints']):
            experiment.myMembrane.myGeometry = []
            experiment.myMembrane.getMyGeometry(experiment.exp_dict['studyDimensions'],
                                                experiment.myMembrane.membranePixelSize,
                                                experiment.exp_dict['overSampling'], pointNum, exp_dict['nbExpPoints'])

            print("\nCalculations point", pointNum)
            if exp_dict['simulation_type'] == "Fresnel":
                SampleImageTmp, ReferenceImageTmp, PropagImageTmp, White = experiment.computeSampleAndReferenceImages_Fresnel(
                    pointNum)
            elif exp_dict['simulation_type'] == "RayT":
                SampleImageTmp, ReferenceImageTmp, PropagImageTmp, White, Dx, Dy, Df = experiment.computeSampleAndReferenceImages_RT(
                    pointNum)
            else:
                raise Exception("simulation Type not defined: ", exp_dict['simulation_type'])
            Nbin = len(SampleImageTmp)

            if pointNum == 0:
                expPathEn = []
                if exp_dict['simulation_type'] == "Fresnel":
                    expImagesFilePath = exp_dict['filepath'] + 'Fresnel_' + str(exp_dict['expID']) + '/'
                if exp_dict['simulation_type'] == "RayT":
                    expImagesFilePath = exp_dict['filepath'] + 'result/'
                os.mkdir(expImagesFilePath)
                os.mkdir(expImagesFilePath + 'membraneThickness/')
                thresholds = experiment.myDetector.det_param['myBinsThersholds'].copy()
                thresholds.insert(0, experiment.mySource.mySpectrum[0][0])
                for ibin in range(Nbin):
                    binstart = '%2.2d' % thresholds[ibin]
                    binend = '%2.2d' % thresholds[ibin + 1]
                    expPathEn.append(f'{expImagesFilePath}{binstart}_{binend}kev/')
                    if len(thresholds) - 1 == 1:
                        expPathEn = [expImagesFilePath]
                    else:
                        os.mkdir(expPathEn[ibin])
                    os.mkdir(expPathEn[ibin] + 'ref/')
                    os.mkdir(expPathEn[ibin] + 'sample/')
                    os.mkdir(expPathEn[ibin] + 'propag/')

            txtPoint = '%2.2d' % pointNum
            print(experiment.myMembrane.myGeometry[0])
            save_image(experiment.myMembrane.myGeometry[0],
                       expImagesFilePath + 'membraneThickness/' + exp_dict['experimentName'] + '_sampling' + str(
                           exp_dict['overSampling']) + '_' + str(pointNum) + "." + saving_format)

            if exp_dict['simulation_type'] == "RayT":
                save_image(Df, expImagesFilePath + "DF" + "." + saving_format)

            for ibin in range(Nbin):
                save_image(SampleImageTmp[ibin], expPathEn[ibin] + 'sample/sampleImage_' + str(
                    exp_dict['expID']) + '_' + txtPoint + "." + saving_format)
                save_image(ReferenceImageTmp[ibin], expPathEn[ibin] + 'ref/ReferenceImage_' + str(
                    exp_dict['expID']) + '_' + txtPoint + "." + saving_format)

                if pointNum == 0:
                    save_image(PropagImageTmp[ibin],
                               expPathEn[ibin] + 'propag/PropagImage_' + str(exp_dict['expID']) + '_' + "." + saving_format)
                    save_image(White[ibin], expPathEn[ibin] + 'White_' + str(exp_dict['expID']) + '_' + "." + saving_format)

        experiment.saveAllParameters(time0, exp_dict)

        print("\nfini")



class experiment(QWidget):
    """
    experiment window
    """
    def __init__(self, father):
        """
        init the window
        Args:
            father: QWindow (class Paresis)
        """
        super().__init__()
        self.father = father
        self.layoutExperiment = QGridLayout()  # layout en grille
        self.setLayout(self.layoutExperiment)

        self.name_text = QLabel("name")
        self.layoutExperiment.addWidget(self.name_text, 0, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_exp_name)
        self.layoutExperiment.addWidget(self.name_value, 0, 1)

        self.distSourceToMembrane_text = QLabel("distSourceToMembrane(m)")
        self.layoutExperiment.addWidget(self.distSourceToMembrane_text,1,0)

        self.distSourceToMembrane_value = QLineEdit("")
        self.layoutExperiment.addWidget(self.distSourceToMembrane_value,1, 1)

        self.distMembraneToObject_text = QLabel("distMembraneToObject(m)")
        self.layoutExperiment.addWidget(self.distMembraneToObject_text, 2, 0)

        self.distMembraneToObject_value = QLineEdit("")
        self.layoutExperiment.addWidget(self.distMembraneToObject_value, 2, 1)

        self.distObjectToDetector_text = QLabel("distObjectToDetector(m)")
        self.layoutExperiment.addWidget(self.distObjectToDetector_text, 3, 0)

        self.distObjectToDetector_value = QLineEdit("")
        self.layoutExperiment.addWidget(self.distObjectToDetector_value, 3, 1)

        self.membraneName_text = QLabel("membraneName")
        self.layoutExperiment.addWidget(self.membraneName_text, 4, 0)

        self.membraneName_value = QLineEdit("")
        self.membraneName_value.textChanged.connect(self.change_membrane_name)
        self.layoutExperiment.addWidget(self.membraneName_value, 4, 1)

        self.sampleName_text = QLabel("sampleName")
        self.layoutExperiment.addWidget(self.sampleName_text, 5, 0)

        self.sampleName_value = QLineEdit("")
        self.sampleName_value.textChanged.connect(self.change_sample_name)
        self.layoutExperiment.addWidget(self.sampleName_value, 5, 1)

        self.sampleType_text = QLabel("sampleType")
        self.layoutExperiment.addWidget(self.sampleType_text, 6, 0)

        self.sampleType_value = QLineEdit("")
        self.layoutExperiment.addWidget(self.sampleType_value, 6, 1)

        self.detectorName_text = QLabel("detectorName")
        self.layoutExperiment.addWidget(self.detectorName_text, 7, 0)

        self.detectorName_value = QLineEdit("")
        self.detectorName_value.textChanged.connect(self.change_detector_name)
        self.layoutExperiment.addWidget(self.detectorName_value, 7, 1)

        self.sourceName_text = QLabel("sourceName")
        self.layoutExperiment.addWidget(self.sourceName_text, 8, 0)

        self.sourceName_value = QLineEdit("")
        self.sourceName_value.textChanged.connect(self.change_source_name)
        self.layoutExperiment.addWidget(self.sourceName_value, 8, 1)

        self.meanShotCount_text = QLabel("meanShotCount")
        self.layoutExperiment.addWidget(self.meanShotCount_text, 9, 0)

        self.meanShotCount_value = QLineEdit("")
        self.layoutExperiment.addWidget(self.meanShotCount_value, 9, 1)

        self.inVacuum_text = QLabel("inVacuum")
        self.layoutExperiment.addWidget(self.inVacuum_text, 10, 0)

        self.inVacuum_value = QCheckBox()
        self.layoutExperiment.addWidget(self.inVacuum_value, 10, 1)

        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        self.layoutExperiment.addWidget(self.reset_button, 11, 0,1,2)

        self.load_button = QPushButton("load")
        self.load_button.clicked.connect(self.load)
        self.layoutExperiment.addWidget(self.load_button, 12, 0, 1, 2)

        self.write_xml_button = QPushButton("write_xml")
        self.write_xml_button.clicked.connect(self.write_xml)
        self.layoutExperiment.addWidget(self.write_xml_button, 13, 0, 1, 2)

    def change_exp_name(self):
        """
        if experiment name is changed, change it in class paresis too
        Returns:

        """
        self.father.name_value.setText(self.name_value.text())

    def change_detector_name(self):
        """
                if detector name is changed, change it in class detector too
                Returns:

                """
        self.father.detector_window.name_value.setText(self.detectorName_value.text())

    def change_sample_name(self):
        """
                if sample name is changed, change it in class samples too
                Returns:

                """
        self.father.samples_window.name_value.setText(self.sampleName_value.text())
    def change_membrane_name(self):
        """
                if membrane name is changed, change it in class membrane too
                Returns:

                """
        self.father.membranes_window.name_value.setText(self.membraneName_value.text())
    def change_source_name(self):
        """
        if source name is changed, change it in class source too
        Returns:

        """
        self.father.sources_window.name_value.setText(self.sourceName_value.text())

    def reset(self):
        """
         reset all window's parameters

        """
        self.distSourceToMembrane_value.setText("")
        self.distMembraneToObject_value.setText("")
        self.distObjectToDetector_value.setText("")
        self.membraneName_value.setText("")
        self.sampleName_value.setText("")
        self.sampleType_value.setText("")
        self.detectorName_value.setText("")
        self.sourceName_value.setText("")
        self.meanShotCount_value.setText("")
        self.inVacuum_value.setChecked(False)

    def load(self):
        """
        load data from xmlFiles/Experiment.xml
        firstly scroll only experiment name then the user choose wich one open and then load it with load_exec
        Args:
            self:

        Returns:

        """
        items = []
        f = open("xmlFiles/Experiment.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('experiment'):
            items.append(experience[0].text)

        exp_name, booleen = QInputDialog.getItem(self, "Load experiment", "enter experiment name",items)
        if not booleen:
            return
        self.father.experiment_window.reset()
        self.father.detector_window.reset()
        self.father.samples_window.reset()
        self.father.membranes_window.reset()
        self.father.sources_window.reset()
        self.load_exec(exp_name)

    def load_exec(self, exp_name):
        """
        Read a xml and load parameters of experiment with <name>=exp_name
        """

        f = open("xmlFiles/Experiment.xml",'r')
        texte    = f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    self.change_value(name,value)
        f.close()
        self.father.detector_window.load_exec(self.detectorName_value.text())
        self.father.samples_window.load_exec(self.sampleName_value.text())
        self.father.membranes_window.load_exec(self.membraneName_value.text())
        self.father.sources_window.load_exec(self.sourceName_value.text())

    def change_value(self, name, value):
        """
        switch name and change the value of the QWidget correspond

        """
        if name == "name":
            self.name_value.setText(value)
        elif name == "distSourceToMembrane":
            self.distSourceToMembrane_value.setText(value)
        elif name == "distMembraneToObject":
            self.distMembraneToObject_value.setText(value)
        elif name == "distObjectToDetector":
            self.distObjectToDetector_value.setText(value)
        elif name == "membraneName":
            self.membraneName_value.setText(value)
        elif name == "sampleName":
            self.sampleName_value.setText(value)
        elif name == "sampleType":
            self.sampleType_value.setText(value)
        elif name == "detectorName":
            self.detectorName_value.setText(value)
        elif name == "sourceName":
            self.sourceName_value.setText(value)
        elif name == "meanShotCount":
            self.meanShotCount_value.setText(value)
        elif name == "inVacuum":
            if value == "True":
                self.inVacuum_value.setCheckState(Qt.CheckState.Checked)
            else:
                self.inVacuum_value.setCheckState(Qt.CheckState.Unchecked)

    def write_xml(self):
        """
        write parameters in Experiment.xml
        """
        f = open("xmlFiles/Experiment.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('experiment'):
            if experience[0].text == self.name_value.text():
                root.remove(experience)

        doc = ET.SubElement(root, "experiment")
        ET.SubElement(doc, "name").text = self.name_value.text()
        ET.SubElement(doc, "distSourceToMembrane").text = self.distSourceToMembrane_value.text()
        ET.SubElement(doc, "distMembraneToObject").text = self.distMembraneToObject_value.text()
        ET.SubElement(doc, "distObjectToDetector").text = self.distObjectToDetector_value.text()
        ET.SubElement(doc, "membraneName").text = self.membraneName_value.text()
        ET.SubElement(doc, "sampleName").text = self.sampleName_value.text()
        ET.SubElement(doc, "sampleType").text = self.sampleType_value.text()
        ET.SubElement(doc, "detectorName").text = self.detectorName_value.text()
        ET.SubElement(doc, "sourceName").text = self.sourceName_value.text()
        ET.SubElement(doc, "meanShotCount").text = self.meanShotCount_value.text()
        ET.SubElement(doc, "inVacuum").text = str(self.inVacuum_value.isChecked())

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("xmlFiles/Experiment.xml", encoding='utf-8')

class detector(QWidget):
    def __init__(self, father):
        super().__init__()
        self.father = father
        self.layoutDetector = QGridLayout()  # layout en grille
        self.setLayout(self.layoutDetector)

        self.name_text = QLabel("name")
        self.layoutDetector.addWidget(self.name_text, 1, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_detector_name)
        self.layoutDetector.addWidget(self.name_value, 1, 1)

        self.dimX_text = QLabel("dimX")
        self.layoutDetector.addWidget(self.dimX_text, 2, 0)

        self.dimX_value = QLineEdit("")
        self.layoutDetector.addWidget(self.dimX_value, 2, 1)

        self.dimY_text = QLabel("dimY")
        self.layoutDetector.addWidget(self.dimY_text, 3, 0)

        self.dimY_value = QLineEdit("")
        self.layoutDetector.addWidget(self.dimY_value, 3, 1)

        self.photonCounting_text = QLabel("photonCounting")
        self.layoutDetector.addWidget(self.photonCounting_text, 4, 0)

        self.photonCounting_value = QCheckBox()
        self.layoutDetector.addWidget(self.photonCounting_value, 4, 1)

        self.myPixelSize_text = QLabel("myPixelSize (um)")
        self.layoutDetector.addWidget(self.myPixelSize_text, 5, 0)

        self.myPixelSize_value = QLineEdit("")
        self.layoutDetector.addWidget(self.myPixelSize_value, 5, 1)

        self.myPSF_text = QLabel("myPSF (pixel)")
        self.layoutDetector.addWidget(self.myPSF_text, 6, 0)

        self.myPSF_value = QLineEdit("")
        self.layoutDetector.addWidget(self.myPSF_value, 6, 1)

        self.myScintillatorMaterial_text = QLabel("myScintillatorMaterial")
        self.layoutDetector.addWidget(self.myScintillatorMaterial_text, 7, 0)

        self.myScintillatorMaterial_value = QLineEdit("")
        self.layoutDetector.addWidget(self.myScintillatorMaterial_value, 7, 1)

        self.myScintillatorThickness_text = QLabel("myScintillatorThickness")
        self.layoutDetector.addWidget(self.myScintillatorThickness_text, 8, 0)

        self.myScintillatorThickness_value = QLineEdit("")
        self.layoutDetector.addWidget(self.myScintillatorThickness_value, 8, 1)

        self.myBinsThersholds_text = QLabel("myBinsThersholds (keV)")
        self.layoutDetector.addWidget(self.myBinsThersholds_text, 9, 0)

        self.myBinsThersholds_value = QLineEdit("")
        self.layoutDetector.addWidget(self.myBinsThersholds_value, 9, 1)

        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        self.layoutDetector.addWidget(self.reset_button, 11, 0,1,2)

        self.load_button = QPushButton("load")
        self.load_button.clicked.connect(self.load)
        self.layoutDetector.addWidget(self.load_button, 12, 0, 1, 2)

        self.write_xml_button = QPushButton("write_xml")
        self.write_xml_button.clicked.connect(self.write_xml)
        self.layoutDetector.addWidget(self.write_xml_button, 13, 0, 1, 2)

    def change_detector_name(self):
        """
        if detector name is change then change detectorName in experiment
        Returns:

        """
        self.father.experiment_window.detectorName_value.setText(self.name_value.text())

    def reset(self):
        """
        reset all detector parameters
        Returns:

        """
        self.name_value.setText("")
        self.dimX_value.setText("")
        self.dimY_value.setText("")
        self.photonCounting_value.setChecked(False)
        self.myPixelSize_value.setText("")
        self.myPSF_value.setText("")
        self.myScintillatorMaterial_value.setText("")
        self.myScintillatorThickness_value.setText("")
        self.myBinsThersholds_value.setText("")

    def load(self):
        """
        choose which detector load and lad it
        Returns:

        """
        items = []
        f = open("xmlFiles/Detectors.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('detector'):
            items.append(experience[0].text)

        exp_name, booleen = QInputDialog.getItem(self, "Load experiment", "enter experiment name", items)
        if not booleen:
            return
        self.father.detector_window.reset()

        self.load_exec(exp_name)

    def load_exec(self,exp_name):
        """
        Read a xml and load parameters
        """
        print(exp_name)
        f = open("xmlFiles/Detectors.xml",'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    if name == "myDimensions":
                        for child3 in child2:
                            name, value = child3.tag, child3.text
                            self.change_value(name, value)
                    else:
                        self.change_value(name,value)
        f.close()


    def change_value(self, name, value):
        """
        switch name and change the value of the QWidget correspond

        """
        if name == "name":
            self.name_value.setText(value)
        elif name == "myDimensions":
            print("a",value)
        elif name == "dimX":
            self.dimX_value.setText(value)
        elif name == "dimY":
            self.dimY_value.setText(value)
        elif name == "photonCounting":
            if value == "True":
                self.photonCounting_value.setCheckState(Qt.CheckState.Checked)
            else:
                self.photonCounting_value.setCheckState(Qt.CheckState.Unchecked)
        elif name == "myPixelSize":
            self.myPixelSize_value.setText(value)
        elif name == "myPSF":
            self.myPSF_value.setText(value)
        elif name == "myScintillatorMaterial":
            self.myScintillatorMaterial_value.setText(value)
        elif name == "myScintillatorThickness":
            self.myScintillatorThickness_value.setText(value)
        elif name == "myBinsThersholds":
            self.myBinsThersholds_value.setText(value)

    def write_xml(self):
        """
        write parameters in Experiment.xml
        """
        f = open("xmlFiles/Detectors.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('detector'):
            if experience[0].text == self.name_value.text():
                root.remove(experience)

        doc = ET.SubElement(root, "detector")
        ET.SubElement(doc, "name").text = self.name_value.text()
        if self.dimX_value!="" and self.dimY_value!="":
            mydoc = ET.SubElement(doc, "myDimensions")
            ET.SubElement(mydoc, "dimX").text = self.dimX_value.text()
            ET.SubElement(mydoc, "dimY").text = self.dimY_value.text()
        if self.photonCounting_value.isChecked():
            ET.SubElement(doc, "photonCounting").text = str(self.photonCounting_value.isChecked())
        if self.myPixelSize_value.text():
            ET.SubElement(doc, "myPixelSize").text = self.myPixelSize_value.text()
        if self.myPSF_value.text():
            ET.SubElement(doc, "myPSF").text = self.myPSF_value.text()
        if self.myScintillatorMaterial_value.text():
            ET.SubElement(doc, "myScintillatorMaterial").text = self.myScintillatorMaterial_value.text()
        if self.myScintillatorThickness_value.text():
            ET.SubElement(doc, "myScintillatorThickness").text = self.myScintillatorThickness_value.text()
        if self.myBinsThersholds_value.text()!="":
            print("eflkefhehehezezezjknj",self.myBinsThersholds_value.text())
            ET.SubElement(doc, "myBinsThersholds").text = self.myBinsThersholds_value.text()

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("xmlFiles/Detectors.xml", encoding='utf-8')


class samples(QWidget):
    """
    sample window
    """
    def __init__(self, father):
        """
        init window
        Args:
            father: class paresis (QWindow)
        """
        super().__init__()
        self.father = father
        self.layoutSample = QGridLayout()  # layout en grille
        self.setLayout(self.layoutSample)

        self.name_text = QLabel("name")
        self.layoutSample.addWidget(self.name_text, 1, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_sample_name)
        self.layoutSample.addWidget(self.name_value, 1, 1)

        self.myType_text = QLabel("myType")
        self.layoutSample.addWidget(self.myType_text, 2, 0)

        self.myType_value = QLineEdit("")
        self.layoutSample.addWidget(self.myType_value, 2, 1)

        self.myGeometryFunction_text = QLabel("myGeometryFunction")
        self.layoutSample.addWidget(self.myGeometryFunction_text, 3, 0)

        self.myGeometryFunction_value = QLineEdit("")
        self.layoutSample.addWidget(self.myGeometryFunction_value, 3, 1)

        self.myGeometryFolder_text = QPushButton("myGeometryFolder")
        self.myGeometryFolder_text.clicked.connect(self.chooseGeometryFolder)
        self.layoutSample.addWidget(self.myGeometryFolder_text, 4, 0)

        self.myGeometryFolder_value = QLineEdit("")
        self.layoutSample.addWidget(self.myGeometryFolder_value, 4, 1)

        self.myMaterials_text = QLabel("myMaterials")
        self.layoutSample.addWidget(self.myMaterials_text, 5, 0)

        self.myMaterials_value = QLineEdit("")
        self.layoutSample.addWidget(self.myMaterials_value, 5, 1)

        self.myRadius_text = QLabel("myRadius (um)")
        self.layoutSample.addWidget(self.myRadius_text, 8, 0)

        self.myRadius_value = QLineEdit("")
        self.layoutSample.addWidget(self.myRadius_value, 8, 1)

        self.myOrientation_text = QLabel("myOrientation (degree)")
        self.layoutSample.addWidget(self.myOrientation_text, 9, 0)

        self.myOrientation_value = QLineEdit("")
        self.layoutSample.addWidget(self.myOrientation_value, 9, 1)

        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        self.layoutSample.addWidget(self.reset_button, 10, 0, 1, 2)

        self.load_button = QPushButton("load")
        self.load_button.clicked.connect(self.load)
        self.layoutSample.addWidget(self.load_button, 12, 0, 1, 2)

        self.write_xml_button = QPushButton("write_xml")
        self.write_xml_button.clicked.connect(self.write_xml)
        self.layoutSample.addWidget(self.write_xml_button, 13, 0, 1, 2)
    def chooseGeometryFolder(self):
        """
        choose a directory thank to QDialog and write him in a QLineEdit (self.myGeometryFolder_value)
        """
        path = QFileDialog.getExistingDirectory(self, str("Choose a Directory"))
        if path != "":
            self.myGeometryFolder_value.setText(path + '/')
            print(self.myGeometryFolder_value.text())
    def change_sample_name(self):
        """
        il sample name change then change it in experiment too
        Returns:

        """
        self.father.experiment_window.sampleName_value.setText(self.name_value.text())

    def reset(self):
        """
        reset all parameters
        Returns:

        """
        self.name_value.setText("")
        self.myType_value.setText("")
        self.myGeometryFunction_value.setText("")
        self.myGeometryFolder_value.setText("")
        self.myMaterials_value.setText("")
        self.myRadius_value.setText("")
        self.myOrientation_value.setText("")

    def load(self):
        """
        choose a sample and load it
        Returns:

        """
        items = []
        f = open("xmlFiles/Samples.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('sample'):
            items.append(experience[0].text)

        exp_name, booleen = QInputDialog.getItem(self, "Load experiment", "enter experiment name", items)
        if not booleen:
            return
        self.father.samples_window.reset()
        self.load_exec(exp_name)
    def load_exec(self,exp_name):
        """
        Read a xml and load parameters
        """
        f = open("xmlFiles/Samples.xml",'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    if name == "myDimensions":
                        for child3 in child2:
                            name, value = child3.tag, child3.text
                            self.change_value(name, value)
                    else:
                        self.change_value(name,value)
        f.close()


    def change_value(self, name, value):
        """
        switch name and change the value of the QWidget correspond

        """
        if name == "name":
            self.name_value.setText(value)
        elif name == "myType":
            self.myType_value.setText(value)
        elif name == "myGeometryFunction":
            self.myGeometryFunction_value.setText(value)
        elif name == "myGeometryFolder":
            self.myGeometryFolder_value.setText(value)
        elif name == "myMaterials":
            self.myMaterials_value.setText(value)
        elif name == "myRadius":
            self.myRadius_value.setText(value)
        elif name == "myOrientation":
            self.myOrientation_value.setText(value)

    def write_xml(self):
        """
        write parameters in Experiment.xml
        """
        f = open("xmlFiles/Samples.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('sample'):
            if experience[0].text == self.name_value.text():
                root.remove(experience)

        doc = ET.SubElement(root, "sample")
        ET.SubElement(doc, "name").text = self.name_value.text()
        if self.myType_value.text()!="":
            ET.SubElement(doc, "myType").text = self.myType_value.text()
        if self.myGeometryFunction_value.text() != "":
            ET.SubElement(doc, "myGeometryFunction").text = self.myGeometryFunction_value.text()
        if self.myGeometryFolder_value.text() != "":
            ET.SubElement(doc, "myGeometryFolder").text = self.myGeometryFolder_value.text()
        if self.myMaterials_value.text() != "":
            ET.SubElement(doc, "myMaterials").text = self.myMaterials_value.text()
        if self.myRadius_value.text() != "":
            ET.SubElement(doc, "myRadius").text = self.myRadius_value.text()
        if self.myOrientation_value.text() != "":
            ET.SubElement(doc, "myOrientation").text = self.myOrientation_value.text()

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("xmlFiles/Samples.xml", encoding='utf-8')
class membranes(QWidget):
    """
    membrane window
    """
    def __init__(self, father):
        """
        init and create button
        Args:
            father: class paresis (QWidget)
        """
        super().__init__()
        self.father = father
        self.layoutMembranes = QGridLayout()  # layout en grille
        self.setLayout(self.layoutMembranes)

        self.name_text = QLabel("name")
        self.layoutMembranes.addWidget(self.name_text, 1, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_membrane_name)
        self.layoutMembranes.addWidget(self.name_value, 1, 1)

        self.myType_text = QLabel("myType")
        self.layoutMembranes.addWidget(self.myType_text, 2, 0)

        self.myType_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myType_value, 2, 1)

        self.myGeometryFunction_text = QLabel("myGeometryFunction")
        self.layoutMembranes.addWidget(self.myGeometryFunction_text, 3, 0)

        self.myGeometryFunction_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myGeometryFunction_value, 3, 1)

        self.myPMMAThickness_text = QLabel("myPMMAThickness (um)")
        self.layoutMembranes.addWidget(self.myPMMAThickness_text, 4, 0)

        self.myPMMAThickness_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myPMMAThickness_value, 4, 1)

        self.myMaterials_text = QLabel("myMaterials")
        self.layoutMembranes.addWidget(self.myMaterials_text, 5, 0)

        self.myMaterials_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myMaterials_value, 5, 1)

        self.myMeanSphereRadius_text = QLabel("myMeanSphereRadius")
        self.layoutMembranes.addWidget(self.myMaterials_text, 6, 0)

        self.myMeanSphereRadius_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myMaterials_value, 6, 1)

        self.myNbOfLayers_text = QLabel("myNbOfLayers")
        self.layoutMembranes.addWidget(self.myNbOfLayers_text, 7, 0)

        self.myNbOfLayers_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myNbOfLayers_value, 7, 1)

        self.myMembraneFile_text = QPushButton("myMembraneFile")
        self.myMembraneFile_text.clicked.connect(self.chooseMembraneFile)
        self.layoutMembranes.addWidget(self.myMembraneFile_text, 8, 0)

        self.myMembraneFile_value = QLineEdit("")
        self.layoutMembranes.addWidget(self.myMembraneFile_value, 8, 1)

        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        self.layoutMembranes.addWidget(self.reset_button, 10, 0, 1, 2)

        self.load_button = QPushButton("load")
        self.load_button.clicked.connect(self.load)
        self.layoutMembranes.addWidget(self.load_button, 12, 0, 1, 2)

        self.write_xml_button = QPushButton("write_xml")
        self.write_xml_button.clicked.connect(self.write_xml)
        self.layoutMembranes.addWidget(self.write_xml_button, 13, 0, 1, 2)

    def chooseMembraneFile(self):
        """
        choose a directory thank to QDialog and write him in a QLineEdit (self.myMembraneFile_value)
        """
        path = QFileDialog.getExistingDirectory(self, str("Choose a Directory"))
        if path != "":
            self.myMembraneFile_value.setText(path + '/')
            print(self.myMembraneFile_value.text())
    def change_membrane_name(self):
        """
        if name change, change membraneName from experiment window too
        Returns:

        """
        self.father.experiment_window.membraneName_value.setText(self.name_value.text())
    def reset(self):
        """
        reset parameters
        Returns:

        """
        self.name_value.setText("")
        self.myType_value.setText("")
        self.myGeometryFunction_value.setText("")
        self.myPMMAThickness_value.setText("")
        self.myMaterials_value.setText("")
        self.myMeanSphereRadius_value.setText("")
        self.myNbOfLayers_value.setText("")
        self.myMembraneFile_value.setText("")

    def load(self):
        """

        Returns:

        """
        items = []
        f = open("xmlFiles/Samples.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('sample'):
            items.append(experience[0].text)

        exp_name, booleen = QInputDialog.getItem(self, "Load experiment", "enter experiment name", items)

        if not booleen:
            return
        self.father.membranes_window.reset()
        self.load_exec(exp_name)
    def load_exec(self,exp_name):
        """
        Read a xml and load parameters
        """
        f = open("xmlFiles/Samples.xml",'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    if name == "myDimensions":
                        for child3 in child2:
                            name, value = child3.tag, child3.text
                            self.change_value(name, value)
                    else:
                        self.change_value(name,value)
        f.close()


    def change_value(self, name, value):
        """
        switch name and change the value of the QWidget correspond

        """
        if name == "name":
            self.name_value.setText(value)
        elif name == "myType":
            self.myType_value.setText(value)
        elif name == "myGeometryFunction":
            self.myGeometryFunction_value.setText(value)
        elif name == "myPMMAThickness":
            self.myPMMAThickness_value.setText(value)
        elif name == "myMaterials":
            self.myMaterials_value.setText(value)
        elif name == "myMeanSphereRadius":
            self.myMeanSphereRadius_value.setText(value)
        elif name == "myNbOfLayers":
            self.myNbOfLayers_value.setText(value)
        elif name == "myMembraneFile":
            self.myMembraneFile_value.setText(value)


    def write_xml(self):
        """
        write parameters in Experiment.xml
        """
        f = open("xmlFiles/Samples.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('sample'):
            if experience[0].text == self.name_value.text():
                root.remove(experience)

        doc = ET.SubElement(root, "sample")
        ET.SubElement(doc, "name").text = self.name_value.text()
        if self.myType_value.text()!="":
            ET.SubElement(doc, "myType").text = self.myType_value.text()
        if self.myGeometryFunction_value.text() != "":
            ET.SubElement(doc, "myGeometryFunction").text = self.myGeometryFunction_value.text()
        if self.myPMMAThickness_value.text() != "":
            ET.SubElement(doc, "myPMMAThickness").text = self.myPMMAThickness_value.text()
        if self.myMaterials_value.text() != "":
            ET.SubElement(doc, "myMaterials").text = self.myMaterials_value.text()
        if self.myMeanSphereRadius_value.text() != "":
            ET.SubElement(doc, "myMeanSphereRadius").text = self.myMeanSphereRadius_value.text()
        if self.myNbOfLayers_value.text() != "":
            ET.SubElement(doc, "myNbOfLayers").text = self.myNbOfLayers_value.text()
        if self.myMembraneFile_value.text() != "":
            ET.SubElement(doc, "myMembraneFile").text = self.myMembraneFile_value.text()
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("xmlFiles/Samples.xml", encoding='utf-8')

class sources(QWidget):
    """
    source window
    """
    def __init__(self, father):
        super().__init__()
        self.father = father
        self.layoutSources = QGridLayout()  # layout en grille
        self.setLayout(self.layoutSources)

        self.name_text = QLabel("name")
        self.layoutSources.addWidget(self.name_text, 1, 0)

        self.name_value = QLineEdit("")
        self.name_value.textChanged.connect(self.change_source_name)
        self.layoutSources.addWidget(self.name_value, 1, 1)

        self.myType_text = QLabel("myType")
        self.layoutSources.addWidget(self.myType_text, 2, 0)

        self.myType_value = QLineEdit("")
        self.layoutSources.addWidget(self.myType_value, 2, 1)

        self.mySize_text = QLabel("mySize")
        self.layoutSources.addWidget(self.mySize_text, 3, 0)

        self.mySize_value = QLineEdit("")
        self.layoutSources.addWidget(self.mySize_value, 3, 1)
        
        self.sourceVoltage_text = QLabel("sourceVoltage")
        self.layoutSources.addWidget(self.sourceVoltage_text, 4, 0)

        self.sourceVoltage_value = QLineEdit("")
        self.layoutSources.addWidget(self.sourceVoltage_value, 4, 1)

        self.filterMaterial_text = QLabel("filterMaterial")
        self.layoutSources.addWidget(self.filterMaterial_text, 5, 0)

        self.filterMaterial_value = QLineEdit("")
        self.layoutSources.addWidget(self.filterMaterial_value, 5, 1)

        self.filterThickness_text = QLabel("filterThickness")
        self.layoutSources.addWidget(self.filterThickness_text, 6, 0)

        self.filterThickness_value = QLineEdit("")
        self.layoutSources.addWidget(self.filterThickness_value, 6, 1)

        self.myEnergySampling_text = QLabel("myEnergySampling")
        self.layoutSources.addWidget(self.myEnergySampling_text, 7, 0)

        self.myEnergySampling_value = QLineEdit("")
        self.layoutSources.addWidget(self.myEnergySampling_value, 7, 1)

        self.myTargetMaterial_text = QLabel("myTargetMaterial")
        self.layoutSources.addWidget(self.myTargetMaterial_text, 8, 0)

        self.myTargetMaterial_value = QLineEdit("")
        self.layoutSources.addWidget(self.myTargetMaterial_value, 8, 1)

        self.spectrumFromXls_text = QLabel("spectrumFromXls")
        self.layoutSources.addWidget(self.spectrumFromXls_text, 9, 0)

        self.spectrumFromXls_value = QLineEdit("")
        self.layoutSources.addWidget(self.spectrumFromXls_value, 9, 1)

        self.pathXlsSpectrum_text = QLabel("pathXlsSpectrum")
        self.layoutSources.addWidget(self.pathXlsSpectrum_text, 10, 0)

        self.pathXlsSpectrum_value = QLineEdit("")
        self.layoutSources.addWidget(self.pathXlsSpectrum_value, 10, 1)

        self.energyUnit_text = QLabel("energyUnit")
        self.layoutSources.addWidget(self.energyUnit_text, 11, 0)

        self.energyUnit_value = QLineEdit("")
        self.layoutSources.addWidget(self.energyUnit_value, 11, 1)

        self.energyColumnKey_text = QLabel("energyColumnKey")
        self.layoutSources.addWidget(self.energyColumnKey_text, 12, 0)

        self.energyColumnKey_value = QLineEdit("")
        self.layoutSources.addWidget(self.energyColumnKey_value, 12, 1)

        self.fluenceColumnKey_text = QLabel("fluenceColumnKey")
        self.layoutSources.addWidget(self.fluenceColumnKey_text, 13, 0)

        self.fluenceColumnKey_value = QLineEdit("")
        self.layoutSources.addWidget(self.fluenceColumnKey_value, 13, 1)

        self.reset_button = QPushButton("reset")
        self.reset_button.clicked.connect(self.reset)
        self.layoutSources.addWidget(self.reset_button, 15, 0, 1, 2)

        self.load_button = QPushButton("load")
        self.load_button.clicked.connect(self.load)
        self.layoutSources.addWidget(self.load_button, 16, 0, 1, 2)

        self.write_xml_button = QPushButton("write_xml")
        self.write_xml_button.clicked.connect(self.write_xml)
        self.layoutSources.addWidget(self.write_xml_button, 17, 0, 1, 2)

    def change_source_name(self):
        """
        if name change, change sourceName from experiment too
        Returns:

        """
        self.father.experiment_window.sourceName_value.setText(self.name_value.text())

    def reset(self):
        self.name_value.setText("")
        self.myType_value.setText("")
        self.mySize_value.setText("")
        self.sourceVoltage_value.setText("")
        self.filterMaterial_value.setText("")
        self.filterThickness_value.setText("")
        self.myEnergySampling_value.setText("")
        self.myTargetMaterial_value.setText("")
        self.spectrumFromXls_value.setText("")
        self.pathXlsSpectrum_value.setText("")
        self.energyUnit_value.setText("")
        self.energyColumnKey_value.setText("")
        self.fluenceColumnKey_value.setText("")

    def load(self):
        """
        chooose a source to load and load it
        Args:
            self:

        Returns:

        """
        items = []
        f = open("xmlFiles/Sources.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('source'):
            items.append(experience[0].text)

        exp_name, booleen = QInputDialog.getItem(self, "Load experiment", "enter experiment name", items)
        if not booleen:
            return
        self.father.sources_window.reset()
        self.load_exec(exp_name)
    def load_exec(self,exp_name):
        """
        Read a xml and load parameters
        """

        f = open("xmlFiles/Sources.xml",'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for child in root:
            if child[0].text==exp_name:
                for child2 in child:
                    name,value=child2.tag, child2.text
                    if name == "myDimensions":
                        for child3 in child2:
                            name, value = child3.tag, child3.text
                            self.change_value(name, value)
                    else:
                        self.change_value(name,value)
        f.close()


    def change_value(self, name, value):
        """
        switch name and change the value of the QWidget correspond

        """
        if name == "name":
            self.name_value.setText(value)
        elif name == "myType":
            self.myType_value.setText(value)
        elif name == "mySize":
            self.mySize_value.setText(value)
        elif name == "sourceVoltage":
            self.sourceVoltage_value.setText(value)
        elif name == "filterMaterial":
            self.filterMaterial_value.setText(value)
        elif name == "filterThickness":
            self.filterThickness_value.setText(value)
        elif name == "myEnergySampling":
            self.myEnergySampling_value.setText(value)
        elif name == "myTargetMaterial":
            self.myTargetMaterial_value.setText(value)
        elif name == "spectrumFromXls":
            self.spectrumFromXls_value.setText(value)
        elif name == "energyUnit":
            self.energyUnit_value.setText(value)
        elif name == "energyColumnKey":
            self.energyColumnKey_value.setText(value)
        elif name == "fluenceColumnKey":
            self.fluenceColumnKey_value.setText(value)

    def write_xml(self):
        """
        write parameters in Experiment.xml
        """
        f = open("xmlFiles/Sources.xml", 'r')
        texte = f.read()
        root = ET.fromstring(texte)
        for experience in root.findall('source'):
            if experience[0].text == self.name_value.text():
                root.remove(experience)

        doc = ET.SubElement(root, "source")
        ET.SubElement(doc, "name").text = self.name_value.text()
        if self.myType_value.text()!="":
            ET.SubElement(doc, "myType").text = self.myType_value.text()
        if self.mySize_value.text() != "":
            ET.SubElement(doc, "mySize").text = self.mySize_value.text()
        if self.sourceVoltage_value.text() != "":
            ET.SubElement(doc, "sourceVoltage").text = self.sourceVoltage_value.text()
        if self.filterMaterial_value.text() != "":
            ET.SubElement(doc, "filterMaterial").text = self.filterMaterial_value.text()
        if self.filterThickness_value.text() != "":
            ET.SubElement(doc, "filterThickness").text = self.filterThickness_value.text()
        if self.myEnergySampling_value.text() != "":
            ET.SubElement(doc, "myEnergySampling").text = self.myEnergySampling_value.text()
        if self.myTargetMaterial_value.text() != "":
            ET.SubElement(doc, "myTargetMaterial").text = self.myTargetMaterial_value.text()
        if self.spectrumFromXls_value.text() != "":
            ET.SubElement(doc, "spectrumFromXls").text = self.spectrumFromXls_value.text()
        if self.energyUnit_value.text() != "":
            ET.SubElement(doc, "energyUnit").text = self.energyUnit_value.text()
        if self.energyColumnKey_value.text() != "":
            ET.SubElement(doc, "energyColumnKey").text = self.energyColumnKey_value.text()
        if self.fluenceColumnKey_value.text() != "":
            ET.SubElement(doc, "fluenceColumnKey").text = self.fluenceColumnKey_value.text()

        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        tree.write("xmlFiles/Sources.xml", encoding='utf-8')


class start_thread(QThread):
    """
    thread for the start paresis
    """
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

    def run(self):

        self.gui.number_execution = self.gui.number_execution + 1
        self.load = load_thread(self.gui)
        self.load.start()
        self.gui.start_exec()
        self.gui.number_execution = self.gui.number_execution - 1


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