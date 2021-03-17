# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from pagailleIO import saveEdf, openSeq
import glob
import random
import os
from scipy.ndimage.filters import gaussian_filter
from MISTII_2 import processProjectionMISTII_2
from MISTII_1 import processProjectionMISTII_1
from MISTI import MISTI
from OpticalFlow2020 import processProjectionOpticalFlow2020
from Pavlov2020 import tie_Pavlovetal2020 as pavlov2020
from LCS import processProjectionLCS
from speckle_matching import processProjectionUMPA
import datetime
from matplotlib import pyplot as plt
import numpy as np
from xml.dom import minidom
from ImageProcessing import deconvolve

class Phase_Retrieval_Experiment:

    """Performs phase retrieval.s for an experiment."""

    def __init__(self, exp_name):
        """Initialize all experiment and algorithm parameters and load images.

        Keyword arguments:
        exp_name -- the experiment name in the parameter xml files (default " ")
        """
        # EXPERIMENT PARAMETERS
        self.xml_experiment_file_name="ExperimentParameters.xml"
        self.experiment_name=exp_name
        self.sample_images=None #Will later be a numpy array
        self.reference_images=None
        self.exp_folder=""
        self.output_folder=""
        self.energy=0.
        self.pixel=0.
        self.dist_object_detector=0.
        self.dist_sample_object=0.
        self.delta=0.
        self.beta=0.
        self.source_size=0.
        self.detector_PSF=0.
        self.crop_on=False
        self.cropDebX= 0
        self.cropDebY= 0
        self.cropEndX= 0
        self.cropEndY= 0
        self.result_LCS={}
        self.result_MISTI={}
        self.result_MISTII_1={}
        self.result_MISTII_2={}
        self.result_OF={}
        self.result_Pavlov2020={}
        self.result_UMPA={}

        # ALGORITHMIC PARAMETERS
        self.xml_algorithmic_file_name="AlgorithmParameter.xml"
        self.nb_of_point=0
        self.pad_size=0
        self.pad_type='reflect'
        self.deconvolution=False
        self.deconvolution_type=''# unsupervised_wiener or richardson_lucy
        self.absorption_correction_sigma=15
        self.max_shift=0
        self.sigma_regularization = 0 # For filtering low frequencies in the fourier space

        self.define_experiment_values()
        self.define_algorithmic_values()

        # We create a folder for each retrieval test
        now=datetime.datetime.now()
        self.expID=now.strftime("%Y%m%d-%H%M%S") #
        self.output_folder+=self.expID
        os.mkdir(self.output_folder)

        self.open_Is_Ir()
        self.preProcessAndPadImages()


    def define_experiment_values(self):
        """Get experiment parameters from xml file.

        Keyword arguments:
        real -- the real part (default 0.0)
        imag -- the imaginary part (default 0.0)
        """
        xml_doc = minidom.parse(self.xml_experiment_file_name)
        for current_exp in xml_doc.documentElement.getElementsByTagName("experiment"):
            correct_exp = self.getText(current_exp.getElementsByTagName("experiment_name")[0])

            if correct_exp == self.experiment_name:
                self.exp_folder=self.getText(current_exp.getElementsByTagName("exp_folder")[0])
                self.output_folder=str(self.getText(current_exp.getElementsByTagName("output_folder")[0]))
                self.energy=float(self.getText(current_exp.getElementsByTagName("energy")[0]))*1e3 #F or eV
                self.pixel=float(self.getText(current_exp.getElementsByTagName("pixel")[0]))
                self.dist_object_detector=float(self.getText(current_exp.getElementsByTagName("dist_object_detector")[0]))
                self.dist_sample_object=float(self.getText(current_exp.getElementsByTagName("dist_sample_object")[0]))
                self.delta=float(self.getText(current_exp.getElementsByTagName("delta")[0]))
                self.beta=float(self.getText(current_exp.getElementsByTagName("beta")[0]))
                self.source_size=float(self.getText(current_exp.getElementsByTagName("source_size")[0]))
                self.detector_PSF=float(self.getText(current_exp.getElementsByTagName("detector_PSF")[0]))
                self.crop_on=bool(self.getText(current_exp.getElementsByTagName("crop_on")[0]))
                self.cropDebX=int(self.getText(current_exp.getElementsByTagName("cropDebX")[0]))
                self.cropDebY=int(self.getText(current_exp.getElementsByTagName("cropDebY")[0]))
                self.cropEndX=int(self.getText(current_exp.getElementsByTagName("cropEndX")[0]))
                self.cropEndY=int(self.getText(current_exp.getElementsByTagName("cropEndY")[0]))
            else:
                raise Exception("Correct experiment not found")
                
        return

    def define_algorithmic_values(self):
        xml_doc = minidom.parse(self.xml_algorithmic_file_name)
        for current_exp in xml_doc.documentElement.getElementsByTagName("experiment"):
            correct_exp = self.getText(current_exp.getElementsByTagName("experiment_name")[0])
            if correct_exp == self.experiment_name:
                self.nb_of_point=int(self.getText(current_exp.getElementsByTagName("nb_of_point")[0]))
                self.pad_size=int(self.getText(current_exp.getElementsByTagName("pad_size")[0]))
                self.pad_type=self.getText(current_exp.getElementsByTagName("pad_type")[0])
                self.deconvolution=bool(self.getText(current_exp.getElementsByTagName("do_deconvolution")[0]))
                self.deconvolution_type=self.getText(current_exp.getElementsByTagName("deconvolution_type")[0])
                self.absorption_correction_sigma=int(self.getText(current_exp.getElementsByTagName("absorption_correction_sigma")[0]))
                self.max_shift=int(self.getText(current_exp.getElementsByTagName("max_shift")[0]))
                self.LCS_median_filter=int(self.getText(current_exp.getElementsByTagName("LCS_median_filter")[0]))
                self.umpaNw=int(self.getText(current_exp.getElementsByTagName("umpaNw")[0]))
                self.MIST_median_filter=int(self.getText(current_exp.getElementsByTagName("MIST_median_filter")[0]))
                self.sigma_regularization=float(self.getText(current_exp.getElementsByTagName("sigma_regularization")[0]))
            else:
                raise Exception("Correct experiment not found")
        return

    def getText(self,node):
        return node.childNodes[0].nodeValue


    def save_image(self):
        return

    def display_and_modify_parameters(self):
        """This function should someday open a window that displays and allows to change algorithm parameters
        It should also save the parameters and propose to modify them in the xml file AlgorithmicParaneters
        """
        return

    def open_Is_Ir(self):
        # Load the reference and sample images
        refFolder = self.exp_folder + 'ref/'
        sampleFolder = self.exp_folder + 'sample/'

        refImages = glob.glob(refFolder + '/*.tif') + glob.glob(refFolder + '/*.tiff') + glob.glob(refFolder + '/*.edf')
        refImages.sort()
        sampImages = glob.glob(sampleFolder + '/*.tif') + glob.glob(sampleFolder + '/*.tiff') + glob.glob(
            sampleFolder + '/*.edf')
        sampImages.sort()
        if self.nb_of_point >= len(refImages):
            print("Nb of points limited to ", len(refImages))
            self.nb_of_point=len(refImages)
            Ir = openSeq(refImages)
            Is = openSeq(sampImages)
        else: # On sellectionne aleatoirement les n points parmi toutes les donnees disponibles
            indexOfImagesPicked = []
            refTaken = []
            sampTaken = []
            while len(indexOfImagesPicked) < self.nb_of_point:
                number = random.randint(0, len(refImages) - 1)
                if not number in indexOfImagesPicked:
                    indexOfImagesPicked.append(number)
                    refTaken.append(refImages[number])
                    sampTaken.append(sampImages[number])
            refTaken.sort()
            sampTaken.sort()
            Ir = openSeq(refTaken)
            Is = openSeq(sampTaken)


        # On cree un white a partir de la reference pour normaliser
        white=gaussian_filter(np.mean(Ir, axis=0),50)
        self.reference_images=np.asarray(Ir/white, dtype=np.float64)
        self.sample_images=np.asarray(Is/white, dtype=np.float64)

        if self.crop_on:
            self.reference_images = self.reference_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
            self.sample_images = self.sample_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
        return self.reference_images, self.sample_images


    def preProcessAndPadImages(self):
        """
        Simply pads images in Is and Ir using parameters in expDict
        Returns Is and Ir padded
        Will eventually do more (Deconvolution, shot noise filtering...)
        """

        nbImages, width, height = self.reference_images.shape
        if self.deconvolution:
            self.set_deconvolution()

        padSize=self.pad_size
        IrToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
        IsToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
        for i in range(nbImages):
            IrToReturn[i] = np.pad(self.reference_images[i], ((padSize, padSize), (padSize, padSize)),mode=self.pad_type)  # voir is edge mieux que reflect
            IsToReturn[i] = np.pad(self.sample_images[i], ((padSize, padSize), (padSize, padSize)),mode=self.pad_type)  # voir is edge mieux que reflect
        self.reference_images=IrToReturn
        self.sample_images=IsToReturn
        return

    def set_deconvolution(self):
        for i in range(self.nb_of_point):
            self.reference_images[i]=deconvolve(self.reference_images[i], self.detector_PSF, self.deconvolution_type)
            self.sample_images[i]=deconvolve(self.sample_images[i], self.detector_PSF, self.deconvolution_type)
        return self.reference_images, self.sample_images

    # *******************************************************
    # ************PHASE RETRIEVAL METHODS******************


    def process_MISTII_2(self):
        """this function calls processMISTII_2() in its file,
        crops the results of the padds added in pre-processing
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        self.result_MISTII_2 = processProjectionMISTII_2(self)
        thickness= self.result_MISTII_2['thickness']
        Deff_xx=self.result_MISTII_2['Deff_xx']
        Deff_yy=self.result_MISTII_2['Deff_yy']
        Deff_xy=self.result_MISTII_2['Deff_xy']
        colouredDeff=self.result_MISTII_2['ColoredDeff']
        excentricity=self.result_MISTII_2['excentricity']
        colouredImageExc=self.result_MISTII_2['colouredImageExc']
        colouredImagearea=self.result_MISTII_2['colouredImagearea']
        colouredImageDir=self.result_MISTII_2['colouredImageDir']
        area=self.result_MISTII_2['area']
        NbIm, _, _=self.sample_images.shape

        padSize = self.pad_size
        if padSize > 0:
            width, height = thickness.shape
            thickness = thickness[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        saveEdf(thickness, self.output_folder + '/MISTII_2_thickness_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_xx, self.output_folder + '/MISTII_2_Deff_xx_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_yy, self.output_folder + '/MISTII_2_Deff_yy_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_xy, self.output_folder + '/MISTII_2_Deff_xy_NPts'+str(NbIm)+'.edf')
        saveEdf(excentricity, self.output_folder + '/MISTII_2_Excentricity_NPts'+str(NbIm)+'.edf')
        saveEdf(area, self.output_folder + '/MISTII_2_area_NPts'+str(NbIm)+'.edf')
        plt.imsave(self.output_folder + '/MISTII_2_colouredDeff_NPts'+str(NbIm)+'.tiff',colouredDeff,format='tiff')
        plt.imsave(self.output_folder + '/MISTII_2_colouredImageExc_NPts'+str(NbIm)+'.tiff',colouredImageExc,format='tiff')
        plt.imsave(self.output_folder + '/MISTII_2_colouredImagearea_NPts'+str(NbIm)+'.tiff',colouredImagearea,format='tiff')
        plt.imsave(self.output_folder + '/MISTII_2_colouredImageDir_NPts'+str(NbIm)+'.tiff',colouredImageDir,format='tiff')
        return self.result_MISTII_2

    def process_MISTII_1(self):
        """this function calls processProjectionMISTII_1() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        self.result_MISTII_1 = processProjectionMISTII_1(self)
        phi= self.result_MISTII_1['phi']
        Deff_xx=self.result_MISTII_1['Deff_xx']
        Deff_yy=self.result_MISTII_1['Deff_yy']
        Deff_xy=self.result_MISTII_1['Deff_xy']
        colouredDeff=self.result_MISTII_1['ColoredDeff']
        NbIm, _, _=self.sample_images.shape
        padSize = self.pad_size
        if padSize > 0:
            width, height = phi.shape
            phi = phi[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        saveEdf(phi, self.output_folder + '/MISTII_1_phi_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_xx, self.output_folder + '/MISTII_1_Deff_xx_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_yy, self.output_folder + '/MISTII_1_Deff_yy_NPts'+str(NbIm)+'.edf')
        saveEdf(Deff_xy, self.output_folder + '/MISTII_1_Deff_xy_NPts'+str(NbIm)+'.edf')
        plt.imsave(self.output_folder + '/MISTII_1_colouredDeff_NPts'+str(NbIm)+'.tiff',colouredDeff,format='tiff')
        return self.result_MISTII_1


    def process_LCS(self):
        """this function calls processProjectionLCS() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        self.result_LCS=processProjectionLCS(self)
        dx=self.result_LCS['dx']
        dy=self.result_LCS['dy']
        phiFC = self.result_LCS['phiFC']
        phiK = self.result_LCS['phiK']
        phiLA = self.result_LCS['phiLA']
        absorption = self.result_LCS['absorption']
        padSize = self.pad_size
        if padSize > 0:
            width, height = dx.shape
            dx = dx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            dy = dy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiFC = phiFC[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiK = phiK[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiLA = phiLA[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            absorption = absorption[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        saveEdf(dx, self.output_folder + '/LCS_dx.edf')
        saveEdf(dy, self.output_folder + '/LCS_dy.edf')
        saveEdf(phiFC, self.output_folder + '/LCS_phiFrankoChelappa.edf')
        saveEdf(phiK, self.output_folder + '/LCS_phiKottler.edf')
        saveEdf(phiLA, self.output_folder + '/LCS_phiLarkin.edf')
        saveEdf(absorption, self.output_folder + '/LCS_absorption.edf')
        return self.result_LCS


    def process_UMPA(self):
        """this function calls processProjectionUMPA() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """

        self.result_UMPA=processProjectionUMPA(self)
        dx=self.result_UMPA['dx']
        dy=self.result_UMPA['dy']
        phiFC = self.result_UMPA['phiFC']
        phiK = self.result_UMPA['phiK']
        phiLA = self.result_UMPA['phiLA']
        thickness = self.result_UMPA['thickness']
        df=self.result_UMPA['df']
        f=self.result_UMPA['f']
        padSize = self.pad_size-self.umpaNw-self.max_shift
        if padSize > 0:
            width, height = dx.shape
            dx = dx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            dy = dy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiFC = phiFC[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiK = phiK[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            phiLA = phiLA[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            thickness = thickness[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            df = df[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
            f = f[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]

        saveEdf(dx, self.output_folder + '/UMPA_dx.edf')
        saveEdf(dy, self.output_folder + '/UMPA_dy.edf')
        saveEdf(phiFC, self.output_folder + '/UMPA_phiFrankoChelappa.edf')
        saveEdf(phiK, self.output_folder + '/UMPA_phiKottler.edf')
        saveEdf(phiLA, self.output_folder + '/UMPA_phiLarkin.edf')
        saveEdf(thickness, self.output_folder + '/UMPA_thickness.edf')
        saveEdf(df, self.output_folder + '/UMPA_darkField.edf')
        saveEdf(f, self.output_folder + '/UMPA_f.edf')
        return self.result_UMPA

    def process_OpticalFlow(self):
        """this function calls opticalFlow2020() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """

        self.result_OF = processProjectionOpticalFlow2020(self)
        dx = self.result_OF['dx']
        dy = self.result_OF['dy']
        phiFC = self.result_OF['phiFC']
        phiK = self.result_OF['phiK']
        phiLA = self.result_OF['phiLA']
        padSize = self.pad_size
        if padSize > 0:
            width, height = dx.shape
            dx = dx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            dy = dy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            phiFC = phiFC[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            phiK = phiK[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            phiLA = phiLA[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]

        saveEdf(dx, self.output_folder + '/OF_dx.edf')
        saveEdf(dy, self.output_folder + '/OF_dy.edf')
        saveEdf(phiFC, self.output_folder + '/OF_phiFrankoChelappa.edf')
        saveEdf(phiK, self.output_folder + '/OF_phiKottler.edf')
        saveEdf(phiLA, self.output_folder + '/OF_phiLarkin.edf')
        return self.result_OF



    def process_Pavlov2020(self):
        """this function calls pavlov2020() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        self.result_Pavlov2020 = pavlov2020(self)
        thicknessPavlov = self.result_Pavlov2020
        padSize = self.pad_size
        if padSize > 0:
            width, height = self.result_Pavlov2020.shape
            thicknessPavlov = self.result_Pavlov2020[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        saveEdf(thicknessPavlov, self.output_folder + '/Pavlov2020_thickness.edf')
        return self.result_Pavlov2020


    def process_MISTI(self):
        """this function calls MISTI() in its file,
        crops the results of the padds added in pre-processin
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        self.result_MISTI = MISTI(self)
        phi = self.result_MISTI['phi']
        Deff = self.result_MISTI['Deff']
        padSize = self.pad_size
        if padSize > 0:
            width, height = phi.shape
            phi = phi[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
            Deff = Deff[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        saveEdf(phi, self.output_folder + '/Phi_MISTI.edf')
        saveEdf(Deff, self.output_folder + '/Deff_MISTI.edf')
        return self.result_MISTI

    def getk(self):
        """
        energy in eV
        """
        h=6.626e-34
        c=2.998e8
        e=1.6e-19
        k=2*np.pi*self.energy*e/(h*c)
        return k

                