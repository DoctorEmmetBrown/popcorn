# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from pagailleIO import save_image, openSeq, openImage
import glob
import random
import os
from scipy.ndimage.filters import gaussian_filter
from MISTII_2 import processProjectionMISTII_2
from MISTII_1 import processProjectionMISTII_1
from MISTI import MISTI
from OpticalFlow2020 import processProjectionOpticalFlow2020
from Pavlov2020 import tie_Pavlovetal2020 as pavlov2020
from LCS_DF import processProjectionLCS_DF
from LCS import processProjectionLCS
from speckle_matching import processProjectionUMPA
from XSVT import processProjectionXSVT
import datetime
from matplotlib import pyplot as plt
import numpy as np
from xml.dom import minidom
from ImageProcessing import deconvolve

class Phase_Retrieval_Experiment:

    """Performs phase retrieval.s for an experiment."""

    def __init__(self, exp_name, do):
        """Initialize all experiment and algorithm parameters and load images.
        

        Args:
            exp_name (STRING): the experiment name in the parameter xml files (default " ").

        Returns:
            None.
        """
        
        # EXPERIMENT PARAMETERS
        self.xml_experiment_file_name="ExperimentParameters.xml"
        self.experiment_name=exp_name
        self.sample_images=None #Will later be a numpy array
        self.reference_images=None
        self.exp_folder=""
        self.tomo=False
        self.output_folder=""
        self.energy=0.
        self.pixel=0.
        self.dist_object_detector=0.
        self.dist_source_object=0.
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
        self.result_XSVT={}
        self.output_images_format="edf" #"edf" or "tif"

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
        self.proj_to_treat_start=0
        self.proj_to_treat_end=1

        self.define_experiment_values()
        self.define_algorithmic_values(do)
        
        self.methods_functions={'LCS':processProjectionLCS,
                                'LCS_DF':processProjectionLCS_DF,
                                'UMPA':processProjectionUMPA,
                                'XSVT':processProjectionXSVT,
                                'MISTI':MISTI,
                                'MISTII_1':processProjectionMISTII_1,
                                'MISTII_2':processProjectionMISTII_2,
                                'OF':processProjectionOpticalFlow2020,
                                'Pavlov':pavlov2020}

    def define_experiment_values(self):
        """Get experiment parameters from xml file.
        

        Raises:
            Exception: Correct experiment not found.

        Returns:
            None.

        """
        xml_doc = minidom.parse(self.xml_experiment_file_name)
        for current_exp in xml_doc.documentElement.getElementsByTagName("experiment"):
            correct_exp = self.getText(current_exp.getElementsByTagName("experiment_name")[0])

            if correct_exp == self.experiment_name:
                for node in current_exp.childNodes:
                    if node.localName=="tomo":
                        self.tomo=self.boolean(self.getText(current_exp.getElementsByTagName("tomo")[0]))
                    if node.localName=="output_images_format":
                        self.output_images_format=self.getText(current_exp.getElementsByTagName("output_images_format")[0])
                
                #if "tomo"=True in the xml file experiment the number of projections must also be defined
                if self.tomo:
                    self.number_of_projections=int(self.getText(current_exp.getElementsByTagName("number_of_projections")[0]))
                    self.proj_to_treat_end=self.number_of_projections
                            
                self.exp_folder=self.getText(current_exp.getElementsByTagName("exp_folder")[0])
                self.output_folder=str(self.getText(current_exp.getElementsByTagName("output_folder")[0]))
                self.energy=float(self.getText(current_exp.getElementsByTagName("energy")[0]))*1e3 #F or eV
                self.pixel=float(self.getText(current_exp.getElementsByTagName("pixel")[0]))
                self.dist_object_detector=float(self.getText(current_exp.getElementsByTagName("dist_object_detector")[0]))
                self.dist_source_object=float(self.getText(current_exp.getElementsByTagName("dist_source_object")[0]))
                self.delta=float(self.getText(current_exp.getElementsByTagName("delta")[0]))
                self.beta=float(self.getText(current_exp.getElementsByTagName("beta")[0]))
                self.source_size=float(self.getText(current_exp.getElementsByTagName("source_size")[0]))
                self.detector_PSF=float(self.getText(current_exp.getElementsByTagName("detector_PSF")[0]))
                self.crop_on=self.boolean(self.getText(current_exp.getElementsByTagName("crop_on")[0]))
                self.cropDebX=int(self.getText(current_exp.getElementsByTagName("cropDebX")[0]))
                self.cropDebY=int(self.getText(current_exp.getElementsByTagName("cropDebY")[0]))
                self.cropEndX=int(self.getText(current_exp.getElementsByTagName("cropEndX")[0]))
                self.cropEndY=int(self.getText(current_exp.getElementsByTagName("cropEndY")[0]))
                return
        
        raise Exception("Correct experiment not found")
        return     
        

    def define_algorithmic_values(self, do):
        """gets algorithmic parameters from xml file

        Raises:
            Exception: Correct experiment not found.

        Returns:
            None.

        """
        xml_doc = minidom.parse(self.xml_algorithmic_file_name)
        for current_exp in xml_doc.documentElement.getElementsByTagName("experiment"):
            correct_exp = self.getText(current_exp.getElementsByTagName("experiment_name")[0])
            if correct_exp == self.experiment_name:
                self.nb_of_point=int(self.getText(current_exp.getElementsByTagName("nb_of_point")[0]))
                self.pad_size=int(self.getText(current_exp.getElementsByTagName("pad_size")[0]))
                self.pad_type=self.getText(current_exp.getElementsByTagName("pad_type")[0])
                self.deconvolution=self.boolean(self.getText(current_exp.getElementsByTagName("do_deconvolution")[0]))
                self.deconvolution_type=self.getText(current_exp.getElementsByTagName("deconvolution_type")[0])
                self.absorption_correction_sigma=int(self.getText(current_exp.getElementsByTagName("absorption_correction_sigma")[0]))
                self.max_shift=int(self.getText(current_exp.getElementsByTagName("max_shift")[0]))
                if do["LCS"]:
                    self.LCS_median_filter=int(self.getText(current_exp.getElementsByTagName("LCS_median_filter")[0]))
                if do["UMPA"]:
                    self.umpaNw=int(self.getText(current_exp.getElementsByTagName("umpaNw")[0]))
                if do["XSVT"]:
                    self.XSVT_Nw=int(self.getText(current_exp.getElementsByTagName("XSVT_Nw")[0]))
                    self.XSVT_median_filter=int(self.getText(current_exp.getElementsByTagName("XSVT_median_filter")[0]))
                if do["MISTI"] or do["MISTII_1"] or do["MISTII_2"]:
                    self.MIST_median_filter=int(self.getText(current_exp.getElementsByTagName("MIST_median_filter")[0]))
                self.sigma_regularization=float(self.getText(current_exp.getElementsByTagName("sigma_regularization")[0]))
                
                #if tomo is true, the range of projections to calculate can be defined in the algorithm xml file
                if self.tomo:
                    for node in current_exp.childNodes:
                        if node.localName=="proj_to_treat_start":
                            self.proj_to_treat_start=int(self.getText(current_exp.getElementsByTagName("proj_to_treat_start")[0]))
                            self.proj_to_treat_end=int(self.getText(current_exp.getElementsByTagName("proj_to_treat_end")[0]))
                return
    
        raise Exception("Correct experiment not found")


    def getText(self,node):
        """get the text situated in the node value

        Args:
            node (minidom node):

        Returns:
            STRING: DESCRIPTION.

        """
        return node.childNodes[0].nodeValue


    def save_image(self):
        """not implemented yet
        """
        return

    def display_and_modify_parameters(self):
        """This function should someday open a window that displays and allows to change algorithm parameters
        It should also save the parameters and propose to modify them in the xml file AlgorithmicParaneters
        """
        return

    def open_Is_Ir(self):
        """Opens sample and reference images

        """
        # Load the reference and sample images
        refFolder = self.exp_folder + 'ref/'
        sampleFolder = self.exp_folder + 'sample/'

        refImages = glob.glob(refFolder + '/*.tif') + glob.glob(refFolder + '/*.tiff') + glob.glob(refFolder + '/*.edf')
        refImages.sort()
        sampImages = glob.glob(sampleFolder + '/*.tif') + glob.glob(sampleFolder + '/*.tiff') + glob.glob(
            sampleFolder + '/*.edf')
        sampImages.sort()
        whiteImage= glob.glob(self.exp_folder+'white.tif')+glob.glob(self.exp_folder+'White.tif')+glob.glob(self.exp_folder+'white.tiff')
        darkImage= glob.glob(self.exp_folder+'dark.tif')+glob.glob(self.exp_folder+'dark.tif')+glob.glob(self.exp_folder+'dark.tiff')
        if self.nb_of_point >= len(refImages):
            print("Nb of points limited to ", len(refImages))
            self.nb_of_point=len(refImages)
            Ir = openSeq(refImages)
            Is = openSeq(sampImages)
        else: # On sellectionne aleatoirement les n points parmi toutes les donnees disponibles
            indexOfImagesPicked = []
            refTaken = []
            sampTaken = []
            number=0
            while len(indexOfImagesPicked) < self.nb_of_point:
                # indexOfImagesPicked.append(number)
                # refTaken.append(refImages[number])
                # sampTaken.append(sampImages[number])
                # number+=1
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
        if len(whiteImage)==0:
            white=gaussian_filter(np.mean(Ir, axis=0),50)
        else:
            white=openSeq(whiteImage)[0]
        if len(darkImage)!=0:
            dark=openSeq(darkImage)[0]
        
        # Ir=(Ir-dark)/(white-dark)
        # Is=(Is-dark)/(white-dark)
        self.reference_images=np.asarray(Ir, dtype=np.float64)#/white
        self.sample_images=np.asarray(Is, dtype=np.float64)#/white

        if self.crop_on:
            self.reference_images = self.reference_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
            self.sample_images = self.sample_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
        return self.reference_images, self.sample_images


    def open_Is_Ir_tomo(self, iproj, Nproj):
        """Opens sample and reference images

        """
        # Load the reference and sample images
        
        # refFolder = self.exp_folder + 'ref/'
        # sampleFolder = self.exp_folder + 'sample/'
        expFolder=self.exp_folder

        refImagesStart = glob.glob(self.exp_folder + '/refHST0000.edf')+ glob.glob(self.exp_folder + '/ref0000_0000.edf') 
        refImagesStart.sort()
        NprojString='%4.2d'%Nproj
        refImagesEnd = glob.glob(self.exp_folder+ '/refHST'+NprojString+'.edf') + glob.glob(self.exp_folder + '/ref0000_0720.edf')
        refImagesEnd.sort()
        
        sampImages=[]
        justfolder=self.exp_folder.split('/')[-2]
        iprojString='%4.4d'%iproj
        samppath=self.exp_folder+justfolder+iprojString+'.edf'
        sampImages=glob.glob(self.exp_folder+'/'+justfolder+iprojString+'.edf')
        
        sampImages.sort()
        
        
        
        whiteImage= glob.glob(self.exp_folder+'white.tif')+glob.glob(self.exp_folder+'White.tif')+glob.glob(self.exp_folder+'white.tiff')
        darkImage= glob.glob(self.exp_folder+'dark.tif')+glob.glob(self.exp_folder+'dark.tif')+glob.glob(self.exp_folder+'dark.tiff')
        if self.nb_of_point >= len(refImagesStart):
            print("Nb of points limited to ", len(refImagesStart))
            self.nb_of_point=len(refImagesStart)
            IrStart = openSeq(refImagesStart)
            IrEnd = openSeq(refImagesEnd)
            Is = openSeq(sampImages)
        else: # On sellectionne aleatoirement les n points parmi toutes les donnees disponibles
            indexOfImagesPicked = []
            refTakenStart = []
            refTakenEnd = []
            sampTaken = []
            number=0
            while len(indexOfImagesPicked) < self.nb_of_point:
                # indexOfImagesPicked.append(number)
                # refTaken.append(refImages[number])
                # sampTaken.append(sampImages[number])
                # number+=1
                number = random.randint(0, len(refImagesStart) - 1)
                if not number in indexOfImagesPicked:
                    indexOfImagesPicked.append(number)
                    refTakenStart.append(refImagesStart[number])
                    refTakenEnd.append(refImagesEnd[number])
                    sampTaken.append(sampImages[number])
            refTakenStart.sort()
            refTakenEnd.sort()
            sampTaken.sort()
            IrStart = openSeq(refTakenStart)
            IrEnd = openSeq(refTakenEnd)
            Is = openSeq(sampTaken)

        Ir=IrStart*(Nproj-iproj)+IrEnd*iproj

        # On cree un white a partir de la reference pour normaliser
        if len(whiteImage)==0:
            white=gaussian_filter(np.mean(Ir, axis=0),100)
        else:
            white=openSeq(whiteImage)[0]
        if len(darkImage)!=0:
            dark=openSeq(darkImage)[0]
        
        # Ir=(Ir-dark)/(white-dark)
        # Is=(Is-dark)/(white-dark)
        self.reference_images=np.asarray(Ir, dtype=np.float64)#/white
        self.sample_images=np.asarray(Is, dtype=np.float64)#/white

        if self.crop_on:
            self.reference_images = self.reference_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
            self.sample_images = self.sample_images[:, self.cropDebX:self.cropEndX,self.cropDebY:self.cropEndY]
        return self.reference_images, self.sample_images

    def preProcessAndPadImages(self):
        """pads images in Is and Ir

        Notes:
            Will eventually do more (Deconvolution, shot noise filtering...)
        """

        nbImages, width, height = self.reference_images.shape
        print("Deconvolution",self.deconvolution)
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
        """Applies deconvolution to every acquisitions
        """
        print("starting deconvolution")
        for i in range(self.nb_of_point):
            self.reference_images[i]=deconvolve(self.reference_images[i], self.detector_PSF, self.deconvolution_type)
            self.sample_images[i]=deconvolve(self.sample_images[i], self.detector_PSF, self.deconvolution_type)
            folderPath=self.output_folder
            
            txtPoint = '%2.2d' % i
            save_image(self.reference_images[i], folderPath+"/refImageDeconvolved_"+txtPoint+".edf")
            save_image(self.sample_images[i], folderPath+"/sampleImageDeconvolved_"+txtPoint+".edf")
        
        return self.reference_images, self.sample_images
    
    def boolean(self, boolStr):
        """turns a string into a boolean

        Args:
            boolStr (str): string "True" or "False"

        Returns:
            boolean
        """
        if boolStr=="True":
            return True
        elif boolStr=="False":
            return False
        else :
            raise Exception("The string you are trying to turn to a boolean is not 'True' or 'False'")

    # *******************************************************
    # ************PHASE RETRIEVAL METHODS******************


    def process_method(self,method):
        """this function calls processMISTII_2() in its file,
        crops the results of the padds added in pre-processing
        and saves the retrieved images.
        Args:
            sampleImage [numpy array]: set of sample images
            referenceImage [numpy array]: set of reference images
            ddict [dictionnary]: experiment dictionnary
        """
        result = self.methods_functions[method](self)
        currentMethod="/"+method+"_"
        padSize = self.pad_size
        for key, value in result.items():
            if padSize >0:
                if value.ndim==2:
                    width, height = value.shape
                if value.ndim==3:
                    width, height, _ = value.shape
                value=value[padSize: width - padSize, padSize: height - padSize]
            currentFolder=self.output_folder+currentMethod+key
            
            if self.tomo:
                if not os.path.exists(currentFolder):
                    os.mkdir(currentFolder)
                iprojString='%4.4d'%self.currentProjection 
                if value.ndim==2:
                    save_image(value,currentFolder+currentMethod+key+"_"+iprojString+'.'+self.output_images_format)
                if value.ndim==3:
                    plt.imsave(currentFolder+currentMethod+key+"_"+iprojString+'.tiff',value,format='tiff')
            else:
                if value.ndim==2:
                    save_image(value,currentFolder+'.'+self.output_images_format)
                if value.ndim==3:
                    plt.imsave(currentFolder+'.tiff',value,format='tiff')
        return None


    def getk(self):
        """calculates the wavenumber of the experiment beam at the current energy

        Returns:
            k (TYPE): DESCRIPTION.
        Note:
            energy in eV
        """
        h=6.626e-34
        c=2.998e8
        e=1.6e-19
        k=2*np.pi*self.energy*e/(h*c)
        return k

                
