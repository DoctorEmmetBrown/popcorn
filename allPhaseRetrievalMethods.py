#! /usr/bin/env python3

from pagailleIO import saveEdf, openImage, openSeq, save3D_Edf
import glob
import random
import os
from scipy.ndimage.filters import gaussian_filter
from MISTII_2 import processProjectionMISTII_2,processProjectionMISTII_2_2
from MISTII_1 import processProjectionMISTII_1
from MISTI import MISTI
from OpticalFlow2020 import processProjectionOpticalFlow2020
from Pavlov2020 import tie_Pavlovetal2020 as pavlov2020
from LCS import processProjectionLCS
from speckle_matching import processProjectionUMPA
import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
from skimage import color, data, restoration
from saveParameters import saveParameters

def preProcessAndPadImages(Is, Ir, expDict):
    """
    Simply pads images in Is and Ir using parameters in expDict
    Returns Is and Ir padded
    Will eventually do more (Deconvolution, shot noise filtering...)
    """
    nbImages, width, height = Ir.shape
    padSize=expDict['padding']
    IrToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
    IsToReturn = np.zeros((nbImages, width + 2 * padSize, height + 2 * padSize))
    for i in range(nbImages):
        IrToReturn[i] = np.pad(Ir[i], ((padSize, padSize), (padSize, padSize)),mode=expDict['padType'])  # voir is edge mieux que reflect
        IsToReturn[i] = np.pad(Is[i], ((padSize, padSize), (padSize, padSize)),mode=expDict['padType'])  # voir is edge mieux que reflect
    return IsToReturn, IrToReturn

    
def deconvolve(Image, sigma, deconvType):
    """
    Deconvolution of the acquisitions to correct detector's PSF
    using unsupervised wiener with a given sigma
    """
    Nx,Ny=Image.shape
    Nblur=int(np.floor(sigma*6))
    x,y= np.meshgrid(np.arange(0, Nblur), np.arange(0, Nblur))
    x = (x - ((Nblur-1) / 2))
    y = (y - ((Nblur-1) / 2))
    blur=np.exp(-(x**2+y**2)/sigma**2/2)
    blur=blur/np.sum(blur)
    
    MAX=np.max(Image)
    # Restore Image using unsupervised_wiener algorithm
    if deconvType=='unsupervised_wiener':
        restoredImage, _ = restoration.unsupervised_wiener(Image/MAX, blur, clip=False)
    # Restore Image using Richardson-Lucy algorithm
    elif deconvType=='richardson_lucy':
        restoredImage = restoration.richardson_lucy(Image/MAX, blur, iterations=10, clip=False)
    else:
        raise Exception('Filter not found')
    restoredImage=restoredImage*MAX
    
    return restoredImage
    
  
def readStudiedCase(sCase, nbImages, machinePrefix='MacLaurene'):
    """
    This function contains and reads the data specific to an experiment,
    opens the acquisitions and normalizes
    Arguments: 
        sCase [string]: the experiment name
        nbImages [int]: the number of pairs of acquisitions to take into account
        machinePrefix [string]: the name of the machine you are working on
    Outputs:
        Is [numpy array]: contains the nbImages sample images
        Ir [numpy array]: contains the nbImages reference images
    """

    expParam = {} #Initialisation of the experiment dictionnary
    now=datetime.datetime.now()
    expParam['expID']=now.strftime("%Y%m%d-%H%M%S") #Creates an ID number for the reconstruction based on time
    
    #Defines the path of data and destination folder on the current mahcine
    if machinePrefix == 'MacLaurene':
        expParam['expFolder']='/Users/quenot/Library/Mobile Documents/com~apple~CloudDocs/Thèse/ReactivIP/Data/' 
        expParam['outputFolder']='/Users/quenot/Library/Mobile Documents/com~apple~CloudDocs/Thèse/ReactivIP/Resultats/'
    elif machinePrefix== 'MacData':
        expParam['expFolder']='/Users/quenot/Data/'
        expParam['outputFolder']='/Users/quenot/Documents/PhaseRetrieval2021/Results/'
    elif machinePrefix== 'MacSimus':
        expParam['expFolder']='/Users/quenot/Documents/Simulations/CodePython_simple_01022021/Results/'
        expParam['outputFolder']='/Users/quenot/Documents/PhaseRetrieval2021/Results/'
    else:
        raise Exception('machine name not found')
    
    #Define the parameters depending on the experiment
    if sCase == 'MoucheSimapAout2017':
        expParam['expFolder'] += 'SIMAP/MoucheSimapAout2017/' #Paths of data Ir and Is which must be in folders named 'ref' and 'sample'
        expParam['outputFolder']+='TestMouche/' #Path of the output folder
        expParam['energy'] = 27
        expParam['pixel'] = 44 * 1e-6 #pixel size on the detector (in m)
        expParam['distOD'] = 0.41#dist object detector
        expParam['distSO'] = 0.12 #dist source object
        expParam['delta'] = 2.97e-7 # optical indices used for single material retrieval methods
        expParam['beta'] = 1.29e-9  
        expParam['sourceSize'] = 2 * 1e-6  # in meters    
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=False #Do we crop the images? To work on smaller windows
        expParam['cropDebX'] = 20  
        expParam['cropDebY'] = 160
        expParam['cropEndX'] = 1520
        expParam['cropEndY'] = 1660
    elif sCase == 'MoucheSimapAout2017Bined':
        expParam['expFolder'] += 'SIMAP/MoucheSimapAout2017/Binedx2/' #Paths of data Ir and Is which must be in folders named 'ref' and 'sample'
        expParam['outputFolder']+='TestMouche/' #Path of the output folder
        expParam['energy'] = 27
        expParam['pixel'] = 88 * 1e-6 #pixel size on the detector (in m)
        expParam['distOD'] = 0.41#dist object detector
        expParam['distSO'] = 0.12 #dist source object
        expParam['delta'] = 2.97e-7 # optical indices used for single material retrieval methods
        expParam['beta'] = 1.29e-9  
        expParam['sourceSize'] = 2 * 1e-6  # in meters    
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=True #Do we crop the images? To work on smaller windows
        expParam['cropDebX'] = 10  
        expParam['cropDebY'] = 80
        expParam['cropEndX'] = 760
        expParam['cropEndY'] = 830
    elif sCase == 'Groseille':
        expParam['expFolder'] += 'md1217/Groseille/'
        expParam['energy'] = 17
        expParam['pixel'] = 5.8 * 1e-6
        expParam['distOD'] = 1
        expParam['distSO'] = 140
        expParam['delta'] = 7.9765465E-07
        expParam['beta'] = 6.9608231E-10
        expParam['sourceSize'] = 10 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 1000
        expParam['cropEndY'] = 1000
        expParam['outputFolder']+='Groseille/'
    elif sCase == 'NylonWireMD1217':
        expParam['expFolder'] += 'md1217/NylonWire_clean/52keV/SecondWires_speckle/070ms/'
        expParam['energy'] = 52
        expParam['pixel'] = 3 * 1e-6
        expParam['distOD'] = 3.6
        expParam['distSO'] = 144
        expParam['delta'] = 9.4962647E-08
        expParam['beta'] = 4.3877157E-11
        expParam['sourceSize'] = 10 * 1e-6  # in meters  
        expParam['detectorPSF'] = 2.5/4
        expParam['cropOn']=True
        expParam['cropDebX'] = 1170
        expParam['cropDebY'] = 2090
        expParam['cropEndX'] = 1370
        expParam['cropEndY'] = 2290
        expParam['outputFolder']+='NylonWireMD1217/'
    elif sCase == 'Patte_XSVTTest':
        expParam['expFolder'] += 'ID17/Patte_XSVTTest/'
        expParam['energy'] = 52
        expParam['pixel'] = 6 * 1e-6
        expParam['distOD'] = 11
        expParam['distSO'] = 140
        expParam['delta'] = 2.03e-7
        expParam['beta'] = 8.53e-11  
        expParam['sourceSize'] = 10 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=True
        expParam['cropDebX'] = 50
        expParam['cropDebY'] = 500
        expParam['cropEndX'] = 1100
        expParam['cropEndY'] = 1250
        expParam['outputFolder']+='Patte_XSVTTest/'
    elif sCase == 'id17_ContrastPhantom':
        expParam['expFolder'] += 'id17_ContrastPhantom_21um_45kev/Fresnel_20210120-104032/'
        expParam['energy'] = 45
        expParam['pixel'] = 21.9 * 1e-6
        expParam['distOD'] = 3.6
        expParam['distSO'] = 146
        expParam['delta'] = 2.03e-7
        expParam['beta'] = 8.53e-11  
        expParam['sourceSize'] = 10 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.5
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Simulations/id17_ContrastPhantom_21um_45kev/'
    elif sCase == 'SIMAP_ContrastPhantom':
        expParam['expFolder'] += 'SIMAP_ContrastPhantom/RayTracing_20210125-082606/'
        expParam['energy'] = 27
        expParam['pixel'] = 100 * 1e-6
        expParam['distOD'] = 1
        expParam['distSO'] = 0.25
        expParam['delta'] = 2.03e-7
        expParam['beta'] = 8.53e-11  
        expParam['sourceSize'] = 2 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Simulations/SIMAP_ContrastPhantom/'
    elif sCase == 'Clinic_ContrastPhantom':
        expParam['expFolder'] += 'Clinic_ContrastPhantom/RayTracing_20210122-082922/' #Folder of the simulated data
        expParam['energy'] = 27
        expParam['pixel'] = 100 * 1e-6
        expParam['distOD'] = 1
        expParam['distSO'] = 1
        expParam['delta'] = 2.03e-7
        expParam['beta'] = 8.53e-11  
        expParam['sourceSize'] = 50 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.5
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Simulations/Clinic_ContrastPhantom/'
    elif sCase == 'Feuille_Xenocs':
        expParam['expFolder'] += 'Xenocs_04022021/Feuille/60s/' #Folder of the simulated data
        expParam['energy'] = 8
        expParam['pixel'] = 75 * 1e-6
        expParam['distOD'] = 1250
        expParam['distSO'] = 550
        expParam['delta'] = 2.03e-7
        expParam['beta'] = 8.53e-11  
        expParam['sourceSize'] = 50 * 1e-6  # in meters  
        expParam['detectorPSF'] = 0
        expParam['cropOn']=True
        expParam['cropDebX'] = 200
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 900
        expParam['cropEndY'] = 514
        expParam['outputFolder']+='Xenocs_04022021/Feuille/60s/'
    elif sCase == 'Alumette_Xenocs':
        expParam['expFolder'] += 'Xenocs_04022021/Alumette/60s/' #Folder of the simulated data
        expParam['energy'] = 8
        expParam['pixel'] = 75 * 1e-6
        expParam['distOD'] = 1250
        expParam['distSO'] = 550
        expParam['delta'] = 5.0118E-06
        expParam['beta'] = 1.3797E-08
        expParam['sourceSize'] = 50 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Xenocs_04022021/Alumette/60s/'
    elif sCase == 'Coquillage_Xenocs':
        expParam['expFolder'] += 'Xenocs_04022021/Coquillage/' #Folder of the simulated data
        expParam['energy'] = 8
        expParam['pixel'] = 75 * 1e-6
        expParam['distOD'] = 1250
        expParam['distSO'] = 550
        expParam['delta'] = 5.0118E-06
        expParam['beta'] = 1.3797E-08
        expParam['sourceSize'] = 50 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Xenocs_04022021/Coquillage/'
    elif sCase == 'FilNylonMD1217':
        expParam['expFolder'] += 'id17_FilNylon_md1217/Fresnel_20210308-095923/' #Folder of the simulated data
        expParam['energy'] = 52
        expParam['pixel'] = 3 * 1e-6
        expParam['distOD'] = 3.6
        expParam['distSO'] = 145.6
        expParam['delta'] = 5.0118E-06
        expParam['beta'] = 1.3797E-08
        expParam['sourceSize'] = 10 * 1e-6  # in meters  
        expParam['detectorPSF'] = 1.2
        expParam['cropOn']=False
        expParam['cropDebX'] = 0
        expParam['cropDebY'] = 0
        expParam['cropEndX'] = 500
        expParam['cropEndY'] = 1500
        expParam['outputFolder']+='Simulations/FilNylonMD1217/'
    else:
        raise Exception('experiment sCase: %g not found',sCase )
    #We create a folder for each retrieval test
    expParam['outputFolder']+=expParam['expID']
    os.mkdir(expParam['outputFolder'])    
    
    ## Load the reference and sample images
    refFolder = expParam['expFolder'] + 'ref/'
    sampleFolder = expParam['expFolder'] + 'sample/'

    refImages = glob.glob(refFolder + '/*.tif') + glob.glob(refFolder + '/*.tiff') + glob.glob(refFolder + '/*.edf')
    refImages.sort()
    sampImages = glob.glob(sampleFolder + '/*.tif') + glob.glob(sampleFolder + '/*.tiff') + glob.glob(
        sampleFolder + '/*.edf')
    sampImages.sort()
    if nbImages >= len(refImages):
        print("Nb of points limited to ", len(refImages))
        Ir = openSeq(refImages)
        Is = openSeq(sampImages)
    else: #On sellectionne aleatoirement les n points parmi toutes les donnees disponibles
        indexOfImagesPicked = []
        refTaken = []
        sampTaken = []
        while len(indexOfImagesPicked) < nbImages:
            number = random.randint(0, len(refImages) - 1)
            if not (number in indexOfImagesPicked):
                indexOfImagesPicked.append(number)
                refTaken.append(refImages[number])
                sampTaken.append(sampImages[number])
        refTaken.sort()
        #print(refTaken)
        sampTaken.sort()
        Ir = openSeq(refTaken)
        Is = openSeq(sampTaken)
        
    # On cree un white a partir de la reference pour normaliser 
    white=gaussian_filter(np.mean(Ir, axis=0),50)
    Ir=np.asarray(Ir/white, dtype=np.float64)
    Is=np.asarray(Is/white, dtype=np.float64)
    
    if expParam['cropOn']:
        Ir = Ir[:, expParam['cropDebX']:expParam['cropEndX'], expParam['cropDebY']:expParam['cropEndY']]
        Is = Is[:, expParam['cropDebX']:expParam['cropEndX'], expParam['cropDebY']:expParam['cropEndY']]

    return Is, Ir, expParam


def processMISTII_2(sampleImage, referenceImage, ddict):
    """
    this function calls processMISTII_2() in its file, 
    crops the results of the padds added in pre-processin 
    and saves the retrieved images.
    """
    result = processProjectionMISTII_2(sampleImage, referenceImage, expParam=ddict)
    thickness= result['thickness']
    Deff_xx=result['Deff_xx']
    Deff_yy=result['Deff_yy']
    Deff_xy=result['Deff_xy']
    colouredDeff=result['ColoredDeff']
    excentricity=result['excentricity']
    colouredImageExc=result['colouredImageExc']
    colouredImagearea=result['colouredImagearea']
    colouredImageDir=result['colouredImageDir']
    area=result['area']
    NbIm, Nx, Ny=sampleImage.shape
    
    padSize = ddict['padding']
    if padSize > 0:
        width, height = thickness.shape
        thickness = thickness[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    saveEdf(thickness, ddict['outputFolder'] + '/MISTII_2_thickness_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_xx, ddict['outputFolder'] + '/MISTII_2_Deff_xx_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_yy, ddict['outputFolder'] + '/MISTII_2_Deff_yy_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_xy, ddict['outputFolder'] + '/MISTII_2_Deff_xy_NPts'+str(NbIm)+'.edf')
    saveEdf(excentricity, ddict['outputFolder'] + '/MISTII_2_Excentricity_NPts'+str(NbIm)+'.edf')
    saveEdf(area, ddict['outputFolder'] + '/MISTII_2_area_NPts'+str(NbIm)+'.edf')
    plt.imsave(ddict['outputFolder'] + '/MISTII_2_colouredDeff_NPts'+str(NbIm)+'.tiff',colouredDeff,format='tiff')
    plt.imsave(ddict['outputFolder'] + '/MISTII_2_colouredImageExc_NPts'+str(NbIm)+'.tiff',colouredImageExc,format='tiff')
    plt.imsave(ddict['outputFolder'] + '/MISTII_2_colouredImagearea_NPts'+str(NbIm)+'.tiff',colouredImagearea,format='tiff')
    plt.imsave(ddict['outputFolder'] + '/MISTII_2_colouredImageDir_NPts'+str(NbIm)+'.tiff',colouredImageDir,format='tiff')
    return 

def processMISTII_1(sampleImage, referenceImage, ddict):
    """
    this function calls processProjectionMISTII_1() in its file, 
    crops the results of the padds added in pre-processing
    and saves the retrieved images.
    """
    result = processProjectionMISTII_1(sampleImage, referenceImage, expParam=ddict)
    phi= result['phi']
    Deff_xx=result['Deff_xx']
    Deff_yy=result['Deff_yy']
    Deff_xy=result['Deff_xy']
    colouredDeff=result['ColoredDeff']
    NbIm, Nx, Ny=sampleImage.shape
    padSize = ddict['padding']
    if padSize > 0:
        width, height = phi.shape
        phi = phi[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xx = Deff_xx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_yy = Deff_yy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff_xy = Deff_xy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    saveEdf(phi, ddict['outputFolder'] + '/MISTII_1_phi_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_xx, ddict['outputFolder'] + '/MISTII_1_Deff_xx_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_yy, ddict['outputFolder'] + '/MISTII_1_Deff_yy_NPts'+str(NbIm)+'.edf')
    saveEdf(Deff_xy, ddict['outputFolder'] + '/MISTII_1_Deff_xy_NPts'+str(NbIm)+'.edf')
    plt.imsave(ddict['outputFolder'] + '/MISTII_1_colouredDeff_NPts'+str(NbIm)+'.tiff',colouredDeff,format='tiff')
    return 


def processLCS(sampleImage, referenceImage, ddict):
    """
    this function calls processProjectionLCS() in its file, 
    crops the results of the padds added in pre-processin 
    and saves the retrieved images.
    """
    result=processProjectionLCS(sampleImage, referenceImage,ddict)
    dx=result['dx']
    dy=result['dy']
    phiFC = result['phiFC']
    phiK = result['phiK']
    phiLA = result['phiLA']
    absorption = result['absorption']
    padSize = ddict['padding']
    if padSize > 0:
        width, height = dx.shape
        dx = dx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        dy = dy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        phiFC = phiFC[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        phiK = phiK[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        phiLA = phiLA[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
        absorption = absorption[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize+1]
    saveEdf(dx, ddict['outputFolder'] + '/LCS_dx.edf')
    saveEdf(dy, ddict['outputFolder'] + '/LCS_dy.edf')
    saveEdf(phiFC, ddict['outputFolder'] + '/LCS_phiFrankoChelappa.edf')
    saveEdf(phiK, ddict['outputFolder'] + '/LCS_phiKottler.edf')
    saveEdf(phiLA, ddict['outputFolder'] + '/LCS_phiLarkin.edf')
    saveEdf(absorption, ddict['outputFolder'] + '/LCS_absorption.edf')
    
#    studyQuantiWireDx(ddict, dy)
    
    return dx, dy, phiFC, phiK, phiLA

    
def processUMPA(sampleImage, referenceImage, ddict):
    """
    this function calls processProjectionUMPA() in its file, 
    crops the results of the padds added in pre-processin 
    and saves the retrieved images.
    """
    result=processProjectionUMPA(sampleImage, referenceImage,ddict)
    dx=result['dx']
    dy=result['dy']
    phiFC = result['phiFC']
    phiK = result['phiK']
    phiLA = result['phiLA']
    thickness = result['thickness']
    df=result['df']
    f=result['f']
    padSize = ddict['padding']-ddict['umpaNw']-ddict['umpaMaxShift']
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

    saveEdf(dx, ddict['outputFolder'] + '/UMPA_dx.edf')
    saveEdf(dy, ddict['outputFolder'] + '/UMPA_dy.edf')
    saveEdf(phiFC, ddict['outputFolder'] + '/UMPA_phiFrankoChelappa.edf')
    saveEdf(phiK, ddict['outputFolder'] + '/UMPA_phiKottler.edf')
    saveEdf(phiLA, ddict['outputFolder'] + '/UMPA_phiLarkin.edf')
    saveEdf(thickness, ddict['outputFolder'] + '/UMPA_thickness.edf')
    saveEdf(df, ddict['outputFolder'] + '/UMPA_darkField.edf')
    saveEdf(f, ddict['outputFolder'] + '/UMPA_f.edf')
    return dx, dy, phiFC, phiK, phiLA

def processOpticalFlow(sampleImage, referenceImage, ddict):
    
    """
    this function calls opticalFlow2020() in its file, 
    crops the results of the padds added in pre-processin 
    and saves the retrieved images.
    """
    result = processProjectionOpticalFlow2020(sampleImage, referenceImage, expParam=ddict)
    dx = result['dx']
    dy = result['dy']
    phiFC = result['phiFC']
    phiK = result['phiK']
    phiLA = result['phiLA']
    padSize = ddict['padding']
    print(dx.shape)
    if padSize > 0:
        width, height = dx.shape
        dx = dx[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        dy = dy[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        phiFC = phiFC[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        phiK = phiK[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        phiLA = phiLA[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]

    saveEdf(dx, ddict['outputFolder'] + '/OF_dx.edf')
    saveEdf(dy, ddict['outputFolder'] + '/OF_dy.edf')
    saveEdf(phiFC, ddict['outputFolder'] + '/OF_phiFrankoChelappa.edf')
    saveEdf(phiK, ddict['outputFolder'] + '/OF_phiKottler.edf')
    saveEdf(phiLA, ddict['outputFolder'] + '/OF_phiLarkin.edf')    
    return dx, dy, phiFC, phiK, phiLA



def processPavlov2020(sampleImage, referenceImage, ddict):
    absMask=gaussian_filter(Is,10)/gaussian_filter(Ir,10)
    result = pavlov2020(sampleImage, referenceImage, absMask, expParam=ddict)
    thicknessPavlov = result
    padSize = ddict['padding']
    if padSize > 0:
        width, height = result.shape
        thicknessPavlov = result[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    saveEdf(thicknessPavlov, ddict['outputFolder'] + '/Pavlov2020_thickness.edf')
    return thicknessPavlov


def processMISTI(sampleImage, referenceImage, ddict):
    result = MISTI(sampleImage, referenceImage, ddict)
    phi = result['phi']
    Deff = result['Deff']
    padSize = ddict['padding']
    if padSize > 0:
        width, height = phi.shape
        phi = phi[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
        Deff = Deff[padSize:padSize + width - 2 * padSize, padSize:padSize + height - 2 * padSize]
    saveEdf(phi, ddict['outputFolder'] + '/Phi_MISTI.edf')
    saveEdf(Deff, ddict['outputFolder'] + '/Deff_MISTI.edf')
    return phi, Deff

if __name__ == "__main__":
    
    #Parameters to tune
    nbImages=10 #nb of pairs (Is, Ir) to use
    studiedCase = 'Patte_XSVTTest'  # name of the experiment we want to work on
    machinePrefix='MacData'  #name of the machine we are working on
    doLCS=False
    doMISTII_2=True
    doMISTII_1=False
    doMISTI=False
    doUMPA=False
    doOF=False
    doPavlov=False
    
    #Load images and parameters:
    Is, Ir, expDict = readStudiedCase(studiedCase, nbImages, machinePrefix)
    
    Nz=len(Is)
    
    #Define phase retrieval parameters
    expDict['nbOfPoint'] = min(nbImages, Nz)
    expDict['padding'] = 0
    expDict['padType'] = 'reflect'  #Peut etre utile pour l'intégration après LCS qui a lieu dans fourier et nécessite des conditions limites
    expDict['studiedCase']=studiedCase
    expDict['Comment']=""
    expDict['sigmaRegularization'] = 0
    expDict['LCS_median_filter'] = 1 #Filtres actuellement en test pour post-processing de LCS
    expDict['LCS_gaussian_filter'] = 0
    expDict['umpaMaxShift'] = 1
    expDict['umpaNw']=7
    expDict['Deconvolution'] = False
    expDict['DeconvType'] = 'unsupervised_wiener' #unsupervised_wiener or richardson_lucy
    expDict['processingtimeLCSv2']=0
    expDict['processingtimePavlovDirDF']=0   
    expDict['PavlovDirDF_MedianFilter']=3
    expDict['Absorption_correction_sigma']=15
    #First processing of the acquisitions before phase retrieval
    Is, Ir = preProcessAndPadImages(Is, Ir, expDict)
    
    if expDict['Deconvolution']:
        for i in range(Nz):
            Is[i]=deconvolve(Is[i], expDict['detectorPSF'],expDict['DeconvType'])
            Ir[i]=deconvolve(Ir[i], expDict['detectorPSF'],expDict['DeconvType'])    
        print("Deconvolution done")
    
    ## PHASE RETRIEVAL
    beforeLCS=time.time() #measure the computation time of each method
    #Compute Dx and Dy using LCS /!\ requires at least 3 points
    if doLCS:
        if expDict['nbOfPoint']<3:
            raise Exception('Not enough points to compute LCS. Required at least 3. Given ', expDict['nbOfPoint']) 
        else:
            dx, dy, phiFC, phiK, phiLA=processLCS(Is, Ir, expDict)    
            afterLCS=time.time()
            expDict['processingtimeLCSv2'] = afterLCS-beforeLCS
    else:
        afterLCS=beforeLCS
            
            
    #Compute directional dark field /!\ requires at least 4 points
    if doMISTII_2:
        if expDict['nbOfPoint']<4:
            raise Exception('Not enough points to compute directional dark field. Required at least 4. Given ', expDict['nbOfPoint']) 
        else:
            processMISTII_2(Is, Ir, expDict)
    afterMISTII_2=time.time()
    expDict['processingtimePavlovDirDF'] = afterMISTII_2-afterLCS
            
    
    if doMISTII_1:
        if expDict['nbOfPoint']<4:
            raise Exception('Not enough points to compute directional dark field. Required at least 4. Given ', expDict['nbOfPoint']) 
        else:
            processMISTII_1(Is, Ir, expDict)
    afterMISTII_1=time.time()
    expDict['processingtimePavlovDirDFa'] = afterMISTII_1-afterMISTII_2
            
    
    if doMISTI:
        if expDict['nbOfPoint']<2:
            raise Exception('Not enough points to compute directional dark field. Required at least 4. Given ', expDict['nbOfPoint']) 
        else:
            processMISTI(Is, Ir, expDict)
    afterMISTI=time.time()
    expDict['processingtimeMISTI'] = afterMISTI-afterMISTII_1
            
    
    if doUMPA:
        processUMPA(Is, Ir, expDict)
    afterUMPA=time.time()
    expDict['processingTimeUMPA'] = afterUMPA-afterMISTI
        
    
    if doOF:
        processOpticalFlow(Is,Ir,expDict)
        
        
    if doPavlov:
        processPavlov2020(Is, Ir, expDict)
        
    #studyQuantiWireDx(expDict, dy)
   
    saveParameters(expDict) #Function that is used to save all parameters in excel file when testing

