import numpy as np
import frankoChellappa  as fc
from scipy.ndimage.filters import gaussian_filter, median_filter
from matplotlib import pyplot as plt
from skimage import color, data, restoration
from phaseIntegration import kottler, LarkinAnissonSheppard


def LCS(sampleImages, refImages, dataDict,nbImages):
    """
    Calculates the displacement images from sample and reference images using the LCS system
    Returns:
        Dx [numpy array]: the displacements along x axis
        Dy [numpy array]: the displacements along y axis
        absoprtion [numpy array]: the absorption 
    """    
    
    Nz, Nx, Ny=refImages.shape
    LHS=np.ones(((nbImages, Nx, Ny)))
    RHS=np.ones((((nbImages,3, Nx, Ny))))
    solution=np.ones(((3, Nx, Ny)))
    
    #Prepare system matrices
    for i in range(nbImages):
        #Right handSide
        gX_IrIr,gY_IrIr=np.gradient(refImages[i])
        RHS[i]=[sampleImages[i],gX_IrIr, gY_IrIr]
        LHS[i]=refImages[i]
        
    #Solving system for each pixel 
    for i in range(Nx):
        for j in range(Ny):
            a=RHS[:,:,i,j]
            b=LHS[:,i,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
            
            if R[2,2]==0 or R[1,1]==0 or R[0,0]==0:
                temp=[1,0,0]
            else:
                temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
            solution[:,i,j]=temp
        
    absoprtion=1/solution[0]
    Dx=solution[1]
    Dy=solution[2]
    
    #Bit of post-processing
    #Limiting displacement to a threshold
    displacementLimit=dataDict.max_shift
    Dx[Dx<-displacementLimit]=-displacementLimit
    Dx[Dx>displacementLimit]=displacementLimit
    Dy[Dy<-displacementLimit]=-displacementLimit
    Dy[Dy>displacementLimit]=displacementLimit
    #Trying different filters
    if dataDict.LCS_median_filter !=0:
        Dx=median_filter(Dx,size=dataDict.LCS_median_filter)
        Dy=median_filter(Dy,size=dataDict.LCS_median_filter)
    
    return Dx, Dy, absoprtion


def processProjectionLCS(Is,Ir,expParam):
    """
    this function calls pre-processing specific to the method (only deconvolution for now)
    then calls the LCS_v2 wich returns displacement images
    then calls 3 different function to integrate Dx, Dy into the phase image (frankotchellappa, kottler and LarkinArnisonSheppard)
    It returns the displacement images, the three phase images and the absorption calculated.
    """
    nbImages, Nx, Ny= Is.shape
    
    dx, dy , absorption =LCS(Is,Ir,expParam,nbImages)

    #Compute the phase gradient from displacements (linear relationship)
    #magnification=(expParam['distSO']+expParam['distOD'])/expParam['distSO'] #Not sure I need to use this yet
    dphix=dx*(expParam.pixel/expParam.dist_object_detector)*expParam.getk()
    dphiy=dy*(expParam.pixel/expParam.dist_object_detector)*expParam.getk()
    
    padForIntegration=True
    padSize=300
    if padForIntegration:
        dphix = np.pad(dphix, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect
        dphiy = np.pad(dphiy, ((padSize, padSize), (padSize, padSize)),mode='reflect')  # voir is edge mieux que reflect

    
    #Compute the phase from phase gradients with 3 different methods (still trying to choose the best one)
    phiFC = fc.frankotchellappa(dphiy, dphix, True)*expParam.pixel
    phiK = kottler(dphiy, dphix)*expParam.pixel
    phiLA = LarkinAnissonSheppard(dphiy, dphix)*expParam.pixel
    
    if padSize > 0:
        phiFC = phiFC[padSize:padSize + Nx, padSize:padSize + Ny]
        phiK = phiK[padSize:padSize + Nx , padSize:padSize + Ny]
        phiLA = phiLA[padSize:padSize + Nx, padSize:padSize + Ny]

    return {'dx': dx, 'dy': dy, 'phiFC': phiFC.real, 'phiK': phiK.real,'phiLA': phiLA.real, 'absorption':absorption}
            
    

