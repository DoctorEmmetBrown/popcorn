from pagailleIO import saveEdf,openImage,openSeq,save3D_Edf
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.sparse.linalg import lsmr
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
import colorsys
from numpy.linalg import eig
import colorsys
from PIL import Image
import math
import multiprocessing


def MISTII_2(sampleImages, refImages, dataDict,nbImages):
    """
    Calculates the tensors of the dark field and the thickness of a single-material object from the acquisitions
    """
        
    Nz, Nx, Ny=refImages.shape
    beta=dataDict.beta
    gamma_mat=dataDict.delta/beta
    distSampDet=dataDict.dist_object_detector
    pixSize=dataDict.pixel
    Lambda=1.2398/dataDict.energy*1e-9
    
    LHS=np.ones(((nbImages, Nx, Ny)))
    RHS=np.ones((((nbImages,4, Nx, Ny))))
    FirstTermRHS=np.ones((Nx,Ny))
    solution=np.ones(((4, Nx, Ny)))
    
    #Prepare system matrices
    for i in range(nbImages):
        #Left hand Side
        IsIr=sampleImages[i]/refImages[i]
        
        #Right handSide
        gX_IrIr,gY_IrIr=np.gradient(refImages[i],pixSize)
        gXX_IrIr,gYX_IrIr=np.gradient(gX_IrIr,pixSize)
        gXY_IrIr,gYY_IrIr=np.gradient(gY_IrIr,pixSize)
        
        gXX_IrIr=gXX_IrIr/refImages[i]
        gXY_IrIr=gXY_IrIr/refImages[i]
        gYX_IrIr=gYX_IrIr/refImages[i]
        gYY_IrIr=gYY_IrIr/refImages[i]
        
        RHS[i]=[FirstTermRHS,gXX_IrIr, gYY_IrIr,gXY_IrIr]
        LHS[i]=IsIr
        
#    Solving system for each pixel 
    for i in range(Nx):
        for j in range(Ny):
            a=RHS[:,:,i,j]
            b=LHS[:,i,j]
            Q,R = np.linalg.qr(a) # qr decomposition of A
            Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
            if R[2,2]==0 or R[1,1]==0 or R[0,0]==0 or R[3,3]==0:
                temp=[1,1,1,1]
            else:
                temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
            solution[:,i,j]=temp
            
    G1=solution[0]
    G2=solution[1]
    G3=solution[2]
    G4=solution[3]
    
    #Calculation of G
    dG2,_=np.gradient(G2,pixSize)
    _,dG3=np.gradient(G3, pixSize)
    dG4,_=np.gradient(G4,pixSize)
    ddG2,_=np.gradient(dG2,pixSize)
    _,ddG3=np.gradient(dG3, pixSize)
    _,ddG4=np.gradient(dG4,pixSize)
    G=G1-ddG2-ddG3-ddG4
     
    sig_scale=dataDict.sigma_regularization
    if sig_scale==0:
        beta=1
    else:
        dqx = 2 * np.pi / (Nx)
        dqy = 2 * np.pi / (Ny)
        Qx, Qy = np.meshgrid((np.arange(0, Ny) - np.floor(Ny / 2) - 1) * dqy, (np.arange(0, Nx) - np.floor(Nx / 2) - 1) * dqx) #frequency ranges of the images in fqcy space
    
        #building filters
        sigmaX = dqx / 1. * np.power(sig_scale,2)
        sigmaY = dqy / 1. * np.power(sig_scale,2)
        #sigmaX=sig_scale
        #sigmaY = sig_scale
    
        g = np.exp(-(((Qx)**2) / 2. / sigmaX + ((Qy)**2) / 2. / sigmaY))
        #g = np.exp(-(((np.power(Qx, 2)) / 2) / sigmaX + ((np.power(Qy, 2)) / 2) / sigmaY))
        beta = 1 - g;
    
    #Calculation of the thickness of the object
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))
    u_m = u / (Nx * pixSize)
    v_m = v / (Ny * pixSize)
    uv_sqr = np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)
    denom=1+np.pi*gamma_mat*distSampDet*Lambda*uv_sqr
    thickness=-Lambda/(4*np.pi)*np.log(ifft2(ifftshift(fftshift(fft2(G))*beta/denom)))
    
    #Calculation of absorption image
    Iob=ifft2(ifftshift(fftshift(fft2(G))/denom)).real

    Deff_xx=G2/distSampDet/Iob
    Deff_yy=G3/distSampDet/Iob
    Deff_xy=G4/distSampDet/Iob
    
    return thickness.real, Deff_xx,Deff_yy,Deff_xy

def normalize(Image):
    Imageb=(Image-np.min(Image))/(np.max(Image)-np.min(Image))
    return Imageb

def processProjectionMISTII_2(Is,Ir,experiment):
    """
    This function calls PavlovDirDF to compute the tensors of the directional dark field and the thickness of the sample
    The function should also convert the tensor into a coloured image
    """
    nbImages, Nx, Ny= Is.shape
    
    #Calculate directional darl field
    thickness, Deff_xx,Deff_yy,Deff_xy=MISTII_2(Is,Ir,experiment,nbImages)
    
    #Post processing tests
    #Median filter
    medFiltSize=experiment.MIST_median_filter
    if medFiltSize!=0:
        thickness=median_filter(thickness, medFiltSize)
        Deff_xx=median_filter(Deff_xx, medFiltSize)
        Deff_yy=median_filter(Deff_yy, medFiltSize)
        Deff_xy=median_filter(Deff_xy, medFiltSize)
    
    alpha=0.00000001
    stdDeff=np.mean([np.std(Deff_xx), np.std(Deff_yy), np.std(Deff_xy)])*2
    Deff_xx=Deff_xx/(3*stdDeff)
    Deff_yy=Deff_yy/(3*stdDeff)
    Deff_xy=Deff_xy/(3*stdDeff)
    maskDeffxx=Deff_xx<0
    maskDeffyy=Deff_yy<0
    Deff_xx[maskDeffxx]=alpha
    Deff_yy[maskDeffyy]=alpha
    
    a11=Deff_xy*Deff_yy
    a22=Deff_xy*Deff_xx
    a12=Deff_xx*Deff_yy/2
    a33=-Deff_xy*Deff_yy*Deff_xx
        
    A=1/Deff_xx
    C=1/Deff_yy
    B=1/Deff_xy
    maskEllipsce=1-((A>0)*(A*C-B**2>0))
    
#
    theta=0.5*np.arctan(2*a12/(a11-a22))
    theta[maskDeffxx*maskDeffyy]=0
    
    Ap1=np.abs(a11*np.sin(theta)**2+a22*np.cos(theta)**2+2*a12*np.sin(theta)*np.cos(theta))
    Bp1=np.abs(a11*np.cos(theta)**2+a22*np.sin(theta)**2-2*a12*np.sin(theta)*np.cos(theta))
    Ap=np.max([Ap1,Bp1], axis=0)
    Bp=np.min([Ap1,Bp1], axis=0)
    a=np.sqrt(Ap)
    b=np.sqrt(Bp)
    theta[Ap==Bp1]+=np.pi/2
    theta[theta<0]+=np.pi
    
    excentricity=(np.sqrt(a**2+b**2)/(a)-1)*2
    excentricity[maskDeffxx*maskDeffyy]=0
    excentricity[maskEllipsce]=0
    area=(a*b)
    area[area<0]=0
    area[maskDeffxx*maskDeffyy]=0
    theta=(theta)/np.pi
    excentricity[excentricity>1]=1
    if medFiltSize!=0:
        area=median_filter(area, medFiltSize)
    stdarea=np.std(area)
    area=abs(area)/(3*stdarea)
    area[area>1]=1
    
    #NORMALIZATION
    Deff_xx[Deff_xx>1]=1
    Deff_yy[Deff_yy>1]=1
    Deff_xy=abs(Deff_xy)
    Deff_xy[Deff_xy>1]=1
    IntensityDeff=np.sqrt((Deff_xx**2+Deff_yy**2+Deff_xy**2)/3)
    
    #Trying to create a coloured image from tensor (method probably wrong for now)
    colouredImage=np.zeros(((Nx, Ny,3)))
    colouredImage[:,:,0]=Deff_xx
    colouredImage[:,:,1]=Deff_yy
    colouredImage[:,:,2]=Deff_xy
    
    colouredImageExc=np.zeros(((Nx, Ny,3)))
    colouredImageExc[:,:,0]=theta
    colouredImageExc[:,:,1]=1
    colouredImageExc[:,:,2]=excentricity
    
    colouredImagearea=np.zeros(((Nx, Ny,3)))
    colouredImagearea[:,:,0]=theta
    colouredImagearea[:,:,1]=1
    colouredImagearea[:,:,2]=area
    
    colouredImageDir=np.zeros(((Nx, Ny,3)))
    colouredImageDir[:,:,0]=theta
    colouredImageDir[:,:,1]=1
    colouredImageDir[:,:,2]=IntensityDeff
    
    colouredImageExc=hsv_to_rgb(colouredImageExc)
    colouredImagearea=hsv_to_rgb(colouredImagearea)
    colouredImageDir=hsv_to_rgb(colouredImageDir)
    

    return {'thickness': thickness, 'Deff_xx': Deff_xx, 'Deff_yy': Deff_yy, 'Deff_xy': Deff_xy, 'ColoredDeff': colouredImage, 'excentricity': excentricity,'area':area, 'colouredImageExc': colouredImageExc, 'colouredImagearea': colouredImagearea, 'colouredImageDir':colouredImageDir}


if __name__ == "__main__":
    im = Image.new("RGB", (300,300))
    radius = min(im.size)/2.0
    cx, cy = im.size[0]/2, im.size[1]/2
    pix = im.load()
 
    for x in range(im.width):
        for y in range(im.height):
            rx = x - cx
            ry = y - cy
            s = (rx ** 2.0 + ry ** 2.0) ** 0.5 / radius
            if s <= 1.0:
                h = ((math.atan2(ry, rx) / math.pi) + 1.0) 
                rgb = colorsys.hsv_to_rgb(h, 1.0, s)
                pix[x,y] = tuple([int(round(c*255.0)) for c in rgb])
 
    im.show()