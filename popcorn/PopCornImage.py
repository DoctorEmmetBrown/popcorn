import numpy as np
from popcorn.input_output import open_image,open_sequence,save_edf_image,save_tif_image

class PopCornImage:

    def __init__(self,**kwargs):
        self.ndim=2
        if 'energy' in kwargs.keys():
            self.energy=kwargs.pop('energy')
        else:
            self.energy = -1
        if 'width' in kwargs.keys():
            self.width=kwargs.pop('energy')
        else:
            self.width = -1
        if 'height' in kwargs.keys():
            self.height = kwargs.pop('height')
            self.ndim=2
        else:
            self.height= -1
        if 'nbSlices' in kwargs.keys():
            self.nbSlices = kwargs.pop('nbSlices')
            self.ndim=3
        else:
            self.nbSlices= -1
        if 'data' in kwargs.keys():
            self.data = np.copy(kwargs.pop('data'))
            self.ndim=self.data.ndim
            self.shape=self.data.shape
            if self.ndim>2:
                self.nbSlices=self.shape[0]
                self.width = self.shape[1]
                self.height = self.shape[2]
            else:
                self.width = self.shape[0]
                self.height = self.shape[1]
        else :
            self.data=None

        self.padded=False

        self.dtype=None
        #self.energy=-1
        self.padWidth=-1
        self.padHeight=-1

        print(self.energy)

    def crop(self,xBeg,yBeg,xEnd,yEnd):
        self.data=self.data[xBeg:xEnd,yBeg:yEnd]

    def show(self):
        print(self.data)

    def __str__(self):
        if self.ndim==2:
            return f'Class PopCorn Image \n ndim:{self.ndim} \n width: {self.width} height:{self.height} \n data: \n {self.data}'
        else:
            return f'Class PopCorn Image \n ndim:{self.ndim} \n nbSlices: {self.nbSlices} width: {self.width} height:{self.height} \n data: \n {self.data}'

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key,value):
        self.data[key]=value



if __name__ == "__main__":
    print("Hello")
    myData=np.zeros((45,35))
    a=PopCornImage(energy=52,data=myData)
    print(a)
    a[0,0]=3
    print(a)
