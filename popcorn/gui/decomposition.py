
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

from datetime import datetime

class Decomposition(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.father = parent
        self.layoutRightTwo = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutRightTwo)
        self.TxtMateriau = QLabel("How many material?")
        self.layoutRightTwo.addWidget(self.TxtMateriau, 0, 0)

        self.inputNbMat = QLineEdit()
        self.layoutRightTwo.addWidget(self.inputNbMat, 0, 1, 1, 4)
        self.inputNbMat.editingFinished.connect(self.addButtonMaterial)

        self.Liste_Mat = []

        self.Liste_Ene = []

        self.Liste_Combo = []

        self.TxtEne = QLabel("How many energie?")
        self.layoutRightTwo.addWidget(self.TxtEne, 3, 0)
        self.inputNbEne = QLineEdit()
        self.layoutRightTwo.addWidget(self.inputNbEne, 3, 1)

        self.inputNbEne.editingFinished.connect(self.addButtonEne)

        self.Table = QTabWidget()

        self.layoutRightTwo.addWidget(self.Table, 9, 0, 1, 5)

        self.xlsx = QTableWidget()

        self.xlsx_densite = QTableWidget()
        self.Table.addTab(self.xlsx, "Value")
        self.Table.addTab(self.xlsx_densite, "Density")
        self.xlsx_densite.setRowCount(1)


        self.buttonStartFonction = QPushButton("Start Decomposition")
        self.buttonStartFonction.clicked.connect(self.call_Fonction)

        self.layoutRightTwo.addWidget(self.buttonStartFonction, 10, 0, 1, 5)

        self.prog = QProgressBar()
        self.prog.setValue(0)
        self.layoutRightTwo.addWidget(self.prog, 11, 0, 1, 5)

    def addButtonMaterial(self):
        """
        Add as many QLineEdit that the number chosse by the user
        Each QLineEdit correspond to a material and his id is store in Liste_Mat  
        """
        n = int(self.inputNbMat.text())
        listeNoLongEnough = True
        for i in range(len(self.Liste_Mat) - 1, n - 1, -1):
            b = self.Liste_Mat[i]
            del self.Liste_Mat[i]
            b.close()
            listeNoLongEnough = False
        if n > 10:
            self.inputNbMat.setText("10")
            n = 10
        self.xlsx.setColumnCount(n)
        self.xlsx_densite.setColumnCount(n)
        i = 0
        j = 0
        while i * 5 + j < n and listeNoLongEnough:
            if i * 5 + j >= len(self.Liste_Mat):
                inputNameMat = QLineEdit()
                self.layoutRightTwo.addWidget(inputNameMat, 1 + i, 0 + j)
                self.Liste_Mat.append(inputNameMat)
                inputNameMat.textChanged.connect(self.ChangColumnName)
            j = (j + 1) % 5
            if j == 0:
                i = i + 1

    def addButtonEne(self):
        """
        Add as many QLineEdit and QComboBox that the number chosse by the user
        Each pair QLineEdit/QComboBox correspond to a energy/image and his id is store in Liste_Ene/Liste_Combo
        """
        n = int(self.inputNbEne.text())
        listeNoLongEnough = True
        for i in range(len(self.Liste_Ene) - 1, n - 1, -1):
            b = self.Liste_Ene[i]
            c = self.Liste_Combo[i]
            del self.Liste_Ene[i]
            del self.Liste_Combo[i]
            b.close()
            c.close()
            listeNoLongEnough = False
        if n > 10:
            self.inputNbEne.setText("10")
            n = 10
        self.xlsx.setRowCount(n)
        i = 0
        j = 0
        while i * 2 + j < n and listeNoLongEnough:
            if i * 2 + j >= len(self.Liste_Ene):
                inputNameEne = QLineEdit()
                self.layoutRightTwo.addWidget(inputNameEne, 4 + i, 2 * j)
                self.Liste_Ene.append(inputNameEne)
                ComboBoxEne = QComboBox(self.father.leftWidget.combobox)
                self.Liste_Combo.append(ComboBoxEne)
                self.layoutRightTwo.addWidget(ComboBoxEne, 4 + i, 2 * j + 1)
                for k in range(self.father.leftWidget.combobox.count()):
                    ComboBoxEne.addItem(self.father.leftWidget.combobox.itemText(k))
                inputNameEne.textChanged.connect(self.ChangRowName)
            j = (j + 1) % 2
            if j == 0:
                i = i + 1


    def ChangColumnName(self):
        """
        Change the name of the column according to the materials
        """
        self.ListColName = []
        for i in self.Liste_Mat:
            self.ListColName.append(i.text())
            self.xlsx.setHorizontalHeaderLabels(self.ListColName)
            self.xlsx_densite.setHorizontalHeaderLabels(self.ListColName)

    def ChangRowName(self):
        """
        Change the name of the row according to the energies
        """
        ListRowName = []
        for i in self.Liste_Ene:
            ListRowName.append(i.text())
            self.xlsx.setVerticalHeaderLabels(ListRowName)

    def get_table(self):
        """
        Read the values from the tables and store them in a np.array
        """
        for i in range(self.xlsx.rowCount()):
            numpy1D = np.zeros(0)
            for j in range(self.xlsx.columnCount()):
                numpy1D = np.append(numpy1D, float(self.xlsx.item(i, j).text()))
            if i == 0:
                numpy2D = numpy1D
            else:
                numpy2D = np.vstack([numpy2D, numpy1D])
        return numpy2D
        
    def get_density(self):
        """
        Read density and store them in a np.array
        """
        numpy1D = np.zeros(0)
        for j in range(self.xlsx_densite.columnCount()):
            numpy1D = np.append(numpy1D, float(self.xlsx_densite.item(0, j).text()))

        
        return numpy1D
    def get_Images(self):
        """
        Get the different image selected in the QComboBox and store them in a np.array where the first dimension is the number of images
        """
        name_liste=[]
        for i in range(int(self.inputNbEne.text())):
            if i == 0:
                image_combo = self.Liste_Combo[i]
                if self.father.leftWidget.liste_image[image_combo.currentIndex()].color:
                    return None,None
                
                liste_decomposition_image = self.father.leftWidget.liste_image[
                    image_combo.currentIndex()
                ].image
                liste_decomposition_image = np.expand_dims(liste_decomposition_image, axis=0)
                name_liste.append(image_combo.currentText())

            else:
                image_combo = self.Liste_Combo[i]
                if self.father.leftWidget.liste_image[image_combo.currentIndex()].color:
                    return None,None
                    
                ima = self.father.leftWidget.liste_image[image_combo.currentIndex()].image

                ima = np.expand_dims(ima, axis=0)
                liste_decomposition_image = np.concatenate(
                    (liste_decomposition_image, ima), axis=0
                )
                name_liste.append(image_combo.currentText())
        return name_liste,liste_decomposition_image
        
    def call_Fonction(self):
        """
        start call_Fonction_exe in a thread
        """
        self.thread=decomp_thread(self)
        self.thread.start()
        
    def call_Fonction_exe(self):
        """
        Read all data and start the fonction decomposition_equation_resolution from popcorn
        """
        name_liste,liste_decomposition_image = self.get_Images()
        if name_liste==None:
            message_error=QErrorMessage()
            message_error.showMessage("At lease one image is in color. This fonction take only gray image")
            message_error.exec()
        else:

            np_density = self.get_density()
            numpy2D = self.get_table()
            print("Arg 1:", liste_decomposition_image.shape)
            print("Arg 2:", np_density.shape)
            print("Arg 3:", numpy2D.shape)

            size=liste_decomposition_image.shape[1]
            for i in range(size):
                self.prog.setValue(int(i/(size-1)*100))
                tmp_liste=liste_decomposition_image[:,i,:,:]
                liste_decomposition_image[:,i,:,:]=decomposition_equation_resolution(
                                                    tmp_liste,
                                                    np_density,
                                                    numpy2D,
                                                    volume_fraction_hypothesis=False,
                                                    verbose=False,
                                                )
            for i in range(liste_decomposition_image.shape[0]):
                self.father.leftWidget.ajouter_image(self.ListColName[i]+"_Decomposition",liste_decomposition_image[i],False)
                date=str(datetime.today()).split(".")[0]
                log=date+"\n"+"Decomposition\n"+self.ListColName[i]+"\n"
                self.father.leftWidget.liste_log[self.ListColName[i]+"_Decomposition"]=log
        return
            
        


def decomposition_equation_resolution(
    images,
    densities,
    material_attenuations,
    volume_fraction_hypothesis=True,
    verbose=False,
):
    """solves the element decomposition system

    Args:
        images (numpy.ndarray): N dim array, each N-1 dim array is an image acquired at 1 given energy (can be 2D or 3D,
        K energies in total)
        densities (numpy.ndarray): 1D array, one density per elements inside a voxel (P elements in total)
        material_attenuations (numpy.ndarray): 2D array, linear attenuation of each element at each energy (K * P array)
        volume_fraction_hypothesis (bool):
        verbose (bool):

    Returns:
        (numpy.ndarray): material decomposition maps, N-dim array composed of P * N-1-dim arrays
    """
    number_of_energies = images.shape[0]
    number_of_materials = densities.size

    if verbose:
        print("-- Material decomposition --")
        print(">Number of energies: ", number_of_energies)
        print(">Number of materials: ", number_of_materials)
        print(
            ">Sum of materials volume fraction equal to 1 hypothesis :",
            volume_fraction_hypothesis,
        )

    system_2d_matrix = np.ones(
        (number_of_energies + volume_fraction_hypothesis * 1, number_of_materials)
    )

    system_2d_matrix[0:number_of_energies, :] = material_attenuations
    vector_2d_matrix = np.ones(
        (number_of_energies + volume_fraction_hypothesis * 1, images[0, :].size)
    )
    for energy_index, image in enumerate(images):
        vector_2d_matrix[energy_index] = image.flatten()
    vector_2d_matrix = np.transpose(vector_2d_matrix)

    solution_matrix = None
    if number_of_energies + volume_fraction_hypothesis * 1 == number_of_materials:
        system_3d_matrix = np.repeat(
            system_2d_matrix[np.newaxis, :], images[0, :].size, axis=0
        )
        solution_matrix = np.linalg.solve(system_3d_matrix, vector_2d_matrix)
    else:
        for nb, vector in enumerate(vector_2d_matrix):

            # Q, R = np.linalg.qr(system_2d_matrix)  # qr decomposition of A
            # Qb = np.dot(Q.T, vector)  # computing Q^T*b (project b onto the range of A)
            # solution_vector = np.linalg.solve(R, Qb)  # solving R*x = Q^T*b
            solution_vector = np.linalg.lstsq(system_2d_matrix, vector, rcond=None)

            if solution_matrix is not None:
                # loading_bar(nb, vector_2d_matrix.size)
                if solution_matrix.ndim == 2:
                    solution_matrix = np.vstack([solution_matrix, solution_vector[0]])
                else:
                    solution_matrix = np.stack(
                        (solution_matrix, solution_vector[0]), axis=0
                    )
            else:
                solution_matrix = solution_vector[0]
    if images.ndim == 3:
        concentration_maps = np.zeros(
            (number_of_materials, images[0, :].shape[0], images[0, :].shape[1])
        )
    else:
        concentration_maps = np.zeros(
            (
                number_of_materials,
                images[0, :].shape[0],
                images[0, :].shape[1],
                images[0, :].shape[2],
            )
        )


    for material_index in range(number_of_materials):

        solution_matrix[:, material_index] = (
            solution_matrix[:, material_index] * densities[material_index] * 1000.0
        ).astype(np.float32)

        concentration_maps[material_index, :] = np.reshape(
            solution_matrix[:, material_index], images[0, :].shape
        )
    return concentration_maps
    
class decomp_thread(QThread):
    """
    A thread class for call_Fonction_exe
    """
    def __init__(self,gui):
        super().__init__()
        self.gui=gui

    def run(self):
        self.gui.call_Fonction_exe()

