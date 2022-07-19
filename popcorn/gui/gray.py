import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *
import numpy as np
from popcorn import input_output

import time
class Image():
    def __init__(self,array,color):
        self.image=np.copy(array)
        self.color=color

class Gray(QWidget):
    def __init__(self, father):
        super().__init__()
        self.father = father

        self.layoutRightOne = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutRightOne)

        # Double slider pour la reduction de niveau de gris
        self.Doubleslider = QRangeSlider(Qt.Orientation.Horizontal, self)
        self.Doubleslider.initStyleOption
        self.Doubleslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.Doubleslider.setTickInterval(5000)
        
        self.Doubleslider.setMinimum(0)
        self.Doubleslider.setMaximum(65535)
        self.Doubleslider.setSliderPosition([0, 65535])
        self.layoutRightOne.addWidget(self.Doubleslider, 0, 0, 1, 2)
        self.Doubleslider.sliderReleased.connect(self.niveau_gris)


        mini=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.min()
        maxi=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.max()
        
        
        self.labelMinDSlider = QLabel("Minimum:")
        self.layoutRightOne.addWidget(self.labelMinDSlider, 1, 0)
        self.inputMinDSlider = QLineEdit()
        self.layoutRightOne.addWidget(self.inputMinDSlider, 1, 1)
        val=self.Doubleslider.sliderPosition()[0]/65535*(maxi-mini)+mini

        self.inputMinDSlider.setText(str(round(self.Doubleslider.sliderPosition()[0]/65535*(maxi-mini)+mini,3)))
        self.inputMinDSlider.returnPressed.connect(self.ChangerDSliderPosition)

        self.labelMaxDSlider = QLabel("Maximum:")
        self.layoutRightOne.addWidget(self.labelMaxDSlider, 2, 0)
        self.inputMaxDSlider = QLineEdit()
        self.layoutRightOne.addWidget(self.inputMaxDSlider, 2, 1)
        self.inputMaxDSlider.setText(str(round(self.Doubleslider.sliderPosition()[1]/65535*(maxi-mini)+mini,3)))
        self.inputMaxDSlider.returnPressed.connect(self.ChangerDSliderPosition)

        self.Doubleslider.sliderReleased.connect(self.ChangerTestDSlider)

        self.opti_contrast=QPushButton("Auto Contrast")
        self.opti_contrast.clicked.connect(self.auto_contrast)
        self.layoutRightOne.addWidget(self.opti_contrast, 3, 0, 1, 2)


        # Construction de l'histogramme

        self.abscisse = QBarSet("Repartition des niveaux de gris")
        self.SetHistValue()

        abs_series = QBarSeries()
        abs_series.append(self.abscisse)

        self.hist = QChart()
        self.hist.addSeries(abs_series)

        self.axisX = QValueAxis()
        self.axisX.setTitleText("Pixel value")
        #self.axisX.setLabelFormat("%g")
        self.axisX.setMin(self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.min())
        self.axisX.setMax(self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.max())
        self.axisX.setTickCount(6)

        self.hist.addAxis(self.axisX, Qt.AlignBottom)

        self.axisY = QLogValueAxis()
        self.axisY.setTitleText("Number of pixel")
        self.axisY.setLabelFormat("%g")
        self.axisY.setBase(1.0)
        self.hist.addAxis(self.axisY, Qt.AlignLeft)
        abs_series.attachAxis(self.axisY)

        chartView = QChartView(self.hist)
        self.layoutRightOne.addWidget(chartView, 5, 0, 1, 2)


        
        self.father.leftWidget.combobox.currentIndexChanged.connect(self.SetHistValue)
        self.father.leftWidget.slider.sliderReleased.connect(self.SetHistValue)
        self.father.leftWidget.slider.sliderReleased.connect(self.niveau_gris)

        self.father.leftWidget.radioA.toggled.connect(self.SetHistValue)
        self.father.leftWidget.radioC.toggled.connect(self.SetHistValue)
        self.father.leftWidget.radioS.toggled.connect(self.SetHistValue)
        
    def auto_contrast(self):
        """
        choose automatically 2 values for percentile: 5% and 99%
        """
        if self.father.leftWidget.image_3D_np.color==False:
            x = self.father.leftWidget.combobox.currentIndex()
            image = Image(self.father.leftWidget.liste_image[x].image, self.father.leftWidget.liste_image[x].color)
            mini=image.image.min()
            maxi=image.image.max()
            z = self.father.leftWidget.slider.sliderPosition()

            if self.father.leftWidget.radioA.isChecked():
                image.image=image.image[z, :, :]
            elif self.father.leftWidget.radioC.isChecked():
                image.image=image.image[:,z, :]
            elif self.father.leftWidget.radioS.isChecked():
                image.image=image.image[:, :, z]

            new_min_pos=np.percentile(image.image,5)
            new_max_pos=np.percentile(image.image,99)
            if new_min_pos!=new_max_pos:
                new_min_pos=round(new_min_pos,3)
                new_max_pos=round(new_max_pos,3)
                self.inputMinDSlider.setText(str(new_min_pos))
                self.inputMaxDSlider.setText(str(new_max_pos))
                

                new_min_pos = int((float(new_min_pos)-mini)*65535/(maxi-mini))
                new_max_pos = int((float(new_max_pos)-mini)*65535/(maxi-mini))
                self.Doubleslider.setValue([new_min_pos, new_max_pos])
                self.niveau_gris()

    def niveau_gris(self):
        """
        Change the values of image_3D_np.image according to the 2 sliders
        """
        if self.father.leftWidget.image_3D_np.color==False:
            self.father.leftWidget.changer_Image()
            valeur_min = 0
            valeur_max = 65535
            new_min = self.Doubleslider.sliderPosition()[0]
            new_max = self.Doubleslider.sliderPosition()[1]
            multi_pix = valeur_max / (new_max - new_min)

            if self.father.leftWidget.radioA.isChecked():
                z = self.father.leftWidget.slider.sliderPosition()
                self.father.leftWidget.image_3D_np.image[z, :, :][
                    self.father.leftWidget.image_3D_np.image[z, :, :] < new_min
                ] = new_min
                self.father.leftWidget.image_3D_np.image[z, :, :][
                    self.father.leftWidget.image_3D_np.image[z, :, :] > new_max
                ] = new_max
                self.father.leftWidget.image_3D_np.image[z] = (
                    self.father.leftWidget.image_3D_np.image[z, :, :] - new_min
                ) * multi_pix
                self.father.leftWidget.image_3D_np.image[z][
                    self.father.leftWidget.image_3D_np.image[z] < 0
                ] = 0
                self.father.leftWidget.image_3D_np.image[z][
                    self.father.leftWidget.image_3D_np.image[z] > 65535
                ] = 65535
            elif self.father.leftWidget.radioC.isChecked():
                y = self.father.leftWidget.slider.sliderPosition()
                self.father.leftWidget.image_3D_np.image[:, y, :][
                    self.father.leftWidget.image_3D_np.image[:, y, :] < new_min
                ] = new_min
                self.father.leftWidget.image_3D_np.image[:, y, :][
                    self.father.leftWidget.image_3D_np.image[:, y, :] > new_max
                ] = new_max
                self.father.leftWidget.image_3D_np.image[:, y, :] = (
                    self.father.leftWidget.image_3D_np.image[:, y, :] - new_min
                ) * multi_pix
                self.father.leftWidget.image_3D_np.image[:, y, :][
                    self.father.leftWidget.image_3D_np.image[:, y, :] < 0
                ] = 0
                self.father.leftWidget.image_3D_np.image[:, y, :][
                    self.father.leftWidget.image_3D_np.image[:, y, :] > 65535
                ] = 65535
            elif self.father.leftWidget.radioS.isChecked():
                x = self.father.leftWidget.slider.sliderPosition()
                self.father.leftWidget.image_3D_np.image[:, :, x][
                    self.father.leftWidget.image_3D_np.image[:, :, x] < new_min
                ] = new_min
                self.father.leftWidget.image_3D_np.image[:, :, x][
                    self.father.leftWidget.image_3D_np.image[:, :, x] > new_max
                ] = new_max
                self.father.leftWidget.image_3D_np.image[:, :, x] = (
                    self.father.leftWidget.image_3D_np.image[:, :, x] - new_min
                ) * multi_pix
                self.father.leftWidget.image_3D_np.image[:, :, x][
                    self.father.leftWidget.image_3D_np.image[:, :, x] < 0
                ] = 0
                self.father.leftWidget.image_3D_np.image[:, :, x][
                    self.father.leftWidget.image_3D_np.image[:, :, x] > 65535
                ] = 65535
            self.father.leftWidget.slider_position()

    def SetHistValue(self):
        """
        Remove all value of the histogram and recounts the pixel values to display the new histogram
        """
        if self.father.leftWidget.image_3D_np.color==False:
            self.abscisse.remove(0, 128)
            compteur_pixel = [0] * 128
            value=0
            if self.father.leftWidget.radioA.isChecked():
                z = self.father.leftWidget.slider.sliderPosition()
                value = self.father.leftWidget.image_3D_np.image[z, :, :]

            elif self.father.leftWidget.radioC.isChecked():
                y = self.father.leftWidget.slider.sliderPosition()
                value = self.father.leftWidget.image_3D_np.image[:, y, :]

            elif self.father.leftWidget.radioS.isChecked():
                x = self.father.leftWidget.slider.sliderPosition()
                value = self.father.leftWidget.image_3D_np.image[:, :, x]

            ind = value / 512
            for i in range(128):
                compteur_pixel[i] = sum(sum(i <= ind)) - sum(sum(i + 1 <= ind)) + 1
            self.abscisse.append(compteur_pixel)
            self.ChangerTestDSlider()
    def ChangerDSliderPosition(self):
        """
        Change sliders positions according to the 2 QLineEdit
        """
        if self.father.leftWidget.image_3D_np.color==False:
            try:
                mini=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.min()
                maxi=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.max()
                new_min_pos = int((float(self.inputMinDSlider.text())-mini)*65535/(maxi-mini))
                new_max_pos = int((float(self.inputMaxDSlider.text())-mini)*65535/(maxi-mini))
                new_min_pos = max(0, new_min_pos)
                new_max_pos = min(65535, new_max_pos)
                if new_min_pos > new_max_pos:
                    copy_min_pos = new_min_pos
                    new_min_pos = new_max_pos
                    new_max_pos = copy_min_pos
                print(new_min_pos, new_max_pos)
                self.Doubleslider.setValue([new_min_pos, new_max_pos])
                self.niveau_gris()
            except ValueError:
                print("erreur")

    def ChangerTestDSlider(self):
        """
        Change QLineEdit according to sliders positions
        """
        
        if self.father.leftWidget.image_3D_np.color==False:
            mini=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.min()
            maxi=self.father.leftWidget.liste_image[self.father.leftWidget.combobox.currentIndex()].image.max()
            self.inputMinDSlider.setText(str(round(self.Doubleslider.sliderPosition()[0]/65535*(maxi-mini)+mini,3)))
            self.inputMaxDSlider.setText(str(round(self.Doubleslider.sliderPosition()[1]/65535*(maxi-mini)+mini,3)))
