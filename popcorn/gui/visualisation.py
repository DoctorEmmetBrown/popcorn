import threading
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys
from qtrangeslider import QRangeSlider
from PyQt6.QtCharts import *


from input_output import *
from time import *
import imagecodecs

class Visualisation(QWidget):
    my_signal = pyqtSignal(int)
    def __init__(self, father):
        super().__init__()
        self.liste_image = []
        self.liste_log = {}
        self.father = father
        self.layoutLeft = QGridLayout()  # Son layout en grille
        self.setLayout(self.layoutLeft)

        # Boutton Load
        self.buttonLoadImage = QPushButton("Load Image")
        self.buttonLoadImage.clicked.connect(self.load)
        self.buttonLoadImage.setShortcut(QKeySequence("Ctrl+o"))
        self.layoutLeft.addWidget(self.buttonLoadImage, 0, 1)

        # Creation du ComboBox qui permet de changer d'image
        self.combobox = QComboBox()

        self.layoutLeft.addWidget(self.combobox, 0, 0)

        # Creation de la liste des filtres
        # Creation du menu deroulant lorsqu'on clique dessus
        self.list_filtre = QToolButton(self)
        self.layoutLeft.addWidget(self.list_filtre, 0, 3, 1, 2)
        self.list_filtre.setText("Filtre")
        self.list_filtre.showMenu()
        self.list_filtre.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        # Menu associe au boutton
        menu = QMenu()
        # boutton 1 du Menu
        filtre1 = QAction("Seuillage", self)
        filtre1.triggered.connect(self.filtre_seuil)
        menu.addAction(filtre1)
        
        # boutton 2 du Menu
        filtre2 = QAction("Rotation", self)
        filtre2.triggered.connect(self.rotation)
        menu.addAction(filtre2)
        # Association du Menu et de self.list_filtre
        self.list_filtre.setMenu(menu)

        filtre3 = QAction("Change Color Bool", self)
        filtre3.triggered.connect(self.change_color)
        menu.addAction(filtre3)


        self.buttonSaveImage = QPushButton("Save Image")
        self.buttonSaveImage.clicked.connect(self.save_image)
        self.buttonSaveImage.setShortcut(QKeySequence("Ctrl+s"))
        self.layoutLeft.addWidget(self.buttonSaveImage, 1, 0)


        self.buttonColor = QPushButton("Change Color")
        self.buttonColor.clicked.connect(self.changeColor)
        self.layoutLeft.addWidget(self.buttonColor, 1, 3)

        # Bar de progression
        self.prog_bar = QProgressBar(self)
        self.prog_bar.setValue(0)
        self.layoutLeft.addWidget(self.prog_bar, 1, 1, 1, 2)

        # Creation du QGraphicsScene qui prendra l'image
        self.angle=0
        self.colorPen=QColor("blue")
        self.scene = QGraphicsScene()
        self.IdImage = None
        self.factor = 1
        self.view = QGV(self.scene, self.my_signal)
        self.view.show()
        self.view.mapFromScene(0, 0)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.view.wheelEvent
        self.view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.view.resize(850, 850)
        self.label = QLabel(self)
        self.layoutLeft.addWidget(self.view, 2, 0, 5, 5)
        self.my_signal.connect(self.scroll)

        # Creer 3 boutons radio (= ne peuvent pqs etre coche en meme temps) pour correspondre aux 3 angles de vu differents

        self.radioA = QRadioButton("Axial", self)

        self.layoutLeft.addWidget(self.radioA, 2, 0)

        self.radioC = QRadioButton("Coronal", self)

        self.layoutLeft.addWidget(self.radioC, 3, 0)

        self.radioS = QRadioButton("Sagittal", self)

        self.layoutLeft.addWidget(self.radioS, 4, 0)

        # Creation du slider pour visualiser les differentes couches
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.initStyleOption
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(10)
        self.layoutLeft.addWidget(self.slider, 7, 0)

        # Creation d'un QLineEdit pouvoir voir la position du pointeur
        self.input = QLineEdit()
        self.layoutLeft.addWidget(self.input, 7, 1)
        self.input.setText(str(self.slider.sliderPosition()))
        self.input.returnPressed.connect(self.ChangeSliderPosition)

        # telechqrgement de la premiere image
        dr = "Image Default"
        color=False
        name_file = "No Image"
        image = open_sequence(dr)  # On load les images dans un numpy.ndarray
        Image_load = np.copy(image)
        self.ajouter_image(name_file, Image_load,color)
        self.changer_Image()


        self.radioA.toggled.connect(lambda: self.change_display(0))
        self.radioA.toggled.connect(self.Set_min_max_slider)
        self.radioC.toggled.connect(lambda: self.change_display(0))
        self.radioC.toggled.connect(self.Set_min_max_slider)
        self.radioS.toggled.connect(lambda: self.change_display(0))
        self.radioS.toggled.connect(self.Set_min_max_slider)

        self.radioA.toggle()

        self.combobox.currentIndexChanged.connect(self.index_changed)
        self.slider.sliderMoved.connect(self.slider_position)
        
    def rotation(self):
        """
        increase self.angle by 90
        Corresponds to the rotation of the image in degree
        """
        self.angle=(self.angle+90)%360
        self.change_display(self.slider.sliderPosition())
        
    def changeColor(self):
        """
        Open a window to change the color of the pen
        """
        self.colorPen=QColorDialog.getColor(self.colorPen)

    def save_image(self):
        """
        Save the current image in a directory that the user can choose
        Image will be save as tif
        """
        x = self.combobox.currentIndex()
        self.image_3D_np = Image(self.liste_image[x].image, self.liste_image[x].color)
        path=QFileDialog.getExistingDirectory(self, str("Open Directory"))+"/"
        save_tif_sequence(np.copy(self.liste_image[x].image),path)
        if self.combobox.currentText() in self.liste_log:
            file_log= open(path+"log.txt", "x")
            file_log.write(self.liste_log[self.combobox.currentText()])
        
    def changer_Image(self):
        """
        Change self.image_3D_np according to which image should be displayed 
        """
        x = self.combobox.currentIndex()
        self.image_3D_np = Image(self.liste_image[x].image, self.liste_image[x].color)
        if self.image_3D_np.color==False:
            mini=self.image_3D_np.image.min()
            maxi=self.image_3D_np.image.max()
            if mini != maxi:
                self.image_3D_np.image=self.image_3D_np.image-mini

                self.image_3D_np.image=self.image_3D_np.image*65535/(maxi-mini)

                self.image_3D_np.image[self.image_3D_np.image>65535]=65535

                self.image_3D_np.image = self.image_3D_np.image.astype("uint16")

    def ajouter_image(self, name, im,color):
        """
        adds an image to the Image list and adds it to the QCombobox with the name given as argument
        name=str
        im=np.array
        color=bool
        """
        new_im=Image(im,color)
        self.liste_image.append(new_im)

        self.combobox.addItems([name])
        if self.father.rightWidgetTwo != None:
            for box in self.father.rightWidgetTwo.Liste_Combo:
                box.addItems([name])
        if self.father.rightWidgetRecalage != None:
            self.father.rightWidgetRecalage.ComboBoxMov.addItems([name])
            self.father.rightWidgetRecalage.ComboBoxRef.addItems([name])
            self.father.rightWidgetRecalage.ComboBoxMaskMov.addItems([name])
            self.father.rightWidgetRecalage.ComboBoxMaskRef.addItems([name])

    def load(self):
        """
        Load a image from a directory (edf or tif)
        You have to say if it's in color or not, but it can modify later
        """
        dr = QFileDialog.getExistingDirectory(self, str("Open Directory"))
        
        dlg = CustomDialogColor()
        if dlg.exec():
            color=True
            print("succes")
        else:
            color=False
        
        
        
        
        name_file = dr.split("/")[-1]

        image = open_sequence(dr)  # On load les images dans un numpy.ndarray
        Image_load = np.copy(image)


        self.ajouter_image(name_file, Image_load,color)

        self.changer_Image()

    def change_display(self, i=int(0)):
        """
        Changes the display. i corresponds to the depth according to which the image is displayed (see slider)
        i=int
        """
        self.clean_display()

        if self.radioA.isChecked():
            im = self.image_3D_np.image[i, :, :]

        elif self.radioC.isChecked():
            im = self.image_3D_np.image[:, i, :]

        elif self.radioS.isChecked():
            im = self.image_3D_np.image[:, :, i]
        im_np = np.copy(im)
        
        im_np=im_np.astype("uint32")

        
        
        if self.image_3D_np.color:
            im_np=im_np.astype("uint32")
            qimage = QImage(
                im_np.data,
                im_np.shape[1],
                im_np.shape[0],
                im_np.strides[0],
                QImage.Format.Format_RGB32,
            )
        else:
            im_np=im_np.astype("uint16")
            qimage = QImage(
                im_np.data,
                im_np.shape[1],
                im_np.shape[0],
                im_np.strides[0],
                QImage.Format.Format_Grayscale16,
            )
        t=QTransform()
        qimage=qimage.transformed(t.rotate(self.angle))
        if qimage.width() > 850 or qimage.height() > 850:
            qimage = qimage.scaled(850, 850, Qt.AspectRatioMode.KeepAspectRatio)
        self.pixmap = QPixmap.fromImage(qimage)
        w = self.pixmap.width()
        h = self.pixmap.height()
        self.pixmap1 = self.pixmap.scaled(int(w * self.factor), int(h * self.factor))
        self.IdImage = self.scene.addPixmap(self.pixmap1)

    def clean_display(self):
        """
        removes all currently displayed items
        """
        if self.IdImage != None:
            self.scene.clear()

    def show_new_image(self):
        """
        remove all elements and display the image for i=0
        """
        self.clean_display()
        self.change_display(0)

    def Set_min_max_slider(self):
        """
        Changes the min and max of the slider according to the dimensions of the current image
        """
        self.angle=0
        self.slider.setMinimum(0)
        if self.radioA.isChecked():
            self.slider.setMaximum(self.image_3D_np.image.shape[0] - 1)
        elif self.radioC.isChecked():
            self.slider.setMaximum(self.image_3D_np.image.shape[1] - 1)
        elif self.radioS.isChecked():
            self.slider.setMaximum(self.image_3D_np.image.shape[2] - 1)
        self.slider.setValue(0)
        self.input.setText(str(self.slider.sliderPosition()))

    def slider_position(self):
        """
        remove all elements and display the image for i=current slider position
        """
        self.clean_display()
        slider_pos = self.slider.sliderPosition()
        self.change_display(slider_pos)
        self.input.setText(str(self.slider.sliderPosition()))

    def ChangeSliderPosition(self):
        """
        Changes the position of the slider according to the value modified with the keyboard
        """
        try:
            new_pos = int(self.input.text())
            new_pos = max(0, new_pos)
            if self.radioA.isChecked():
                new_pos = min(new_pos, self.image_3D_np.image.shape[0])
            elif self.radioC.isChecked():
                new_pos = min(new_pos, self.image_3D_np.image.shape[1])
            elif self.radioS.isChecked():
                new_pos = min(new_pos, self.image_3D_np.image.shape[2])
            self.slider.setValue(new_pos)
            self.input.setText(str(self.slider.sliderPosition()))
            self.slider_position(self)
            self.change_display(new_pos)
        except ValueError:
            pass

    def index_changed(self):
        """
        Changes the values of the axes of the contrast histogram
        """
        self.changer_Image()
        self.Set_min_max_slider()
        self.father.rightWidgetOne.axisY.setMax(max(self.image_3D_np.image.shape[0]*self.image_3D_np.image.shape[1],
                                                    self.image_3D_np.image.shape[0]*self.image_3D_np.image.shape[2],
                                                    self.image_3D_np.image.shape[1]*self.image_3D_np.image.shape[2]))
        self.father.rightWidgetOne.axisX.setMin(self.liste_image[self.combobox.currentIndex()].image.min())
        self.father.rightWidgetOne.axisX.setMax(self.liste_image[self.combobox.currentIndex()].image.max())
        self.slider_position()

    def filtre_seuil(self):
        """
        Selects the value of the threshold and launches the filter_threshold_exec in a thread
        """
        self.buttonLoadImage.setEnabled(False)
        self.list_filtre.setEnabled(False)
        self.combobox.setEnabled(False)
        num, ok = QInputDialog.getDouble(self, "integer input dualog", "enter a number")

    
        self.thread = threading.Thread(target=self.filtre_seuil_exec, args=([num]))
        self.thread.start()


    def filtre_seuil_exec(self, num):
        """
        Do the threshold in the current image with the value num
        """
        x = self.combobox.currentIndex()
        image_seuil = np.copy(self.liste_image[x].image)
        mini=image_seuil.min()
        maxi=image_seuil.max()
        num = max(mini, num)
        seuil = min(maxi, num)
        for z in range(image_seuil.shape[0]):
            self.prog_bar.setValue(int(100 * z / (image_seuil.shape[0] - 1)))
            image_seuil[z][image_seuil[z] < seuil] = 0
            image_seuil[z][image_seuil[z] >= seuil] = 1
     
        self.buttonLoadImage.setEnabled(True)
        self.list_filtre.setEnabled(True)
        self.combobox.setEnabled(True)
        self.prog_bar.setValue(0)
        name_file = self.combobox.currentText()
        name_file = name_file + "_seuillage_" + str(num)
        self.ajouter_image(name_file, image_seuil,False)
    
    def change_color(self):
        """
        Change the bool correpondent of if the image is in color or not
        """
        x = self.combobox.currentIndex()
        self.liste_image[x].color=not(self.liste_image[x].color)
        self.changer_Image()
        self.change_display()
        
    @pyqtSlot(int)
    def scroll(self, value):
        """
        Changes the size of the image allowing to scroll
        self.factor is the current zoom
        """
        if value == 2:
            self.factor = self.factor * 1.25
        else:
            self.factor = self.factor * 0.8
        w = self.pixmap.width()
        h = self.pixmap.height()
        self.pixmap1 = self.pixmap.scaled(int(w * self.factor), int(h * self.factor))
        self.clean_display()
        self.IdImage = self.scene.addPixmap(self.pixmap1)


class QGV(QGraphicsView):
    """
    corresponds to the display area
    """
    my_signal2 = pyqtSignal(int)

    def __init__(self, x, my_signal):
        """
        init with the good QGraphicScene and with black background
        """
        super().__init__(x)
        self.scene = x
        self.my_signal2 = my_signal
        self.press = False
        self.setBackgroundBrush(QBrush(QColor("black")))
    def wheelEvent(self, ev):
        """
        emit 2 or 0 when scrolling
        """
        if ev.angleDelta().y() > 0:  # up Wheel
            self.my_signal2.emit(2)

        elif ev.angleDelta().y() < 0:  # down Wheel
            self.my_signal2.emit(0)

    def mousePressEvent(self, event):
        """
        draw when mouse is press (1 pixel)
        """
        self.press = True
        point = event.pos()
        x = self.horizontalScrollBar().value()
        y = self.verticalScrollBar().value()
        if self.rect().contains(point):
            pen = QPen(self.parent().colorPen)
            self.scene.addEllipse(point.x() + x, point.y() + y, 1, 1, pen)

    def mouseReleaseEvent(self, event):
        """
        turn self.press to false. Call when a mouse button is release
        """
        self.press = False

    def mouseMoveEvent(self, event):
        """
        draw then mouse is move AND self.press=true (<=> mouse is pressed)
        """
        if self.press:
            point = event.pos()
            x = self.horizontalScrollBar().value()
            y = self.verticalScrollBar().value()
            if self.rect().contains(point):
                pen = QPen(self.parent().colorPen)
                self.scene.addLine(point.x() + x, point.y() + y, point.x() + x, point.y() + y, pen)
                
                
                
class Image():
    """
    corresponds to an image. There is a numpy array for the values and a boolean to say if the image is in color or not
    """
    def __init__(self,array,color):
        """
        array=np.array
        color=bool
        """
        self.image=np.copy(array)
        self.color=color
        
class CustomDialogColor(QDialog):
    """
    QDialog to ask if the image is in color or not
    """
    def __init__(self):
        super().__init__()
        QBtn = QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout = QVBoxLayout()
        message = QLabel("Is that image in color?")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
