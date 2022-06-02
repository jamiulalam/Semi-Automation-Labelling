import os
from glob import glob
import numpy as np
import napari
from skimage.io import imread
from magicgui import magicgui
from napari.types import ImageData, LabelsData, LayerDataTuple
from magicgui.widgets import FunctionGui


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QFormLayout, QLabel, QMessageBox,\
                            QLineEdit, QFileDialog, QPushButton, QDialog, QCheckBox, QVBoxLayout, QMainWindow,QTabWidget, QMessageBox
# from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor, QIntValidator
from PyQt5.QtCore import *

from lib.salct_utils import napari_view_volume, generate_mask_from_stl
# from lib.salct_registration import *
from lib.salct_registration_lib import *




## DIALOG Windows 




class App(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.title = 'SALCT: Semi-Automated Labelling'
        self.left = 100
        self.top = 100
        self.width = 600
        self.height = 300
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.table_widget = Salct_window(self)
        self.setCentralWidget(self.table_widget)
        
        self.show()

class Salct_window(QWidget):
    
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.root_folder= '~'
        self.stl_data_path = ''
        self.destination_data_path = ''
        self.mask_size=0
        
        # Initialize tab screen
        self.tabs = QTabWidget()
        self.mask_tab = QWidget()
        self.label_tab = QWidget()
        # self.tabs.resize(300,200)
        
        # Add tabs
        self.tabs.addTab(self.mask_tab,"Mask Generation")
        self.tabs.addTab(self.label_tab,"Label Data")
        
        # Create mask tab
        self.mask_tab.layout = QVBoxLayout(self)

        self.description_text=QLabel()
        self.description_text.setText('Generate Mask Data From STL files')
        self.description_text.setFont(QFont('Arial', 15))
        self.description_text.setAlignment(Qt.AlignCenter)
        self.mask_tab.layout.addWidget(self.description_text)


        self.stl_data_button = QPushButton('Click')# browse file
        self.stl_data_button.clicked.connect(self.get_stl_data_path)
        self.destination_data_button = QPushButton('Click')# browse file
        self.destination_data_button.clicked.connect(self.get_destination_data_path)

        self.mask_res_input_field=QLineEdit()
        self.mask_res_input_field.setValidator(QIntValidator())
        # self.mask_res_input_field.setMaxLength(4000)
        self.mask_res_input_field.setAlignment(Qt.AlignRight)
        self.mask_res_input_field.setText('0')
        # self.mask_res_input_field.setFont(QFont("Arial",20)) 
        self.mask_res_input_field.textChanged.connect(self.textchanged)
        

        self.formGroupBox = QGroupBox("")
        layout = QFormLayout()
        layout.addRow(QLabel("STL data path   :"), self.stl_data_button)
        layout.addRow(QLabel("Destination path:"), self.destination_data_button)
        layout.addRow("Mask Resolution:", self.mask_res_input_field)
        self.formGroupBox.setLayout(layout)
        self.mask_tab.layout.addWidget(self.formGroupBox)





        self.generate_mask_btn = QPushButton('Generate Mask Files')
        # self.generate_mask_btn.setFont(QFont("Times", 10, QFont.Bold))
        self.generate_mask_btn.setStyleSheet("font-weight: bold")
        self.generate_mask_btn.setEnabled(False)
        self.generate_mask_btn.clicked.connect(self.run_mask_generation)
        self.mask_tab.layout.addWidget(self.generate_mask_btn)

        self.mask_tab.setLayout(self.mask_tab.layout)


        
        # Create label tab
        self.label_tab.layout = QVBoxLayout(self)

        self.mask_data_path=''
        self.volume_data_path=''

        self.description_text2=QLabel()
        self.description_text2.setText('Generate Mask Data for Volume file')
        self.description_text2.setFont(QFont('Arial', 15))
        self.description_text2.setAlignment(Qt.AlignCenter)
        self.label_tab.layout.addWidget(self.description_text2)

        self.mask_data_button = QPushButton('Click')# browse file
        self.mask_data_button.clicked.connect(self.get_mask_data_path)
        self.volume_data_button = QPushButton('Click')# browse file
        self.volume_data_button.clicked.connect(self.get_volume_data_path)

        self.formGroupBox2 = QGroupBox("")
        layout = QFormLayout()
        layout.addRow(QLabel("Mask data path   :"), self.mask_data_button)
        layout.addRow(QLabel("Volume path:"), self.volume_data_button)
        self.formGroupBox2.setLayout(layout)
        self.label_tab.layout.addWidget(self.formGroupBox2)

        self.generate_labeled_data_btn = QPushButton('Generate Labeled File')
        # self.generate_labeled_data_btn.setFont(QFont("Times", 10, QFont.Bold))
        self.generate_labeled_data_btn.setStyleSheet("font-weight: bold")
        self.generate_labeled_data_btn.setEnabled(False)
        self.generate_labeled_data_btn.clicked.connect(self.run_labled_data_generation)
        self.label_tab.layout.addWidget(self.generate_labeled_data_btn)


        self.label_tab.setLayout(self.label_tab.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def get_mask_data_path(self):
        self.mask_data_path = self.get_file_path()
        self.mask_data_button.setText(self.mask_data_path)
        self.check_data_ready_label()
    
    def get_volume_data_path(self):
        self.volume_data_path = self.get_file_path2()
        self.volume_data_button.setText(self.volume_data_path)
        self.check_data_ready_label()
    
    def get_file_path(self):
        # open select folder dialog
        data_path = QFileDialog.getOpenFileName(self, 'Select mask volume ', self.root_folder, "Mask volume (*.nrrd)")[0]
        self.root_folder = os.path.dirname(data_path)
        return data_path

    def get_file_path2(self):
        # open select folder dialog
        data_path = QFileDialog.getOpenFileName(self, 'Select data volume ', self.root_folder, "CT volume (*.nrrd)")[0]
        self.root_folder = os.path.dirname(data_path)
        return data_path


    def textchanged(self,text):
        if text=='':
            text='0'
        self.mask_size=int(text)
        self.check_data_ready()

    def get_stl_data_path(self):
        self.stl_data_path = self.get_data_path()
        self.stl_data_button.setText(self.stl_data_path)
        self.check_data_ready()

    def get_destination_data_path(self):
        self.destination_data_path = self.get_data_path2()
        self.destination_data_button.setText(self.destination_data_path)
        self.check_data_ready()

    def get_data_path(self):
        # open select folder dialog
        data_path = QFileDialog.getExistingDirectory(self, 'Select STL files directory ', self.root_folder)

        self.root_folder = os.path.dirname(data_path)
        return data_path

    def get_data_path2(self):
        # open select folder dialog
        data_path = QFileDialog.getExistingDirectory(self, 'Select Destination directory ', self.root_folder)

        self.root_folder = os.path.dirname(data_path)
        return data_path

    def check_data_ready(self):
        # disale go inteactive if one of the data is empthy
        if  os.path.exists(self.stl_data_path) and  os.path.exists(self.destination_data_path) and self.mask_size>0:
            self.generate_mask_btn.setEnabled(True)
        else:
            self.generate_mask_btn.setEnabled(False)

    def check_data_ready_label(self):
        if  os.path.exists(self.mask_data_path) and  os.path.exists(self.volume_data_path):
            self.generate_labeled_data_btn.setEnabled(True)
        else:
            self.generate_labeled_data_btn.setEnabled(False)

    def run_mask_generation(self):
        print('stl_data path :',self.stl_data_path)
        print('des_data path :',self.destination_data_path)
        print('mask size :',self.mask_size)

        mask_volume, ColorTable = generate_mask_from_stl(self.stl_data_path, volume_path=self.destination_data_path, ColorTable_file=self.destination_data_path, resolution=self.mask_size)
        napari_view_volume(mask_volume)

    def run_labled_data_generation(self):
        print('mask_data path :',self.mask_data_path)
        print('vol_data path :',self.volume_data_path)

        volume_file,_=nrrd.read(self.volume_data_path)
        mask_file,_=nrrd.read(self.mask_data_path)

        if volume_file.shape!=mask_file.shape:
            QMessageBox.warning(self, "Error", "Mask and CT volume have different dimensions!")
            return 1
        # ref_reg, volume_reg, psnr, rotation_angle, msg_reg=register_3D_volume(mask_file, volume_file, registration_type=3,normalize=True)#-21
        msg_reg='angle derived.'
        rotation_angle=-21
        if msg_reg=='angle derived.':
            ref_reg, volume_reg, psnr, rotation_angle, msg_reg=register_3D_volume(mask_file, volume_file, registration_type=2,normalize=False,rotation_angle=rotation_angle)#-21

        if msg_reg=='Registration completed.':
            des_path_file = self.volume_data_path[:len(self.volume_data_path) - 5]+'_masked.nrrd'
            nrrd.write(des_path_file, volume_reg)
            print(msg_reg)

            return 1
        
        print(msg_reg)
        return 1






def tab_GUI():
    app = QApplication(sys.argv)
    # app style
    # set_GUI_style(app)
    ex = App()
    sys.exit(app.exec_())

if __name__ == '__main__':
    tab_GUI()
