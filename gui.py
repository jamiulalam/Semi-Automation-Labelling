# importing the required libraries
import os
import shutil
import sys
import cv2
import time
from glob import glob
import nibabel as nib
import numpy as np
from PIL import Image
import json


from PyQt5.QtWidgets import *
from PyQt5.QtGui import * 
from PyQt5.QtCore import QDateTime, Qt, QTimer,pyqtSignal,QEvent
from PyQt5 import   uic
from lib.utils import *
from lib import Jutils_3D_pre_proc

# VTK
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support



## MAIN WINDOWS
class MainApp(QMainWindow):
    def __init__(self):
        #Parent constructor
        super(MainApp,self).__init__()
        self.hide()
        self.file_path = ""
        self.msg = ''
        # 2D plots 
        self.plot_1 = None
        self.plot_2 = None

        # 3D plots 
        self.vtk_plot_1 = None
        self.vtk_plot_2 = None

        self.vtk_widget_1 = None
        self.vtk_widget_2 = None
   
        self.createMenu()
        self.windows_format()

  
    def create_new_folder(self, DIR):
        if not os.path.exists(DIR):
            os.makedirs(DIR)

    def showdialog_newScan(self):
        self.dialog_widget = Dialog_ScanForm(self.root_folder, user='NA')
        self.dialog_widget.show()

    def showdialog_updateScan(self, tool_name, scan_name):
        # using the update_flag (update) to differentiate between new tool and update tool
        self.dialog_widget = Dialog_ScanForm(tool_name, scan_name=scan_name, user=self.user.user_name, update_flag=1)
        self.dialog_widget.show()
                  
    def showdialog_registration(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Do you accept the registration?")
        msg.setInformativeText("Info: Registration is the operation of geometrically  aligning two tools")
        msg.setWindowTitle("Expert validation")
        msg.setDetailedText("If you select yes, that data will be saved to database.")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.No)
        msg.buttonClicked.connect(self.msgbtn)
            
        retval = msg.exec_()

        if retval == 65536:
                print("Saving without registration\n --> outputs = ", self.output_dic)

                #save data
                # Matthew: call function to save data without preprocessing
        else: 
                print("Saving with registration\n --> outputs = ", self.output_dic)
                main_Win.showdialog_operation_success("The preprocessed data is saved to the database.")
                # Matthew: call function to save data after preprocessing
        
        print("value of pressed message box button:", retval)
    
    def msgbtn(self, i):
        print("Button pressed is:",i.text())
    
    def plot_3D_volume(self, path, vtk_widget):
        # display the selected Rek volume 
        volume_arr, vol0  = self.get_volume(path)
        volume_segments = np.array(np.unique(volume_arr.ravel()))
        self.Plot_volume(vtk_widget , vol0, volume_segments=volume_segments)
        print(' vizualized volume path: ', path)
         
        
    def windows_format(self):
        # set the title
        self.setWindowTitle("Semi-Automated Labeling for CT Images (SALCT) 2021")
        # setting  the geometry of window
        # self.setGeometry(0, 0, 400, 300)
        self.setWindowIcon(QIcon('files/icon.png'))
        # self.resize(1800, 1000)
        self.showMaximized()
        self.hide()

    def createMenu(self):
        self.hide()
        self.text = QPlainTextEdit()
        #/// menu: dataset
        self.menu_stdrd = self.menuBar().addMenu("&File")
      
        self.open_new_scan  = QAction("&Upload data")
        self.open_new_scan.triggered.connect(self.add_new_scan)
        self.menu_stdrd.addAction(self.open_new_scan)

        self.close_windows_action = QAction("&Exit")
        self.close_windows_action.triggered.connect(self.close)
        self.menu_stdrd.addAction(self.close_windows_action)

    
        # sub-menu: Vizualisation
        self.dataset_menu = self.menuBar().addMenu("&Annotation")
        self.viz_data_action = self.dataset_menu.addAction("&Vizualization")
        self.viz_data_action.triggered.connect(self.visualize_dataset)

        #/// menu: help
        self.help_menu = self.menuBar().addMenu("&Help")
        self.about_action = QAction("About")
        self.help_menu.addAction(self.about_action)
        self.help_Tutorials = QAction("Tutorials")
        self.help_menu.addAction(self.help_Tutorials)
        self.help_contact = QAction("Contact")
        self.help_menu.addAction(self.help_contact)
        self.about_action.triggered.connect(self.show_about_dialog)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def Plot_array(self, img, figure):

        dim=(500,500)
        # resize image
        image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        image = QImage(image.data, image.shape[0], image.shape[1], QImage.Format_RGB888).rgbSwapped()
        figure.setPixmap(QPixmap.fromImage(image))

    def Plot_image(self, path, figure):
        
        if self.file_image_or_volume(path) == '2D':
            msg = ''
            img = cv2.imread(path)
            dim=(500,500)
            # resize image
            image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            image = QImage(image.data, image.shape[0], image.shape[1], QImage.Format_RGB888).rgbSwapped()
            figure.setPixmap(QPixmap.fromImage(image))

            # pixmap = QPixmap(path)
            # figure.setPixmap(pixmap)
            # self.resize(100,100)
            figure.show()

        else: 
            msg = '\n Warning(2) : The selected image file is not supported!! \n - file path = '+ path

        return msg

    def get_image(self, path):
        import numpy as np
        print(' The selected image is :', path)
        filename, file_extension = os.path.splitext(path)
        img = cv2.imread(path,0)

        return img

    def get_supported_volume(self, root, ext_list = ['.rek' , '.nrrd', '.nii']):
        from glob import glob 
        vol_list = []
        for ext in ext_list:
            vol_list = vol_list + glob( os.path.join(root, '*' + ext) )

        return vol_list

    def get_volume(self, path):
        import numpy as np
        print(' Loading the file :', path)
 
        filename, file_extension = os.path.splitext(path)
        
        if os.path.exists(path):
            # read 3D volume from rek file
            if file_extension=='.nii':
                volume_array = nib.load(path).get_data()
                # volume_array =np.asanyarray( nib.load(path).dataobj)

            elif file_extension=='.nrrd':
                import nrrd
                volume_array, _ = nrrd.read(path)

            else: 
                print(' Warning: The file format is not supported. a ones volume will be generated!!!!')
                volume_array = np.ones((10, 10, 10))

        else:
            print(' Warning: The file is not found. a ones volume will be generated!!!!')
            volume_array = np.ones((10, 10, 10))
        
        # print('volume_array values= ', np.unique(volume_array.ravel()) )

        volume_vtk = get_vtk_volume_from_3d_array(volume_array)            
        # print(' volume shape = ' , volume_array.shape)
        # print(' volume unique = ' , np.unique(volume_array))
        # print(' volume segments = ' , len(np.unique(volume_array)) )
        # print(' volume type = ' , type(volume_array) )
        
        return volume_array, volume_vtk

    def file_image_or_volume(self, path):
        filename, file_extension = os.path.splitext(path)
        if file_extension =='.rek' or file_extension =='.bd' or file_extension =='.nii' or file_extension =='.nrrd':
            return '3D' 
        elif file_extension =='.tif' or file_extension =='.tiff' or file_extension =='.jpg' or file_extension =='.png':
            return '2D'

        elif file_extension =='.txt':
            return 'txt'

        else: 
            print('Error : the input format is not supported as image of volume:', path)

    def Plot_volume(self, vtk_widget, volume, volume_segments=[0]):
        # update the plot
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_widget.start()

    def initiate_widget(self, vtk_widget, vtk_plot):
        vtk_widget = QGlyphViewer()      
        vtk_layout_1 = QHBoxLayout()
        vtk_layout_1.addWidget(vtk_widget)
        vtk_layout_1.setContentsMargins(0,0,0,0)
        vtk_plot.setLayout(vtk_layout_1)
        vtk_widget.start()
        return vtk_widget, vtk_plot

    def create_vtk_widget(self, vtk_widget, vtk_plot, path0):
        vtk_widget = QGlyphViewer()      
        volume_arr, volume = self.get_volume( path0)
        volume_segments = np.array(np.unique(volume_arr.ravel()) )
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_layout_1 = QHBoxLayout()
        vtk_layout_1.addWidget(vtk_widget)
        vtk_layout_1.setContentsMargins(0,0,0,0)
        vtk_plot.setLayout(vtk_layout_1)
        vtk_widget.start()
        return vtk_widget, vtk_plot

    def save_as(self):
        # Matthew : save to database 
        path = QFileDialog.getSaveFileName(self, "Save As")[0]
        print("Save as function path:", path)
        if path:
            self.file_path = path
            self.save()

    def save(self):
        if self.file_path is None:
            print("Save path missing:", self.file_path)
            self.save_as()
        else:
            print("Save function path:", self.file_path)
            with open(self.file_path, "w") as f:
                f.write(self.text.toPlainText())
            self.text.document().setModified(False)

    def show_about_dialog(self):
        
        text = "<center>" \
            "<h1> Semi-Automated Labeling for CT Images (SALCT) 2021 </h1>" \
            "&#8291;" \
            "<img src=files/logo.png>" \
            "</center>" \
            "<p>  This software generate Semi-Automatically masks and annotations for 3D CT volume using CAD files<br/>" \
            "Email:Hossam.Gaber@ontariotechu.ca <br/></p>"
        QMessageBox.about(self, "About SALCT 2021", text)
    
    def add_new_scan(self):
        print('Add new scan',)
        Dataset_Win.show()
        main_Win.hide()

    
        
        Dataset_Win.showdialog_newScan()
    
    def visualize_dataset(self):
        print('visualize the data stored in : \n', self.file_path)
        Dataset_Win.show()
        main_Win.hide()
   

    def close_vtk_widget(self, vtk_widget):
        if not vtk_widget == None:
            ren = vtk_widget.interactor.GetRenderWindow()
            iren = ren.GetInteractor()
            ren.Finalize()
            iren.TerminateApp()

    def closeEvent(self, e):
        print('Closing the windows')
        self.close_vtk_widget(self.vtk_widget_1)
        self.close_vtk_widget(self.vtk_widget_2)
        self.close_vtk_widget(self.vtk_widget_3)

## 3D Viewer:
class QGlyphViewer(QFrame):
    arrow_picked = pyqtSignal(float)

    def __init__(self):
        super(QGlyphViewer,self).__init__()
        # Make tha actual QtWidget a child so that it can be re parented
        interactor = QVTKRenderWindowInteractor(self)
        layout = QHBoxLayout()
        layout.addWidget(interactor)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        self.text=None
       
        # Setup VTK environment
        renderer = vtk.vtkRenderer()
        render_window = interactor.GetRenderWindow()
        render_window.AddRenderer(renderer)

        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        render_window.SetInteractor(interactor)
        renderer.SetBackground(0.2,0.2,0.2)

        # set Renderer
        self.renderer = renderer
        self.interactor = interactor
        self.render_window = render_window
        # self.picker = vtk.vtkCellPicker()
        # self.picker.AddObserver("EndPickEvent", self.process_pick)
        # self.interactor.SetPicker(self.picker)

    def start(self):
        self.interactor.Initialize()
        self.interactor.Start()
        # self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.click_to_pick, 10)

    def volume_properties_setup(self, volume, volume_segments=0, max_segment=300):
        print('len(volume_segments) = ', len(volume_segments))
        # volume_segments = np.linspace(volume_segments.min(), volume_segments.max(), num=max_segment)
        if len(volume_segments)>1 and len(volume_segments)<=max_segment:
            #volume property
            volume_property = vtk.vtkVolumeProperty()
            volume_color = vtk.vtkColorTransferFunction()

            # The opacity 
            volume_scalar_opacity = vtk.vtkPiecewiseFunction()
            # The gradient opacity 
            volume_gradient_opacity = vtk.vtkPiecewiseFunction()
            volume_color = vtk.vtkColorTransferFunction()
            for i in volume_segments:
                if i==0:
                    volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
                else:
                    color_i = tuple(np.random.randint(255, size=3))
                    volume_color.AddRGBPoint(i, color_i[0]/255.0, color_i[1]/255.0, color_i[2]/255.0)
                    # volume_scalar_opacity.AddPoint(i, 0.15)
                    # volume_gradient_opacity.AddPoint(0i, 0.5)
            volume_property.SetColor(volume_color)
            # volume_property.SetScalarOpacity(volume_scalar_opacity)
            # volume_property.SetGradientOpacity(volume_gradient_opacity)

            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            # volume_property.SetDiffuse(0.6)
            # volume_property.SetSpecular(0.2)

            # # setup the properties
            volume.SetProperty(volume_property)

        return volume

    def update_volume(self, volume, volume_segments):
        
        # assign disply propoerties
        volume = self.volume_properties_setup(volume, volume_segments)
        # Finally, add the volume to the renderer
        self.renderer.RemoveAllViewProps()
        # self.renderer.AddViewProp(volume)
        self.renderer.AddVolume(volume)

        # setup the camera
        camera = self.renderer.GetActiveCamera()
        c = volume.GetCenter()
        camera.SetViewUp(0, 0, -1)
        camera.SetPosition(c[0], c[1]-1000, c[2])
        camera.SetFocalPoint(c[0], c[1], c[2])
        camera.Azimuth(0.0)
        camera.Elevation(90.0)

        # # Set a background color for the renderer
        # colors = vtk.vtkNamedColors()
        # colors.SetColor('BkgColor', [51, 77, 102, 255])
        # self.renderer.SetBackground(colors.GetColor3d('BkgColor'))

        # message = ' Volume loaded successfully!!'
        # self.showdialog_information(message)

        

    def disp_fault_volume(self, volume,fault):
        # self.renderer.RemoveAllViewProps()
        # self.renderer.AddVolume(volume)
        ##-------------------------------------------------------

        # # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
        # # to be of the colors red green and blue.
        # colorFunc = vtk.vtkColorTransferFunction()
        # colorFunc.AddRGBPoint(5, 1, 0.0, 0.0)  # Red

        # # The previous two classes stored properties. Because we want to apply these properties to the volume we want to render,
        # # we have to store them in a class that stores volume properties.
        # volumeProperty = vtk.vtkVolumeProperty()
        # volumeProperty.SetColor(colorFunc)
        # # color the faults
        # fault.SetProperty(volumeProperty)
        self.renderer.AddVolume(fault)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval


## Preprocessing
class DatasetApp(MainApp):
    def __init__(self):
        #Parent constructor
        super(DatasetApp,self).__init__()
        self.ui = None
        self.root_folder = 'data'
        self.path = self.root_folder
        self.setup()
        self.msg ='\n\n - Please click on the appropriate CT image/volume to be visualized or text to be edited . '
        self.ui.msg_label.setText(self.msg)
        self.filename =  None
        self.file_before = 'files/reference.jpg'
        self.createMenu()
        self.windows_format()
        # self.plot_sample_image()


    def setup(self):
        
        import UI.VisualizationUI 
        self.ui = UI.VisualizationUI.Ui_Run_Visualization()
        self.ui.setupUi(self)
        self.plot_1 = self.ui.Label_plot_1
        self.edit_1 = self.ui.plainTextEdit_1

    
        # list tree
        self.path = self.root_folder 
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        
        # VTK rendrer 1
        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_widget_1 = None
        path0 ='files/volume_after.rek'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)

        # VTK rendrer 2
        self.vtk_plot_2 = self.ui.Frame_plot_2
        self.vtk_widget_2 = None
        path0 ='files/volume_after.rek'
        self.vtk_widget_2, self.vtk_plot_2 = self.create_vtk_widget(self.vtk_widget_2, self.vtk_plot_2, path0)
        # buttons
        self.run_annotation = self.ui.run_annotation
        self.run_annotation.clicked.connect(self.run_annotation_algorithm)
        self.save_edit_1 = self.ui.pushButton_save_1
        self.save_edit_1.clicked.connect(self.save_txt)

    def get_selected_path(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        path = model.fileInfo(index).absoluteFilePath()
        msg = ''
        if os.path.isfile(path):
            self.file_path = path 
            if path[-5:] == '.ctbl' or path[-4:] == '.txt':
                # display the selected text 
                self.edit_1.setPlainText(open(self.file_path ).read())

            else:
            
                self.plot_3D_volume( path, self.vtk_widget_1)

        else:
            msg=  '- Warrning (1):  You selected a folder not a file!!\n - [ ' + path + ' ]' 
                      
        msg = msg + self.msg
        print(msg);self.ui.msg_label.setText(msg)
    
    def run_annotation_algorithm(self):
        a=1



    def show_scans_data(self, val):

        tool_name = self.list_tools.currentText()
        #self.treeView.setRootIndex(self.treeModel.index(self.root_folder + tool_name))
        self.path = self.root_folder + tool_name
        self.treeModel = QFileSystemModel()
        self.treeModel.setRootPath(self.path)
        self.treeView = self.ui.treeView_1
        self.treeView.setModel(self.treeModel)
        self.treeView.setRootIndex(self.treeModel.index(self.path))
        self.treeView.setColumnHidden(1,True)
        self.treeView.setColumnHidden(2,True) 
        self.treeView.setColumnHidden(3,True)
        self.treeView.installEventFilter(self) # QEvent.ContextMenu

        # for test -----------------------------------
        self.treeView.clicked.connect(lambda index: self.get_selected_path(index))
        #self.treeView.setRootIndex(self.treeModel.index(self.path))

        # SQL load details on list of tools
        if tool_name == "":
            self.table.clear()
            return
        else:
            # SQL load details on list of scans of select tool
            table_scans = sqlcon.get_all_scan_table(tool_name)

        #self.table = self.update_table(self.table, table_scans)
        self.update_table(self.table, table_scans)

    def plot_sample_image(self):
        self.Plot_image(self.file_before, self.plot_1)

    def upload_image(self):           
        self.Plot_image(self.file_before, self.plot_1)
 
    def save_txt(self):
        with open(self.file_before, "w") as f:
            f.write(self.edit_1.toPlainText())
        self.edit_1.document().setModified(False)

    def change_name(self, index):
        """ rename """
        if not index.isValid():
            return

        model = index.model()
        old_name = model.fileName(index)
        path = model.fileInfo(index).absoluteFilePath()

        # ask new name
        name, ok = QInputDialog.getText(self, "New Name", "Enter a name", QLineEdit.Normal, old_name)
        if not ok or not name:
            return
        
        # rename
        model = index.model()
        wasReadOnly = model.isReadOnly()
        model.setReadOnly(False)
        model.setData(index, name)
        model.setReadOnly(wasReadOnly)
    
    def get_selected_scan(self, index):
        """ for test """
        if not index.isValid():
            return
        model = index.model()
        raw = model.fileInfo(index).absoluteFilePath()

        msg = 'Uploading in process .... \n file = ' + raw
        print(msg);self.ui.msg_label.setText(msg)

        self.treeModel.setRootPath(self.path)

    def onStateChanged(self):
        ch = self.sender()
        print(ch.parent())
        ix = self.table.indexAt(ch.pos())
        print(ix.row(), ix.column(), ch.isChecked())


## RUN CT inspection algorithms  
class VerifyRegistration(MainApp):
    def __init__(self):
        #Parent constructor
        super(VerifyRegistration,self).__init__()
        self.config_algo = 0
        self.ui = None
        self.setup()
        self.filename =  None
        self.createMenu()
        self.windows_format()
        self.scan_sample = None
        ref_vol_array, _ = self.get_volume('xx')
        self.ref_vol_array = ref_vol_array
        vol_array, _ = self.get_volume('xx')
        self.vol_array = vol_array
        self.update_flag = 0

        self.plot_random_slice()

    def setup(self):
        
        import UI.Compare_registrationUI 
        self.ui = UI.Compare_registrationUI.Ui_Run_compare_reg()
        self.ui.setupUi(self)

        self.file_before = 'files/img_before.tif'
        self.file_after = 'files/img_after.tif'

        self.plot_1 = self.ui.Label_plot_1
        self.plot_2 = self.ui.Label_plot_2

        self.vtk_plot_1 = self.ui.Frame_plot_1
        self.vtk_plot_2 = self.ui.Frame_plot_2

        # Initialize the 3D plots
        path0 ='data/demo_data/reg.nrrd'
        self.vtk_widget_1, self.vtk_plot_1 = self.create_vtk_widget(self.vtk_widget_1, self.vtk_plot_1, path0)
        self.plot_3D_volume(path0, self.vtk_widget_1)
        path1 ='data/demo_data/vol_reg.nrrd'
        self.vtk_widget_2, self.vtk_plot_2 = self.create_vtk_widget(self.vtk_widget_2, self.vtk_plot_2, path1)
        self.plot_3D_volume(path1, self.vtk_widget_2)

        # Show 2D images first
        self.plot_1.show();self.plot_2.show()
        self.vtk_plot_1.show();self.vtk_plot_2.show()
   
        self.ui.accept_results.clicked.connect(self.accept_registration)
        self.ui.reject_results.clicked.connect(self.reject_registration)
        self.ui.get_new_slice.clicked.connect(self.plot_random_slice)    
        self.ui.go_back.clicked.connect(self.go_back_windows) 

    def plot_3D_volume(self, path, vtk_widget):
        # display the selected Rek volume 
        volume_arr, vol0  = self.get_volume(path)
        volume_segments = np.array(np.unique(volume_arr.ravel()))
        self.Plot_volume(vtk_widget , vol0, volume_segments=volume_segments)
        print(' vizualized volume path: ', path)
         
    def Plot_volume(self, vtk_widget, volume, volume_segments=[0]):
        # update the plot
        vtk_widget.update_volume(volume, volume_segments=volume_segments)
        vtk_widget.start()

    def save_array_image(self, img_array, file_path = "output/image.jpg"):
        img = Image.fromarray(img_array);  
        img.mode = 'I'; img.point(lambda i:i*(1./256)).convert('L').save(file_path) 
  
    def apply_volumes_registration_flowchart(self, ref_vol_array, vol_array, isref=False):
        # vol_reg = vol_array
        # ref_vol_reg = ref_vol_array
        save_ref_volume = isref

        #resize ratio
        res_ratio=ref_vol_array.shape[1]/vol_array.shape[1]

        ref_vol_reg, vol_reg= Jutils_3D_pre_proc.pre_proc_3d(ref_vol_array,vol_array,bg_remove_mode=2,resize_ratio=res_ratio,crop_point=0, pre_proc_ref=False, angle_step=5)

        self.vol_array=vol_reg
        return ref_vol_reg, vol_reg, save_ref_volume

    def plot_random_slice(self):
        self.ui.get_new_slice.hide()

        msg = '\n - Reference volume =  '  + str(self.ref_vol_array.shape) + '\n - Input volume =  '  + str(self.vol_array.shape) 
        print(msg)

        

        if self.ref_vol_array.shape != self.vol_array.shape:
            self.close_verification_error(msg)
        else:    

            import random
            idx = random.randint(0,self.vol_array.shape[1]-1)
            file_path = "output/image.jpg"
            self.save_array_image(self.ref_vol_array[:,idx,:], file_path) 
            # file_path =  'files/img_before.tif'
            self.Plot_image(file_path, self.plot_1)

            file_path = "output/image_reg.jpg"
            self.save_array_image(self.vol_array[:,idx,:], file_path) 
            # file_path =  'files/img_before.tif'
            self.Plot_image(file_path , self.plot_2)
            print('plotted a new slice')

        self.ui.get_new_slice.show()

    def accept_registration(self):

        self.showdialog_information('The registration is accepted. \nThe registred data will saved in the database.')
        if self.update_flag ==0:
            Dataset_Win.SQL_add_new_scan(self.scan_sample)

        else:
            Dataset_Win.SQL_update_scan(self.scan_sample)

        self.hide(); Dataset_Win.show()

    def reject_registration(self):
        self.showdialog_information('The registration was not accepted. \nThe Process will be repeated with other parameters. \
                                     \n The process will take few minutes. Please wait :)')
        self.upload_new_volume(self.scan_sample, self.update_flag)

    def close_verification_error(self, msg):
        self.hide(); Dataset_Win.show()
        self.showdialog_warnning('The selected scans does not have the same size: \n' + msg + \
                                  '\nPlease recheck the data or upload it as a new scan in a new tool!!')

    def go_back_windows(self):
        self.hide(); Dataset_Win.show()

class Dialog_ScanForm(QDialog):

    def __init__(self,  root_folder, scan_name="", user='', update_flag=0):
        super(Dialog_ScanForm, self).__init__()
        self.user = user
        self.dict_tool= None
        self.root_folder = root_folder
        self.file_path = ""
        self.update_flag = update_flag
        # the form attributs.
        self.tool_name = os.path.basename(root_folder)
        self.scan_name = QLineEdit()
        self.scan_part_id = QSpinBox()
        self.before_after_status = QCheckBox()
        self.data_verification = QCheckBox()
        self.data_verification.toggled.connect(self.Expert_verification_options)

        self.data_directory = QPushButton('Click')# browse full data folder
        self.data_directory.clicked.connect(self.get_data_directory)
        
        self.defect = QPushButton('Click')# browse file
        self.defect.clicked.connect(self.get_defect_file)

        self.image_folder = QPushButton('Click')# browse file
        self.image_folder.clicked.connect(self.get_image_folder)

        self.volume_file = QPushButton('Click')# browse file
        self.volume_file.clicked.connect(self.get_volume_file)

        self.volume_mask_file = QPushButton('Click')# browse file
        self.volume_mask_file.clicked.connect(self.get_volume_mask_file)

        self.object_material =  QLineEdit()
        self.filter_type = QLineEdit()
        self.filter_size = QLineEdit()
        self.technician_name = QLineEdit()
        self.technician_name.setText(self.user)
        self.scan_data_label = QComboBox()
        
     
        self.root_folder = root_folder
        # fill optional fileds
        self.object_material.setText("lead")
        self.filter_type.setText("copper")
        self.filter_size.setText("2.0mm")
        self.data_labels = ["I don't know"]
        self.scan_data_label.addItems(self.data_labels)
            
        # create the form .
        self.createFormGroupBox()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.get_form_data)
        buttonBox.rejected.connect(self.cancel_saving)
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)
        
        self.setWindowTitle("Add new data")
       


    def Expert_verification_options(self):
        if not self.data_verification.isChecked() :
            self.data_labels = ["I don't know"]
        else:
            self.data_labels = ["Good", "Faulty Acceptable", "Faulty Unacceptable", "I don't know"]
        self.scan_data_label.clear()
        self.scan_data_label.addItems(self.data_labels)

    def get_data_directory(self):
        # open select folder dialog
        fname = QFileDialog.getExistingDirectory(self, 'Select the folder containing all scan data', '~')
        print(fname)

        if os.path.isdir(fname):
            self.data_directory.setText(fname)
            message = 'The data folder  is selected successfully!'
            print(message)
            self.showdialog_information(message)
            scan_name = os.path.basename(fname)
            self.scan_name.setText(scan_name)
            print('the scan name is  :  ', scan_name)

        else: 
            message = 'You did not select a folder. Please select folder containing all required scan data'
            print(message)
            self.showdialog_warnning(message)
            
    def get_defect_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open defect file", "~", "Text Files (*.txt)")[0]
        print(filepath)
        self.defect.setText(filepath)

        message = 'The defect file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def get_image_folder(self):
        # open select folder dialog
        fname = QFileDialog.getExistingDirectory(self, 'Select the tif images directory', '~')

        print(fname)

        if os.path.isdir(fname):
            self.image_folder.setText(fname)
            message = 'The scan image folder  is selected successfully!'
            print(message)
            self.showdialog_information(message)

        else: 
            message = 'You did not select a folder. Please seelect folder where the tif images are saved '
            print(message)
            self.showdialog_warnning(message)

    def get_volume_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open 3D volume file", "~", "3D volume Files (*.rek *.nrrd *.nii *.bd)")[0]
        print(filepath)
        self.volume_file.setText(filepath)

        message = 'The 3D volume file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def get_volume_mask_file(self):
        filepath = QFileDialog.getOpenFileName(self, "Open 3D mask volume file", "~", "3D volume Files (*.rek *.nrrd *.nii *.bd)")[0]
        print(filepath)
        self.volume_mask_file.setText(filepath)

        message = 'The 3D volume mask file is selected successfully!'
        print(message)
        self.showdialog_information(message)

    def showdialog_information(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Information message")
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)

    def showdialog_warnning(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Error message")
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval
        
    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Scan form [tool = " + self.tool_name+ "]")
        layout = QFormLayout()   

        if self.update_flag == 0:
            layout.addRow(QLabel("Data directory:"), self.data_directory)

        else:
            layout.addRow(QLabel("Defect file:"), self.defect)
            layout.addRow(QLabel("Tif images folder:"), self.image_folder)
            layout.addRow(QLabel("Volume file:"), self.volume_file)
            layout.addRow(QLabel("Volume mask file:"), self.volume_mask_file)
        
        layout.addRow(QLabel("Was the scan verified by an expert?:"), self.data_verification)
        layout.addRow(QLabel("Was the scan before the tool usage?:"), self.before_after_status)
        layout.addRow(QLabel("Tool status:"), self.scan_data_label)

        layout.addRow(QLabel("Scan name:"), self.scan_name)
        layout.addRow(QLabel("Scan part:"), self.scan_part_id)
        
           
        layout.addRow(QLabel("Object material type:"), self.object_material)
        layout.addRow(QLabel("Filter type:"), self.filter_type)
        layout.addRow(QLabel("Filter size:"), self.filter_size)
        layout.addRow(QLabel("Technician name:"), self.technician_name)
        self.formGroupBox.setLayout(layout)

    def get_path_button(self, var_name):

        path = var_name.text()

        if path == 'Click':
            path = ''

        return path

    def isImg(self, fname):
        img_ext = ['png', 'bmp', 'jpg', 'tif', 'tiff']
        if fname.split('.')[-1] in img_ext:
            return True
        else:
            return False
    
    def get_supported_volume(self, root, ext_list = ['.rek' , '.nrrd', '.nii']):
        from glob import glob 
        vol_list = []
        for ext in ext_list:
            vol_list = vol_list + glob( os.path.join(root, '*' + ext) )

        return vol_list

    def get_form_data(self):
        tool_name = self.tool_name
        scan_name = self.scan_name.text() 
        scan_part_id = self.scan_part_id.text()
        before_after_status = self.before_after_status.isChecked()
        data_verification = self.data_verification.isChecked()
        
        if self.update_flag == 0:
            scan_data_label = self.scan_data_label.currentText()
            data_directory = self.get_path_button(self.data_directory)
            if data_directory in ["", "Click"]:
                msg = "Path to data directory has not been provided!!"
                Dataset_Win.showdialog_warnning(msg)
                return
            
            list_files = [os.path.join(data_directory, x) for x in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, x))]
            list_folders = [os.path.join(data_directory, x) for x in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, x)) and len(os.listdir(os.path.join(data_directory, x))) > 0]
            list_txt = [x for x in list_files if '.txt' in x]
            list_img = [x for x in list_folders if self.isImg(os.listdir(x)[0])]
            list_rek = []

            list_rek = self.get_supported_volume(data_directory)

            # flag : which folder/files to choose from the list. defaut 0
            defect_file = list_txt[0] if len(list_txt) > 0 else ""
            image_folder = list_img[0] if len(list_img) > 0 else ""
            volume_file = list_rek[0] if len(list_rek) > 0 else ""
            volume_mask_file = ""
            
        else:
            data_directory = None
            scan_data_label = None
            defect_file = "" if self.defect != "" else self.get_path_button(self.defect)
            image_folder = self.get_path_button(self.image_folder)
            volume_file = self.get_path_button(self.volume_file)
            volume_mask_file = self.get_path_button(self.volume_mask_file)
        
        object_material =  self.object_material.text()
        filter_type = self.filter_type.text()
        filter_size = self.filter_size.text()
        technician_name = self.technician_name.text()
            
        if scan_data_label == "Good":
            scan_usage = "This tool is safe for reuse in the reactor."
            defect = 'defect-free'
        elif scan_data_label == "Faulty Acceptable":
            scan_usage = "This tool is defective but it still safe for reuse in the reactor."
            defect = 'defective-acceptable'
        elif scan_data_label == "Faulty Unacceptable":
            scan_usage = "Do not use it to rector. Need fixing/visual inspection "
            defect = 'defective'
        elif scan_data_label == "I don't know":
            scan_usage = "Unknown, needs inspection (software + visual verification)"
            defect = 'Unknown'
        else:
            scan_usage = None
            defect = ""

        self.dict_scan = {
             'tool_name':tool_name, 'scan_name':scan_name, 'scan_part_id':scan_part_id, 'data_directory':data_directory,  'before_after_status':before_after_status,
             'data_verification':data_verification, 'defect_file':defect_file, 'defect':defect , 'image_location':image_folder, 'volume_location':volume_file,
             'mask_location':volume_mask_file, 'object_material':object_material, 'filter_type':filter_type, 'filter_size':filter_size,
             'technician_name':technician_name, 'scan_usage':scan_usage, 'scan_data_label':scan_data_label
        }

        if self.update_flag == 1:
            # Scan sanity check
            err, mssg = self.check_scan_data()
            if err == 0  :
                self.close()
            else:
                Dataset_Win.showdialog_warnning(mssg)
                return
        else:
            self.close()
        VerifyReg_Win.upload_new_volume(self.dict_scan, self.update_flag)

    def check_scan_data(self): 
        print('==> save the new scan of tool [%s] in MySQL : \n'%(self.tool_name))
        mssg = ''
        err = 0
        print('scan_name:',self.scan_name.text())
        if self.scan_name.text() == '' :
            mssg = mssg + '- Please type a non-empty  scan name !\n'
            err = err + 1

        defect = self.get_path_button(self.defect)
        print('defect:',defect)
  
        if  os.path.isfile(defect ) == False:
            mssg = mssg + '- Please select the defect file !\n'
            err = err + 1
        
        import glob 
        image_folder = self.get_path_button(self.image_folder)
        print('image_folder :',image_folder)
        if  len(glob.glob( os.path.join(image_folder , "*.tif" ) ) ) == 0 and len(glob.glob( os.path.join(image_folder , "*.tiff" ) ) ) == 0 :
            mssg = mssg + '- Please select folder containing tif images!\n'
            err = err + 1

        volume_file = self.get_path_button(self.volume_file)
        print('volume_file :',volume_file)

        if  os.path.isfile(volume_file) == False :
            mssg = mssg + '- Please select the volume file !\n'
            err = err + 1

        print('volume_mask_file :',self.get_path_button(self.volume_mask_file))
        print('object_material :',self.object_material.text())
        print('filter_type:',self.filter_type.text())
        print('filter_size :',self.filter_size.text())
        print('technician_name :',self.technician_name.text())

        return err, mssg


    def cancel_saving(self):

        print('cancel the saving of the new tool in MySQL')
        self.close()

def compile_ui(path_):
    from glob import glob
    list_ui = glob(path_+'.ui')
    print('list_ui=', list_ui)
    for filename in list_ui:
    # Recompile ui
        with open(filename) as ui_file:
            with open(filename.replace('.ui','.py'),"w") as py_ui_file:
                uic.compileUi(ui_file,py_ui_file)

def init_windows():
    # main_Win.show_about_dialog()
    # main_Win.paintEngine()
    main_Win.show()
    Dataset_Win.hide()
    VerifyReg_Win.hide()


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    compile_ui("UI/*")
    App = QApplication(sys.argv)
    main_Win = MainApp()
    Dataset_Win = DatasetApp()
    VerifyReg_Win = VerifyRegistration()
    # show the starting window
    init_windows()
    # start the app
    App.setStyle(QStyleFactory.create('Fusion'))
    sys.exit(App.exec())
