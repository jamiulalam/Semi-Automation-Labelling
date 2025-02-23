# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI\MainWindowsUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1552, 679)
        MainWindow.setWindowTitle("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.threshold_slider = QtWidgets.QSlider(self.centralwidget)
        self.threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.threshold_slider.setObjectName("threshold_slider")
        self.gridLayout_5.addWidget(self.threshold_slider, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_5.addWidget(self.label_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem, 6, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem1, 7, 0, 1, 1)
        self.vector_size = QtWidgets.QLabel(self.centralwidget)
        self.vector_size.setObjectName("vector_size")
        self.gridLayout_5.addWidget(self.vector_size, 9, 0, 1, 1)
        self.label_picked = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.label_picked.setFont(font)
        self.label_picked.setObjectName("label_picked")
        self.gridLayout_5.addWidget(self.label_picked, 8, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_5.addWidget(self.comboBox, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_5)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.frame_1 = QtWidgets.QFrame(self.centralwidget)
        self.frame_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_1.setObjectName("frame_1")
        self.gridLayout.addWidget(self.frame_1, 0, 0, 1, 1)
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.gridLayout.addWidget(self.pushButton_1, 1, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_4)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2.addWidget(self.frame_2, 1, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_2.addWidget(self.pushButton_2, 2, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_2)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_3.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_3.addWidget(self.frame_3, 0, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1552, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_data = QtWidgets.QAction(MainWindow)
        self.actionLoad_data.setObjectName("actionLoad_data")
        self.actionSave_to_database = QtWidgets.QAction(MainWindow)
        self.actionSave_to_database.setObjectName("actionSave_to_database")
        self.actionregistration = QtWidgets.QAction(MainWindow)
        self.actionregistration.setObjectName("actionregistration")
        self.actionbackground_suppression = QtWidgets.QAction(MainWindow)
        self.actionbackground_suppression.setObjectName("actionbackground_suppression")
        self.actionAnnotation = QtWidgets.QAction(MainWindow)
        self.actionAnnotation.setObjectName("actionAnnotation")
        self.actionTrain_a_model = QtWidgets.QAction(MainWindow)
        self.actionTrain_a_model.setObjectName("actionTrain_a_model")
        self.actionDeploy_a_model = QtWidgets.QAction(MainWindow)
        self.actionDeploy_a_model.setObjectName("actionDeploy_a_model")
        self.action2D_fault_detection = QtWidgets.QAction(MainWindow)
        self.action2D_fault_detection.setObjectName("action2D_fault_detection")
        self.action3D_Fault_detection = QtWidgets.QAction(MainWindow)
        self.action3D_Fault_detection.setObjectName("action3D_Fault_detection")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.label_2.setText(_translate("MainWindow", "Threshold"))
        self.vector_size.setText(_translate("MainWindow", "vector size"))
        self.label_picked.setText(_translate("MainWindow", "Picked data"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Algorithm1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Algorithm2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Algorithm3"))
        self.pushButton_1.setText(_translate("MainWindow", "Upload Before"))
        self.pushButton_2.setText(_translate("MainWindow", "Upload After"))
        self.pushButton_3.setText(_translate("MainWindow", "Run Algorithm"))
        self.actionLoad_data.setText(_translate("MainWindow", "Load data"))
        self.actionSave_to_database.setText(_translate("MainWindow", "Save to database"))
        self.actionregistration.setText(_translate("MainWindow", "Registration"))
        self.actionbackground_suppression.setText(_translate("MainWindow", "Background suppression"))
        self.actionAnnotation.setText(_translate("MainWindow", "Annotation"))
        self.actionTrain_a_model.setText(_translate("MainWindow", "Train a model"))
        self.actionDeploy_a_model.setText(_translate("MainWindow", "Deploy a model"))
        self.action2D_fault_detection.setText(_translate("MainWindow", "2D fault detection"))
        self.action3D_Fault_detection.setText(_translate("MainWindow", "3D Fault detection"))
