# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI\VisualizationUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Run_Visualization(object):
    def setupUi(self, Run_Visualization):
        Run_Visualization.setObjectName("Run_Visualization")
        Run_Visualization.resize(1653, 742)
        self.centralwidget = QtWidgets.QWidget(Run_Visualization)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_picked = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_picked.setFont(font)
        self.label_picked.setObjectName("label_picked")
        self.gridLayout_5.addWidget(self.label_picked, 7, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_5.addWidget(self.comboBox, 1, 0, 1, 1)
        self.treeView_1 = QtWidgets.QTreeView(self.centralwidget)
        self.treeView_1.setObjectName("treeView_1")
        self.gridLayout_5.addWidget(self.treeView_1, 6, 0, 1, 1)
        self.SQL_table = QtWidgets.QTableWidget(self.centralwidget)
        self.SQL_table.setEnabled(True)
        self.SQL_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.SQL_table.setDragEnabled(True)
        self.SQL_table.setObjectName("SQL_table")
        self.SQL_table.setColumnCount(0)
        self.SQL_table.setRowCount(0)
        self.SQL_table.horizontalHeader().setSortIndicatorShown(True)
        self.gridLayout_5.addWidget(self.SQL_table, 3, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_5.addWidget(self.label_6, 0, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout_5.addWidget(self.label, 5, 0, 1, 1)
        self.add_scan = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.add_scan.setFont(font)
        self.add_scan.setObjectName("add_scan")
        self.gridLayout_5.addWidget(self.add_scan, 4, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 2, 0, 1, 1)
        self.msg_label = QtWidgets.QLabel(self.centralwidget)
        self.msg_label.setText("")
        self.msg_label.setObjectName("msg_label")
        self.gridLayout_5.addWidget(self.msg_label, 8, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout_5)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.plainTextEdit_1 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_1.setObjectName("plainTextEdit_1")
        self.gridLayout.addWidget(self.plainTextEdit_1, 2, 4, 1, 1)
        self.pushButton_save_1 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_save_1.setFont(font)
        self.pushButton_save_1.setObjectName("pushButton_save_1")
        self.gridLayout.addWidget(self.pushButton_save_1, 3, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 2, 1, 1)
        self.Label_plot_1 = QtWidgets.QLabel(self.centralwidget)
        self.Label_plot_1.setText("")
        self.Label_plot_1.setObjectName("Label_plot_1")
        self.gridLayout.addWidget(self.Label_plot_1, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 4, 1, 1)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.Frame_plot_1 = QtWidgets.QFrame(self.centralwidget)
        self.Frame_plot_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Frame_plot_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Frame_plot_1.setObjectName("Frame_plot_1")
        self.gridLayout_3.addWidget(self.Frame_plot_1, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_3, 2, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        Run_Visualization.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Run_Visualization)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1653, 26))
        self.menubar.setObjectName("menubar")
        Run_Visualization.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Run_Visualization)
        self.statusbar.setObjectName("statusbar")
        Run_Visualization.setStatusBar(self.statusbar)
        self.actionLoad_data = QtWidgets.QAction(Run_Visualization)
        self.actionLoad_data.setObjectName("actionLoad_data")
        self.actionSave_to_database = QtWidgets.QAction(Run_Visualization)
        self.actionSave_to_database.setObjectName("actionSave_to_database")
        self.actionregistration = QtWidgets.QAction(Run_Visualization)
        self.actionregistration.setObjectName("actionregistration")
        self.actionbackground_suppression = QtWidgets.QAction(Run_Visualization)
        self.actionbackground_suppression.setObjectName("actionbackground_suppression")
        self.actionAnnotation = QtWidgets.QAction(Run_Visualization)
        self.actionAnnotation.setObjectName("actionAnnotation")
        self.actionTrain_a_model = QtWidgets.QAction(Run_Visualization)
        self.actionTrain_a_model.setObjectName("actionTrain_a_model")
        self.actionDeploy_a_model = QtWidgets.QAction(Run_Visualization)
        self.actionDeploy_a_model.setObjectName("actionDeploy_a_model")
        self.action2D_fault_detection = QtWidgets.QAction(Run_Visualization)
        self.action2D_fault_detection.setObjectName("action2D_fault_detection")
        self.action3D_Fault_detection = QtWidgets.QAction(Run_Visualization)
        self.action3D_Fault_detection.setObjectName("action3D_Fault_detection")

        self.retranslateUi(Run_Visualization)
        QtCore.QMetaObject.connectSlotsByName(Run_Visualization)

    def retranslateUi(self, Run_Visualization):
        _translate = QtCore.QCoreApplication.translate
        Run_Visualization.setWindowTitle(_translate("Run_Visualization", "CT Inspection"))
        self.label_picked.setText(_translate("Run_Visualization", "Logfile:"))
        self.SQL_table.setSortingEnabled(True)
        self.label_6.setText(_translate("Run_Visualization", "Select the Tool:"))
        self.label.setText(_translate("Run_Visualization", "Scan\'s files"))
        self.add_scan.setText(_translate("Run_Visualization", "Add new scan"))
        self.label_5.setText(_translate("Run_Visualization", "List of available Scans"))
        self.pushButton_save_1.setText(_translate("Run_Visualization", "Save"))
        self.label_4.setText(_translate("Run_Visualization", "3D volume"))
        self.label_3.setText(_translate("Run_Visualization", "Editing Meta Data"))
        self.label_2.setText(_translate("Run_Visualization", "2D images"))
        self.actionLoad_data.setText(_translate("Run_Visualization", "Load data"))
        self.actionSave_to_database.setText(_translate("Run_Visualization", "Save to database"))
        self.actionregistration.setText(_translate("Run_Visualization", "Registration"))
        self.actionbackground_suppression.setText(_translate("Run_Visualization", "Background suppression"))
        self.actionAnnotation.setText(_translate("Run_Visualization", "Annotation"))
        self.actionTrain_a_model.setText(_translate("Run_Visualization", "Train a model"))
        self.actionDeploy_a_model.setText(_translate("Run_Visualization", "Deploy a model"))
        self.action2D_fault_detection.setText(_translate("Run_Visualization", "2D fault detection"))
        self.action3D_Fault_detection.setText(_translate("Run_Visualization", "3D Fault detection"))
