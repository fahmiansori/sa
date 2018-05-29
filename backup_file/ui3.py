# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

# WARNING: check point > tab word fix, next and back button to finish

from PyQt5 import QtCore, QtGui, QtWidgets
from app import App
import re

class Ui_MainWindow(object):
    def __init__(self):
        self.app = App()

        # tab setup
        self.selectedTable = ""
        self.setupTextCol = "text"
        self.setupClassCol = "clas"
        self.exceptCol = []

        # tab fix word
        # self.fixWord_db = "sentiment_analysis"
        self.fixWord_cols = ['id','word','word_fix']
        self.fixWord_fixtable = "fix_word"
        self.fixWord_indexData = None
        self.fixWord_idData = 0

        # tab preprocessing
        self.preprocessing_doPreprocessing = True
        self.preprocessing_doFeatureSelection = False
        self.preprocessing_numFeatureToRetain = 0
        self.preprocessing_thresholdFeatureIgnore = 0
        self.vsmFeature = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setToolTip("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_setup = QtWidgets.QWidget()
        self.tab_setup.setObjectName("tab_setup")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_setup)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem, 2, 0, 1, 1)
        self.pushButton_setup_next = QtWidgets.QPushButton(self.tab_setup)
        self.pushButton_setup_next.setObjectName("pushButton_setup_next")
        self.gridLayout_3.addWidget(self.pushButton_setup_next, 2, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_setup)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lineEdit_setup_except_col = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_setup_except_col.setObjectName("lineEdit_setup_except_col")
        self.gridLayout_6.addWidget(self.lineEdit_setup_except_col, 2, 1, 1, 1)
        self.lineEdit_setup_clas_col = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_setup_clas_col.setObjectName("lineEdit_setup_clas_col")
        self.gridLayout_6.addWidget(self.lineEdit_setup_clas_col, 1, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 1, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 2, 0, 1, 1)
        self.lineEdit_setup_text_col = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_setup_text_col.setObjectName("lineEdit_setup_text_col")
        self.gridLayout_6.addWidget(self.lineEdit_setup_text_col, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem1, 1, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem2, 3, 1, 1, 1)
        self.label_6.raise_()
        self.label_7.raise_()
        self.label_8.raise_()
        self.lineEdit_setup_except_col.raise_()
        self.lineEdit_setup_clas_col.raise_()
        self.lineEdit_setup_text_col.raise_()
        self.gridLayout_3.addWidget(self.groupBox_2, 1, 0, 1, 2)
        self.groupBox = QtWidgets.QGroupBox(self.tab_setup)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
        self.comboBox_setup_table = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_setup_table.setObjectName("comboBox_setup_table")
        self.gridLayout_4.addWidget(self.comboBox_setup_table, 6, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 3, 0, 1, 1)
        self.lineEdit_setup_db = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_setup_db.setObjectName("lineEdit_setup_db")
        self.gridLayout_4.addWidget(self.lineEdit_setup_db, 3, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 6, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem3, 4, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.lineEdit_setup_password = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_setup_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_setup_password.setObjectName("lineEdit_setup_password")
        self.gridLayout_4.addWidget(self.lineEdit_setup_password, 2, 1, 1, 1)
        self.lineEdit_setup_host = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_setup_host.setText("")
        self.lineEdit_setup_host.setObjectName("lineEdit_setup_host")
        self.gridLayout_4.addWidget(self.lineEdit_setup_host, 0, 1, 1, 1)
        self.lineEdit_setup_user = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_setup_user.setObjectName("lineEdit_setup_user")
        self.gridLayout_4.addWidget(self.lineEdit_setup_user, 1, 1, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem4, 0, 0, 1, 1)
        self.pushButton_setup_connect = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_setup_connect.setObjectName("pushButton_setup_connect")
        self.gridLayout_5.addWidget(self.pushButton_setup_connect, 0, 1, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem5, 0, 2, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 4, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout.addWidget(self.label_12)
        self.label_setup_status_connect = QtWidgets.QLabel(self.groupBox)
        self.label_setup_status_connect.setObjectName("label_setup_status_connect")
        self.horizontalLayout.addWidget(self.label_setup_status_connect)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.gridLayout_4.addLayout(self.horizontalLayout, 5, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 0, 0, 1, 2)
        self.tabWidget.addTab(self.tab_setup, "")
        self.tab_fix_word = QtWidgets.QWidget()
        self.tab_fix_word.setObjectName("tab_fix_word")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_fix_word)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.tableWidget_fixword = QtWidgets.QTableWidget(self.tab_fix_word)
        self.tableWidget_fixword.setObjectName("tableWidget_fixword")
        self.tableWidget_fixword.setColumnCount(0)
        self.tableWidget_fixword.setRowCount(0)
        self.gridLayout_10.addWidget(self.tableWidget_fixword, 0, 0, 1, 1)
        self.gridLayout_7.addLayout(self.gridLayout_10, 0, 2, 4, 3)
        self.pushButton_fixword_next = QtWidgets.QPushButton(self.tab_fix_word)
        self.pushButton_fixword_next.setObjectName("pushButton_fixword_next")
        self.gridLayout_7.addWidget(self.pushButton_fixword_next, 4, 4, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem7, 4, 2, 1, 1)
        self.pushButton_fixword_save = QtWidgets.QPushButton(self.tab_fix_word)
        self.pushButton_fixword_save.setObjectName("pushButton_fixword_save")
        self.gridLayout_7.addWidget(self.pushButton_fixword_save, 2, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem8, 3, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.tab_fix_word)
        self.label_9.setObjectName("label_9")
        self.gridLayout_7.addWidget(self.label_9, 0, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.tab_fix_word)
        self.label_10.setObjectName("label_10")
        self.gridLayout_7.addWidget(self.label_10, 1, 0, 1, 1)
        self.lineEdit_fixword_word = QtWidgets.QLineEdit(self.tab_fix_word)
        self.lineEdit_fixword_word.setReadOnly(True)
        self.lineEdit_fixword_word.setPlaceholderText("")
        self.lineEdit_fixword_word.setObjectName("lineEdit_fixword_word")
        self.gridLayout_7.addWidget(self.lineEdit_fixword_word, 0, 1, 1, 1)
        self.lineEdit_fixword_wordfix = QtWidgets.QLineEdit(self.tab_fix_word)
        self.lineEdit_fixword_wordfix.setObjectName("lineEdit_fixword_wordfix")
        self.gridLayout_7.addWidget(self.lineEdit_fixword_wordfix, 1, 1, 1, 1)
        self.pushButton_fixword_back = QtWidgets.QPushButton(self.tab_fix_word)
        self.pushButton_fixword_back.setObjectName("pushButton_fixword_back")
        self.gridLayout_7.addWidget(self.pushButton_fixword_back, 4, 3, 1, 1)
        self.tabWidget.addTab(self.tab_fix_word, "")
        self.tab_preprocessing = QtWidgets.QWidget()
        self.tab_preprocessing.setObjectName("tab_preprocessing")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.tab_preprocessing)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.pushButton_preprocessing_next = QtWidgets.QPushButton(self.tab_preprocessing)
        self.pushButton_preprocessing_next.setObjectName("pushButton_preprocessing_next")
        self.gridLayout_11.addWidget(self.pushButton_preprocessing_next, 2, 2, 1, 1)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_11.addItem(spacerItem9, 2, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_preprocessing)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.tableWidget_preprocessing_features_list_after = QtWidgets.QTableWidget(self.groupBox_4)
        self.tableWidget_preprocessing_features_list_after.setObjectName("tableWidget_preprocessing_features_list_after")
        self.tableWidget_preprocessing_features_list_after.setColumnCount(0)
        self.tableWidget_preprocessing_features_list_after.setRowCount(0)
        self.gridLayout_13.addWidget(self.tableWidget_preprocessing_features_list_after, 3, 0, 1, 2)
        self.tableWidget_preprocessing_vsm = QtWidgets.QTableWidget(self.groupBox_4)
        self.tableWidget_preprocessing_vsm.setObjectName("tableWidget_preprocessing_vsm")
        self.tableWidget_preprocessing_vsm.setColumnCount(0)
        self.tableWidget_preprocessing_vsm.setRowCount(0)
        self.gridLayout_13.addWidget(self.tableWidget_preprocessing_vsm, 3, 2, 1, 6)
        self.gridLayout_14 = QtWidgets.QGridLayout()
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.label_preprocessing_feature_after = QtWidgets.QLabel(self.groupBox_4)
        self.label_preprocessing_feature_after.setObjectName("label_preprocessing_feature_after")
        self.gridLayout_14.addWidget(self.label_preprocessing_feature_after, 2, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox_4)
        self.label_21.setObjectName("label_21")
        self.gridLayout_14.addWidget(self.label_21, 2, 5, 1, 1)
        self.label_preprocessing_feature_before = QtWidgets.QLabel(self.groupBox_4)
        self.label_preprocessing_feature_before.setObjectName("label_preprocessing_feature_before")
        self.gridLayout_14.addWidget(self.label_preprocessing_feature_before, 1, 2, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_4)
        self.label_17.setObjectName("label_17")
        self.gridLayout_14.addWidget(self.label_17, 1, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox_4)
        self.label_19.setObjectName("label_19")
        self.gridLayout_14.addWidget(self.label_19, 2, 3, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.groupBox_4)
        self.label_18.setObjectName("label_18")
        self.gridLayout_14.addWidget(self.label_18, 2, 1, 1, 1)
        self.label_preprocessing_feature_reduction = QtWidgets.QLabel(self.groupBox_4)
        self.label_preprocessing_feature_reduction.setObjectName("label_preprocessing_feature_reduction")
        self.gridLayout_14.addWidget(self.label_preprocessing_feature_reduction, 2, 4, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_4)
        self.label_14.setObjectName("label_14")
        self.gridLayout_14.addWidget(self.label_14, 1, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_4)
        self.label_15.setObjectName("label_15")
        self.gridLayout_14.addWidget(self.label_15, 2, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_14.addItem(spacerItem10, 2, 6, 1, 1)
        self.gridLayout_13.addLayout(self.gridLayout_14, 0, 0, 1, 8)
        self.label_16 = QtWidgets.QLabel(self.groupBox_4)
        self.label_16.setObjectName("label_16")
        self.gridLayout_13.addWidget(self.label_16, 2, 2, 1, 3)
        self.label_20 = QtWidgets.QLabel(self.groupBox_4)
        self.label_20.setObjectName("label_20")
        self.gridLayout_13.addWidget(self.label_20, 2, 0, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox_4, 1, 0, 1, 3)
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_preprocessing)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setObjectName("label_11")
        self.gridLayout_12.addWidget(self.label_11, 2, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setObjectName("label_13")
        self.gridLayout_12.addWidget(self.label_13, 3, 1, 1, 1)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem11, 2, 4, 1, 1)
        self.lineEdit_preprocessing_threshold = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_preprocessing_threshold.setObjectName("lineEdit_preprocessing_threshold")
        self.gridLayout_12.addWidget(self.lineEdit_preprocessing_threshold, 3, 2, 1, 2)
        self.lineEdit_preprocessing_numberoffeature = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_preprocessing_numberoffeature.setObjectName("lineEdit_preprocessing_numberoffeature")
        self.gridLayout_12.addWidget(self.lineEdit_preprocessing_numberoffeature, 2, 2, 1, 2)
        self.pushButton_preprocessing_process = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_preprocessing_process.setObjectName("pushButton_preprocessing_process")
        self.gridLayout_12.addWidget(self.pushButton_preprocessing_process, 4, 3, 1, 2)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem12, 4, 2, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_12.addItem(spacerItem13, 4, 5, 1, 1)
        self.progressBar_preprocessing = QtWidgets.QProgressBar(self.groupBox_3)
        self.progressBar_preprocessing.setProperty("value", 0)
        self.progressBar_preprocessing.setObjectName("progressBar_preprocessing")
        self.gridLayout_12.addWidget(self.progressBar_preprocessing, 5, 2, 1, 4)
        self.gridLayout_20 = QtWidgets.QGridLayout()
        self.gridLayout_20.setObjectName("gridLayout_20")
        self.radioButton_preprocessing_nofeatureselection = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_preprocessing_nofeatureselection.setObjectName("radioButton_preprocessing_nofeatureselection")
        self.buttonGroup_2 = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radioButton_preprocessing_nofeatureselection)
        self.gridLayout_20.addWidget(self.radioButton_preprocessing_nofeatureselection, 0, 0, 1, 1)
        self.radioButton_preprocessing_featureselection_ig = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_preprocessing_featureselection_ig.setObjectName("radioButton_preprocessing_featureselection_ig")
        self.buttonGroup_2.addButton(self.radioButton_preprocessing_featureselection_ig)
        self.gridLayout_20.addWidget(self.radioButton_preprocessing_featureselection_ig, 1, 0, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_20, 0, 2, 1, 1)
        self.gridLayout_19 = QtWidgets.QGridLayout()
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.radioButton_preprocessing_nopreprocessing = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_preprocessing_nopreprocessing.setObjectName("radioButton_preprocessing_nopreprocessing")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton_preprocessing_nopreprocessing)
        self.gridLayout_19.addWidget(self.radioButton_preprocessing_nopreprocessing, 0, 0, 1, 1)
        self.radioButton_preprocessing_withpreprocessing = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_preprocessing_withpreprocessing.setObjectName("radioButton_preprocessing_withpreprocessing")
        self.buttonGroup.addButton(self.radioButton_preprocessing_withpreprocessing)
        self.gridLayout_19.addWidget(self.radioButton_preprocessing_withpreprocessing, 1, 0, 1, 1)
        self.gridLayout_12.addLayout(self.gridLayout_19, 0, 1, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox_3, 0, 0, 1, 3)
        self.pushButton_preprocessing_back = QtWidgets.QPushButton(self.tab_preprocessing)
        self.pushButton_preprocessing_back.setObjectName("pushButton_preprocessing_back")
        self.gridLayout_11.addWidget(self.pushButton_preprocessing_back, 2, 1, 1, 1)
        self.tabWidget.addTab(self.tab_preprocessing, "")
        self.tab_training = QtWidgets.QWidget()
        self.tab_training.setObjectName("tab_training")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.tab_training)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.pushButton_training_back = QtWidgets.QPushButton(self.tab_training)
        self.pushButton_training_back.setObjectName("pushButton_training_back")
        self.gridLayout_15.addWidget(self.pushButton_training_back, 2, 1, 1, 1)
        self.pushButton_training_analys = QtWidgets.QPushButton(self.tab_training)
        self.pushButton_training_analys.setObjectName("pushButton_training_analys")
        self.gridLayout_15.addWidget(self.pushButton_training_analys, 2, 2, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_15.addItem(spacerItem14, 2, 0, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_training)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_16.setObjectName("gridLayout_16")
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_16.addItem(spacerItem15, 5, 4, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.groupBox_5)
        self.label_34.setObjectName("label_34")
        self.gridLayout_16.addWidget(self.label_34, 6, 4, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.groupBox_5)
        self.label_25.setObjectName("label_25")
        self.gridLayout_16.addWidget(self.label_25, 3, 0, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.groupBox_5)
        self.label_23.setObjectName("label_23")
        self.gridLayout_16.addWidget(self.label_23, 4, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.groupBox_5)
        self.label_24.setObjectName("label_24")
        self.gridLayout_16.addWidget(self.label_24, 5, 0, 1, 1)
        self.label_training_total_data = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_total_data.setObjectName("label_training_total_data")
        self.gridLayout_16.addWidget(self.label_training_total_data, 4, 3, 1, 1)
        self.label_training_total_feature = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_total_feature.setObjectName("label_training_total_feature")
        self.gridLayout_16.addWidget(self.label_training_total_feature, 5, 3, 1, 1)
        self.label_training_time = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_time.setObjectName("label_training_time")
        self.gridLayout_16.addWidget(self.label_training_time, 6, 3, 1, 1)
        self.label_training_total_class = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_total_class.setObjectName("label_training_total_class")
        self.gridLayout_16.addWidget(self.label_training_total_class, 3, 3, 1, 1)
        self.label_training_data_perclass = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_data_perclass.setText("")
        self.label_training_data_perclass.setObjectName("label_training_data_perclass")
        self.gridLayout_16.addWidget(self.label_training_data_perclass, 4, 4, 1, 3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem16 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem16)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        spacerItem17 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem17)
        self.gridLayout_16.addLayout(self.horizontalLayout_2, 0, 0, 1, 8)
        self.label_training_status = QtWidgets.QLabel(self.groupBox_5)
        self.label_training_status.setText("")
        self.label_training_status.setObjectName("label_training_status")
        self.gridLayout_16.addWidget(self.label_training_status, 2, 0, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.groupBox_5)
        self.label_26.setObjectName("label_26")
        self.gridLayout_16.addWidget(self.label_26, 6, 0, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.groupBox_5)
        self.label_27.setObjectName("label_27")
        self.gridLayout_16.addWidget(self.label_27, 3, 2, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.groupBox_5)
        self.label_28.setObjectName("label_28")
        self.gridLayout_16.addWidget(self.label_28, 4, 2, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.groupBox_5)
        self.label_30.setObjectName("label_30")
        self.gridLayout_16.addWidget(self.label_30, 6, 2, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.groupBox_5)
        self.label_29.setObjectName("label_29")
        self.gridLayout_16.addWidget(self.label_29, 5, 2, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_16.addItem(spacerItem18, 5, 6, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_16.addItem(spacerItem19, 5, 5, 1, 1)
        self.gridLayout_15.addWidget(self.groupBox_5, 0, 0, 1, 3)
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_training)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.label_22 = QtWidgets.QLabel(self.groupBox_6)
        self.label_22.setObjectName("label_22")
        self.gridLayout_17.addWidget(self.label_22, 0, 0, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_17.addItem(spacerItem20, 4, 0, 1, 4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem21 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem21)
        self.pushButton_training_test = QtWidgets.QPushButton(self.groupBox_6)
        self.pushButton_training_test.setObjectName("pushButton_training_test")
        self.horizontalLayout_3.addWidget(self.pushButton_training_test)
        spacerItem22 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem22)
        self.gridLayout_17.addLayout(self.horizontalLayout_3, 2, 0, 1, 4)
        self.label_31 = QtWidgets.QLabel(self.groupBox_6)
        self.label_31.setObjectName("label_31")
        self.gridLayout_17.addWidget(self.label_31, 1, 0, 1, 1)
        spacerItem23 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_17.addItem(spacerItem23, 1, 2, 1, 1)
        self.lineEdit_training_fold = QtWidgets.QLineEdit(self.groupBox_6)
        self.lineEdit_training_fold.setObjectName("lineEdit_training_fold")
        self.gridLayout_17.addWidget(self.lineEdit_training_fold, 1, 1, 1, 1)
        spacerItem24 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_17.addItem(spacerItem24, 1, 3, 1, 1)
        self.gridLayout_18 = QtWidgets.QGridLayout()
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.label_38 = QtWidgets.QLabel(self.groupBox_6)
        self.label_38.setObjectName("label_38")
        self.gridLayout_18.addWidget(self.label_38, 1, 1, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.groupBox_6)
        self.label_37.setObjectName("label_37")
        self.gridLayout_18.addWidget(self.label_37, 1, 0, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.groupBox_6)
        self.label_32.setObjectName("label_32")
        self.gridLayout_18.addWidget(self.label_32, 0, 0, 1, 1)
        spacerItem25 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem25, 0, 4, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.groupBox_6)
        self.label_33.setObjectName("label_33")
        self.gridLayout_18.addWidget(self.label_33, 0, 1, 1, 1)
        self.label_training_accuration = QtWidgets.QLabel(self.groupBox_6)
        self.label_training_accuration.setObjectName("label_training_accuration")
        self.gridLayout_18.addWidget(self.label_training_accuration, 0, 2, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.groupBox_6)
        self.label_36.setObjectName("label_36")
        self.gridLayout_18.addWidget(self.label_36, 0, 3, 1, 1)
        self.label_training_precision = QtWidgets.QLabel(self.groupBox_6)
        self.label_training_precision.setObjectName("label_training_precision")
        self.gridLayout_18.addWidget(self.label_training_precision, 1, 2, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.groupBox_6)
        self.label_40.setObjectName("label_40")
        self.gridLayout_18.addWidget(self.label_40, 2, 0, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.groupBox_6)
        self.label_41.setObjectName("label_41")
        self.gridLayout_18.addWidget(self.label_41, 2, 1, 1, 1)
        self.label_training_recall = QtWidgets.QLabel(self.groupBox_6)
        self.label_training_recall.setObjectName("label_training_recall")
        self.gridLayout_18.addWidget(self.label_training_recall, 2, 2, 1, 1)
        self.gridLayout_17.addLayout(self.gridLayout_18, 3, 0, 1, 4)
        spacerItem26 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_17.addItem(spacerItem26, 5, 0, 1, 4)
        self.groupBox_5.raise_()
        self.groupBox_5.raise_()
        self.groupBox_5.raise_()
        self.label_22.raise_()
        self.label_31.raise_()
        self.lineEdit_training_fold.raise_()
        self.gridLayout_15.addWidget(self.groupBox_6, 1, 0, 1, 3)
        self.tabWidget.addTab(self.tab_training, "")
        self.tab_analysis = QtWidgets.QWidget()
        self.tab_analysis.setObjectName("tab_analysis")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_analysis)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget.addTab(self.tab_analysis, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.setupDefault()
        self.connectAction()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Sentiment Analysis"))
        self.pushButton_setup_next.setText(_translate("MainWindow", "Next"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Additional"))
        self.lineEdit_setup_except_col.setToolTip(_translate("MainWindow", "<html><head/><body><p>Column that not neccesary for process, eg : id column.</p></body></html>"))
        self.lineEdit_setup_except_col.setPlaceholderText(_translate("MainWindow", "seperate with comma (,)"))
        self.lineEdit_setup_clas_col.setToolTip(_translate("MainWindow", "<html><head/><body><p>Class column of data. Default : clas</p></body></html>"))
        self.lineEdit_setup_clas_col.setPlaceholderText(_translate("MainWindow", "default : clas"))
        self.label_7.setText(_translate("MainWindow", "Class column"))
        self.label_6.setText(_translate("MainWindow", "Text column"))
        self.label_8.setText(_translate("MainWindow", "Exceptional Column"))
        self.lineEdit_setup_text_col.setToolTip(_translate("MainWindow", "<html><head/><body><p>Column for text to be process. Default : text</p></body></html>"))
        self.lineEdit_setup_text_col.setPlaceholderText(_translate("MainWindow", "default : text"))
        self.groupBox.setTitle(_translate("MainWindow", "Data source"))
        self.label.setText(_translate("MainWindow", "Host"))
        self.label_2.setText(_translate("MainWindow", "User"))
        self.label_4.setText(_translate("MainWindow", "Database"))
        self.lineEdit_setup_db.setToolTip(_translate("MainWindow", "<html><head/><body><p>Database to use</p></body></html>"))
        self.lineEdit_setup_db.setPlaceholderText(_translate("MainWindow", "eg : training_db"))
        self.label_5.setText(_translate("MainWindow", "Table for training data"))
        self.label_3.setText(_translate("MainWindow", "Password"))
        self.lineEdit_setup_password.setToolTip(_translate("MainWindow", "<html><head/><body><p>User password</p></body></html>"))
        self.lineEdit_setup_password.setPlaceholderText(_translate("MainWindow", "password"))
        self.lineEdit_setup_host.setToolTip(_translate("MainWindow", "<html><head/><body><p>Host of database server</p></body></html>"))
        self.lineEdit_setup_host.setPlaceholderText(_translate("MainWindow", "eg : localhost"))
        self.lineEdit_setup_user.setToolTip(_translate("MainWindow", "<html><head/><body><p>Database user</p></body></html>"))
        self.lineEdit_setup_user.setPlaceholderText(_translate("MainWindow", "eg : root"))
        self.pushButton_setup_connect.setText(_translate("MainWindow", "Connect"))
        self.label_12.setText(_translate("MainWindow", "Status :"))
        self.label_setup_status_connect.setText(_translate("MainWindow", "Disconnected"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_setup), _translate("MainWindow", "Setup"))
        self.pushButton_fixword_next.setText(_translate("MainWindow", "Next"))
        self.pushButton_fixword_save.setText(_translate("MainWindow", "Save"))
        self.label_9.setText(_translate("MainWindow", "Word"))
        self.label_10.setText(_translate("MainWindow", "Word fix"))
        self.lineEdit_fixword_wordfix.setToolTip(_translate("MainWindow", "Word fix"))
        self.pushButton_fixword_back.setText(_translate("MainWindow", "Back"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_fix_word), _translate("MainWindow", "Fix word"))
        self.pushButton_preprocessing_next.setText(_translate("MainWindow", "Next"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Result"))
        self.label_preprocessing_feature_after.setText(_translate("MainWindow", "0"))
        self.label_21.setText(_translate("MainWindow", "%"))
        self.label_preprocessing_feature_before.setText(_translate("MainWindow", "0"))
        self.label_17.setText(_translate("MainWindow", ":"))
        self.label_19.setText(_translate("MainWindow", "Reduction : "))
        self.label_18.setText(_translate("MainWindow", ":"))
        self.label_preprocessing_feature_reduction.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "Total feature before"))
        self.label_15.setText(_translate("MainWindow", "Total feature"))
        self.label_16.setText(_translate("MainWindow", "Vector Space Model"))
        self.label_20.setText(_translate("MainWindow", "Feature"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Preprocessing setting"))
        self.label_11.setText(_translate("MainWindow", "Number of feature"))
        self.label_13.setText(_translate("MainWindow", "Threshold"))
        self.lineEdit_preprocessing_threshold.setToolTip(_translate("MainWindow", "<html><head/><body><p>Threshold feature to be taken, if below threshold then feature will be removed.</p></body></html>"))
        self.lineEdit_preprocessing_threshold.setPlaceholderText(_translate("MainWindow", "default : 0. no threshold - all feature selected"))
        self.lineEdit_preprocessing_numberoffeature.setToolTip(_translate("MainWindow", "<html><head/><body><p>Number of feature taken.</p></body></html>"))
        self.lineEdit_preprocessing_numberoffeature.setPlaceholderText(_translate("MainWindow", "default : 0, all feature selected"))
        self.pushButton_preprocessing_process.setText(_translate("MainWindow", "Process"))
        self.radioButton_preprocessing_nofeatureselection.setToolTip(_translate("MainWindow", "<html><head/><body><p>No feature selection process.</p></body></html>"))
        self.radioButton_preprocessing_nofeatureselection.setText(_translate("MainWindow", "No feature selection"))
        self.radioButton_preprocessing_featureselection_ig.setToolTip(_translate("MainWindow", "<html><head/><body><p>Perform a feature selection.</p></body></html>"))
        self.radioButton_preprocessing_featureselection_ig.setText(_translate("MainWindow", "Feature selection (Information Gain)"))
        self.radioButton_preprocessing_nopreprocessing.setToolTip(_translate("MainWindow", "<html><head/><body><p>Will only remove symbol and lower case the text.</p></body></html>"))
        self.radioButton_preprocessing_nopreprocessing.setText(_translate("MainWindow", "No preprocessing"))
        self.radioButton_preprocessing_withpreprocessing.setToolTip(_translate("MainWindow", "<html><head/><body><p>Remove symbols, case folding, remove stopword, and do stemming.</p><p><br/></p></body></html>"))
        self.radioButton_preprocessing_withpreprocessing.setText(_translate("MainWindow", "Preprocessing"))
        self.pushButton_preprocessing_back.setText(_translate("MainWindow", "Back"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_preprocessing), _translate("MainWindow", "Preprocessing"))
        self.pushButton_training_back.setText(_translate("MainWindow", "Back"))
        self.pushButton_training_analys.setText(_translate("MainWindow", "Analysis Data"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Training"))
        self.label_34.setText(_translate("MainWindow", "second"))
        self.label_25.setText(_translate("MainWindow", "Total class"))
        self.label_23.setText(_translate("MainWindow", "Total training data"))
        self.label_24.setText(_translate("MainWindow", "Total feature processed"))
        self.label_training_total_data.setText(_translate("MainWindow", "0"))
        self.label_training_total_feature.setText(_translate("MainWindow", "0"))
        self.label_training_time.setText(_translate("MainWindow", "0"))
        self.label_training_total_class.setText(_translate("MainWindow", "0"))
        self.pushButton.setText(_translate("MainWindow", "Train data"))
        self.label_26.setText(_translate("MainWindow", "Time elapsed"))
        self.label_27.setText(_translate("MainWindow", ":"))
        self.label_28.setText(_translate("MainWindow", ":"))
        self.label_30.setText(_translate("MainWindow", ":"))
        self.label_29.setText(_translate("MainWindow", ":"))
        self.groupBox_6.setTitle(_translate("MainWindow", "Evaluation"))
        self.label_22.setText(_translate("MainWindow", "K-Fold Cross Validation"))
        self.pushButton_training_test.setText(_translate("MainWindow", "Test"))
        self.label_31.setText(_translate("MainWindow", "Folds"))
        self.label_38.setText(_translate("MainWindow", ":"))
        self.label_37.setText(_translate("MainWindow", "Precision"))
        self.label_32.setText(_translate("MainWindow", "Accuration"))
        self.label_33.setText(_translate("MainWindow", ":"))
        self.label_training_accuration.setText(_translate("MainWindow", "0"))
        self.label_36.setText(_translate("MainWindow", "%"))
        self.label_training_precision.setText(_translate("MainWindow", "0"))
        self.label_40.setText(_translate("MainWindow", "Recall"))
        self.label_41.setText(_translate("MainWindow", ":"))
        self.label_training_recall.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_training), _translate("MainWindow", "Training"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_analysis), _translate("MainWindow", "Analysis"))

    def setupDefault(self):
        self.tabWidget.setTabEnabled(1,False)
        self.tabWidget.setTabEnabled(2,False)
        self.tabWidget.setTabEnabled(3,False)
        self.tabWidget.setTabEnabled(4,False)
        # tab setup
        self.pushButton_setup_next.setEnabled(False)
        self.pushButton_preprocessing_next.setEnabled(False)
        # tab preprocessing
        self.radioButton_preprocessing_withpreprocessing.setChecked(True)
        self.radioButton_preprocessing_nofeatureselection.setChecked(True)
        self.lineEdit_preprocessing_numberoffeature.setEnabled(False)
        self.lineEdit_preprocessing_threshold.setEnabled(False)

    def connectAction(self):
        # tab setup
        regexLetterUnderscoreOnly = QtCore.QRegExp(r'^[a-zA-Z0-9\_]*$')
        validator1 = QtGui.QRegExpValidator(regexLetterUnderscoreOnly,self.lineEdit_setup_host)
        self.lineEdit_setup_host.setValidator(validator1)
        validator2 = QtGui.QRegExpValidator(regexLetterUnderscoreOnly,self.lineEdit_setup_user)
        self.lineEdit_setup_user.setValidator(validator2)
        validator3 = QtGui.QRegExpValidator(regexLetterUnderscoreOnly,self.lineEdit_setup_db)
        self.lineEdit_setup_db.setValidator(validator3)
        self.pushButton_setup_connect.clicked.connect(lambda: self.buttonConnectDb())
        self.comboBox_setup_table.currentTextChanged.connect(self.tableTrainingCheck)
        self.lineEdit_setup_text_col.keyReleaseEvent = self.setupHandleKeyReleaseText
        self.lineEdit_setup_clas_col.keyReleaseEvent = self.setupHandleKeyReleaseClass
        self.lineEdit_setup_except_col.keyReleaseEvent = self.setupHandleKeyReleaseEx
        self.pushButton_setup_next.clicked.connect(lambda: self.setupNext())

        # tab fixword
        regexLetter = QtCore.QRegExp(r'^[a-z]*$')
        validatorWordFix = QtGui.QRegExpValidator(regexLetter,self.lineEdit_fixword_wordfix)
        self.lineEdit_fixword_wordfix.setValidator(validatorWordFix)
        self.pushButton_fixword_save.clicked.connect(self.fixWordSave)
        self.pushButton_fixword_back.clicked.connect(self.fixWordBack)
        self.pushButton_fixword_next.clicked.connect(self.fixWordNext)

        # tab preprocessing
        self.radioButton_preprocessing_nopreprocessing.toggled.connect(self.preprocessCheckNoPre)
        self.radioButton_preprocessing_withpreprocessing.toggled.connect(self.preprocessCheckPre)
        self.radioButton_preprocessing_nofeatureselection.toggled.connect(self.preprocessCheckNoFS)
        self.radioButton_preprocessing_featureselection_ig.toggled.connect(self.preprocessCheckFS)
        regexNumber = QtCore.QRegExp(r'^[0-9]*$')
        validatorPreNum = QtGui.QRegExpValidator(regexNumber,self.lineEdit_preprocessing_numberoffeature)
        self.lineEdit_preprocessing_numberoffeature.setValidator(validatorPreNum)
        regexDecimal = QtCore.QRegExp(r'^[0-9]\d*(\.\d+)?$')
        validatorPreDec = QtGui.QRegExpValidator(regexDecimal,self.lineEdit_preprocessing_threshold)
        self.lineEdit_preprocessing_threshold.setValidator(validatorPreDec)
        self.pushButton_preprocessing_process.clicked.connect(self.preprocessProcess)
        self.pushButton_preprocessing_back.clicked.connect(self.preprocessBack)
        self.pushButton_preprocessing_next.clicked.connect(self.preprocessNext)

        # tab training

# tab setup
    def buttonConnectDb(self):
        host = self.lineEdit_setup_host.text()
        user = self.lineEdit_setup_user.text()
        password = self.lineEdit_setup_password.text()
        db = self.lineEdit_setup_db.text()
        connectStatus = self.app.connectDb(host,user,password,db)
        if connectStatus['success'] == True:
            if connectStatus['tables'] != None:
                for tb in connectStatus['tables']:
                    self.comboBox_setup_table.addItem(tb[0])
        else:
            pass
        self.label_setup_status_connect.setText(connectStatus['msg'])

    def setupCheckTrainingTable(self):
        if self.app.con != None:
            if self.setupTextCol == self.setupClassCol:
                print("Text column cannot be same as class column!")
                self.pushButton_setup_next.setEnabled(False)
                self.tabWidget.setTabEnabled(1,False)
                self.tabWidget.setTabEnabled(2,False)
            else:
                query = "SELECT {0},{1} from {2}"
                testCol = self.app.con.queryWithError(query.format(self.setupTextCol,self.setupClassCol,self.selectedTable))
                if testCol['errorcode'] == 1054:
                    print("Table does not contain \'{0}\' or \'{1}\' column, please select another table!".format(self.setupTextCol,self.setupClassCol))
                    print("Or change column below.")
                    self.pushButton_setup_next.setEnabled(False)
                    self.tabWidget.setTabEnabled(1,False)
                    self.tabWidget.setTabEnabled(2,False)
                elif testCol['errorcode'] == 0:
                    self.pushButton_setup_next.setEnabled(True)
                    print("Column text and class confirmed.")
                else:
                    print("Error in check column for table. No {0}".format(testCol['errorcode']))
                    self.pushButton_setup_next.setEnabled(False)
                    self.tabWidget.setTabEnabled(1,False)
                    self.tabWidget.setTabEnabled(2,False)

    def tableTrainingCheck(self,value):
        self.selectedTable = value
        self.setupCheckTrainingTable()

    def setupHandleKeyReleaseText(self,event):
        self.setupTextCol = self.lineEdit_setup_text_col.text()
        QtWidgets.QLineEdit.keyReleaseEvent(self.lineEdit_setup_text_col,event)
        self.setupCheckTrainingTable()

    def setupHandleKeyReleaseClass(self,event):
        self.setupClassCol = self.lineEdit_setup_clas_col.text()
        QtWidgets.QLineEdit.keyReleaseEvent(self.lineEdit_setup_clas_col,event)
        self.setupCheckTrainingTable()

    def setupHandleKeyReleaseEx(self,event):
        exceptCol = self.lineEdit_setup_except_col.text()
        if re.search(r'^[a-zA-Z0-9,\_]*$',exceptCol):
            # print("check to db, before it parse to list with comma separator")
            exceptCol = exceptCol.split(",")
            self.exceptCol = list(filter(None,exceptCol)) #remove empty string in list
            if self.setupTextCol in self.exceptCol or self.setupClassCol in self.exceptCol:
                self.pushButton_setup_next.setEnabled(False)
                self.tabWidget.setTabEnabled(1,False)
                self.tabWidget.setTabEnabled(2,False)
            else:
                self.pushButton_setup_next.setEnabled(True)
            # print(self.exceptCol)
            # print(len(self.exceptCol))
        else:
            print("Character contain illegal symbol, only comma (,) allowed!")
        QtWidgets.QLineEdit.keyReleaseEvent(self.lineEdit_setup_except_col,event)

    def setupNext(self):
        self.selectedTable = str(self.comboBox_setup_table.currentText())
        if self.selectedTable:
            self.app.setTrainingTable(self.selectedTable)
            self.app.setTextCol(self.setupTextCol)
            self.app.setClassCol(self.setupClassCol)
            self.app.setExceptionalFeature(self.exceptCol)
            self.fixWordTableRefresh(self.gridLayout_10)
            self.tabWidget.setTabEnabled(1,True)
            self.tabWidget.setTabEnabled(2,True)
            self.tabWidget.setCurrentIndex(1)

# tab fixword
    def fixWordTableRefresh(self,lay):
        import sip

        lay.removeWidget(self.tableWidget_fixword)
        sip.delete(self.tableWidget_fixword)
        self.tableWidget_fixword= None

        colCount = 0
        rowCount = 0
        data = None

        if self.app.con != None:
            colCount = len(self.fixWord_cols)
            cols_q = ""
            j = 0
            for i in self.fixWord_cols:
                cols_q+=i
                j+=1
                if j < colCount:
                    cols_q+=","

            query = "SELECT {0} FROM {1}"
            data = self.app.con.query(query.format(cols_q,self.fixWord_fixtable))
            rowCount = len(data)

        self.tableWidget_fixword = QtWidgets.QTableWidget(rowCount, colCount)
        self.tableWidget_fixword.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget_fixword.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        vheader = QtWidgets.QHeaderView(QtCore.Qt.Vertical)
        vheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_fixword.setVerticalHeader(vheader)
        hheader = QtWidgets.QHeaderView(QtCore.Qt.Horizontal)
        hheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_fixword.setHorizontalHeader(hheader)
        self.tableWidget_fixword.setHorizontalHeaderLabels(self.fixWord_cols)

        if data != None:
            for i in range(rowCount):
                for j in range(colCount):
                    if data[i][j]:
                        itemText = str(data[i][j])
                    else:
                        itemText = ""
                    item = QtWidgets.QTableWidgetItem(itemText)
                    self.tableWidget_fixword.setItem(i, j, item)
        self.tableWidget_fixword.doubleClicked.connect(lambda: self.fixWordgetItem(cols_q))
        lay.addWidget(self.tableWidget_fixword)

    def fixWordgetItem(self,cols_q):

        self.fixWord_indexData = self.tableWidget_fixword.currentRow()
        self.fixWord_idData = self.tableWidget_fixword.item(self.fixWord_indexData,0).text()

        query = "SELECT {0} FROM {1} where id={2}"
        data = self.app.con.query(query.format(cols_q,self.fixWord_fixtable,self.fixWord_idData))

        self.lineEdit_fixword_word.setText(data[0][1])
        self.lineEdit_fixword_wordfix.setText(data[0][2])

    def fixWordSave(self):
        if self.fixWord_idData == 0:
            print("No data selected!")
        elif not self.lineEdit_fixword_wordfix.text():
            print("Empty!")
        else:
            print("Saving ...")
            newText = self.lineEdit_fixword_wordfix.text()
            query = "UPDATE {0} set word_fix='{1}' where id={2}"
            self.app.con.queryInsert(query.format(self.fixWord_fixtable,newText,self.fixWord_idData))
            self.tableWidget_fixword.item(self.fixWord_indexData,2).setText(newText)
            self.lineEdit_fixword_word.setText("")
            self.lineEdit_fixword_wordfix.setText("")
            self.fixWord_idData = 0
            print("Saved")

    def fixWordBack(self):
        self.tabWidget.setCurrentIndex(0)

    def fixWordNext(self):
        self.tabWidget.setCurrentIndex(2)

# tab preprocessing
    def preprocessCheckNoPre(self,enabled):
        if enabled:
            self.preprocessing_doPreprocessing = False

    def preprocessCheckPre(self,enabled):
        if enabled:
            self.preprocessing_doPreprocessing = True

    def preprocessCheckNoFS(self,enabled):
        if enabled:
            self.preprocessing_doFeatureSelection = False
            self.lineEdit_preprocessing_numberoffeature.setEnabled(False)
            self.lineEdit_preprocessing_threshold.setEnabled(False)

    def preprocessCheckFS(self,enabled):
        if enabled:
            self.preprocessing_doFeatureSelection = True
            self.lineEdit_preprocessing_numberoffeature.setEnabled(True)
            self.lineEdit_preprocessing_threshold.setEnabled(True)

    def preprocessProcess(self):
        if self.preprocessing_doFeatureSelection:
            if self.lineEdit_preprocessing_numberoffeature.text():
                self.preprocessing_numFeatureToRetain = self.lineEdit_preprocessing_numberoffeature.text()
            else:
                self.preprocessing_numFeatureToRetain = 0

            if self.lineEdit_preprocessing_threshold.text():
                self.preprocessing_thresholdFeatureIgnore = self.lineEdit_preprocessing_threshold.text()
            else:
                self.preprocessing_thresholdFeatureIgnore = 0

        features = self.app.preprocessing(self.preprocessing_doPreprocessing,self.preprocessing_doFeatureSelection,self.preprocessing_numFeatureToRetain,self.preprocessing_thresholdFeatureIgnore)
        self.preprocessTable(self.gridLayout_13,features)

    def preprocessTable(self,lay,features):
        import sip

        lay.removeWidget(self.tableWidget_preprocessing_features_list_after)
        sip.delete(self.tableWidget_preprocessing_features_list_after)
        self.tableWidget_preprocessing_features_list_after= None

        lay.removeWidget(self.tableWidget_preprocessing_vsm)
        sip.delete(self.tableWidget_preprocessing_vsm)
        self.tableWidget_preprocessing_vsm= None

        colCount = 0
        rowCount = 0
        data = None
        cols = []

        vms_colCount = 0
        vms_rowCount = 0
        vms_data = None
        vms_cols = []

        if features != None:
            feature = features['vsm']['feature']
            vsm = features['vsm']['vsm']

            cols = [i for i in feature]
            colCount = len(cols)
            rowCount = len(feature.index)

            vms_cols = [i for i in vsm]
            vms_colCount = len(vms_cols)
            vms_rowCount = len(vsm.index)

            # CHECK POINT > kurang hitung dibawah ini dan tampilkan di window
            totalFeatureBefore = features['featurebefore']
            totalFeatureAfter = rowCount
            reduction = 0
            if totalFeatureBefore != 0:
                reduction = totalFeatureAfter/totalFeatureBefore*100 # rumus masih salah
                reduction = round(100-reduction) # rumus masih salah
            self.label_preprocessing_feature_before.setText(str(totalFeatureBefore))
            self.label_preprocessing_feature_after.setText(str(totalFeatureAfter))
            self.label_preprocessing_feature_reduction.setText(str(reduction))

        #for table feature
        self.tableWidget_preprocessing_features_list_after = QtWidgets.QTableWidget(rowCount, colCount)
        self.tableWidget_preprocessing_features_list_after.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget_preprocessing_features_list_after.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        vheader = QtWidgets.QHeaderView(QtCore.Qt.Vertical)
        vheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_preprocessing_features_list_after.setVerticalHeader(vheader)
        hheader = QtWidgets.QHeaderView(QtCore.Qt.Horizontal)
        hheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_preprocessing_features_list_after.setHorizontalHeader(hheader)

        self.tableWidget_preprocessing_features_list_after.setHorizontalHeaderLabels(cols) # !

        #for table VSM
        self.tableWidget_preprocessing_vsm = QtWidgets.QTableWidget(vms_rowCount, vms_colCount)
        self.tableWidget_preprocessing_vsm.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget_preprocessing_vsm.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        vms_vheader = QtWidgets.QHeaderView(QtCore.Qt.Vertical)
        vheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_preprocessing_vsm.setVerticalHeader(vms_vheader)
        vms_hheader = QtWidgets.QHeaderView(QtCore.Qt.Horizontal)
        hheader.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget_preprocessing_vsm.setHorizontalHeader(vms_hheader)

        self.tableWidget_preprocessing_vsm.setHorizontalHeaderLabels(vms_cols) # !


        if features != None:
            feature = features['vsm']['feature']
            vsm = features['vsm']['vsm']

            for index,row in feature.iterrows():
                jj = 0
                for j in cols:
                    feat = row[j]
                    if not feat:
                        feat = ""
                    item = QtWidgets.QTableWidgetItem(str(feat))
                    self.tableWidget_preprocessing_features_list_after.setItem(index, jj, item)
                    jj+=1

            for index,row in vsm.iterrows():
                jj = 0
                for j in vms_cols:
                    feat = row[j]
                    item = QtWidgets.QTableWidgetItem(str(feat))
                    self.tableWidget_preprocessing_vsm.setItem(index, jj, item)
                    jj+=1

            # for button next, to process in training data
            if len(vsm.index) > 0:
                self.vsmFeature = features
                self.tabWidget.setTabEnabled(3,True)
                self.pushButton_preprocessing_next.setEnabled(True)
            else:
                print("VSM has no data!")
                self.pushButton_preprocessing_next.setEnabled(False)
                self.tabWidget.setTabEnabled(3,False)
        else:
            self.pushButton_preprocessing_next.setEnabled(False)
            self.tabWidget.setTabEnabled(3,False)

        lay.addWidget(self.tableWidget_preprocessing_features_list_after, 3, 0, 1, 2)
        lay.addWidget(self.tableWidget_preprocessing_vsm, 3, 2, 1, 6)

    def preprocessBack(self):
        self.tabWidget.setCurrentIndex(1)

    def preprocessNext(self):
        self.tabWidget.setCurrentIndex(3)

#tab training
    def trainData(self):
        return True

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
