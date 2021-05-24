# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'client.ui'
##
## Created by: Qt User Interface Compiler version 6.1.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import covid19.ui._resource_rc

class Ui_Client(object):
    def setupUi(self, Client):
        if not Client.objectName():
            Client.setObjectName(u"Client")
        Client.resize(1450, 1022)
        icon = QIcon()
        icon.addFile(u":/images/logo.jpeg", QSize(), QIcon.Normal, QIcon.Off)
        Client.setWindowIcon(icon)
        Client.setStyleSheet(u"* {\n"
"	font-size: 18px;\n"
"}\n"
"\n"
"QPushButton#predict {\n"
"	background-color: rgb(75, 116, 22);\n"
"	border-radius: 10px;\n"
"	color: white;\n"
"	padding: 5px;\n"
"}\n"
"\n"
"QPushButton#predict:hover {\n"
"	background-color: rgb(22, 57, 9);\n"
"}\n"
"\n"
"QPushButton#predict:pressed {\n"
"	border-radius: 12px;\n"
"}\n"
"\n"
"QPushButton#predict:disabled {\n"
"	background-color: rgba(75, 116, 22, 50%)\n"
"}\n"
"\n"
"QPushButton#select_image {\n"
"	padding: 5px;\n"
"}\n"
"\n"
"QPushButton#select_image:hover {\n"
"	background-color: rgb(136, 138, 133);\n"
"}\n"
"\n"
"QPushButton#select_image:pressed {\n"
"	border-radius: 5px;\n"
"}\n"
"\n"
"QGroupBox * {\n"
"	margin: 10px;\n"
"	padding: 5px;\n"
"}")
        self.centralwidget = QWidget(Client)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.output_panel = QVBoxLayout()
        self.output_panel.setObjectName(u"output_panel")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.explanation_label = QLabel(self.centralwidget)
        self.explanation_label.setObjectName(u"explanation_label")
        self.explanation_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.explanation_label)

        self.explanation = QLabel(self.centralwidget)
        self.explanation.setObjectName(u"explanation")
        self.explanation.setMinimumSize(QSize(640, 640))
        self.explanation.setMaximumSize(QSize(640, 640))
        self.explanation.setPixmap(QPixmap(u":/images/default.png"))

        self.verticalLayout_2.addWidget(self.explanation)


        self.output_panel.addLayout(self.verticalLayout_2)

        self.results = QGroupBox(self.centralwidget)
        self.results.setObjectName(u"results")
        self.formLayout_2 = QFormLayout(self.results)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.prediction_label = QLabel(self.results)
        self.prediction_label.setObjectName(u"prediction_label")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.prediction_label)

        self.prediction = QLabel(self.results)
        self.prediction.setObjectName(u"prediction")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.prediction)

        self.confidence_label = QLabel(self.results)
        self.confidence_label.setObjectName(u"confidence_label")

        self.formLayout_2.setWidget(3, QFormLayout.LabelRole, self.confidence_label)

        self.confidence = QLabel(self.results)
        self.confidence.setObjectName(u"confidence")

        self.formLayout_2.setWidget(3, QFormLayout.FieldRole, self.confidence)


        self.output_panel.addWidget(self.results)


        self.gridLayout.addLayout(self.output_panel, 0, 3, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_3, 0, 4, 1, 1)

        self.input_panel = QVBoxLayout()
        self.input_panel.setObjectName(u"input_panel")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.input_label = QLabel(self.centralwidget)
        self.input_label.setObjectName(u"input_label")
        self.input_label.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.input_label)

        self.input = QLabel(self.centralwidget)
        self.input.setObjectName(u"input")
        self.input.setMinimumSize(QSize(640, 640))
        self.input.setMaximumSize(QSize(640, 640))
        self.input.setPixmap(QPixmap(u":/images/default.png"))

        self.verticalLayout.addWidget(self.input)


        self.input_panel.addLayout(self.verticalLayout)

        self.settings = QGroupBox(self.centralwidget)
        self.settings.setObjectName(u"settings")
        self.formLayout = QFormLayout(self.settings)
        self.formLayout.setObjectName(u"formLayout")
        self.image_label = QLabel(self.settings)
        self.image_label.setObjectName(u"image_label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.image_label)

        self.select_image = QPushButton(self.settings)
        self.select_image.setObjectName(u"select_image")
        self.select_image.setMaximumSize(QSize(200, 50))

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.select_image)

        self.architecture_label = QLabel(self.settings)
        self.architecture_label.setObjectName(u"architecture_label")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.architecture_label)

        self.architecture = QComboBox(self.settings)
        self.architecture.addItem("")
        self.architecture.addItem("")
        self.architecture.setObjectName(u"architecture")
        self.architecture.setMaximumSize(QSize(200, 50))

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.architecture)

        self.explainer_label = QLabel(self.settings)
        self.explainer_label.setObjectName(u"explainer_label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.explainer_label)

        self.explainer = QComboBox(self.settings)
        self.explainer.addItem("")
        self.explainer.addItem("")
        self.explainer.setObjectName(u"explainer")
        self.explainer.setMaximumSize(QSize(200, 50))

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.explainer)

        self.predict = QPushButton(self.settings)
        self.predict.setObjectName(u"predict")
        self.predict.setEnabled(False)
        self.predict.setMinimumSize(QSize(0, 0))
        self.predict.setMaximumSize(QSize(200, 50))

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.predict)


        self.input_panel.addWidget(self.settings)


        self.gridLayout.addLayout(self.input_panel, 0, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 0, 1, 1)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer_2, 0, 2, 1, 1)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer, 1, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 1, 3, 1, 1)

        Client.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(Client)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1450, 26))
        Client.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(Client)
        self.statusbar.setObjectName(u"statusbar")
        Client.setStatusBar(self.statusbar)

        self.retranslateUi(Client)

        QMetaObject.connectSlotsByName(Client)
    # setupUi

    def retranslateUi(self, Client):
        Client.setWindowTitle(QCoreApplication.translate("Client", u"MainWindow", None))
        self.explanation_label.setText(QCoreApplication.translate("Client", u"Explanation", None))
        self.explanation.setText("")
        self.results.setTitle(QCoreApplication.translate("Client", u"Results", None))
        self.prediction_label.setText(QCoreApplication.translate("Client", u"Prediction:", None))
        self.prediction.setText("")
        self.confidence_label.setText(QCoreApplication.translate("Client", u"Confidence:", None))
        self.confidence.setText("")
        self.input_label.setText(QCoreApplication.translate("Client", u"Chest X-Ray Image", None))
        self.input.setText("")
        self.settings.setTitle(QCoreApplication.translate("Client", u"Settings", None))
        self.image_label.setText(QCoreApplication.translate("Client", u"Input image:", None))
        self.select_image.setText(QCoreApplication.translate("Client", u"Select image", None))
        self.architecture_label.setText(QCoreApplication.translate("Client", u"Architecture:", None))
        self.architecture.setItemText(0, QCoreApplication.translate("Client", u"ResNet50", None))
        self.architecture.setItemText(1, QCoreApplication.translate("Client", u"COVID-Net", None))

        self.explainer_label.setText(QCoreApplication.translate("Client", u"Explainer:", None))
        self.explainer.setItemText(0, QCoreApplication.translate("Client", u"Grad-CAM", None))
        self.explainer.setItemText(1, QCoreApplication.translate("Client", u"Integrated Gradients", None))

        self.predict.setText(QCoreApplication.translate("Client", u"Predict", None))
    # retranslateUi

