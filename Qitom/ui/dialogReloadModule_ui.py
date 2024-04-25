# Form implementation generated from reading ui file 'c:\itom\sources\itom\Qitom\ui\dialogReloadModule.ui'
#
# Created by: PyQt6 UI code generator 6.6.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_DialogReloadModule:
    def setupUi(self, DialogReloadModule):
        DialogReloadModule.setObjectName("DialogReloadModule")
        DialogReloadModule.resize(551, 481)
        self.horizontalLayout = QtWidgets.QHBoxLayout(DialogReloadModule)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.treeWidget = QtWidgets.QTreeWidget(parent=DialogReloadModule)
        self.treeWidget.setTextElideMode(QtCore.Qt.TextElideMode.ElideLeft)
        self.treeWidget.setAutoExpandDelay(1)
        self.treeWidget.setColumnCount(1)
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "1")
        self.treeWidget.header().setDefaultSectionSize(200)
        self.treeWidget.header().setHighlightSections(False)
        self.treeWidget.header().setMinimumSectionSize(50)
        self.treeWidget.header().setSortIndicatorShown(True)
        self.verticalLayout.addWidget(self.treeWidget)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=DialogReloadModule)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.checkShowBuildin = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkShowBuildin.setObjectName("checkShowBuildin")
        self.horizontalLayout_2.addWidget(self.checkShowBuildin)
        self.checkShowFromPythonPath = QtWidgets.QCheckBox(parent=self.groupBox_2)
        self.checkShowFromPythonPath.setObjectName("checkShowFromPythonPath")
        self.horizontalLayout_2.addWidget(self.checkShowFromPythonPath)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(parent=DialogReloadModule)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.lblModuleName = QtWidgets.QLabel(parent=self.groupBox)
        self.lblModuleName.setObjectName("lblModuleName")
        self.gridLayout.addWidget(self.lblModuleName, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.lblPath = QtWidgets.QLabel(parent=self.groupBox)
        self.lblPath.setObjectName("lblPath")
        self.gridLayout.addWidget(self.lblPath, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btnReload = QtWidgets.QPushButton(parent=DialogReloadModule)
        self.btnReload.setDefault(True)
        self.btnReload.setObjectName("btnReload")
        self.verticalLayout_2.addWidget(self.btnReload)
        self.btnCancel = QtWidgets.QPushButton(parent=DialogReloadModule)
        self.btnCancel.setObjectName("btnCancel")
        self.verticalLayout_2.addWidget(self.btnCancel)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(DialogReloadModule)
        self.btnCancel.clicked.connect(DialogReloadModule.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(DialogReloadModule)

    def retranslateUi(self, DialogReloadModule):
        _translate = QtCore.QCoreApplication.translate
        DialogReloadModule.setWindowTitle(_translate("DialogReloadModule", "Python Modules"))
        self.treeWidget.setSortingEnabled(True)
        self.groupBox_2.setTitle(_translate("DialogReloadModule", "Filter"))
        self.checkShowBuildin.setText(_translate("DialogReloadModule", "show built-in modules"))
        self.checkShowFromPythonPath.setText(_translate("DialogReloadModule", "show modules lying in python-folder(s)"))
        self.groupBox.setTitle(_translate("DialogReloadModule", "Information"))
        self.label.setText(_translate("DialogReloadModule", "Module Name:"))
        self.lblModuleName.setText(_translate("DialogReloadModule", "<click on item to see information>"))
        self.label_2.setText(_translate("DialogReloadModule", "Path:"))
        self.lblPath.setText(_translate("DialogReloadModule", "-"))
        self.btnReload.setText(_translate("DialogReloadModule", "Reload Modules"))
        self.btnCancel.setText(_translate("DialogReloadModule", "Cancel"))
