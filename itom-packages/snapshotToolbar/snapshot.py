from itom import ui
from itomUi import ItomUi
import itom
import __main__
import inspect, os
import weakref
import glob


class Snapshot(ItomUi):
    def __init__(self, name):
        self.name = name
        self.dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        self.absPath = os.path.join(self.dir, "snapshot.ui")
        ItomUi.__init__(
            self, self.absPath, ui.TYPEWINDOW,
        )
        self.gui.setAttribute(55, True)

        self.direct = None
        self.dataTyp = {
            "BMP": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
            "IDC": ["single", "stack"],
            "JPG": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
            "PGM": ["gray"],
            "PNG": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
            "PPM": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
            "RAS": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
            "Tiff": [
                "gray",
                "rgb",
                "rgba",
                "grayMarked",
                "falseColor",
                "falseColorIR",
                "hotIron",
            ],
        }
        self.gui.spinMulti["enabled"] = 0
        self.gui.checkTimer["enabled"] = 0
        self.gui.spinInterval["enabled"] = 0
        self.gui.btnCancel["enabled"] = 0
        self.gui.checkMulti["checked"] = 0
        self.syncGUI()
        self.show()

        self.gui.connect("destroyed()", self.closeEvent)

    #########################################################
    def closeEvent(self):
        try:
            del __main__.__dict__["snapFactory"].__dict__[self.name]
        except KeyError:
            del __main__.__dict__[self.name]

    #########################################################
    def syncGUI(self):
        self.devices = self.searchDevices()
        self.gui.comboType.call("clear")
        self.gui.comboSource.call("clear")
        self.gui.comboSource.call("addItems", list(self.devices.keys()))
        self.gui.comboType.call("addItems", list(self.dataTyp.keys()))
        self.gui.comboType["currentIndex"] = list(self.dataTyp.keys()).index("IDC")
        self.setColorCombo()

    #########################################################
    def disableElements(self, val):
        val = bool(val)
        self.gui.groupBox["enabled"] = val
        self.gui.groupBox_5["enabled"] = val
        self.gui.btnSnap["enabled"] = val
        self.gui.btnCancel["enabled"] = not val
        self.gui.groupBox_3["enabled"] = val

    #########################################################
    def setColorCombo(self):
        idx = self.gui.comboType["currentIndex"]
        map = self.dataTyp[list(self.dataTyp.keys())[idx]]
        self.gui.comboColor.call("clear")
        self.gui.comboColor.call("addItems", map)

    #########################################################
    def showGui(self):
        self.show()

    #########################################################
    def searchDevices(self):
        if "devices" in self.__dict__:
            del self.devices
        devices = {}
        for x in __main__.__dict__:
            # gettype to check type of dataIO. maybe not implemented
            if type(__main__.__dict__[x]) == itom.dataIO:
                if __main__.__dict__[x].getType() == 129:
                    devices[x] = weakref.ref(__main__.__dict__[x])
        return devices

    #########################################################
    def callPathDialog(self):
        if self.direct is None:
            options = 1 | 2
            self.direct = itom.ui.getExistingDirectory("save images", self.dir, options)
        else:
            options = 1 | 2
            self.direct = itom.ui.getExistingDirectory(
                "save images", self.direct, options
            )
        if self.direct is not None:
            self.gui.call("statusBar").call("showMessage", str(self.direct))

    ##########################################################
    def takeSnapshot(self):
        if self.gui.checkMulti["checked"]:
            if not self.gui.checkTimer["checked"]:
                try:
                    self.selectedDevice.setAutoGrabbing(False)
                    self.selectedDevice.startDevice()
                    self.selectedDevice.acquire()
                    firstpic = itom.dataObject()
                    self.selectedDevice.copyVal(firstpic)
                    typ = firstpic.dtype
                    dat = dataObject(
                        [
                            self.gui.spinMulti["value"],
                            self.selectedDevice.getParam("sizey"),
                            self.selectedDevice.getParam("sizex"),
                        ],
                        dtype=typ,
                    )
                    dat[0, :, :] = firstpic
                    i = 1
                    while i < dat.shape[0]:
                        self.selectedDevice.acquire()
                        self.selectedDevice.copyVal(dat[i, :, :])
                        i = i + 1
                    self.selectedDevice.stopDevice()
                    return dat
                except Exception:
                    ui.msgCritical("Error", "Not able to acquire Snapshot")
                    raise RuntimeError("Error", "Not able to acquire Snapshot")
            else:
                self.cnt = 0
                step = int(self.gui.spinInterval["value"] * 1e3)
                self.tim = timer(step, self.timerPic)
                return
        elif not self.gui.checkMulti["checked"]:
            try:
                data = itom.dataObject()
                self.selectedDevice.setAutoGrabbing(False)
                self.selectedDevice.startDevice()
                self.selectedDevice.acquire()
                self.selectedDevice.copyVal(data)
                self.selectedDevice.stopDevice()
                return data
            except Exception:
                ui.msgCritical("Error", "Not able to acquire Snapshot")
                raise RuntimeError("Error", "Not able to acquire Snapshot")

    ##########################################################
    def timerPic(self):
        """ takes pictures and is called by a timer instance"""
        try:
            if self.cnt == 0:
                self.selectedDevice.setAutoGrabbing(False)
                self.selectedDevice.startDevice()
                self.selectedDevice.acquire()
                firstpic = itom.dataObject()
                self.selectedDevice.copyVal(firstpic)
                typ = firstpic.dtype
                self.dat = dataObject(
                    [
                        self.gui.spinMulti["value"],
                        self.selectedDevice.getParam("sizey"),
                        self.selectedDevice.getParam("sizex"),
                    ],
                    dtype=typ,
                )
                self.dat[0, :, :] = firstpic
                self.cnt = self.cnt + 1
            elif self.cnt < self.gui.spinMulti["value"]:
                self.selectedDevice.acquire()
                self.selectedDevice.copyVal(self.dat[self.cnt, :, :])

                self.cnt = self.cnt + 1
            else:
                self.selectedDevice.stopDevice()
                if self.gui.checkSaveAfterSnap["checked"]:
                    self.safeMultiFiles(self.dat)
                self.disableElements(1)
                del self.tim, self.dat, self.cnt
        except Exception:
            self.disableElements(1)
            del self.tim, self.dat, self.cnt

    ##########################################################
    def saveFiles(self, file):
        if self.direct is None:
            self.callPathDialog()
        os.chdir(self.direct)
        filestr = "save%s" % self.Typ
        if filestr == "saveIDC":
            idx = 1
            filename = "DataObject_%03i.idc" % idx
            while self.checkName(filename):
                idx = idx + 1
                filename = "DataObject_%03i.idc" % idx
            dict = {"data": file}
            try:
                saveIDC(self.direct + "\\" + filename, dict)
            except Exception:
                ui.msgCritical("Error", "Not able to save files")
        else:
            idx = 1
            filename = "pic_%03i.%s" % (idx, self.Typ)
            while self.checkName(filename):
                idx = idx + 1
                filename = "pic_%03i.%s" % (idx, self.Typ)
            try:
                filter(
                    filestr,
                    file,
                    filename,
                    self.dataTyp[self.Typ][self.gui.comboColor["currentIndex"]],
                )
            except Exception:
                ui.msgCritical(
                    "Error",
                    "Not able to save files. Maybe the selcted color palette does not fit to data type",
                )

    ##########################################################
    def checkName(self, name):
        files = glob.glob("./*." + self.Typ)
        files = sorted(files)
        return ".\\" + name in files

    ##########################################################
    def safeMultiFiles(self, files):
        if self.direct is None:
            self.callPathDialog()
        os.chdir(self.direct)
        if self.Typ == "IDC":
            mode = self.dataTyp["IDC"][self.gui.comboColor["currentIndex"]]
            if mode == "single":
                idx = 1
                folderName = "fold_%03i" % (idx)
                while os.path.exists(self.direct + "\\" + folderName):
                    idx = idx + 1
                    folderName = "fold_%03i" % (idx)
                os.makedirs(self.direct + "\\" + folderName)
                os.chdir(self.direct + "\\" + folderName)
                i = 1
                while i <= files.shape[0]:
                    filename = "DataObject_%03i.idc" % i
                    dat = files[i - 1, :, :].squeeze()
                    dict = {"data": dat}
                    saveIDC(self.direct + "\\" + folderName + "\\" + filename, dict)
                    i = i + 1
            elif mode == "stack":
                i = 1
                filename = "DataStack_%03i.idc" % i
                while self.checkName(filename):
                    i = i + 1
                    filename = "DataStack_%03i.idc" % i
                dict = {"data": files}
                saveIDC(self.direct + "\\" + filename, dict)
        else:
            idx = 1
            folderName = "fold_%03i" % (idx)
            while os.path.exists(self.direct + "\\" + folderName):
                idx = idx + 1
                folderName = "fold_%03i" % (idx)
            os.makedirs(self.direct + "\\" + folderName)
            olddir = self.direct
            self.direct = self.direct + "\\" + folderName
            i = 0
            while i < files.shape[0]:
                dat = files[i, :, :].squeeze()
                self.saveFiles(dat)
                i = i + 1
            self.direct = olddir
        os.chdir(self.direct)

    ##########################################################

    ##########################################################
    @ItomUi.autoslot("int")
    def on_comboSource_currentIndexChanged(self, var):
        try:
            self.selectedDevice = self.devices[list(self.devices.keys())[var]]()
            self.on_btnLive_clicked()
        except Exception:
            pass

    ##########################################################
    @ItomUi.autoslot("")
    def on_btnFolder_clicked(self):
        self.callPathDialog()

    ##########################################################
    @ItomUi.autoslot("")
    def on_btnSnap_clicked(self):
        self.disableElements(0)
        data = self.takeSnapshot()
        if self.gui.checkMulti["checked"] == 0:
            try:
                if self.gui.checkSaveAfterSnap["checked"]:
                    self.saveFiles(data)
                else:
                    globals()["snapshot"] = data
                self.disableElements(1)
            except Exception:
                self.disableElements(1)
        elif self.gui.checkMulti["checked"] == 1:
            if self.gui.checkTimer["checked"] == 0:
                try:
                    if self.gui.checkSaveAfterSnap["checked"]:
                        self.safeMultiFiles(data)
                    else:
                        globals()["snapshot"] = data
                    self.disableElements(1)
                except Exception:
                    self.disableElements(1)

    ##########################################################
    @ItomUi.autoslot("int")
    def on_comboType_currentIndexChanged(self, val):
        self.Typ = list(self.dataTyp.keys())[val]
        self.setColorCombo()

    ##########################################################
    @ItomUi.autoslot("bool")
    def on_checkMulti_toggled(self, var):
        if var:
            self.gui.spinMulti["enabled"] = 1
            self.gui.checkTimer["enabled"] = 1
            if self.gui.checkTimer["checked"]:
                self.gui.spinInterval["enabled"] = 1
        else:
            self.gui.spinMulti["enabled"] = 0
            self.gui.checkTimer["enabled"] = 0
            self.gui.spinInterval["enabled"] = 0

    ##########################################################
    @ItomUi.autoslot("")
    def on_pushRefresh_clicked(self):
        self.syncGUI()

    ##########################################################
    @ItomUi.autoslot("bool")
    def on_checkTimer_toggled(self, var):
        if var:
            self.gui.spinInterval["enabled"] = 1
        else:
            self.gui.spinInterval["enabled"] = 0

    ##########################################################
    @ItomUi.autoslot("")
    def on_btnCancel_clicked(self):
        del self.tim, self.cnt
        self.disableElements(1)

    ##########################################################
    @ItomUi.autoslot("")
    def on_btnLive_clicked(self):
        self.gui.liveImagePlot["camera"] = self.selectedDevice
        self.selectedDevice.setAutoGrabbing(True)


if __name__ == "__main__":
    clc()
    close("all")
    snap = Snapshot("snap")
