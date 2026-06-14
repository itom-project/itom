# coding=iso-8859-15
from itomUi import ItomUi
from itom import dataObject, ui, filter
import inspect
import os
import numpy as np
import __main__
import warnings


class ProfileRoughness(ItomUi):
    def __init__(self):
        dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        absPath = os.path.join(dir, "profile_roughness.ui")
        ItomUi.__init__(self, absPath, ui.TYPEWINDOW)
        self.gui.plotSource1d["visible"] = False
        self.gui.groupSelectDisplayedRow["visible"] = False
        self.gui.tabWidget["currentIndex"] = 0
        self.gui.lblRoughnessWarning["visible"] = False
        self.scanWorkspace()
        self.roughness = None
        self.waviness = None

    def show(self):
        self.gui.show()

    def scanWorkspace(self):
        workspace = __main__.__dict__
        self.gui.comboBox.call("clear")
        vals = []
        for (key, val) in workspace.items():
            if type(val) is dataObject:
                vals.append(key)
        self.gui.comboBox.call("addItems", vals)

    def loadItem(self, dataObj: dataObject):
        possibleTypes = ["int8", "int16", "int32", "float32", "float64"]
        possibleUnits = ["µm", "mm", "nm"]
        if not type(dataObj) is dataObject:
            ui.msgWarning(
                "wrong type",
                "given object must be of type dataObject",
                parent=self.gui,
            )
        elif not dataObj.dtype in possibleTypes:
            ui.msgWarning(
                "wrong type",
                "Only data types int8, int16, int32, float32 or float64 are allowed",
                parent=self.gui,
            )
        elif dataObj.dims != 2:
            ui.msgWarning(
                "wrong size", "Only 1D or 2D data objects are allowed", parent=self.gui,
            )
        elif dataObj.shape[1] <= 1:
            ui.msgWarning(
                "wrong size",
                "The data object must have more than one column.",
                parent=self.gui,
            )
        elif (not dataObj.axisUnits[1] in possibleUnits) or (
            not dataObj.valueUnit in possibleUnits
        ):
            ui.msgWarning(
                "wrong unit",
                "Value and horizontal axis unit must be nm, µm or mm",
                parent=self.gui,
            )
        else:
            self.sourceObj = dataObj.astype("float64").copy()
            self.sourceObjCropped = self.sourceObj.astype("float64")
            self.gui.plotSource2d["visible"] = self.sourceObj.shape[0] > 1
            self.gui.plotSource1d["visible"] = self.sourceObj.shape[0] <= 1
            if self.sourceObj.shape[0] > 1:
                self.gui.plotSource2d["source"] = self.sourceObj
            else:
                self.gui.plotSource1d["source"] = self.sourceObj
            self.gui.btnSourceCropReset["enabled"] = True
            self.gui.btnSourceCropCurrentView["enabled"] = True

    def filterProfile(self):
        self.gui.groupFiltering["enabled"] = False
        if self.sourceObjCropped is not None:

            if self.gui.comboFilterForm["currentText"] == "tilt correction":
                filter(
                    "subtract1DRegression",
                    self.sourceObjCropped,
                    self.sourceObjCropped,
                    1,
                )
                # eliminate nan and inf
                filter(
                    "replaceInfAndNaN",
                    self.sourceObjCropped,
                    dataObject.zeros(
                        self.sourceObjCropped.shape, self.sourceObjCropped.dtype
                    ),
                    self.sourceObjCropped,
                )

            self.roughness = dataObject()
            self.waviness = dataObject()
            try:
                Lc = float(self.gui.comboFilterLc["currentText"])
            except Exception:
                ui.msgWarning("wrong type", "Lc is no number", parent=self.gui)
                return
            if self.gui.comboFilterLs["currentText"] == "None":
                Ls = 0.0
            else:
                Ls = float(self.gui.comboFilterLs["currentText"])
            periodicity = 1
            if self.gui.radioFilterNonPeriodic["checked"]:
                periodicity = 0
            if "auto" in self.gui.comboMode["currentText"]:
                mode = "auto"
            elif "dft" in self.gui.comboMode["currentText"]:
                mode = "dft"
            else:
                mode = "convolution"
            self.endeffect = filter(
                "calcRoughnessProfile",
                self.sourceObjCropped,
                self.roughness,
                self.waviness,
                Lc,
                Ls,
                mode=mode,
                periodicity=periodicity,
                cutoff_factor=1.0,
            )

            self.gui.groupSelectDisplayedRow["visible"] = self.roughness.shape[0] > 1
            self.gui.spinFilterRow["value"] = 0
            self.gui.spinFilterRow["maximum"] = self.roughness.shape[0] - 1
            try:
                self.showFilteredProfile(0)
                self.calcRoughness()
                self.calcAbbott(self.gui.radioAbbottRoughness["checked"])
            except Exception as ex:
                raise ex
            finally:
                self.gui.groupFiltering["enabled"] = True

    def calcRoughness(self):
        if self.roughness is not None:
            try:
                Lc = float(self.gui.comboFilterLc["currentText"])
            except Exception:
                ui.msgWarning("wrong type", "Lc is no number", parent=self.gui)
                return
            rangeRdc = (
                self.gui.rangeRdc["minimumValue"],
                self.gui.rangeRdc["maximumValue"],
            )
            samplingMode = 2
            if self.gui.radioRoughnessMode0["checked"]:
                samplingMode = 0
            if self.gui.radioRoughnessMode1["checked"]:
                samplingMode = 1

            params = [
                "Ra",
                "Rq",
                "Rp",
                "Rv",
                "Rt",
                "Rz",
                "Rsk",
                "Rku",
                "Rdq",
                "Rda",
                "Rdc",
            ]
            self.gui.roughnessTable["verticalLabels"] = params
            result = dataObject([len(params), 5], "float64")
            c = 0
            self.gui.lblRoughnessWarning["visible"] = False
            warns = []

            for param in params:
                with warnings.catch_warnings(record=True) as w:
                    try:
                        [temp, samples] = filter(
                            "evalRoughnessProfile",
                            self.roughness,
                            param,
                            Lc,
                            self.endeffect,
                            sampling_length_mode=samplingMode,
                            mr_range=rangeRdc,
                        )
                    except RuntimeError as ex:
                        ui.msgCritical("Roughness", str(ex))
                        break
                    if w:
                        warns.append("%s: %s" % (param, w[-1].message))

                result[c, 0] = temp[0]
                result[c, 1] = temp[1]
                result[c, 2] = temp[2]
                if len(temp) > 3:
                    result[c, 3] = temp[3]
                else:
                    result[c, 3] = float("nan")
                result[c, 4] = samples
                c += 1
            if len(warns) > 0:
                self.gui.lblRoughnessWarning["text"] = "\n".join(warns)
                self.gui.lblRoughnessWarning["visible"] = True

            self.gui.roughnessTable["data"] = result

    def calcAbbott(self, roughnessNotWaviness: bool):
        abbott = dataObject()
        if self.roughness is not None and roughnessNotWaviness:
            filter("calcAbbottCurve", self.roughness, abbott, self.endeffect)
        elif self.waviness is not None and not roughnessNotWaviness:
            filter("calcAbbottCurve", self.waviness, abbott, self.endeffect)
        self.gui.plotAbbott["source"] = abbott

    def __calcRoughness(self, param, Lc, mode):
        result = dataObject()

    def showFilteredProfile(self, row: int = 0):
        if self.roughness is not None:
            obj = dataObject([3, self.roughness.shape[1]], self.roughness.dtype)

            if self.roughness.shape[0] > 1:
                row = self.gui.spinFilterRow["value"]
                obj[0, :] = self.sourceObjCropped[row, :]
                obj[1, :] = self.roughness[row, :]
                obj[2, :] = self.waviness[row, :]
            else:
                obj[0, :] = self.sourceObjCropped
                obj[1, :] = self.roughness
                obj[2, :] = self.waviness
            obj.axisUnits = [
                obj.axisUnits[0],
                self.sourceObjCropped.axisUnits[1],
            ]
            obj.axisDescriptions = [
                obj.axisDescriptions[0],
                self.sourceObjCropped.axisDescriptions[1],
            ]
            obj.axisScales = [
                obj.axisScales[0],
                self.sourceObjCropped.axisScales[1],
            ]
            obj.axisOffsets = [
                obj.axisOffsets[0],
                self.sourceObjCropped.axisOffsets[1],
            ]
            obj.valueUnit = self.sourceObjCropped.valueUnit
            obj.valueDescription = self.sourceObjCropped.valueDescription
            self.gui.plotFiltering["source"] = obj
            self.gui.plotFiltering["legendTitles"] = (
                "Raw Profile",
                "Roughness",
                "Waviness",
            )
            print(self.gui.plotFiltering["legendTitles"])
        else:
            self.gui.plotFiltering["source"] = dataObject()

    @ItomUi.autoslot("")
    def on_btnSourceRefresh_clicked(self):
        self.scanWorkspace()

    @ItomUi.autoslot("")
    def on_btnSourceSelect_clicked(self):
        dataObj = None
        itemName = self.gui.comboBox["currentText"]
        try:
            dataObj = __main__.__dict__[itemName]
        except Exception:
            ui.msgWarning(
                "not available",
                "The object %s is not available" % itemName,
                parent=self.gui,
            )
        if dataObj is not None:
            self.loadItem(dataObj)

    @ItomUi.autoslot("")
    def on_btnLoadDemo_clicked(self):
        dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        absPath = os.path.join(dir, "Ground_rawdata_872015436.sdf")
        dataObj = dataObject()
        filter("loadSDF", dataObj, absPath, "mm", "mm")
        self.loadItem(dataObj)

    @ItomUi.autoslot("")
    def on_btnSourceCropReset_clicked(self):
        self.sourceObjCropped = self.sourceObj.astype("float64")
        self.filterProfile()

    @ItomUi.autoslot("")
    def on_btnSourceCropCurrentView_clicked(self):
        if self.gui.plotSource2d["visible"]:
            self.sourceObjCropped = self.gui.plotSource2d["displayed"].astype("float64")
        else:
            self.sourceObjCropped = self.gui.plotSource1d["displayed"].astype("float64")
        self.filterProfile()

    @ItomUi.autoslot("int")
    def on_spinFilterRow_valueChanged(self, idx):
        self.showFilteredProfile(idx)

    @ItomUi.autoslot("")
    def on_btnFilterGo_clicked(self):
        self.filterProfile()

    @ItomUi.autoslot("")
    def on_btnRoughnessGo_clicked(self):
        if self.roughness is None:
            self.filterProfile()
        else:
            self.calcRoughness()
            self.calcAbbott(self.gui.radioAbbottRoughness["checked"])

    @ItomUi.autoslot("bool")
    def on_radioAbbottRoughness_toggled(self, roughnessNotWaviness):
        self.calcAbbott(roughnessNotWaviness)


if __name__ == "__main__":
    profile_roughness = ProfileRoughness()
    profile_roughness.show()
