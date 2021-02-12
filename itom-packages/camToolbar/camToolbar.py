# coding=iso-8859-15
"""
This file contains a toolbar with basic live image and snap shot interactions
"""

from abstractObjToolbar.abstractObjToolbar import abstractObjInteractionToolBar
import itom


class camToolbar(abstractObjInteractionToolBar):
    """
    This is the basic camera toolbar.
    """

    def __init__(
        self,
        myName,
        appendButtons=True,
        appendMenu=True,
        defaultCam="cam",
        defaultVar="dObj",
    ):
        """
        Initialisation of camera toolbar
        """

        abstractObjInteractionToolBar.__init__(self, myName, defaultVar)
        self.defaultCamName = defaultCam
        self.defaultVarName = defaultVar
        self.myNameDelete = myName
        hashName = str(hash(self.myNameDelete))

        self.hasMenu = appendMenu
        self.hasButtons = appendButtons

        if appendMenu == True:
            addMenu(MENU, hashName, self.myNameDelete)
            addMenu(BUTTON, hashName + "/LiveImage", "Open live image", self.live)
            addMenu(BUTTON, hashName + "/TakeImage", "Take image(s)", self.takeSnapshot)
            addMenu(
                BUTTON, hashName + "/Quikesnap", "Take a snap shot", self.takeSpeedShot
            )

        # addButton("toolBarRoughnesStatistics","Rz/Sz","surfaceAnalysisTools.calcRzSz()", "")
        # addButton("toolBarRoughnesStatistics","Ra/Sa","surfaceAnalysisTools.calcRaSa()", "")
        # addButton("toolBarRoughnesStatistics","Rq/Sq","surfaceAnalysisTools.calcRqSq()", "")
        if self.hasButtons == True:
            addButton(
                self.myNameDelete,
                "Live Image",
                self.live,
                ":/plots/icons/itom_icons/monitor.png",
            )
            addButton(
                self.myNameDelete,
                "Snapshot",
                self.takeSnapshot,
                ":/measurement/icons/itom_icons/snap.png",
            )
            addButton(
                self.myNameDelete,
                "Speed Shot",
                self.takeSpeedShot,
                ":/measurement/icons/itom_icons/snapQuick.png",
            )

    def __del__(self):
        """
        Toolbar deleter which cleans up the button bar
        """
        # removeButton("toolBarCam","Live Image")
        # removeButton("toolBarCam","Snapshot")
        # removeButton("toolBarCam","Speed Shot")
        if self.hasMenu == True:
            hashName = str(hash(self.myNameDelete))
            removeMenu(hashName)

        try:
            if self.hasButtons == True:
                removeButton(self.myNameDelete, "Live Image")
                removeButton(self.myNameDelete, "Snapshot")
                removeButton(self.myNameDelete, "Speed Shot")
        except:
            pass

    def getCamName(self, defaultCamName, valuename, textstring="{} for Camera:"):
        """
        Basic function to get a grabber name by text input
        """
        [varname, check] = ui.getText(
            "Variable input", textstring.format(valuename), defaultCamName
        )
        return [varname, check]

    def getGrabbers(self, title, camname):
        """This function gets the list of current grabbers and offers the user a drop down menu"""

        grabbers = []
        key = ""
        gvar = globals().keys()
        global typ
        typ = 0
        script = "globals()['typ']={}.getType()\n"
        for key in gvar:
            if type(globals()[key]) == itom.dataIO:
                exec(script.format(key))
                if (
                    typ & 128
                ):  # 0x01 = dataIO, # 0x80 = grabber --> only grabber allowed
                    grabbers.append(key)

        camindex = [i for i, x in enumerate(grabbers) if x == camname]
        if len(camindex) > 0:
            camindex = camindex[0]
        else:
            camindex = 0

        if len(grabbers) > 1:
            [myGrabber, done] = ui.getItem(title, "Devices", grabbers, camindex, False)
        elif len(grabbers) == 1:
            myGrabber = grabbers[0]
            done = True
        else:
            ui.msgCritical("Cam", "No dataIO-devices found", ui.MsgBoxOk)
            myGrabber = ""
            done = False

        del key
        del gvar
        del grabbers
        del typ
        return [myGrabber, done]

    def getVarName(self, title, VarName):
        """
        Generate a dropdown menu for selection a DataObject
        """
        dObjects = []
        key = ""
        gvar = globals().keys()

        for key in gvar:
            if type(globals()[key]) == itom.dataObject:
                dObjects.append(key)

        if len(dObjects) > 0:
            objindex = [i for i, x in enumerate(dObjects) if x == VarName]
            if len(objindex) > 0:
                objindex = objindex[0]
            else:
                objindex = 0
            [varname, done] = ui.getItem(
                title, "DataObjects", dObjects, objindex, False
            )
        else:
            ui.msgCritical("Plot", "No dataObjects found", ui.MsgBoxOk)
            varname = ""
            done = False
        del key
        del gvar
        del dObjects

        return [varname, done]

    def live(self, skipBox=False, defaultCamName=None):
        """
        Open a liveImage for the camera defined in defaultCamName. If skipBox is false and more than a grabber is open,
        a drop down menu will be displayed.
        """
        if defaultCamName == None:
            camname = self.defaultCamName
        else:
            camname = defaultCamName

        result = 0
        [camname, check] = self.getGrabbers("Live image", camname)

        if check == True:
            try:

                global rows
                script = "globals()['rows'] = {}.getParam('sizey')"
                exec(script.format(camname))

                if rows == 1:
                    # script = "liveLine({})\n" # command was removed by Marc Gronle in 2013
                    script = "liveImage({})\n"
                else:
                    script = "liveImage({})\n"
                    # Evaluate String with {} changed to varname
                result = eval(script.format(camname))

            except:
                if skipBox == False:
                    ui.msgCritical("Cam", "Execution error", ui.MsgBoxOk)
                check = False
                return [check, result]
        else:
            return [check, result]

        self.defaultCamName = camname
        return [check, result]

    def takeSnapshot(self, skipBox=False, defaultCamName=None, defaultVarName=None):
        if defaultCamName == None:
            camname = self.defaultCamName
        else:
            camname = defaultCamName
        if defaultVarName == None:
            defaultVar = self.defaultVarName
        else:
            defaultVar = defaultVarName

        result = ""

        grabbers = []
        dObjects = []
        key = ""
        gvar = globals().keys()
        global typ
        typ = 0
        s = "globals()['typ']={}.getType()\n"
        for key in gvar:
            if type(globals()[key]) == itom.dataIO:
                exec(s.format(key))
                if (
                    typ & 128
                ):  # 0x01 = dataIO, # 0x80 = grabber --> only grabber allowed
                    grabbers.append(key)
            if type(globals()[key]) == itom.dataObject:
                dObjects.append(key)

        if len(grabbers) == 0:
            ui.msgCritical("Cam", "No dataIO-devices found", ui.MsgBoxOk)
            myGrabber = ""
            done = False
            return [False, result]

        del key
        del gvar
        del typ

        camindex = [i for i, x in enumerate(grabbers) if x == camname]
        if len(camindex) > 0:
            camindex = camindex[0]
        else:
            camindex = 0

        objindex = [i for i, x in enumerate(dObjects) if x == defaultVar]
        if len(objindex) > 0:
            objindex = objindex[0]
        else:
            dObjects.append(defaultVar)
            objindex = 0

        dialogSnapshot = ui(
            itom.getAppPath() + "/itom-packages/camToolbar/snapshot.ui",
            ui.TYPEDIALOG,
            ui.BUTTONBAR_HORIZONTAL,
            dialogButtons={"AcceptRole": "OK", "RejectRole": "Cancel"},
        )
        dialogSnapshot.comboBoxGrabber.call("addItems", grabbers)
        dialogSnapshot.comboBoxGrabber.call("setCurrentIndex", camindex)
        dialogSnapshot.comboBoxdObj.call("addItems", dObjects)
        dialogSnapshot.comboBoxdObj.call("setCurrentIndex", objindex)
        dialogSnapshot.comboBoxdObj[
            "editable"
        ] = True  # .setProperty("comboBoxdObj", "editable", True)
        dialogSnapshot.spinBoxTempBin[
            "value"
        ] = 1  # .setProperty("txtBinning", "text", "1")
        dialogSnapshot.spinBoxNumImages[
            "value"
        ] = 1  # .setProperty("txtBinning", "text", "1")
        dialogSnapshot.checkBoxShow[
            "checked"
        ] = True  # .setProperty("checkBoxShow", "checked", True)
        retdialog = dialogSnapshot.show(1)

        if retdialog == 1:
            dataObj = dialogSnapshot.comboBoxdObj[
                "currentText"
            ]  # .getProperty("comboBoxdObj", "currentText")
            # dataObj = dataObj[0]
            camname = dialogSnapshot.comboBoxGrabber[
                "currentText"
            ]  # getProperty("comboBoxGrabber", "currentText")
            # camname = camname[0]
            binning = dialogSnapshot.spinBoxTempBin[
                "value"
            ]  # .getProperty("txtBinning","text")
            stacked = dialogSnapshot.spinBoxNumImages[
                "value"
            ]  # .getProperty("txtBinning","text")
            # binning = binning[0]
            show = dialogSnapshot.checkBoxShow[
                "checked"
            ]  # .getProperty("checkBoxShow", "checked")
            # show = show[0] #show is bool now....
        try:
            eval(dataObj)
            check = True
        except:
            try:
                s = "global {}\n{}=dataObject()"
                exec(s.format(dataObj, dataObj))

            except:
                if skipBox == False:
                    ui.msgCritical(
                        "DataObject", "Could not create DataObject", ui.MsgBoxOk
                    )
                    check = False
                    return [check, result]
            ui.msgInformation("DataObject", "Created DataObject {}".format(dataObj))
            check = True

        if check == True:
            try:
                # Check if variable exists
                eval(camname)
                eval(dataObj)
            except:
                if skipBox == False:
                    ui.msgCritical("Cam", "Variable does not exist", ui.MsgBoxOk)
                return [False, result]

            try:
                # Evaluate String with {} changed to varname

                checkGrabbing = eval("{}.getAutoGrabbing()".format(camname))
                eval("{}.startDevice()".format(camname))
                if checkGrabbing == True:
                    eval("{}.disableAutoGrabbing()".format(camname))

                if stacked == 1:
                    if binning == 1:

                        eval("{}.acquire()".format(camname))

                        script = 'tmpObj=dataObject()\n{camname}.getVal(tmpObj)\nglobals()["{dataObj}"]=tmpObj.copy()\ndel tmpObj'
                        exec(script.format(camname=camname, dataObj=dataObj), globals())

                    else:
                        tmpObj = dataObject()
                        script = "{camname}.acquire()\n{camname}.getVal(tmpObj)"

                        exec(script.format(camname=camname), globals(), locals())
                        tmpObj2 = tmpObj.astype("float64")
                        script = "{camname}.acquire()\n{camname}.getVal(tmpObj)\n".format(
                            camname=camname
                        )
                        for i in range(1, binning):
                            exec(script, globals(), locals())
                            tmpObj2 = tmpObj2 + tmpObj.astype("float64")
                        bin = 1 / (binning)
                        script = "{}=tmpObj2*{}".format(dataObj, bin)
                        exec(script)
                        del tmpObj

                else:
                    if binning > 1:
                        print("Warning, binning not compatible with stack")
                    exec(
                        'tmpObj=dataObject()\n{camname}.acquire()\n{camname}.getVal(tmpObj)\nglobals()["{dataObj}"] = dataObject([{cnt},tmpObj.shape[0], tmpObj.shape[1]], tmpObj.dtype)\nglobals()["{dataObj}"][0,:, :] = tmpObj'.format(
                            cnt=stacked, camname=camname, dataObj=dataObj
                        )
                    )
                    for i in range(1, stacked):
                        exec(
                            'tmpObj=dataObject()\n{camname}.acquire()\n{camname}.getVal(tmpObj)\nglobals()["{dataObj}"][{cnt},:, :] = tmpObj'.format(
                                cnt=i, camname=camname, dataObj=dataObj
                            )
                        )

                if checkGrabbing == True:
                    eval("{}.enableAutoGrabbing()".format(camname))
                eval("{}.stopDevice()".format(camname))

                if show == True:
                    script = "plot({})"
                    exec(script.format(dataObj))

                result = dataObj

            except:
                if skipBox == False:
                    ui.msgCritical("Cam", "Execution error", ui.MsgBoxOk)
                check = False

        self.defaultCamName = camname
        self.defaultVarName = dataObj

        return [check, result]

    def takeSpeedShot(self):
        camname = self.defaultCamName
        dataObj = self.defaultVarName

        result = ""

        try:
            # Check if variable exists
            eval(camname)
            check = True
        except:
            [camname, check] = self.getGrabbers("default cam not defined", camname)
            check = False
            return [check, ""]
            # ui.msgCritical("Cam", "Cam does not exist", ui.MsgBoxOk)

        try:
            eval(dataObj)
            check = True
        except:
            try:
                s = "global {}\n{}=dataObject()"
                exec(s.format(dataObj, dataObj))
                check = True

            except:
                ui.msgCritical("DataObject", "Could not create DataObject", ui.MsgBoxOk)
                check = False
                return [check, ""]

        script = '{camname}.acquire()\ntmpObj=dataObject()\n{camname}.getVal(tmpObj)\nglobals()["{dataObj}"]=tmpObj.copy()\ndel tmpObj\n'

        try:
            checkGrabbing = eval("{}.getAutoGrabbing()".format(camname))
            eval("{}.startDevice()".format(camname))
            if checkGrabbing == True:
                eval("{}.disableAutoGrabbing()".format(camname))
            exec(script.format(camname=camname, dataObj=dataObj), globals())
            if checkGrabbing == True:
                eval("{}.enableAutoGrabbing()".format(camname))
            eval("{}.stopDevice()".format(camname))

        except:
            ui.msgCritical("Cam", "Execution error during Grabbing", ui.MsgBoxOk)
            check = False

        try:
            s = "plot({})".format(dataObj)
            eval(s)
            result = dataObj

        except:
            ui.msgCritical("Cam", "Execution error during Plotting", ui.MsgBoxOk)
            check = False

        self.defaultCamName = camname
        return [check, result]


if __name__ == "__main__":
    toolbarCamera = camToolbar("Camera Access", not (userIsUser()))

# Tobias Boettcher, ITO, & Wolfram Lyda, twip optical solutions
