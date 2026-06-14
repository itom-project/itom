# coding=iso-8859-15
"""
This file contains a toolbar with basic surface analysis methods e.g. Rz, Sa, Polynomical Fittings,
INIT via
toolBarRoughnesStatistics = roughnessToolBar("toolBarRoughnesStatistics")
...
...
"""


from abstractObjToolbar.abstractObjToolbar import abstractObjInteractionToolBar
import itom

if not (itom.pluginLoaded("FittingFilters")):
    raise RuntimeError(
        "FittingFilters-plugin not available. Loading roughness toolbar cancelled"
    )


class surfaceAnalysisTools(abstractObjInteractionToolBar):
    """
    This class contains function for the analysis of technical surfaces.
    It wrapps functions from the M++Filter, the FittingFilters and the ITOMFilters plugins to a menu entry and a button bar.
    """

    def __init__(
        self,
        myName,
        appendButtons=True,
        appendMenu=True,
        defPrecision=3,
        defaultVar="dObj",
    ):

        # self.defaultVarName = defaultVar
        self.defaultPrecision = defPrecision
        self.defXGradPoly = 3
        self.defYGradPoly = 3
        # self.myNameDelete = myName
        self.myNameDelete = myName
        abstractObjInteractionToolBar.__init__(self, myName, defaultVar)

        hashName = str(hash(self.myNameDelete))

        self.hasMenu = appendMenu
        self.hasButtons = appendButtons

        if appendMenu == True:
            addMenu(MENU, hashName, self.myNameDelete)
            addMenu(MENU, hashName + "/alignObject", "align object")
            addMenu(
                BUTTON, hashName + "/alignObject/polyfit", "Fit polynom", self.planefit
            )
            addMenu(
                BUTTON,
                hashName + "/alignObject/planefit",
                "Fit line or plane",
                self.polyfit,
            )
            addMenu(MENU, hashName + "/roughness", "roughness calculation")
            addMenu(
                BUTTON,
                hashName + "/roughness/Rz_Sz",
                "Calculate Rz or Sz",
                self.calcRzSz,
            )
            addMenu(
                BUTTON,
                hashName + "/roughness/Ra_Sa",
                "Calculate Ra or Sa",
                self.calcRaSa,
            )
            addMenu(
                BUTTON,
                hashName + "/roughness/Rq_Sq",
                "Calculate Rq or Sq",
                self.calcRqSq,
            )
            addMenu(
                BUTTON,
                hashName + "/roughness/Sz_10P",
                "Calculate former Sz",
                self.calcSz10P,
            )

        # addButton("toolBarRoughnesStatistics","Rz/Sz","surfaceAnalysisTools.calcRzSz()", "")
        # addButton("toolBarRoughnesStatistics","Ra/Sa","surfaceAnalysisTools.calcRaSa()", "")
        # addButton("toolBarRoughnesStatistics","Rq/Sq","surfaceAnalysisTools.calcRqSq()", "")
        if self.hasButtons == True:
            addButton(
                self.myNameDelete,
                "Subtract Plane",
                self.planefit,
                ":/measurement/icons/itom_icons/subtractPlane.png",
            )
            addButton(
                self.myNameDelete,
                "Subtract Polynome",
                self.polyfit,
                ":/measurement/icons/itom_icons/subtractPolynom.png",
            )

    def __del__(self):
        # removeButton("roughness","Rz/Sz")
        # removeButton("roughness","Ra/Sa")
        # removeButton("roughness","Rq/Sq")

        # removeButton("roughness","Subtract Plane")
        # removeButton("roughness","Subtract Polynome")

        # removeButton(self.myNameDelete,"Rz/Sz")
        # removeButton(self.myNameDelete,"Ra/Sa")
        # removeButton(self.myNameDelete,"Rq/Sq")
        if self.hasMenu == True:
            hashName = str(hash(self.myNameDelete))
            removeMenu(hashName)

        if self.hasButtons == True:
            removeButton(self.myNameDelete, "Subtract Plane")
            removeButton(self.myNameDelete, "Subtract Polynome")

    def __getVarNameText(
        self, defaultVarName, valuename, textstring="Calculate {} for Object:"
    ):
        [varname, check] = ui.getText(
            "Variable input", textstring.format(valuename), defaultVarName
        )
        return [varname, check]

    def calcRzSz(self, skipBox=False, defaultVarName=None):
        """
        calcRzSz([skipBox [, defaultVarName]) -> give a list with all objects and calculate Rz or Sz, Sp, Sv
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
            - List with parameter: [Rz, Skipped Pixel] or [Sp, Sv, Sz]
        """
        if defaultVarName == None:
            varname = self.defaultVarName
        else:
            varname = defaultVarName

        # Get the varaiblename as string
        # [varname, check]= self.__getVarNameText(varname, "Rz/Sz")
        [varname, obtype, check] = self.getVarNameDialog(
            "Rz/Sz", varname, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

        if plane == False and line == False:
            ui.msgCritical("Roughness", "Object is no line or plane", ui.MsgBoxOk)
            result = [0, 0]

        if check == True:
            try:
                # Check if variable exists
                eval(varname)
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return [False, [0, 0]]

            # Create a command to calculate Rz as executeable string
            if line == True:
                script = 'filter("calcRz",{},5)\n'
            else:
                script = 'filter("calcSvSpSz",{})\n'

            try:
                # Evaluate String with {} changed to varname
                result = eval(script.format(varname))
                valueUnit = eval("{}.valueUnit".format(varname))
                if line == True:
                    if skipBox == False:
                        [scaledResult, scaledUnit] = scaleDoubleUnit(
                            getDefaultScaleableUnits(), result[0], valueUnit
                        )
                        ui.msgInformation(
                            "Roughness",
                            "Rz = {:.4f} {}\nSkipped {} Pixel".format(
                                scaledResult, scaledUnit, result[1]
                            ),
                            ui.MsgBoxOk,
                        )
                else:
                    if skipBox == False:
                        [scaledResult0, scaledUnit0] = scaleDoubleUnit(
                            getDefaultScaleableUnits(), result[0], valueUnit
                        )
                        [scaledResult1, scaledUnit1] = scaleDoubleUnit(
                            getDefaultScaleableUnits(), result[1], valueUnit
                        )
                        [scaledResult2, scaledUnit2] = scaleDoubleUnit(
                            getDefaultScaleableUnits(), result[2], valueUnit
                        )
                        ui.msgInformation(
                            "Roughness",
                            "Sv = {:.4f} {}\nSp = {:.4f} {}\nSz = {:.4f} {}".format(
                                scaledResult0,
                                scaledUnit0,
                                scaledResult1,
                                scaledUnit1,
                                scaledResult2,
                                scaledUnit2,
                            ),
                            ui.MsgBoxOk,
                        )
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False
                result = [0, 0]
        else:
            result = [0, 0]

        self.defaultVarName = varname

        return [check, result]

    def calcSz10P(self, skipBox=False, defaultVarName=None):
        """
        calcSz10P([skipBox [, defaultVarName]) -> give a list with all objects and calculate former Sz (10 Point height)
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
            - List with parameter: [Sz, Skipped Pixel in y, Skipped Pixel in x]
        """
        if defaultVarName == None:
            varname = self.defaultVarName
        else:
            varname = defaultVarName

        # Get the varaiblename as string
        # [varname, check]= self.__getVarNameText(varname, "Rz/Sz")
        [varname, obtype, check] = self.getVarNameDialog(
            "TenPointHeigth (former Sz)", varname, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

        if plane == False:
            ui.msgCritical("Roughness", "Object is no plane", ui.MsgBoxOk)
            result = [0, 0]

        if check == True:
            try:
                # Check if variable exists
                eval(varname)
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return [False, [0, 0]]

            # Create a command to calculate Rz as executeable string
            script = 'filter("calcTenPointHeigth",{})\n'

            try:
                # Evaluate String with {} changed to varname
                result = eval(script.format(varname))
                valueUnit = eval("{}.valueUnit".format(varname))
                if skipBox == False:
                    [scaledResult, scaledUnit] = scaleDoubleUnit(
                        getDefaultScaleableUnits(), result[0], valueUnit
                    )
                    ui.msgInformation(
                        "Roughness",
                        "Sz 10P = {:.4f} {}\nSkipped {} Pixel in y\nSkipped {} Pixel in x".format(
                            scaledResult, scaledUnit, result[1], result[2]
                        ),
                        ui.MsgBoxOk,
                    )
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False
                result = [0, 0]
        else:
            result = [0, 0]

        self.defaultVarName = varname

        return [check, result]

    def calcRaSa(self, skipBox=False, defaultVarName=None):
        """
        calcRaSa([skipBox [, defaultVarName]) -> give a list with all objects and calculate Ra or Sa
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
            - List with parameter: [Ra] or [Sa]
        """
        if defaultVarName == None:
            varname = self.defaultVarName
        else:
            varname = defaultVarName

        # [varname, check]= self.getVarName(varname, "Ra/Sa")
        [varname, obtype, check] = self.getVarNameDialog(
            "Ra/Sa", varname, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

        if plane == False and line == False:
            ui.msgCritical("Roughness", "Object is no line or plane", ui.MsgBoxOk)
            result = [
                0,
            ]

        if check == True:
            try:
                eval(varname)
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return [False, [0,]]

            # Create a command to calculate Rz as executeable string
            if line == True:
                script = 'filter("calcRa",{})\n'
                printScript = "Ra = {:.4f} {}"
            else:
                script = 'filter("calcSa",{})\n'
                printScript = "Sa = {:.4f} {}"

            try:
                result = eval(script.format(varname))
                valueUnit = eval("{}.valueUnit".format(varname))
                if skipBox == False:
                    [scaledResult, scaledUnit] = scaleDoubleUnit(
                        getDefaultScaleableUnits(), result, valueUnit
                    )
                    ui.msgInformation(
                        "Roughness",
                        printScript.format(scaledResult, scaledUnit),
                        ui.MsgBoxOk,
                    )
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False
                result = [
                    0,
                ]
        else:
            result[0]

        self.defaultVarName = varname

        return [check, [result]]

    def calcRqSq(self, skipBox=False, defaultVarName=None):
        """
        calcRqSq([skipBox [, defaultVarName]) -> give a list with all objects and calculate Rq or Sq
        Parameters:
            - skipBox             If True, the dialog will be skipped
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
            - List with parameter: [Rq] or [Sq]
        """
        if defaultVarName == None:
            defaultVar = self.defaultVarName
        else:
            defaultVar = defaultVarName

        # [varname, check]= self.getVarName(defaultVar, "Rq/Sq")
        [varname, obtype, check] = self.getVarNameDialog(
            "Rq/Sq", defaultVar, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

        if plane == False and line == False:
            ui.msgCritical("Roughness", "Object is no line or plane", ui.MsgBoxOk)
            result = [
                0,
            ]

        if check == True:
            try:
                eval(varname)
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return [False, [0,]]

            # Create a command to calculate Rz as executeable string
            if line == True:
                script = 'filter("calcRq",{})\n'
                printScript = "Rq = {:.4f} {}"
            else:
                script = 'filter("calcSq",{})\n'
                printScript = "Sq = {:.4f} {}"

            try:
                result = eval(script.format(varname))
                valueUnit = eval("{}.valueUnit".format(varname))
                if skipBox == False:
                    [scaledResult, scaledUnit] = scaleDoubleUnit(
                        getDefaultScaleableUnits(), result, valueUnit
                    )
                    ui.msgInformation(
                        "Roughness",
                        printScript.format(scaledResult, scaledUnit),
                        ui.MsgBoxOk,
                    )
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False
                result = [
                    0,
                ]
        else:
            result[
                0,
            ]

        self.defaultVarName = varname
        return [check, [result,]]

    def planefit(self, skipBox=False, defaultVarName=None):
        """
        planefit([skipBox [, defaultVarName]) -> Fit a plane or a line into the object and substract the values. The function works in place and the object will be changed to float64.
        Parameters:
            - skipBox             If True, the dialog will be skipped
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
        """
        if defaultVarName == None:
            defaultVar = self.defaultVarName
        else:
            defaultVar = defaultVarName

        # [varname, check]= self.getVarName(defaultVar, "Plane fit")
        [varname, obtype, check] = self.getVarNameDialog(
            "Plane-Fit", defaultVar, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

            if plane == False and line == False:
                ui.msgCritical("Roughness", "Object is no line or plane", ui.MsgBoxOk)

        if check == True:
            try:
                dims = eval("{}.dims".format(varname))
                if dims > 2:
                    if skipBox == False:
                        ui.msgCritical(
                            "Roughness", "Dimension must be 1D or 2D", ui.MsgBoxOk
                        )
                dim = eval("{}.shape".format(varname))

            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return False
            try:
                script = 'filter("subtractRegressionPlane",{}, {})\n'
                result = eval(script.format(varname, varname))

            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False

        self.defaultVarName = varname
        return check

    def polyfit(self, skipBox=False, defaultVarName=None):
        """
        polyfit([skipBox [, defaultVarName]) -> Fit a mxn-th Order or n-th Order into a 1D or 2D object and substract the values. The function works in place and the object will be changed to float64.
        Parameters:
            - skipBox             If True, the dialog will be skipped
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:                   True finishied correctly
            - List with parameter: [Rq] or [Sq]
        """
        if defaultVarName == None:
            defaultVar = self.defaultVarName
        else:
            defaultVar = defaultVarName

        # [varname, check] = self.getVarName(defaultVar, "Poly fit")
        [varname, obtype, check] = self.getVarNameDialog(
            "Polynome-Fit", defaultVar, globals(), "ND"
        )

        if check == True:
            [line, plane, dir, check] = self.checkIsLineOrPlane(varname, globals())

            if plane == False and line == False:
                ui.msgCritical("Roughness", "Object is no line or plane", ui.MsgBoxOk)

        if check == False:
            return check

        [self.defXGradPoly, check] = ui.getInt(
            "Polyfit", "Order in x", self.defXGradPoly
        )

        tempObjVar = dataObject()

        if check == True:
            try:
                dims = eval("{}.dims".format(varname))
                if dims > 2:
                    if skipBox == False:
                        ui.msgCritical(
                            "Roughness", "Dimension must be 1D or 2D", ui.MsgBoxOk
                        )
                    return False

            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Variable does not exist", ui.MsgBoxOk)
                return False
            try:
                if line == True:
                    if dir == "x":
                        script = 'filter("fitPolynom2D",{}, tempObjVar, {}, 0)\n'
                    else:
                        script = 'filter("fitPolynom2D",{}, tempObjVar, 0, {})\n'
                    result = eval(script.format(varname, str(self.defXGradPoly)))
                    script = "globals()[\"{}\"] = {}.astype('float64') - tempObjVar".format(
                        varname, varname
                    )
                    exec(script, globals(), {"tempObjVar": tempObjVar})
                else:
                    [self.defYGradPoly, check] = ui.getInt(
                        "Polyfit", "Grade in y", self.defYGradPoly
                    )
                    script = 'filter("fitPolynom2D",{}, {}, {}, {})\n'
                    result = eval(
                        script.format(
                            varname,
                            "tempObjVar",
                            str(self.defXGradPoly),
                            str(self.defYGradPoly),
                        )
                    )
                    script = "globals()[\"{}\"] = {}.astype('float64') - tempObjVar".format(
                        varname, varname
                    )
                    exec(script)
            except:
                if skipBox == False:
                    ui.msgCritical("Roughness", "Calculation error", ui.MsgBoxOk)
                check = False

            if check == True:
                # still everything worked fine
                script = "{}.addToProtocol('Filtered )"

        del tempObjVar
        self.defaultVarName = varname
        return check


if __name__ == "__main__":
    toolbarSurface = surfaceAnalysisTools("Surface analysis", not (userIsUser()))
