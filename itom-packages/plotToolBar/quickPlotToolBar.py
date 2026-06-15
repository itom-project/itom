# coding=iso-8859-15
"""
This file contains a toolbar with basic plot interactions e.g. linePlot, 2D-Plot
INIT via
toolBarQuickPlot = quickPlotToolBar('toolBarQuickPlot')
addButton("toolBarQuickPlot","Show Image","toolBarQuickPlot.show2D()", "icons_m/2d.png")
addButton("toolBarQuickPlot","Show Line","toolBarQuickPlot.show1D()", "icons_m/1d.png")
"""

from abstractObjToolbar.abstractObjToolbar import abstractObjInteractionToolBar

import itom

hasMatPlotLib = False
hasBASICFILTERS = True

loadWarnings = []

if not (plotLoaded("twipOGLFigure")):
    loadWarnings.append(
        "- 3D plot 'twipOGLFigure' not found. Try to use deprecated 'ItomIsoGLWidget' instead."
    )

if not (itom.pluginLoaded("BasicFilters")):
    loadWarnings.append(
        "- Not all functions available since plugin 'BasicFilters' not available."
    )
    hasBASICFILTERS = False

if hasMatPlotLib == True:
    try:
        import matplotlib

        matplotlib.use("module://mpl_itom.backend_itomagg", False)
    except:
        loadWarnings.append(
            "- Not all functions available since Python package 'matplotlib' not available."
        )
        hasMatPlotLib = False

if hasMatPlotLib == True:
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from matplotlib import cm
    from matplotlib import pyplot as plt

if len(loadWarnings) > 0:
    print("Warning while loading 'QuickPlotToolBar':\n", "\n".join(loadWarnings))


class quickPlotToolBar(abstractObjInteractionToolBar):
    """
    This is toolbar for plotting dataObjects.
    """

    def __init__(
        self,
        myName,
        hasISO,
        hasMCPP,
        appendButtons=True,
        appendMenu=True,
        defaultVar="dObj",
    ):
        """
        set up the basic variable, this means the default dObj
        """
        abstractObjInteractionToolBar.__init__(self, myName, defaultVar)
        self.hasISO = hasISO
        self.hasMCPP = hasMCPP

        self.myNameDelete = myName
        hashName = str(hash(self.myNameDelete))

        self.hasMenu = appendMenu
        self.hasButtons = appendButtons

        if appendMenu == True:
            addMenu(MENU, hashName, self.myNameDelete)
            addMenu(BUTTON, hashName + "/ShowImage", "2D-Plot", self.show2D)
            addMenu(BUTTON, hashName + "/ShowLine", "Line-Plot", self.show1D)

            if plotLoaded("ItomIsoGLWidget") or plotLoaded("twipOGLFigure"):
                addMenu(BUTTON, hashName + "/ShowISO", "Iso-Plot", self.show25D)

            if self.hasMCPP == True:
                addMenu(
                    BUTTON, hashName + "/ShowHistogramm", "Histogramm", self.showHist,
                )
        if self.hasButtons == True:
            addButton(
                self.myNameDelete,
                "Show Image",
                self.show2D,
                ":/plots/icons/itom_icons/2d.png",
            )
            addButton(
                self.myNameDelete,
                "Show Line",
                self.show1D,
                ":/plots/icons/itom_icons/1d.png",
            )

            if plotLoaded("ItomIsoGLWidget") or plotLoaded("twipOGLFigure"):
                addButton(
                    self.myNameDelete,
                    "Show ISO",
                    self.show25D,
                    ":/plots/icons/itom_icons/3d.png",
                )

            if self.hasMCPP == True:
                addButton(
                    self.myNameDelete,
                    "Show Histogramm",
                    self.showHist,
                    ":/plots/icons/itom_icons/histogra.png",
                )

    def __del__(self):
        """
        Delete the toolBar content before getting killed
        """

        if self.hasMenu == True:
            hashName = str(hash(self.myNameDelete))
            removeMenu(hashName)

        try:
            if self.hasButtons == True:
                removeButton(self.myNameDelete, "Show Image")
                removeButton(self.myNameDelete, "Show Line")
                if plotLoaded("ItomIsoGLWidget") or plotLoaded("twipOGLFigure"):
                    emoveButton(self.myNameDelete, "Show ISO")
                if self.hasMCPP == True:
                    removeButton(self.myNameDelete, "Show Histogramm")
        except:
            pass

    def __getVarNameOld(self, title, VarName, objType="2D"):
        """
        get a list with suitable variables
        """
        dObjects = []
        key = ""
        gvar = globals().keys()

        dimsTestString = ""

        if objType == "1D":
            dimsTestString = "{}.shape({}.dims-1) == 1 or {}.shape({}.dims-2) == 1"
        else:
            dimsTestString = "{}.shape({}.dims-1)>1 and {}.shape({}.dims-2)>1"

        for key in gvar:
            if type(globals()[key]) == itom.dataObject:
                if eval(dimsTestString.format(key, key, key, key)):
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

    def showHist(self, skipBox=False, defaultVarName=None):
        """
        showHist([skipBox [, defaultVarName]) -> plot the histogramm of a given dataObject
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to calc the histogramm
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        if self.hasMCPP == False:
            ui.msgCritical(
                "DataObject",
                "Execution error, MCPPFilterFuncs not loaded",
                ui.MsgBoxOk,
            )
            check = False
            result = 0
            return [check, result]

        if defaultVarName == None:
            dataObj = self.defaultVarName
        else:
            dataObj = defaultVarName

        if skipBox == False:
            [dataObj, obtype, check] = self.getVarNameDialog(
                "Choose Object:", dataObj, globals(), "lineORplane", 1
            )
        else:
            check = True

        if check == True:
            try:
                # Check if variable exists
                eval(dataObj)
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Variable does not exist", ui.MsgBoxOk)
                return [False, 0]
            try:
                tempHist = dataObject()

                s = 'filter("calcHist",{}, tempHist)'
                eval(s.format(dataObj))
                result = plot(tempHist)
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Execution error", ui.MsgBoxOk)
                check = False
                result = 0
        else:
            result = 0

        self.defaultVarName = dataObj
        return [check, result]

    def show2D(self, skipBox=False, defaultVarName=None, typeFilter=3):
        """
        show2D([skipBox [, defaultVarName]) -> give a list with all 2D-objects an plot one of this
        Parameters:
            - skipBox         If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
            - typeFilter      BitMask of allowed dataTypes (1: DataObjects, 2: numpyArrays / npObjects, 3:pointclouds)
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        if defaultVarName == None:
            dataObj = self.defaultVarName
        else:
            dataObj = defaultVarName

        if skipBox == False:
            [dataObj, obtype, check] = self.getVarNameDialog(
                "Choose Object:", dataObj, globals(), "plane", typeFilter
            )
        else:
            check = True

        if check == True:
            try:
                # Check if variable exists
                eval(dataObj)
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Variable does not exist", ui.MsgBoxOk)
                return [False, 0]
            try:
                if obtype == 2:
                    s = "plot(dataObject({}))"
                else:
                    s = "plot({})"
                result = eval(s.format(dataObj))
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Execution error", ui.MsgBoxOk)
                check = False
                result = 0
        else:
            result = 0

        self.defaultVarName = dataObj
        return [check, result]

    def show25DOGL(self, dataObj):
        """
        show25DOGL([skipBox [, defaultVarName]) -> plot 2D-Object via ITOM-DesignerWidget 3DPlot-Widget
        Parameters:
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        check = True
        result = 0

        try:
            if plotLoaded("twipOGLFigure"):
                s = 'plot({},"twipOGLFigure")'
            else:
                s = 'plot({},"ItomIsoGLWidget")'
            result = eval(s.format(dataObj))
        except:
            result = 0
            ui.msgCritical(
                "DataObject", "Open GL-Window for 3D-Plot failed", ui.MsgBoxOk
            )

        return [check, result]

    def show25DMPL(self, dataObj):
        """
        show25DMPL([skipBox [, defaultVarName]) -> plot 2D-Object plot 2D-Object via Matplot-Lib ISO-representation
        Parameters:
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        check = True
        result = 0
        skipBox = False

        try:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            shape = eval("{}.shape()".format(dataObj))
            scale = eval("{}.axisScales".format(dataObj))
            offset = eval("{}.axisOffsets".format(dataObj))
            unit = eval("{}.axisUnits".format(dataObj))
            desc = eval("{}.axisDescriptions".format(dataObj))
            valUn = eval("{}.valueUnit".format(dataObj))
            valDes = eval("{}.valueDescription".format(dataObj))

            minValue = eval('filter("minValue",{})'.format(dataObj))
            maxValue = eval('filter("maxValue",{})'.format(dataObj))

            npdObj = eval("npDataObject({})".format(dataObj))

            # create supporting points cartesian system
            x = np.linspace(
                (0 - offset[dims - 1]) * scale[dims - 1],
                (shape[dims - 1] - 1 - offset[dims - 1]) * scale[dims - 1],
                shape[dims - 1],
            )
            y = np.linspace(
                (0 - offset[dims - 2]) * scale[dims - 2],
                (shape[dims - 2] - 1 - offset[dims - 2]) * scale[dims - 2],
                shape[dims - 2],
            )

            # transform them to cartesian system
            X, Y = np.meshgrid(x, y)
            ystride = int(len(npdObj) / 60)
            xstride = int(len(npdObj[0]) / 60)
            ax.plot_surface(
                X,
                Y,
                npdObj,
                rstride=ystride,
                cstride=xstride,
                cmap=cm.jet,
                linewidth=0,
                antialiased=False,
            )
            # ax.plot_surface(X, Y, npdObj, rstride=5, cstride=5, linewidth=0, antialiased=False)
            ax.set_zlim3d(minValue, maxValue)
            ax.set_xlabel(r"${} in {}$".format(desc[dims - 1], unit[dims - 1]))
            ax.set_ylabel(r"${} in {}$".format(desc[dims - 2], unit[dims - 2]))
            ax.set_zlabel(r"${} in {}$".format(valDes, valUn))
            plt.show()

            del shape
            del scale
            del offset
            del unit
            del desc
            del valUn
            del valDes
            del Y
            del y
            del X
            del x
            del npdObj

            result = fig
        except:
            if skipBox == False:
                ui.msgCritical("DataObject", "Execution error", ui.MsgBoxOk)
            check = False
            result = 0

        return [check, result]

    def show25D(self, skipBox=False, defaultVarName=None):
        """
        show25D([skipBox [, defaultVarName])  -> give a list with all 2D-objects an plot one of this either via ITOM-Filters 3DPlot-Widget or MatplotLib
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        if defaultVarName == None:
            dataObj = self.defaultVarName
        else:
            dataObj = defaultVarName

        if skipBox == False:
            [dataObj, obtype, check] = self.getVarNameDialog(
                "Choose Object:", dataObj, globals(), "plane"
            )
        else:
            check = True

        if check == True:
            dims = 2
            try:
                # Check if variable exists
                dims = eval("{}.dims".format(dataObj))
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Variable does not exist", ui.MsgBoxOk)
                return [False, 0]

            if True:  # pluginLoaded("ITOM-Filter"):
                [check, result] = self.show25DOGL(dataObj)
            else:
                [check, result] = self.show25DMPL(dataObj)

        else:
            result = 0

        self.defaultVarName = dataObj
        return [check, result]

    def show1D(self, skipBox=False, defaultVarName=None, typeFilter=3):
        """
        show1D([skipBox [, defaultVarName]) -> give a list with all 1D-objects an plot one of this
        Parameters:
            - skipBox             If True, the dialog will be skipped and
            - defaultVarName  The name of the variable to plot
            - typeFilter      BitMask of allowed dataTypes (1: DataObjects, 2: numpyArrays / npObjects, 3:pointclouds)
        Return parameters:
            - check:       True finishied correctly
            - plotHandle: Handle to the figure (0 if failed)
        """
        if defaultVarName == None:
            dataObj = self.defaultVarName
        else:
            dataObj = defaultVarName

        if skipBox == False:
            [dataObj, obtype, check] = self.getVarNameDialog(
                "Choose Object:", dataObj, globals(), "line", typeFilter
            )
        else:
            check = True

        if check == True:
            try:
                # Check if variable exists
                eval(dataObj)
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Variable does not exist", ui.MsgBoxOk)
                return [False, 0]
            try:
                if obtype == 2:
                    s = "plot(dataObject({}))"
                else:
                    s = "plot({})"
                exec(s.format(dataObj))
                result = None
            except:
                if skipBox == False:
                    ui.msgCritical("DataObject", "Execution error", ui.MsgBoxOk)
                check = False
                result = [0, 0]
        else:
            result = [0, 0]

        self.defaultVarName = dataObj
        return [check, result]


if __name__ == "__main__":
    toolBarQuickPlot = quickPlotToolBar(
        "Plotting Tools", hasMatPlotLib or True, hasBASICFILTERS, not (userIsUser()),
    )

del hasMatPlotLib
del hasBASICFILTERS
del loadWarnings
# Wolfram Lyda, ITO, 2012
