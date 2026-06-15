# coding=iso-8859-15
"""
This file contains an abstract toolbar with basic functions for ToolBar / Menu Interaction.
This contains parser for the global workspace to find DataObjects.
By Wolfram Lyda, ITO, 2012
"""
import itom
from itom import ui

reloadModules = 1


class abstractObjInteractionToolBar:
    """
    This is an abstract class which can be used as a templed for Object interaction Toolbar.
    The roughness Tools and the plotTools inherit this class.
    """

    defaultVarName = None
    myNameDelete = None

    def __init__(self, myName, defaultVar):
        """
        __init__(myName , defaultVar) -> set up the basic variable, this means the default dObject

        Parameters
        ------------
        myName : {str}
            Name of this toolBar. Nessecary for the deleter function to kill the b-bar.
        defaultVar :  {str}
            default variable name.

        Returns
        -------
        None

        Notes
        -------
        This function initializes an abstract toolbar. You can derive toolBars with object variable selection from this class.
        """

        self.defaultVarName = defaultVar
        self.myNameDelete = myName
        return

    def __del__(self):
        """
        Delete the toolBar content before getting killed
        e.g. removeButton(self.myNameDelete,"Show Image")
        e.g. removeButton(self.myNameDelete,"Show Line")
        """
        return

    def getVarNameDialog(
        self, title, VarName, workSpace, objType="ND", acceptedObjectType=1
    ):
        """
        getVarName(title, VarName, workSpace [, objType]) -> opens a drop down window with suitable DataObjects / npOjects / PointClouds and returns selected variable

        Parameters
        ------------
        title : {str}
            Title of the dialog.
        VarName :  {str}
            default variable name.
        VarName :  {PythonDict}
            Kontent of the global workspace of caller (use globals())
        objType : {str}, optinal
            ObjectTypes to filter. Can be ' line', 'plane' , ' empty' or 'ND'
        acceptedObjectType: {int}, optional
            BitSet 1 = DataObjects, 2 = NumPy-Array, 4 = PointClouds accepted


        Returns
        -------
        varname: {str}
            string value of the variable
        done:  {bool}
            Check, if dialog was closed with 'ok' or 'cancel'

        Notes
        -------
        This function opens a drop down window with suitable DataObject according to 'objType'.
        The selected Object-Name is returned together with a boolean expression representing
        the dialog status. If only one element fits to 'objType', the dialog is skipped.
        """

        dObjects = []
        npObjects = []
        pcObjects = []

        key = ""
        objtype = 1
        dimsTestString = ""
        dimsNPTestString = ""

        try:
            if (acceptedObjectType & 2) == 2:
                eval("numpy.ndarray")
        except:
            import numpy

        try:
            if (acceptedObjectType & 4) == 4:
                eval("itom.PointCloud")
        except:
            acceptedObjectType = acceptedObjectType - 4

        if (acceptedObjectType & 1) == 1:
            if objType == "empty":
                dimsTestString = "{}.dims == 0"
            elif objType == "plane":
                dimsTestString = "{}.shape[{}.dims-1]>1 and {}.shape[{}.dims-2]>1"
            elif objType == "line":
                dimsTestString = "{}.shape[{}.dims-1] == 1 or {}.shape[{}.dims-2] == 1"
            elif objType == "lineORplane":
                dimsTestString = "{}.ndim > 0 and ({}.ndim < 3 or (({}.ndim < 4) and ({}.shape[0] == 1)))"
            else:
                dimsTestString = "{}.dims > 0"

        if (acceptedObjectType & 2) == 2:
            if objType == "empty":
                dimsNPTestString = "{}.ndim == 0"
            elif objType == "plane":
                dimsNPTestString = (
                    "{}.ndim > 1 and {}.shape[{}.ndim-1]>1 and {}.shape[{}.ndim-2]>1"
                )
            elif objType == "line":
                dimsNPTestString = "({}.ndim == 1) or ({}.shape[{}.ndim-1] == 1) or ({}.shape[{}.ndim-2] == 1)"
            elif objType == "lineORplane":
                dimsNPTestString = "{}.ndim > 0 and ({}.ndim < 3 or (({}.ndim < 4) and ({}.shape[0] == 1)))"
            else:
                dimsNPTestString = "{}.ndim > 0"

        for key in workSpace.keys():
            try:
                if ((acceptedObjectType & 1) == 1) and (
                    type(workSpace[key]) == itom.dataObject
                ):
                    try:
                        if eval(dimsTestString.format(key, key, key, key), workSpace):
                            dObjects.append(key)
                    except:
                        temp = 0
                        del temp

                if ((acceptedObjectType & 2) == 2) and (
                    type(workSpace[key]) == numpy.ndarray
                    or type(workSpace[key]) == itom.npDataObject
                ):
                    try:
                        if eval(
                            dimsNPTestString.format(key, key, key, key, key), workSpace,
                        ):
                            npObjects.append(key)
                    except:
                        temp = 0
                        del temp
            except:
                temp = 0
                del temp

        if (len(npObjects) < 1) and (len(pcObjects) < 1) and (len(dObjects) < 1):
            ui.msgCritical("Plot", "No dataObjects found", ui.MsgBoxOk)
            varname = ""
            done = False
            objtype = 0

        elif (len(npObjects) < 1) and (len(pcObjects) < 1):
            objtype = 1
            if len(dObjects) > 1:
                objindex = [i for i, x in enumerate(dObjects) if x == VarName]

                if len(objindex) > 0:
                    objindex = objindex[0]
                else:
                    objindex = 0

                [varname, done] = itom.ui.getItem(
                    title, "DataObjects", dObjects, objindex, False
                )
            else:
                varname = dObjects[0]
                done = True

        elif (len(dObjects) < 1) and (len(pcObjects) < 1):
            objtype = 2
            if len(npObjects) > 1:
                objindex = [i for i, x in enumerate(npObjects) if x == VarName]

                if len(objindex) > 0:
                    objindex = objindex[0]
                else:
                    objindex = 0

                [varname, done] = itom.ui.getItem(
                    title, "numpyArrays", npObjects, objindex, False
                )
            else:
                varname = npObjects[0]
                done = True
        else:
            allDics = []
            allDics.extend(dObjects)
            allDics.extend(npObjects)
            allDics.extend(pcObjects)

            objindex = [i for i, x in enumerate(allDics) if x == VarName]

            if len(objindex) > 0:
                objindex = objindex[0]
            else:
                objindex = 0

            [varname, done] = itom.ui.getItem(
                title, "numpyArrays", allDics, objindex, False
            )

            if varname in pcObjects:
                objtype = 4
            elif varname in npObjects:
                objtype = 2
            else:
                objtype = 1

            del allDics

        del key
        del dObjects
        del npObjects
        del pcObjects
        del dimsTestString
        del dimsNPTestString

        return [varname, objtype, done]

    def checkIsLineOrPlane(self, varname, workSpace):
        """
        checkIsLineOrPlane(title, VarName, workSpace) -> check if the object is a 1D-line / 2D-plane
        Parameters:
            - title               Title of the dialog
            - VarName         default variable name
            - workSpace      KeyList with local or global variables
        Return parameters:
            - isLine:   True if obj is a 1D-Line
            - isPlane: True if obj is a 2D-plane (max 3D, z == 1)
            - dir:       If obj is a line, this will be 'x' or 'y'
            - done:    True if nothing failed
        """
        isLine = False
        isPlane = False
        dir = ""
        done = False

        try:
            dims = eval("{}.dims".format(varname), workSpace)
            if dims > 3:
                del dims
                return [False, False, "", True]

            elif dims == 3:
                if eval("{}.shape[0] > 1".format(varname, varname), workSpace):
                    del dims
                    return [False, False, 0, ""]

            if eval(
                "({}.shape[{}.dims-1] == 1) or ({}.shape[{}.dims-2] == 1)".format(
                    varname, varname, varname, varname
                ),
                workSpace,
                {"dimensions": dims},
            ):
                isLine = True
                if eval(
                    "({}.shape[{}.dims-1] == 1)".format(varname, varname), workSpace,
                ):
                    dir = "y"
                else:
                    dir = "x"
                done = True
            else:
                isPlane = True
                done = True

        except:
            ui.msgCritical("Script Error", "Checking object dims failed", ui.MsgBoxOk)
            done = False

        del dims
        return [isLine, isPlane, dir, done]
