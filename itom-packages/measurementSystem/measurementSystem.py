# coding=iso-8859-15
"""
This file contains a template class for measurement systems used under itom.
By using this template further packages, e.g. unified stitching shall be implemented.

This system has been developed by Institut fuer Technische Optik (ITO), Universitaet Stuttgart
"""

from itomEnum import ItomEnum
from datetime import datetime
import json
import itom


class InstrumentType:
    """
    Structure providing information about the instrument type.
    This consist of different strings, giving information about
    manufacturer, the model identification string, the serial number and
    an unique version string.
    The entire type is based on the definition of the openGPS-project.
    """

    def __init__(self, manufacturer, model, serial, version):
        self.manufacturer = manufacturer
        self.model = model
        self.serial = serial
        self.version = version


class ProbingSystemType:
    """
    Structure providing information about the system's probing type.
    This consist of a type-enumeration and an identification string.
    The entire type is based on the definition of the openGPS-project.
    """

    tType = ItomEnum(
        "ProbingSystem", ("unknown", "Contacting", "NonContacting", "Software")
    )

    def __init__(self, type, identification):
        self.type = type
        self.identification = identification


class MeasurementSystemBase:
    """
    This class is a base class for any 1d, 2d, 2.5d or 3d measurement system.

    Let your system class inherit from this class and implement the functions to let your
    system be compatible to other generic interfaces that can access such systems.
    """

    tFeatureType = ItomEnum("FeatureType", ("unknown", "SUR", "PRF", "PCL"))

    def __init__(self, name, probingSystem, instrument, username):
        """The basic constructor. In real implementations, it should verify parameters amd load configuration parameters."""

        # Parameter Validation
        if not (isinstance(probingSystem, ProbingSystemType)):
            raise RuntimeError(
                "probingSystem must be an instance of class 'ProbingSystemType'"
            )
        if not (isinstance(instrument, InstrumentType)):
            raise RuntimeError(
                "instrument must be an instance of class 'InstrumentType'"
            )

        self.name = name
        self.instrumentType = instrument
        self.probingSystemType = probingSystem

        self.username = username
        self.calibrationDate = "{unknown}"
        self.__calibrationDate = None

    def __repr__(self):
        """representation string for this instance of measurement system"""
        return self.instrumentType.model + ": " + self.name

    def setCalibrationDate(self, calibrationDate):
        """
        If you load a new calibration to your system, use this method of the base class to tell the system
        the timestamp of the calibration data.
        """
        if not (isinstance(calibrationDate, datetime)):
            raise RuntimeError("calibrationDate must be a datetime object")
        self.__calibrationDate = calibrationDate
        self.calibrationDate = calibrationDate.strftime("%Y-%m-%dT%H:%M:%S.%f")

    # ---------------------------------------------------------------------------------------------------------------------------------------

    def addSystemTags(self, dObj, title=None, configurationDict=None):
        """
        adds system related tags to the given data object 'dObj' (by reference). Additionally you can give
        a dictionary with further configuration parameters, which is serialized as a string using the module json
        and then stored in the tag 'configuration'.
        """

        if not (isinstance(dObj, itom.dataObject)):
            raise RuntimeError("dObj must a dataObject")

        dObj.setTag("calibrationDate", self.calibrationDate)
        dObj.setTag(
            "probingSystemType",
            self.probingSystemType.tType.whatis(self.probingSystemType.type),
        )
        # dObj.setTag("probingSystemType", self.probingSystemType.type.value)
        dObj.setTag("probingSystemID", self.probingSystemType.identification)
        dObj.setTag("manufacturer", self.instrumentType.manufacturer)
        dObj.setTag("model", self.instrumentType.model)
        dObj.setTag("serial", self.instrumentType.serial)
        dObj.setTag("version", self.instrumentType.version)

        if configurationDict is not None:
            dObj.setTag("configuration", json.dumps(configurationDict))

        if title is not None:
            dObj.setTag("title", title)

    # ---------------------------------------------------------------------------------------------------------------------------------------

    def add25DMeasurementTags(
        self,
        dObj,
        axisScales,
        axisUnits,
        axisOffsets,
        axisDescriptions,
        valueUnit,
        valueDescription,
        measurementDate=None,
    ):
        """
        adds tags to the given dataObject (by reference) which are related to a 2.5D measurement. The measurementDate must be
        given as datetime object. If it is set to None (default), the actual timestamp is assigned to the tag 'measurementDate'
        of the data object.
        """
        if not (isinstance(dObj, itom.dataObject)):
            raise RuntimeError("dObj must a dataObject")

        if measurementDate is None:
            measurementDate = datetime.today()

        if isinstance(measurementDate, datetime):
            dObj.setTag(
                "measurementDate", measurementDate.strftime("%Y-%m-%dT%H:%M:%S.%f")
            )
        elif isinstance(measurementDate, str):
            dObj.setTag("measurementDate", measurementDate)
        else:
            raise RuntimeError("measurementDate must be a datetime or str object")

        dObj.axisScales = axisScales
        dObj.axisOffsets = axisOffsets
        dObj.axisUnits = axisUnits
        dObj.axisDescriptions = axisDescriptions
        dObj.valueUnit = valueUnit
        dObj.valueDescription = valueDescription

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def setParam(self, paramDict):
        """
        overwrite this method to verify and set all the parameters contained in paramDict. In case of error you may throw an exception.
        """
        raise RuntimeError("setParam not implemented")

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def getSystemCapabilities(self):
        """
        overwrite this method to return your system capability as dictionary with the following entries:
        - 'field2D' : {'available':True or False, 'size':(sizex,sizey), 'scale':(scalex,scaley) or scale <in mm/px>, 'bitdepth':<8 or 16>)
        - 'field25D' : {'available':True or False, 'size':(sizex,sizey), 'scale':(scalex,scaley) or scale <in mm/px>, 'bitdepth':<32 or 64>)
        - 'field3D' : {'available':True or False}
        ...
        """
        raise RuntimeError(
            "getSystemCapabilites has to be implemented by your measurement system"
        )

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def measurementField2DSnap(self, *args, **kwds):
        """
        performs one single snap of a 2D-scene with the camera
        usually an intensity image of the surface ...
        @param args is an infinite number of further parameters, accessible as tuple.
        @return dictionary with the following mandatory and optional values:
            - 'intensityMap' : dataObject [mand] is the mandatory intensity image (or modulation image)
            - <user defined> : you can add further elements to the dictionary
        """
        raise RuntimeError("System has no implementation for a 2D scaled snapshot")

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def measurementField25D(self, *args, **kwds):
        """
        performs one single 2,5D-measurement
        @param args is an infinite number of further parameters, accessible as tuple.
        @return dictionary with the following mandatory and optional values:
            - 'topologyMap' : dataObject [mand] is the mandatory topology image
            - 'intensityMap' : dataObject [opt] is the optional intensity image (or modulation image)
            - <user defined> : you can add further elements to the dictionary
        """
        raise RuntimeError("System has no implementation for a 2.5D measurement")

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def measurementField3D(self, *args, **kwds):
        """
        performs one single 3D-measurement
        @param args is an infinite number of further parameters, accessible as tuple.
        @return dictionary with the following mandatory and optional values:
            - 'points3D': pointCloud [mand] is the (optionally organized) point cloud consisting of XYZ-tuples and optionally further information about intensity, color...
            - 'X': dataObject [optional] are the x-values of every point in the regular grid, only possible if point-cloud is organized, hence NaN values are still available.
            - 'Y': dataObject [optional] are the y-values of every point in the regular grid, only possible if point-cloud is organized, hence NaN values are still available.
            - 'Z': dataObject [optional] are the z-values of every point in the regular grid, only possible if point-cloud is organized, hence NaN values are still available.
            - 'intensity': dataObject [optional] are further intensity-values of every point.
            - <user defined> : you can add further elements to the dictionary
        """
        raise RuntimeError("System has no implementation for a 3D measurement")

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def getCameras(self):
        """
        overwrite this method and return list of cameras, used in this system. This might be useful to show a general live image of your main camera
        """
        return []

    # ---------------------------------------------------------------------------------------------------------------------------------------
    def getSinglePoint3DCoordsFrom25D(self, mPx, nPx, height):
        """
        this method returns the 3D-coordinates in the sensor's coordinate system for one single value in the topology map.
        Overwrite this method, if your height value together with its pixel position cannot be simply transformed into
        a real 3D coordinate (e.g. microscopic fringe projection)
        """
        cap = self.getSystemCapabilites()["field25D"]
        scales = cap["scale"]
        if type(scales) is list or type(scales) is tuple:
            [sx, sy] = scales
        else:
            [sx, sy] = [scales, scales]

        return [sx * nPx, sy * mPx, height]
