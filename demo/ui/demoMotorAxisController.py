gui = ui("demoMotorAxisController.ui", ui.TYPEWINDOW)
#please notice, that some signals/slots are already connected via Designer!

c = gui.controller
motor = actuator("DummyMotor", 4)
c["actuator"] = motor
c["numAxis"] = 3 #the first 4 axes from the motor are considered in this case.
c["defaultAxisUnit"] = "UnitMum" #available: UnitNm (0), UnitMum (1), UnitMm (2), UnitCm (3), UnitM (4) or UnitDeg (5)
c["defaultAxisType"] = "TypeLinear" #available: TypeRotational (0), TypeLinear (1)
c["defaultRelativeStepSize"] = 0.010 #always in mm or deg
c["axisNames"] = ("x","y","z","alpha")
c["defaultDecimals"] = 2

if version(1)["itom"]["QT_Version"] < "5.0.0":
    raise Warning("It is not possible to call slots with enumeration data types as arguments with Qt < 5.0.0")
#the fourth axis is a rotational axis:
if version(1)["itom"]["QT_Version"]>="5.5.0":
    #c.call("setAxisType", 3, "TypeRotational")
    #c.call("setAxisUnit", 3, "UnitDeg")
    pass
else:
    #before Qt 5.5.0, slots with enumeration data types can only be called with their integer value
    c.call("setAxisType", 3, 0) #TypeRotational
    c.call("setAxisUnit", 3, 5) #UnitDeg

gui.show()