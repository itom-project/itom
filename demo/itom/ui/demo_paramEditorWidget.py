"""Parameter editor widget
==========================

This example shows the usage of the generic ParamEditorWidget.

The widget can be used in various ways. Some of them are shwon in
this example.

In the first tab, we have a DummyGrabber live image with two
ParamEditorWidgets on the right side. The left one is a default one
with an information text field below. If a parameter is clicked, its
description is shown there. The right one is configured such that
changes are not directly applied to the camera. Instead the Apply button
must be clicked to apply all recent changes.

In the second tab, we have a DummyMotor with 3 axes. There is one
read-only ParamEditorWidget and one standard one, that only shows
a subset of parameters (depending on their category). The category
of a parameter is an optional meta information and is part of the 
plugin.
"""
from itom import dataIO
from itom import actuator
from itom import ui
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoParamEditorWidget.png'


cam = dataIO("DummyGrabber", 1280, 1024, 8)
cam.setParam("frame_time", 0.100)  # limit to 10 Hz
mot = actuator("DummyMotor", 3)

gui = ui("paramEditorWidget.ui", ui.TYPEWINDOW)


def onApply():
    """Call 'setParam' of the DummyGrabber device
    for all changed parameters in the right
    ParamEditorWidget of the camera tab."""
    gui.pewGrabber2.call("applyChangedParameters")


# configure the DummyGrabber camera tab
gui.plot["camera"] = cam  # assign the camera to the plot
gui.pewGrabber[
    "plugin"
] = cam  # assign the camera to the left ParamEditorWidget

# if a parameter is changed in this ParamEditorWidget, directly call
# setParam of the camera
gui.pewGrabber["immediatelyModifyPluginParamsAfterChange"] = True

gui.pewGrabber2[
    "plugin"
] = cam  # assign the camera to the right ParamEditorWidget

# do not directly change the parameters in the camera, instead click
# the Apply button...
gui.pewGrabber2["immediatelyModifyPluginParamsAfterChange"] = False

# connect the click signal of the Apply button to the method ``onApply``.
gui.btnApplyChangesGrabber.connect("clicked()", onApply)

# configure the DummyMotor tab
gui.motorController["actuator"] = mot
gui.pewMotor1["plugin"] = mot

# do only show parameters that belong to these two categories
gui.pewMotor1["filteredCategories"] = ("General", "Motion")

# make the left ParamEditorWidget readonly
gui.pewMotor1["readonly"] = True

gui.pewMotor2["plugin"] = mot

gui.show()
