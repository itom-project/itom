cam = dataIO("DummyGrabber", 1280, 1024, 8)
mot = actuator("DummyMotor", 3)

gui = ui("paramEditorWidget.ui", ui.TYPEWINDOW)

gui.plot["camera"] = cam
gui.pewGrabber["plugin"] = cam
gui.pewGrabber["immediatelyModifyPluginParamsAfterChange"] = False

gui.motorController["actuator"] = mot
gui.pewMotor1["plugin"] = mot
gui.pewMotor1["filteredCategories"] = ("General","Motion")
gui.pewMotor1["readonly"] = True
gui.pewMotor2["plugin"] = mot

gui.show()