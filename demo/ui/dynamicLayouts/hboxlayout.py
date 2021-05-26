from itom import ui, uiItem
import time

t0 = time.time()

gui: ui = ui("layoutExample.ui", type=ui.TYPEWINDOW)


t = time.time()
for i in range(0,100):
    hlayout: uiItem = gui.horLayout  # access the layout item
# print(time.time()-t)
t = time.time()
for i in range(0,100):
    hlayout = gui.getChild("horLayout")
# print(time.time()-t)

# remove the 2nd widget at index position 1
hlayout.call("removeItemAt", 1)

# add a new radio button at the end
className: str = "QRadioButton"
objName: str = "newRadioButton"
radioBtn: uiItem = hlayout.call("addItem", className, objName)
radioBtn["text"] = "new option"
radioBtn["checked"] = True

# insert a spin box at index position 1
idx: int = 1  # insert at this position
className: str = "QSpinBox"
objName: str = "mySpinBox"
spinBox: uiItem = hlayout.call("insertItem", idx, className, objName)
spinBox["value"] = 7

print(time.time() - t)
gui.show()  # show the user interface