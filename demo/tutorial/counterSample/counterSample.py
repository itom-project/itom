def inc():
    val = int(gui.lblResult["text"])
    gui.lblResult["text"] = str(val+1)

def dec():
    val = int(gui.lblResult["text"])
    gui.lblResult["text"] = str(val-1)
    pass

gui = ui("counter.ui",ui.TYPEWINDOW)
gui.btnInc.connect("clicked()", inc)
gui.btnDec.connect("clicked()", dec)

#gui.show()

if "btnIndex" in globals():
    removeButton(btnIndex)

btnIndex = addButton("demo", "start", "gui.show()",":/dialogue/reset.png")
