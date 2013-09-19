gui = ui("counter.ui",ui.TYPEWINDOW)

def inc():
    val = int(gui.lblResult["text"])
    gui.lblResult["text"] = str(val+1)

def dec():
    val = int(gui.lblResult["text"])
    gui.lblResult["text"] = str(val-1)
    pass

gui.btnInc.connect("clicked()", inc)
gui.btnDec.connect("clicked()", dec)


gui.show()