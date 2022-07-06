"""Interrupt
============

"""
from itom import ui
import time
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoInterrupt.png'

gui = ui("interruptDemo.ui", ui.TYPEWINDOW)


def do():
    gui.btnDo["enabled"] = False
    try:
        # it is not possible to interrupt a simple sleep command
        # therefore, it is put into a loop over shorter sleeps...
        for i in range(0, 200):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("the sleep loop has been interrupted.")
        gui.btnDo["enabled"] = True
        raise
    gui.btnDo["enabled"] = True


gui.btnDo.connect("clicked()", do)
gui.btnInterrupt.invokeKeyboardInterrupt("clicked()")
gui.show()
