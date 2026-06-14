"""Window
=========

Window is a stand-alone window. The window is only hidden
if the user closes it. Call show again to re-show it. It is only
deleted, if the window variable is deleted."""
from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoWindow.png'


window = ui("mainWindow.ui", ui.TYPEWINDOW)
window.show()

###############################################################################
# Window_destroy is a stand-alone window,
# that is deleted if the user closes it, not only hidden"""

def window_destroyed():
    print("window_destroy destroyed")


window_destroy = ui("mainWindow.ui", ui.TYPEWINDOW, deleteOnClose=True)
# deleteOnClose can also be set or unset using setAttribute
print("state of deleteOnClose flag:", window_destroy.getAttribute(55))
window_destroy.setAttribute(55, True)

window_destroy[
    "windowTitle"
] = "Self-destroyable main window"  # change title of main window
window_destroy.connect("destroyed()", window_destroyed)
window_destroy.show()

###############################################################################
# Window_no_child is a stand-alone window that is no child of the main window with own symbol in the task bar
window_no_child = ui("mainWindow.ui", ui.TYPEWINDOW, childOfMainWindow=False)
window_no_child[
    "windowTitle"
] = "Stand-alone main window"  # change title of main window
window_no_child.show()

###############################################################################
# Widget_window is a stand-alone window obtained from a widget that was created in QtDesigner.
widget_window = ui("widget.ui", ui.TYPEWINDOW)
widget_window.show()

###############################################################################
# Configure a window to not have a maximize button and to always stay on top.
window_on_top = ui("mainWindow.ui", ui.TYPEWINDOW)
window_on_top["windowTitle"] = "On-top window"  # change title of main window
# 0x00040000 is the flag Qt::WindowStaysOnTopHint and needs to be added if not yet done (or)
# 0x00008000 is the flag Qt::WindowMaximizeButtonHint and needs to be removed if not yet done (xor)
window_on_top.setWindowFlags(
    (window_on_top.getWindowFlags() | 0x00040000) ^ 0x00008000
)

# it is also possible to disable the close button by xor-ing Qt::WindowCloseButtonHint (0x08000000)

window_on_top.show()
