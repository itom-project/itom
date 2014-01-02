def closeGui():
    gui.hide()

gui = ui("gui_icons_images.ui",type=ui.TYPEWINDOW)
gui.actionClose.connect("triggered()", closeGui)
gui.show()