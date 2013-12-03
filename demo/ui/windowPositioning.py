'''
Demo for getting/setting the size and position of any ui-window
'''
mainWin = ui("windowPositioning.ui", type = ui.TYPEWINDOW)

#show window
mainWin.show()

#frame of window is the entire window including any title bar and window frame
# properties: frameGeometry, x, y
print("window frame geometry (x,y,w,h):", mainWin["frameGeometry"])
print("window position (x,y):", mainWin["pos"])
print("x, y:", mainWin["x"], mainWin["y"])

#the real area of the window is accessible by geometry, size, width, height
print("window geometry (x,y,w,h):", mainWin["geometry"])
print("window size (w,h):", mainWin["size"])
print("window width:", mainWin["width"])
print("window height:", mainWin["height"])

#in order to change the outer position use the property 'pos'
mainWin["pos"] = (0,0)

#size change: property 'size'
mainWin["size"] = (500,400)

#in order to change the inner position and size use the property 'geometry'
mainWin["geometry"] = (100,200,300,200)


