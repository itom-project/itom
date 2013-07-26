
obj = dataObject.randN([1024,1024],'int16')

[nr,h] = plot(obj, "itom2dqwtplot")

#pick point demo
#------------------

pickedPoints = dataObject()
#this command let the user pick maximum 4 points (earlier break with space, esc aborts the selection)
h.pickPoints(pickedPoints, 4)

print("selected points")
pickedPoints.data()

#marker demo
#--------------

'''the marker syntax may change in some future releases'''
markers = dataObject([2,3],'float32', data = [10.1, 20.2, 30.3, 7, 100, 500])
[nr,h] = plot(obj, "itom2dqwtplot")
h.call("plotMarkers", markers, "b+10", "setName") #'setName' is the name for this set of markers (optional)

#the second argument of plotMarkers is a style-string (this may change)
#[color,symbol,size]
# color = {b,g,r,c,m,y,k,w}
# symbol = {.,o,s,d,>,v,^,<,x,*,+,h}
# size = any integer number

#delete marker set
#h.call("deleteMarkers","setName") #deletes given set
#h.call("deleteMarkers","") #deletes all sets

