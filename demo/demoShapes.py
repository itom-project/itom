#unrotated shapes
point = shape(shape.Point, (0,0))
print(point)
line = shape(shape.Line, (0,0), (100,50))
print(line)
rect = shape(shape.Rectangle, (20,20), (70,100))
print(rect)
square = shape(shape.Square, (30,-50), 20)
print(square)
ellipse = shape(shape.Ellipse, (-50,-70), (-20, 0))
print(ellipse)
circle = shape(shape.Circle, (-30, 100), 40)
print(circle)

#create a pixel-based region from all shapes (union of all shapes)
area = region()
area += point.region()
area += line.region()
area += rect.region()
area += square.region()
area += ellipse.region()
area += circle.region()

mask = area.createMask(region(-100, -100, 300, 300, region.RECTANGLE))

#plot all contours inside of plot
area = region()
contour_point = point.contour()
contour_line = line.contour()
contour_rect = rect.contour()
contour_square = square.contour()
contour_ellipse = ellipse.contour(tol = 0.02)
contour_circle = circle.contour(tol = 0.01)

#plot mask and contour points
[i,h] = plot(mask, "2D", properties = {"colorMap":"falseColorIR"})
h.call("plotMarkers", contour_point, "b+")
h.call("plotMarkers", contour_line, "rs")
h.call("plotMarkers", contour_rect, "g.")
h.call("plotMarkers", contour_square, "cd")
h.call("plotMarkers", contour_ellipse, "y>")
h.call("plotMarkers", contour_circle, "kx")


