from itom import *


def demo_shapesTrafo():
    d = dataObject.zeros([300,300])
    d.axisOffsets = (150,150)
    [i,h] = plot(d, properties = {"keepAspectRatio":True})

    rect = shape(shape.Rectangle, (-30,-20), (30,20))

    rect2 = rect.copy()
    rect2.translate([10,20])

    rect3 = rect.copy()
    rect3.rotateDeg(30)
    rect3.translate([10,20])

    rect4 = rect.copy()
    rect4.translate([10,20])
    rect4.rotateDeg(30)

    h["geometricShapes"] = (rect,rect2,rect3,rect4)

if __name__ == "__main__":
    demo_shapesTrafo()
