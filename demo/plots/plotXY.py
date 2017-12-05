# coding=iso-8859-15
import numpy as np

xVec = np.arange(0, 6*np.pi, np.pi/100)
yVec = np.sin(xVec)

xVec = dataObject(xVec)
yVec = dataObject(yVec)

yVec.valueDescription = "yVec value description"
yVec.valueUnit = "yVec value unit"
yVec.setAxisDescription(1, "yVec x axis description")
yVec.setAxisUnit(1, "yVec x unit")

plot1(yVec, xVec)