# coding=iso-8859-15
import numpy as np

xVec = np.arange(0, 6*np.pi, np.pi/100)
yVec = np.sin(xVec)

xVec = dataObject(xVec)
yVec = dataObject(yVec)

yVec.valueDescription = "amplitude"
yVec.valueUnit = "a. u."
yVec.setAxisDescription(1, "time")
yVec.setAxisUnit(1, "s")
yVec.setTag("title", "sin function")

plot1(yVec, xVec)