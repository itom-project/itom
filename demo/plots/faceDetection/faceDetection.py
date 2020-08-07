# coding=iso-8859-15
from itom import *

import cv2
import numpy as np
import time

import gc

def detectFace(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def detectEyes(img, cascade):
    eyes = cascade.detectMultiScale(img)
    if len(eyes) == 0:
        return []
    
    if eyes.shape[0] > 2:
        eyes = eyes[0:1,:]
        
    return eyes

def drawRects(img, faces, color):
    for x1, y1, x2, y2 in faces:
        rect = shape.createRectangle((x1, y1),(x2, y2), index = 11)
        rect.color = color
        win.plot.call('updateGeometricShape', rect)

def drawEyes(img, eyes, color):
    cnt = 21
    for x, y, w, h in eyes:
        eye = shape.createEllipse((x, y + h),(x + w, y), index = cnt)
        eye.color = color
        try:
            win.plot.call('updateGeometricShape', eye)
        except AttributeError:
            break
        cnt = cnt + 1

def snap():
    #image acquisition
    if "cam" in globals():
        d = dataObject()
        cam.disableAutoGrabbing()
        cam.acquire()
        cam.getVal(d)
    else:
        return
    
    img=np.array(d)
    
    win.plot["source"] = img
    
    # detect face and eyes
    faces = detectFace(img, faceCascade)
    eyes = detectEyes(img, eyeCascade)
    
    # overlay rect and eyes
    drawRects(img, faces, rgba(255,0,0,255)) # in color red
    drawEyes(img, eyes, rgba(0,255,0,255)) # in color green
    
def guiClosed():
    tDetect.stop()
    global cam, win
    del win
    del cam
    gc.collect()

if __name__ == '__main__':
    win = ui("FaceDetect.ui", ui.TYPEWINDOW, childOfMainWindow = True, deleteOnClose = True)
    
    faceCascade = cv2.CascadeClassifier()
    eyeCascade = cv2.CascadeClassifier()
    
    faceCascade.load('haarcascades/haarcascade_frontalface_alt.xml')
    eyeCascade.load('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
        
    cam= dataIO("OpenCVGrabber",0, 'gray')
    cam.startDevice()
    cam.disableAutoGrabbing()

    tDetect = timer(100, snap)

    win.connect("destroyed()", guiClosed)
    
    #start GUI
    win.show(0)
