# coding=iso-8859-15
''' This demo shows how to set an optional x-vector to an 1D-plot.

The optional x-vector can be set by passing the optional x-vector as second argument to the plot1 function
or by setting the property xVector of an existing plot.
If you want to add an x-vector to a plot of an nxm dataObject your x-vector has to be an dataObject of shape 1xm.
Once you add an x-vector with a last dimension greater m the last points will be ignored.'''
import numpy as np
angle=np.linspace(0,10*np.pi)
x = dataObject([1,angle.shape[0]],dtype='float32')
y = dataObject([1,angle.shape[0]],dtype='float32')
cnt=0
for theta in angle:
    r = ((theta)**2)
    x[0,cnt]=(r*np.cos(theta))
    y[0,cnt]=(r*np.sin(theta))
    cnt+=1

plot1(y,x)

fig=plot(y)
fig[-1]['xObject']=x