import itom
fig = itom.figure(1)

a = itom.dataObject.randN([1,100],'float32')
a.axisScales= (0.2, 0.3)
a.axisOffsets = (0.0, 50)
a.setTag("title", "nice plot")

fig.plot(a)