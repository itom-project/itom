#Demofile for DataObject functions
#create an empty DataObject (for ITOM-Filter or FileIO) with the name dObj
dObj = dataObject()

#Delete an anykind of dataObject
del dObj

#Methods to create a dataObject
yDim = 480
xDim = 640
zDim = 10
# dObj = dataObject.rand([yDim, xDim], "float32")       # 2D Random white noise, for integer-types between min and max, for floatingpoint between 0..1
# dObj = dataObject.randN([zDim, yDim, xDim], "int32")       # 3D Random white noise for integer-types 6*Sigma is between min and max, for floatingpoint 6Sigma is between 0..1
# dObj = dataObject.ones([yDim, xDim], "int8")       #2D Object filled with 1
# dObj = dataObject.zeros([zDim, yDim, xDim], "float64")       #3D Object filled with 0

dObj = dataObject.rand([yDim, xDim], "float64") 

# plot the dataObject
plot(dObj)

# Make a shallow-Copy of dObj (both objects share the same data)
dObjTemp = dObj

# Delete the shallowCopy, data will exist until dObj is deleted (Works also wise versa)
del dObjTemp

# Make a shallow-Copy of dObj with a ROI (both objects share the same data)
dObjTemp = dObj[round(yDim/2):yDim, round(xDim/2):xDim]

# set the values of the dataObject within the ROI to1
dObjTemp[:,:] = 1

# getValues (Point or Slice) from the dataObject
pythonTuple = dObj[0,0] #one element is directly returned as integer or float
pythonTuple = dObj[:,640-1].value #multiple elements are returned as dataObject. Therefore the value attribute is called to transform it into a tuple.
pythonTuple = dObj[round(yDim/2),:].value

# plot the ROI and the object
plot(dObj)
plot(dObjTemp)

# make a DeepCopy of the Object or the ROI
dObjCopy = dObj.copy()
del dObjCopy

# Set additional informations (meta data) as tags
# Use set tag "key", "value". Key / value are user defined.
dObj.setTag('Creator', 'Lyda')
dObj.setTag('Type', 'Measurement')
dObj.setTag('DummyTag', 'Delete Me')

# Retrieve tags by their key and print them
print(dObj.tags['Creator'])
print(dObj.tags['Type'])

# Get the complete tag space as a python dictionary
tagdic = dObj.tags

# Get the size of the tagspace
print(dObj.getTagListSize())
# or
print(len(tagdic))

# List up all tags in the tagspace
print('\nMy Taglist')
for key in tagdic.keys():
    print(key + ' -> ' + dObj.tags[key])

# Delete a tag
dObj.deleteTag('DummyTag')

# List up all tags in the tagspace
print('\nMy Taglist')
for key in dObj.tags.keys():
    print(key + ' -> ' + dObj.tags[key])

print('\nUnits / Offsets and Scales')
# Set the meta-data for units and lateral offset / scale for plotting and storing informations
# Set the axis Units to mm
dObj.axisUnits = ('mm', 'mm')
# Set the axisScale as 0.1 units / px
dObj.axisScales = (0.1, 0.1)
# Set offset so coordinate offset is in the center of the image
dObj.axisOffsets= (yDim/2, xDim/2)
# Set the axis descriptions to axis names
dObj.axisDescriptions = ('y-Axis', 'x-Axis')
# Set the value units to a.u.
dObj.valueUnit = 'a.u.'
# Set the value description to 'Intensity'
dObj.valueDescription = 'Intensity'

plot(dObj)

# Get Axis Units and print them
print('The axis descriptions are '+ dObj.axisDescriptions[1] +' and '+ dObj.axisDescriptions[0])
print('The axis scales are '+ str(dObj.axisScales[1]) + dObj.axisUnits[1]+'/px and '+ str(dObj.axisScales[0])+ dObj.axisUnits[1]+'/px')

# The iTOM-dataObject has a protocol function. It is a tag and every filter in c++ should add its properties after computation to this string by using addToProtocol
# Add a protocol to the Object, if object is a ROI-ShallowCopy the ROI is automatically added to the protocol-String
print("\nProtocol function:")
dObj.addToProtocol('Created today for test reasons')

dObjTemp.addToProtocol('Values set to 1 (Show the effect of ROI in case of the protocol)')
# Read protocol string
print(dObj.tags["protocol"])

