# Initialisation of DummyMotor
stage = actuator("DummyMotor")

#show the toolbox
stage.showToolbox()

# Access the DummyStage with variable-name stage

# Set parameter e.g. speed of the stage to 1000 mm / s
stage.setParam("speed", 1000)

# Get the current speed, should be 1000 mm/s
speed = stage.getParam("speed")

# Set pos of 1. axis (index 0) to the absolute value 10.2 mm
stage.setPosAbs(0,10.2)

# Read the axis position of 1. axis, should be 10 (mm)
stage.getPos(0)

# Change the position of 1. axis by -6 mm relative to the current position
stage.setPosRel(0,-6)

# Read the axis position of 1. axis, should be 4 (mm)
stage.getPos(0)

# Address n-axis
# Set the position of 1. and 3. axis to 5mm
stage.setPosAbs(0,5.0, 2, 5.0)

# Read the axis position of 1./3. axis, should be 5 (mm) and 5 (mm)
[x, z] = stage.getPos(0, 2)
print('x = ' + str(x) + ' z = ' + str(z))

# Change the position of 1. axis and 2. by 2 mm relative to the current position
stage.setPosRel(0,2, 1, 2)

# Read the axis position of 1./2./3. axis, should be 7 (mm), 2 (mm), 5 (mm)
[x, y, z] =  stage.getPos(0, 1, 2)
print('x = ' + str(x) + ' y = ' + str(y)+ ' z = ' + str(z))

