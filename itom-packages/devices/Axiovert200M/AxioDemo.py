from axiovert import *
import time

Mikro =  dataIO("SerialIO",1,9600,"\r",8,1,0,4,0,4.000000)

# Run the AxioInit before you use the microscope
AxioInit(Mikro)

halogen_on(Mikro)
halogen_setInt(Mikro,60)
time.sleep(2)
halogen_off(Mikro)
time.sleep(3)
halogen_on(Mikro)
time.sleep(0.5)
print('Halogen intensity is set to '+str(halogen_getInt(Mikro)))

del Mikro
