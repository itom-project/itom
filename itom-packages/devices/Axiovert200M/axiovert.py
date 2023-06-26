# Main functions for Using the Axiovert 200M
# For using these function the Microscope should be connected to the PC via the COM port.
# For communication with the Axiovert the SerialIO Plugin is used:
#
# serial =  dataIO("SerialIO",1,9600,"\r",8,1,0,4,0,4.000000)
#
# Overview:
# AxioInit(serial)
# halogen_switch(serial)
# halogen_getState(serial)
# halogen_on(serial)
# halogen_off(serial)
# halogen_getInt(serial)
# halogen_setInt(serial,Int) # 1-255
# focus_busy(serial)
# focus_getPos(serial)
# focus_setPos(serial,Pos) # in um (Stepsize 0.025 um)
# focus_origin(serial)
# focus_down(serial)
# focus_up(serial)
# focus_checkLoad(serial)
# mo_busy(serial)
# mo_getNum(serial)
# mo_setNum(serial,Num,Pos) # 1-6  # 0 OR 1 = same position OR 'best' position
# tube_getNum(serial)
# tube_setNum(serial,Num) # 1-3
# reflector_getNum(serial)
# reflector_setNum(serial,Num) # 1-5
# shutter_getState(serial)
# shutter_switch(serial)
# shutter_open(serial)
# shutter_close(serial)
# ############################################################
import time # for sleep
import numpy as np


# IMPORTANT Initialization of the Microscope
# Run the Initialization before you use the Microscope.
def AxioInit(serial):
    print('Searching for a minimum for the stage and set as origin.')
    halogen_on(serial)
    halogen_setInt(serial,50)
    actMO=mo_getNum(serial)
    mo_setNum(serial,1,0)
    print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
    focus_setPos(serial,-10000)
    print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
    focus_origin(serial)
    print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
    for k in range(2,7):
        mo_setNum(serial,k,0)
        print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
        focus_setPos(serial,int(focus_getPos(serial))+500)
        print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
        focus_down(serial)
        print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
        if focus_getPos(serial)<0:
            focus_origin(serial)
            print('Focus ' + str(mo_getNum(serial)) + ' = '+ str(focus_getPos(serial)))
    mo_setNum(serial,actMO,1)
    halogen_setInt(serial,50)
    initializing=0


# IMPORTANT Calibration of the Microscope (depend on the MOs in the turret and the stageadapter)
AxioMO={1:{'min':-10000, 'max':9450, 'best':5290, 'name':'Name1'}}
AxioMO[2]={'min':0, 'max':9200, 'best':6745, 'name':'Name2'}
AxioMO[3]={'min':0, 'max':5500, 'best':5415, 'name':'Name3'}
AxioMO[4]={'min':0, 'max':6100, 'best':5275, 'name':'Name4'}
AxioMO[5]={'min':0, 'max':5900, 'best':5766, 'name':'Name5'}
AxioMO[6]={'min':0, 'max':6100, 'best':5684, 'name':'Name6'}



# Halogen Settings
# ############

# Halogen switsch ON / OFF
def halogen_switch(serial):
    state=halogen_getState(serial)
    if state==0:
        serial.setVal('HPCT8,1')
    if state==1:
        serial.setVal('HPCT8,0')

# Get Halogen State 0 = OFF, 1 = ON
def halogen_getState(serial):
    serial.setVal('HPCt8')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    state=str(buf).split('PH')[1][0]
    return int(state)

# Halogen ON
def halogen_on(serial):
    serial.setVal('HPCT8,0')

# Halogen OFF
def halogen_off(serial):
    serial.setVal('HPCT8,1')

# Get Intensity
def halogen_getInt(serial):
    serial.setVal('HPCv1')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    Int = str(buf).split('PH')[1][0:3].split('\\')[0]
    return int(Int)         # Vorsicht, unterschied 1,2,3 Stellen???

# Set Intensity, Int = 1-255
def halogen_setInt(serial,Int):
    serial.setVal('HPCV1,'+str(Int))

# Focus Settings
# ##########

# Focus busy function
def focus_busy(serial):
    busy='2'
    a=0
    while busy!='0' and busy!='1':
        serial.setVal('FPZFs')
        c=0
        buf=bytearray(10)
        while str(buf).find('\\r')==-1:
            serial.getVal(buf)
            c=c+1
            if c>1000:
                buf='\\r'
        busy=str(buf).split('PF')[1][3]
        a=a+1
        if a>200:
            print('a ' +str(a))
            busy='0'

# Get Focus Position, Pos in um (Stepsize 0.025 um)
def focus_getPos(serial):
    serial.setVal("FPZp")
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    if str(buf).split('PF')[1][0] == 'F':
        Pos = (int('FFFFFF',16)-int(str(buf).split('PF')[1][0:6],16)+1)*-0.025
    else:
        Pos = int(str(buf).split('PF')[1][0:6],16)*0.025
    return Pos

# Set Focus Position, Pos in um (Stepsize 0.025 um)
def focus_setPos(serial,Pos):
    if AxioMO[mo_getNum(serial)]['min'] <= Pos <= AxioMO[mo_getNum(serial)]['max']:
        step=0.025
        if np.floor(round(Pos/step*1000)/1000)==np.ceil(round(Pos/step*1000)/1000):
            if Pos<0:
                steps=int(abs(Pos)/step)-1
                Max=0xFFFFFF
                steps_hex=hex(Max-steps).split('x')[1].zfill(6)
            else:
                steps=int(Pos/step)
                steps_hex=hex(steps).split('x')[1].zfill(6)
            serial.setVal('FPZT'+steps_hex.upper())
            focus_busy(serial)
        else:
            print('Wrong step size')
    else:
        print('Focus must be between min and max! See AxioMO.')

# Set / Define Origin, current position -> z=0
def focus_origin(serial):
    serial.setVal('FPZP0')

# LoadPosition Down
def focus_down(serial):
    serial.setVal('FPZW0')
    focus_busy(serial)

# LoadPosition Up
def focus_up(serial):
    serial.setVal('FPZW1')
    focus_busy(serial)

# Check Load Position
def focus_checkLoad(serial):
    serial.setVal('FPZw')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    updown=str(buf).split('PF')[1][0:2]
    if updown=='01':
        return 'Up'
    if updown=='04':
        return 'Down'

# Objective Turret
# ##########
def mo_busy(serial):
    busy='123'
    a=0
    while busy!='0':
        serial.setVal('HPSb1')
        c=0
        buf=bytearray(10)
        while str(buf).find('\\r')==-1:
            serial.getVal(buf)
            c=c+1
            if c>1000:
                buf='\\r'
        busy=str(buf).split('PH')[1][0]
        a=a+1
        if a>200:
            print('a ' +str(a))
            busy='0'

# Check Objective Nummer
def mo_getNum(serial):
    serial.setVal('HPCr2,1')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    Num=str(buf).split('PH')[1][0:1]
    return int(Num)

# Change Objective
def mo_setNum(serial,Num,Pos):
    if 1 <= Num <= 6:
        actInt=halogen_getInt(serial)
        serial.setVal('HPCR2,'+str(Num))
        mo_busy(serial)
        if Pos==1:
            focus_setPos(serial,AxioMO[Num]['best'])
        halogen_setInt(serial,actInt)
    else:
        print('Wrong Number!')


# Tube Lens Turret
# ############

# Check Tube Lens Nummer
def tube_getNum(serial):
    serial.setVal('HPCr36,1')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    Num=str(buf).split('PH')[1][0:1]
    return int(Num)

# Change Tube Lens
def tube_setNum(serial,Num):
    if 1 <= Num <= 3:
        serial.setVal('HPCR36,'+str(Num))
    else:
        print('Wrong Number!')

# Reflector Turret
# ##########

# Check Reflector Nummer
def reflector_getNum(serial):
    serial.setVal('HPCr1,1')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    Num=str(buf).split('PH')[1][0:1]
    return int(Num)

# Change Reflector
def reflector_setNum(serial,Num):
    if 1 <= Num <= 5:
        currShut=shutter_getState(serial)
        currInt=halogen_getInt(serial)
        serial.setVal('HPCR1,'+str(Num))
        time.sleep(1.5)
        serial.setVal('HPCK1,'+currShut)
        halogen_on(serial)
        halogen_setInt(serial,currInt)
    else:
        print('Wrong Number!')

# Shutter (FL)
# #########

# Get Shutter State, '1' = close; '2'=open
def shutter_getState(serial):
    serial.setVal('HPCk1,1')
    buf = bytearray(20)
    time.sleep(0.1)
    serial.getVal(buf)
    state=str(buf).split('PH')[1][0:1]
    if state=='1':
        print('close')
    if state=='2':
        print('open')
    return state

# Shutter switch
def shutter_switch(serial):
    state=shutter_getState(serial)
    if state=='open':
        shutter_close(serial)
    if state=='close':
        shutter_open(serial)

# Shutter Open
def shutter_open(serial):
    serial.setVal('HPCK1,2')

# Shutter Close
def shutter_close(serial):
    serial.setVal('HPCK1,1')
