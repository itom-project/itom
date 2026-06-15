from axiovert import *
import time

if 'Mikro' not in globals():
    Mikro =  dataIO("SerialIO",1,9600,"\r",8,1,0,4,0,4.000000)

if 'cam' not in globals():
    cam=dataIO('Vistek')

mainWin = ui("simpleGUI.ui", ui.TYPEWINDOW, childOfMainWindow = True)

mainWin.Live["camera"] = cam
S=dataObject()


# Initializing
def initializing():
    AxioInit(Mikro)
    StatusUpdate()

mainWin.btnInit.connect('clicked()', initializing)

# Update Status
def StatusUpdate():
    SynchronizeGain()
    SynchronizeExp()
    SynchronizeMO()
    SynchronizeFocus()
    SynchronizeInt()

mainWin.btnUpdate.connect('clicked()', StatusUpdate)


# Exposure Time
def changeExp():
    cam.setParam('exposure',mainWin.spinExp['value']/1000)
    SynchronizeExp()
def SynchronizeExp():
    exp=cam.getParam('exposure')
    mainWin.spinExp['value']=exp*1000
mainWin.spinExp.connect('editingFinished()', changeExp)

# Gain
def changeGain():
    cam.setParam('gain',mainWin.spinGain['value'])
    SynchronizeGain()
def SynchronizeGain():
    gain=cam.getParam('gain')
    mainWin.spinGain['value']=gain
mainWin.spinGain.connect('editingFinished()', changeGain)

# Extern Plot/Video
def externVid():
    liveImage(cam)
def externPlot():
    cam.startDevice()
    autoGrabbingStatus = cam.getAutoGrabbing()
    cam.disableAutoGrabbing()
    cam.acquire()
    cam.copyVal(S)
    if(autoGrabbingStatus):
        cam.enableAutoGrabbing()
    cam.stopDevice()
    plot(S)

mainWin.btnExSnap.connect('clicked()', externPlot)
mainWin.btnExLive.connect('clicked()', externVid)

# Change MO
for k in range(1,7):
    mainWin.comBox_MO.call('addItem',(str(k)+': '+AxioMO[k]['name'])) # index 0-5 = MO 1-6

def changeMO(ind):
    mo_setNum(Mikro,ind+1,1)
    SynchronizeMO()
    SynchronizeFocus()

def SynchronizeMO():
    currMO=mo_getNum(Mikro)
    mainWin.comBox_MO.call('setCurrentIndex',currMO-1)

mainWin.comBox_MO.connect('activated(int)',changeMO)

# Focus Settings
def changeFocus():
    focus_setPos(Mikro,mainWin.spinFocus['value'])
    SynchronizeFocus()
def SynchronizeFocus():
    mainWin.spinFocus['value']=focus_getPos(Mikro)

mainWin.spinFocus.connect('editingFinished()', changeFocus)

def focusPlus01():
    focus_setPos(Mikro,focus_getPos(Mikro)+0.1)
    SynchronizeFocus()
mainWin.btn01plus.connect('clicked()', focusPlus01)
def focusMinus01():
    focus_setPos(Mikro,focus_getPos(Mikro)-0.1)
    SynchronizeFocus()
mainWin.btn01minus.connect('clicked()', focusMinus01)

def focusPlus1():
    focus_setPos(Mikro,focus_getPos(Mikro)+1)
    SynchronizeFocus()
mainWin.btn1plus.connect('clicked()', focusPlus1)
def focusMinus1():
    focus_setPos(Mikro,focus_getPos(Mikro)-1)
    SynchronizeFocus()
mainWin.btn1minus.connect('clicked()', focusMinus1)

def focusPlus10():
    focus_setPos(Mikro,focus_getPos(Mikro)+10)
    SynchronizeFocus()
mainWin.btn10plus.connect('clicked()', focusPlus10)
def focusMinus10():
    focus_setPos(Mikro,focus_getPos(Mikro)-10)
    SynchronizeFocus()
mainWin.btn10minus.connect('clicked()', focusMinus10)

def focusPlus50():
    focus_setPos(Mikro,focus_getPos(Mikro)+50)
    SynchronizeFocus()
mainWin.btn50plus.connect('clicked()', focusPlus50)
def focusMinus50():
    focus_setPos(Mikro,focus_getPos(Mikro)-50)
    SynchronizeFocus()
mainWin.btn50minus.connect('clicked()', focusMinus50)

def focusPlus100():
    focus_setPos(Mikro,focus_getPos(Mikro)+100)
    SynchronizeFocus()
mainWin.btn100plus.connect('clicked()', focusPlus100)
def focusMinus100():
    focus_setPos(Mikro,focus_getPos(Mikro)-100)
    SynchronizeFocus()
mainWin.btn100minus.connect('clicked()', focusMinus100)

def focusPlus500():
    focus_setPos(Mikro,focus_getPos(Mikro)+500)
    SynchronizeFocus()
mainWin.btn500plus.connect('clicked()', focusPlus500)
def focusMinus500():
    focus_setPos(Mikro,focus_getPos(Mikro)-500)
    SynchronizeFocus()
mainWin.btn500minus.connect('clicked()', focusMinus500)

# Halogen Lamp 1-255
def changeInt():
    halogen_setInt(Mikro,mainWin.spinInt['value'])
    time.sleep(0.5)
    SynchronizeInt()
def SynchronizeInt():
    mainWin.spinInt['value']=halogen_getInt(Mikro)
    if halogen_getState(Mikro)==0:
        mainWin.btnHalogen['checked']=True
    if halogen_getState(Mikro)==1:
        mainWin.btnHalogen['checked']=False
mainWin.spinInt.connect('editingFinished()', changeInt)

def halogenOnOff():
    halogen_switch(Mikro)
    time.sleep(0.5)
    SynchronizeInt()
mainWin.btnHalogen.connect('clicked()', halogenOnOff)

# ###########################
# START

StatusUpdate()
mainWin.show(0)

do_init=ui.msgQuestion('Do you want to initialize the Mikroscope?', 'If you already have initialized the Microscope in the running session you can click NO. Otherwise do the initialization to prevent damage.', ui.MsgBoxYes | ui.MsgBoxNo, ui.MsgBoxYes)
if do_init[1]=='Yes':
    AxioInit(Mikro)

# Test Area...
