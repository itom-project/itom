import os
import sys
import weakref
import itom

import matplotlib
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase, \
     NavigationToolbar2, cursors
from matplotlib.backend_bases import ShowBase

from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool

from itom import uiItem, ui, timer

figureoptions = None

def fn_name(): return sys._getframe(1).f_code.co_name

DEBUG = False

cursord = {
    -1 : -1,
    cursors.MOVE          : 9,
    cursors.HAND          : 13,
    cursors.POINTER       : 0,
    cursors.SELECT_REGION : 2,
    }

def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager != None:
            figManager.canvas.draw_idle()

class Show(ShowBase):
    def mainloop(self):
        pass
        #print("mainloop")
        #QtGui.qApp.exec_()

show = Show()

def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    #thisFig = Figure( *args, **kwargs )
    FigureClass = kwargs.pop('FigureClass', Figure)
    existingCanvas = kwargs.pop('canvas', None)
    if(existingCanvas is None):
        embeddedCanvas = False
        itomUI = ui("itom://matplotlib")
        #itomUI.show() #in order to get the right size
    else:
        embeddedCanvas = True
        if(isinstance(existingCanvas,uiItem)):
            itomUI = existingCanvas
        else:
            raise("keyword 'canvas' must contain an instance of uiItem")
    thisFig = FigureClass( *args, **kwargs )
    canvas = FigureCanvasItom( thisFig, num, itomUI, embeddedCanvas )
    manager = FigureManagerItom( canvas, num, itomUI, embeddedCanvas )
    return manager

class FigureCanvasItom( FigureCanvasBase ):
    keyvald = { 0x01000021 : 'control',
                0x01000020 : 'shift',
                0x01000023 : 'alt',
                0x01000004 : 'enter',
                0x01000012 : 'left',
                0x01000013 : 'up',
                0x01000014 : 'right',
                0x01000015 : 'down',
                0x01000000 : 'escape',
                0x01000030 : 'f1',
                0x01000031 : 'f2',
                0x01000032 : 'f3',
                0x01000033 : 'f4',
                0x01000034 : 'f5',
                0x01000035 : 'f6',
                0x01000036 : 'f7',
                0x01000037 : 'f8',
                0x01000038 : 'f9',
                0x01000039 : 'f10',
                0x0100003a : 'f11',
                0x0100003b : 'f12',
                0x01000010 : 'home',
                0x01000011 : 'end',
                0x01000016 : 'pageup',
                0x01000017 : 'pagedown',
               }
    # left 1, middle 2, right 3
    buttond = {1:1, 2:3, 4:2}
    def __init__(self, figure, num, itomUI, embeddedCanvas):
        FigureCanvasBase.__init__(self, figure)
        self._idle = True
        self._idle_callback = None
        self.num = num
        t1,t2,w,h = self.figure.bbox.bounds
        w, h = int(w), int(h)
        
        self.initialized = False
        self.embeddedCanvas = embeddedCanvas
        self._destroying = False
        self._timer = None
        
        if(embeddedCanvas == False):
            self.canvas = itomUI.canvasWidget  #this object is deleted in the destroy-method of manager, due to cyclic garbage collection
            win = self.canvas
            #win["width"]=w
            #win["height"]=h
            win["mouseTracking"] = False #by default, the itom-widget only sends mouse-move events if at least one button is pressed or the tracker-button is is checked-state
        else:
            self.canvas = itomUI.canvasWidget
            itomUI["mouseTracking"] = False #by default, the itom-widget only sends mouse-move events if at least one button is pressed or the tracker-button is is checked-state
            
        #dpival = self.figure.dpi
        #winch = self.canvas["width"]/dpival
        #hinch = self.canvas["height"]/dpival
        #self.figure.set_size_inches( winch, hinch )
        
        self.canvas.connect("eventLeaveEnter(bool)", self.leaveEnterEvent)
        self.canvas.connect("eventMouse(int,int,int,int)", self.mouseEvent)
        self.canvas.connect("eventWheel(int,int,int,int)", self.wheelEvent)
        self.canvas.connect("eventKey(int,int,QString,bool)", self.keyEvent)
        self.canvas.connect("eventResize(int,int)", self.resizeEvent)
        self.canvas.connect("eventIdle()", self.idle_event)
        
        self.initialized = True
        
    def destroy(self):
        if(self.initialized == True):
            del self.canvas
            self.canvas = None
            del self.figure #from base class
            self.figure = None 
            #del self.itomUIDialog
            #self.itomUIDialog = None
            
        self.initialized = False
    
    def __timerEvent(self, event):
        # hide until we can test and fix
        self.mpl_idle_event(event)

    def leaveEnterEvent(self, enter):
        if(enter):
            FigureCanvasBase.enter_notify_event(self, None)
        else:
            FigureCanvasBase.leave_notify_event(self, None)
    
    def mouseEvent(self, type, x, y, button):
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - y
        
        try:
            #button: left 1, middle 2, right 3
            if(type == 0): #mousePressEvent
                FigureCanvasBase.button_press_event( self, x, y, button )
                if DEBUG: print('button pressed:', button)
            elif(type == 1): #mouseDoubleClickEvent
                FigureCanvasBase.button_press_event( self, x, y, button, dblclick=True )
                if DEBUG: print ('button doubleclicked:', button)
            elif(type == 2): #mouseMoveEvent
                if(button == 0): #if move without button press, reset timer since no other visualization is given to Qt, which could then reset the timer
                    self.canvas.call("stopTimer")
                FigureCanvasBase.motion_notify_event( self, x, y )
                if DEBUG: print('mouse move, (x,y):', x, y)
            elif(type == 3): #mouseReleaseEvent
                FigureCanvasBase.button_release_event( self, x, y, button )
                if DEBUG: print('button released, (x,y):', x, y)
        except RuntimeError:
            self.signalDestroyedWidget()

    def wheelEvent( self, x, y, delta, orientation ):
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - y
        # from QWheelEvent::delta doc
        steps = delta/120
        if (orientation == 1): #vertical
            FigureCanvasBase.scroll_event( self, x, y, steps)
            if DEBUG: print('scroll event : delta = %i, steps = %i ' % (delta,steps))
    
    def keyEvent(self, type, keyId, keyString, autoRepeat):
        key = self._get_key(keyId, keyString, autoRepeat)
        if(key is None):
            return
        
        if(type == 0): #keyPressEvent
            FigureCanvasBase.key_press_event( self, key )
            if DEBUG: print('key press', key)
        elif(type == 1): #keyReleaseEvent
            FigureCanvasBase.key_release_event( self, key )
            if DEBUG: print('key release', key)

    def resizeEvent( self, w, h ):
        if DEBUG: print("FigureCanvasQt.resizeEvent(", w, ",", h, ")")
        dpival = self.figure.dpi
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_size_inches( winch, hinch )
        self.draw()
        #self.update()
        #QtGui.QWidget.resizeEvent(self, event)

    def sizeHint( self ):
        w, h = self.get_width_height()
        return w, h

    def minumumSizeHint( self ):
        return 10, 10

    def _get_key( self, keyId, keyString, autoRepeat ):
        if autoRepeat:
            return None
        if keyId < 256:
            key = str(keyString)
        elif keyId in self.keyvald:
            key = self.keyvald[ keyId ]
        else:
            key = None

        return key

    def flush_events(self):
        #QtGui.qApp.processEvents()
        pass

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__
    
    def idle_draw(self):
        if self._timer:
            self._timer.stop()
        self.draw()
        self._idle = True

    def draw_idle(self):
        'update drawing area only if idle'
        d = self._idle
        self._idle = False
        
        if d: 
            if not self._timer:
                self._timer = timer(0, self.idle_draw) #auto-start, continuous mode
            else:
                self._timer.start()
            #print("singleShot draw_idle timer")
    
    def signalDestroyedWidget(self):
        '''
        if the figure has been closed (e.g. by the user - clicking the close button),
        this might either be registered by the destroyed-event, catched by FigureManagerItom,
        or by any method of this class which tries to access the figure (since the destroyed
        signal is delivered with a time gap). This function should be called whenever the widget
        is not accessible any more, then the manager is closed as quick as possible, such that
        a new figure can be opened, if desired.
        '''
        
        if(self._destroying == False):
            self._destroying = True
            try:
                Gcf.destroy(self.num)
            except AttributeError:
                pass
                # It seems that when the python session is killed,
                # Gcf can get destroyed before the Gcf.destroy
                # line is run, leading to a useless AttributeError.
            
        

class FigureManagerItom( FigureManagerBase ):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The qt.QToolBar
    window      : The qt.QMainWindow
    """

    def __init__( self, canvas, num, itomUI, embeddedCanvas ):
        if DEBUG: 
            print('FigureManagerItom.%s' % fn_name())
        FigureManagerBase.__init__( self, canvas, num )
        self.embeddedCanvas = embeddedCanvas
        
        if(embeddedCanvas == False):
            self.itomUI = itomUI
            itomUI["windowTitle"] = ("Figure %d" % num)
            itomUI["focusPolicy"] = 0x2 #QtCore.Qt.ClickFocus
            itomUI.connect("destroyed()", self._widgetclosed)
        else:
            self.itomUI = itomUI
            #winWidget["windowTitle"] = ("Figure %d" % num)
            itomUI["focusPolicy"] = 0x2 #QtCore.Qt.ClickFocus
            itomUI.connect("destroyed()", self._widgetclosed)
        
        #image = os.path.join( matplotlib.rcParams['datapath'],'images','matplotlib.png' )
        #self.window.setWindowIcon(QtGui.QIcon( image ))

        self.canvas._destroying = False
        
        #the size of the toolbar is not handled by matplotlib, therefore ignore it.
        #any resize command only addresses the size of the canvas. not more.
        self.toolbar = self._get_toolbar(self.canvas)
        # if self.toolbar is not None:
            # [tbs_width, tbs_height] = [0,0] #itomUI.toolbar["sizeHint"]
            # pass
        # else:
            # tbs_width = 0
            # tbs_height = 0

        # resize the main window so it will display the canvas with the
        # requested size:
        cs_width, cs_height = self.canvas.get_width_height() #canvas.sizeHint()
        #sbs_width, sbs_height = 0,0 #self.window.statusBar().sizeHint()
        self.resize(cs_width, cs_height) #+tbs_height+sbs_height)

        if matplotlib.is_interactive():
            self.show()
            #self.window.show()

        # attach a show method to the figure for pylab ease of use
        #self.canvas.figure.show = lambda *args: self.window.show()
        self.canvas.figure.show = lambda *args: self.show()

        def notify_axes_change( fig ):
           # This will be called whenever the current axes is changed
           if self.toolbar is not None:
               self.toolbar.update()
        self.canvas.figure.add_axobserver( notify_axes_change )


    def _widgetclosed( self ):
        if DEBUG: print("_widgetclosed called")
        if self.canvas._destroying: return
        self.canvas._destroying = True
        try:
            Gcf.destroy(self.num)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.


    def _get_toolbar(self, figureCanvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar'] == 'classic':
            print("Classic toolbar is not supported")
        elif matplotlib.rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2Itom(figureCanvas, self.itomUI, self.embeddedCanvas, True)
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        'set the canvas size in pixels'
        #print("resize: ", width, height)
        #win = self.window.win()
        
        self.canvas.canvas.call("externalResize",width,height)
        self.canvas.resizeEvent(width,height)

    def show(self):
        
        #experimental:
        itom.processEvents()
        
        self.canvas.draw()
        if(self.embeddedCanvas == False):
            try:
                self.itomUI.show()
            except RuntimeError:
                self._widgetclosed()
            except:
                pass

    def destroy( self, *args ):
        if DEBUG: print("destroy figure manager (1)")
        
        # check for qApp first, as PySide deletes it in its atexit handler
        #if self.canvas._destroying: return
        if(self.canvas._destroying == False):
            if(self.embeddedCanvas == False):
                try:
                    self.itomUI.disconnect( "destroyed()", self._widgetclosed )
                except:
                    pass
                try:
                    self.itomUI.hide()
                except:
                    pass
            else:
                try:
                    self.itomUI.disconnect( "destroyed()", self._widgetclosed )
                except:
                    pass
                try:
                    #self.itomUIDialog.hide()
                    pass
                except:
                    pass
        del self.itomUI
        self.itomUI = None
        if self.toolbar: self.toolbar.destroy()
        if DEBUG: print("destroy figure manager (2)")
        self.canvas.destroy()
        self.canvas._destroying = True
        
        
        #f.close()

    def set_window_title(self, title):
        if(self.embeddedCanvas == False):
            self.itomUI["windowTitle"] = ("%s (Figure %d)" % (title,self.num))

class NavigationToolbar2Itom( NavigationToolbar2 ):
    def __init__(self, figureCanvas, itomUI, embeddedCanvas, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        
        self.embeddedCanvas = embeddedCanvas
        self.itomUI = weakref.ref(itomUI)
        self.locLabel = None
        
        self.coordinates = coordinates
        self._idle = True
        self.subplotConfigDialog = None
        
        self.defaultSaveFileName = None
        
        NavigationToolbar2.__init__( self, figureCanvas )

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')
        
        r = self.itomUI()
        if(not r is None):
            r.actionHome.connect("triggered()", self.home)
            r.actionBack.connect("triggered()", self.back)
            r.actionForward.connect("triggered()", self.forward)
            r.actionPan.connect("triggered()", self.pan)
            r.actionZoomToRect.connect("triggered()", self.zoom)
            r.actionSubplotConfig.connect("triggered()", self.configure_subplots)
            r.actionSave.connect("triggered()", self.save_figure)

        if figureoptions is not None:
            a = self.addAction(self._icon("qt4_editor_options.png"),
                               'Customize', self.edit_parameters)
            a.setToolTip('Edit curves line and axes parameters')


        self.buttons = {}

        # reference holder for subplots_adjust window
        self.adj_window = None

    if figureoptions is not None:
        def edit_parameters(self):
            allaxes = self.canvas.figure.get_axes()
            if len(allaxes) == 1:
                axes = allaxes[0]
            else:
                titles = []
                for axes in allaxes:
                    title = axes.get_title()
                    ylabel = axes.get_ylabel()
                    if title:
                        fmt = "%(title)s"
                        if ylabel:
                            fmt += ": %(ylabel)s"
                        fmt += " (%(axes_repr)s)"
                    elif ylabel:
                        fmt = "%(axes_repr)s (%(ylabel)s)"
                    else:
                        fmt = "%(axes_repr)s"
                    titles.append(fmt % dict(title = title,
                                         ylabel = ylabel,
                                         axes_repr = repr(axes)))
                item, ok = QtGui.QInputDialog.getItem(self, 'Customize',
                                                      'Select axes:', titles,
                                                      0, False)
                if ok:
                    axes = allaxes[titles.index(str(item))]
                else:
                    return

            figureoptions.figure_edit(axes, self)
            
    def pan( self, *args ):
        if self._active != 'PAN':
            self.set_cursor(cursors.MOVE)
        else:
            self.set_cursor(-1)
        self.itomUI().actionZoomToRect['checked'] = False
        #self.window.actionMarker['checked'] = False
        NavigationToolbar2.pan( self, *args )

    def zoom( self, *args ):
        if self._active != 'ZOOM':
            self.set_cursor(cursors.SELECT_REGION)
        else:
            self.set_cursor(-1)
        self.itomUI().actionPan['checked'] = False
        #self.window.actionMarker['checked'] = False
        NavigationToolbar2.zoom( self, *args )

    def dynamic_update( self ):
        d = self._idle
        self._idle = False
        if d:
            self.canvas.draw()
            self._idle = True

    def set_message( self, s ):
        if self.coordinates:
            r = self.itomUI()
            if(not r is None):
                r.call("setLabelText", (s.replace(', ', '\n')))

    def set_cursor( self, cursor ):
        self.canvas.canvas.call("setCursors", cursord[cursor])

    def draw_rubberband( self, event, x0, y0, x1, y1 ):
        if DEBUG: 
            print('draw_rubberband: ', event, x0, y0, x1, y1)
        
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        
        rect = [ int(val) for val in (min(x0,x1), min(y0, y1), w, h) ]
        self.canvas.drawRectangle( rect )

    def configure_subplots(self):
        if(self.subplotConfigDialog is None):
            self.subplotConfigDialog = SubplotToolItom(self.canvas.figure, self.itomUI(), self.embeddedCanvas)
        
        self.subplotConfigDialog.showDialog()

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(filetypes.items())
        sorted_filetypes.sort()
        
        
        if not self.defaultSaveFileName is None:
            start = self.defaultSaveFileName
            default_filetype = os.path.splitext(start)[1]
            if default_filetype == "":
                default_filetype = self.canvas.get_default_filetype()
            elif default_filetype.startswith("."):
                default_filetype = default_filetype[1:]
        else:
            default_filetype = self.canvas.get_default_filetype()
            start = "image." + default_filetype
        
        filters = []
        selectedFilterIndex = 0
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilterIndex = len(filters)
            filters.append(filter)
        filters = ';;'.join(filters)
        
        fname = ui.getSaveFileName("Choose a filename to save to", start, filters, selectedFilterIndex, parent = self.itomUI())
        
        if fname:
            try:
                self.canvas.print_figure( str(fname) )
                self.defaultSaveFileName = fname
            except Exception as e:
                ui.msgCritical("Error saving file", str(e), parent = self.itomUI())
    
    def set_history_buttons(self):
        can_backward = (self._views._pos > 0)
        can_forward = (self._views._pos < len(self._views._elements) - 1)
        
        itomUI = self.itomUI()
        if(not itomUI is None):
            itomUI.actionBack["enabled"] = can_backward
            itomUI.actionForward["enabled"] = can_forward
    
    def destroy(self):
        del self.canvas #in base class
        self.canvas = None
        

class SubplotToolItom( SubplotTool ):
    def __init__(self, targetfig, itomUI, embeddedCanvas):
        self.targetfig = targetfig
        self.embeddedCanvas = embeddedCanvas        
        self.itomUI = weakref.ref(itomUI)
        itomUI.connect("subplotConfigSliderChanged(int,int)", self.funcgeneral)
        
    def showDialog(self):
        valLeft = int(self.targetfig.subplotpars.left*1000)
        valBottom = int(self.targetfig.subplotpars.bottom*1000)
        valRight = int(self.targetfig.subplotpars.right*1000)
        valTop = int(self.targetfig.subplotpars.top*1000)
        valWSpace = int(self.targetfig.subplotpars.wspace*1000)
        valHSpace = int(self.targetfig.subplotpars.hspace*1000)
        
        r = self.itomUI()
        if(not r is None):
            r.call("showSubplotConfig", valLeft, valTop, valRight, valBottom, valWSpace, valHSpace)
    
    def funcgeneral(self, type, val):
        if(type == 0):
            self.funcleft(val)
        elif(type == 1):
            self.functop(val)
        elif(type == 2):
            self.funcright(val)
        elif(type == 3):
            self.funcbottom(val)
        elif(type == 4):
            self.funcwspace(val)
        elif(type == 5):
            self.funchspace(val)

    def funcleft(self, val):
        #if val == self.sliderright.value():
        #    val -= 1
        self.targetfig.subplots_adjust(left=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcright(self, val):
        #if val == self.sliderleft.value():
        #    val += 1
        self.targetfig.subplots_adjust(right=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcbottom(self, val):
        #if val == self.slidertop.value():
        #    val -= 1
        self.targetfig.subplots_adjust(bottom=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def functop(self, val):
        #if val == self.sliderbottom.value():
        #    val += 1
        self.targetfig.subplots_adjust(top=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

        
FigureManager = FigureManagerItom




