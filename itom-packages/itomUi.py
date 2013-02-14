from itom import ui

class ItomUi():
    '''
    Base class which can be inherited in order to show an user defined
    user-interface. This class provides possibilites for auto-connecting
    decorated methods in your implementation with certain signals of widgets
    in the user interface.
    
    Example:
    - User interface contains a button 'pushButton'
    - Write a method which should be connected to the buttons clicked(bool)-signal:
    
        @ItomUi.autoslot("bool")
        def on_pushButton_clicked(self,arg):
            #this method is auto-connected in the constructor of ItomUi
            pass
    
    - This step is the same than typing:
        self.ui.pushButton.connect("clicked(bool)", self.on_pushButton_clicked)
    '''
    
    def __init__(self,filename,*args,**kwds):
        '''
        constructor with the same parameters than class 'ui':
        - filename [string]: path to user interface file (*.ui) 
        OPTIONAL:
        - showDialogButtons [bool]: indicates whether dialog buttons should automatically be added (true, default), else false 
        - dialogButtonsOrientation [int]: 0: horizontal above ui-widget, 1: vertical on the right side of ui-widget 
        - dialogButtons [dict]: every dictionary-entry is one button. key is the role, value is the button text
        '''
        self.gui = ui(filename,*args,**kwds)
        self.autoconnect()
    
    def show(self,modal=0):
        return self.gui.show(modal)
    
    def autoconnect(self):
        '''
        checks all methods of your class and if they have the decorator @autoslot,
        connect them with the widget's signal, if the name of the method fits to the
        requirements (see doc of autoslot-decorator)
        '''
        for key in dir(self):
            value = getattr(self, key)
            if(getattr(value, "hasAutoSlot",False)):
                wid = getattr(value, "widgetName", [])
                sig = getattr(value, "signature", [])
                for w,s in zip(wid,sig):
                    try:
                        widget = eval("self.gui." + w)
                    except:
                        print("Auto-connection failed: Widget",w,"could not be found.")
                        continue
                    
                    try:
                        widget.connect(s, value)
                    except:
                        print("Auto-connection failed. Widget ",w," has no slot ",s,"(",sig,").",sep = '')
                
    
    def autoslot(*attr):
        '''
        For auto-connecting your method with a signal of a widget in the
        user interface, your method must have as name 'on_WIDGETNAME_SIGNALNAME' and
        you have to decorate your method with the decorator '@autoslot('parameters').
        '''
        def decorate(func):
            parts = func.__name__.split("_")
            if(len(parts) == 3 and parts[0] == "on"):
                setattr(func,"hasAutoSlot",True)
                newSig = "{0}({1})".format(parts[2],attr[0])
                sig = getattr(func, "signature", [])
                sig.append(newSig)
                wid = getattr(func, "widgetName", [])
                wid.append(parts[1])
                setattr(func,"signature",sig)
                setattr(func,"widgetName",wid)
            return func
        return decorate
