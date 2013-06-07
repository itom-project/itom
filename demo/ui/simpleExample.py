'''
This script shows an example how to integrate the user interface
"simpleExampleui" into itom, connect it with the necessary methods
and show it
'''

#create dialog instance, that represents the user interface
# - our ui-file is a simple widget, that should be embedded in an auto-created
# window, that itom provides for us (type: ui.TYPEDIALOG)
# - We want to have an auto-created vertical button bar at the right side (dialogButtonBar = ui.BUTTONBAR_VERTICAL)
# - The button bar should consist of one OK button, therefore its role is "AcceptRole"
#
mainWin = ui("simpleExample.ui", ui.TYPEDIALOG, ui.BUTTONBAR_VERTICAL, {"AcceptRole":"OK"})

# 1. first group box: show text of textfield in a message box
# -----------------------------------------------------------


# set default text of TextField with name txtMessage
field = mainWin.txtMessage #you can get an instance (class uiItem) of any sub-element of your user interface
field["text"] = "hello world"

#the same can be done directly:
mainWin.txtMessage["text"] = "hello world"

# now we want to connect the clicked-signal of the button with the method 'showMessage'
# How do I know which signal any GUI-element can emit?
#  --> see Qt help -> goto to class corresponding to your element and check the available signals
#               of that class and all inherited classes (QPushButton >> QAbstractPushButton >> QWidget...)
#  --> note the name of the signal you want to connect and the argument list (clicked(bool checked = false))
#  --> In our case, we want to connect to the clicked-signal, its argument can be a boolean argument (optional)
#
#  --> Finally, we connect the clicked-signal without argument (not interesting for us) with a method 'showMessage'
#     This method needs to have the same number of arguments than the signal (here: 0)
def showMessage():
    text = mainWin.txtMessage["text"]
    ui.msgInformation("your text", text, parent = mainWin)
    
mainWin.btnShowText.connect("clicked()", showMessage)


# 2. second group box: show getDirectory-dialog and print the chosen directory in the text field
# -----------------------------------------------------------

def showGetDirectory():
    directory = ui.getExistingDirectory("chose directory", itom.getCurrentPath(), parent = mainWin)
    if(directory is None):
        pass
        #cancel has been clicked
    else:
        mainWin.txtDirectory["text"] = directory

#connect the clicked-signal of toolSelectDir with showGetDirectory
mainWin.toolSelectDir.connect("clicked()", showGetDirectory)

# if txtDirectory has the property 'readOnly' set to true, the button btnReadOnly should be 'checked'.
mainWin.btnReadOnly["checked"] = mainWin.txtDirectory["readOnly"]

# if the button is toggled (check-state is changed), then we want to change the ready-only property.
def btnReadOnlyToggled(checked):
    mainWin.txtDirectory["readOnly"] = checked

mainWin.btnReadOnly.connect("clicked(bool)", btnReadOnlyToggled)
    


# 3. At first, the third group box should be hidden. If the checkbox checkShowThirdExample is clicked, the visibility should change
# -----------------------------------------------------------

def checkShowThirdExample_clicked(checked):
    # we have two possibilities: 1. change visible-property of group3, 2. use the slot (special, accessible method of any widget, that can be addressed by python)
    # 'hide' or 'show' from group3 (the slots are part of any widget) in order to show or hide the element.
    # Here: we want to call the slots
    if(checked):
        mainWin.group3.call("show")
    else:
        mainWin.group3.call("hide")

mainWin.checkShowThirdExample.connect("clicked(bool)", checkShowThirdExample_clicked)

#initially, hide group3
mainWin.group3.call("hide")


# 4. Connect radio-button list with list-box
# -----------------------------------------------------------

# get handle to list box
listBox = mainWin.listWidget

# if you see Qt help for QListWidget, you will see, that there is no slot, that you can use to add items
#  to a listWidget. However, for some special widgets, a selection of public methods is nevertheless accessible in the same way than
#  a slot. In case of a listWidget, this is 'addItem', 'addItems'
#  this is used, to add the names of all radio buttons to the list:
listBox.call("addItem", mainWin.radio1["text"])
listBox.call("addItems", [ mainWin.radio2["text"], mainWin.radio3["text"] ])

#let us pre-select the second radio button and the second list item
mainWin.radio2["checked"] = True
listBox["currentRow"] = 1

def listCurrentChanged(row):
    if(row == 0):
        mainWin.radio1["checked"] = True
    elif(row == 1):
        mainWin.radio2["checked"] = True
    else:
        mainWin.radio3["checked"] = True
        
#if the current item in the list is changed, the corresponding radio button should be checked
listBox.connect("currentRowChanged(int)", listCurrentChanged)


#show dialog
# you have three options.
# 1. show dialog in a non-modal form (user can still click something else in itom)
#     >> mainWin.show(0) or mainWin.show()
# 2. show dialog in a model form (user cannot interact with itom, python script execution stops until
#      dialog has been closed. The return value is the role-number of the button, that has been clicked
#      for closing ("AcceptRole":0, "RejectRole":1) (see Qt-enum QDialogButtonBox::ButtonRole)
#     >> mainWin.show(1)
# 3. show dialog in a model form, python script execution continues, you don't have access to the return value
#     >> mainWin.show(2)
mainWin.show(1)




