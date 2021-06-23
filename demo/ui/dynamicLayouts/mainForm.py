from itom import ui, uiItem

gui: ui = ui("mainForm.ui", type=ui.TYPEWINDOW)
print("a")
vlayout: uiItem = gui.vlayout
print("b")
#raise
import time



# add 10 items from item.ui
for i in range(0, 10):
    # all object names of the added widget including
    # its child widgets and layouts are modified by
    # the following suffix:
    objNameSuffix: str = f"_{i}"
    
    # ctrlItem is the reference to the newly added outer widget
    ctrlItem: uiItem = vlayout.call("addItemFromUiFile", "item.ui", objNameSuffix)
    
    # print the name of all newly added child widgets
    print(ctrlItem.children())
    
    # access the newly added label
    lbl: uiItem = ctrlItem.getChild("label" + objNameSuffix)
    lbl["text"] = f"Item {i+1}"
    
    # alternate the check state of the LedStatus
    led: uiItem = ctrlItem.getChild("led" + objNameSuffix)
    led["checked"] = i % 2
    
    # change the checkstate of some checkboxes
    # the enable state of the corresponding spinboxes
    # is automatically changed due to the signal/slot 
    # connection, created in QtDesigner.
    checkbox: uiItem = ctrlItem.getChild("checkBox" + objNameSuffix)
    checkbox["checked"] = i % 3

# show the gui
gui.show()