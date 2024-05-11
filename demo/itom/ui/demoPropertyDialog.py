"""Property dialog
==================

"""

from itom import ui
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPropertyDialog.png'

gui = ui(
    "propertyDialog.ui",
    ui.TYPEDIALOG,
    ui.BUTTONBAR_HORIZONTAL,
    {"AcceptRole": "OK", "RejectRole": "Cancel"},
)

# pre-initialize values
gui.spinValueA["value"] = 5
gui.spinValueB["value"] = 4.5

# show dialog and wait until it is closed (argument: 1 -> modal)
ret = gui.show(1)

if ret == 1:
    # evaluate your input
    print("ValueA:", gui.spinValueA["value"])
    print("ValueB:", gui.spinValueB["value"])

    if gui.radioItem1["checked"]:
        radioNr = 1
    elif gui.radioItem2["checked"]:
        radioNr = 2
    else:
        radioNr = 3
    print("selected radio button:", radioNr)

    print("your text:", gui.txtText["text"])
else:
    print("the dialog has been rejected")
