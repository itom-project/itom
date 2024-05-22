"""Dialog
=========

The dialog is created from a QDialog, designed in QtDesigner.
The dialog already has OK and Cancel buttons whose clicked
signal is connected with the accept and reject slot of the
dialog. If you show the dialog in a modal way, you can then
obtain the result (if OK or Cancel has been clicked).
Use deleteOnClose=true such in order to close the dialog once
the user pressed OK or Cancel."""

from itom import ui

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoDialog.png'


dialog = ui("dialog.ui", ui.TYPEDIALOG)
print("Result of the dialog:", dialog.show(1))  # show a modal dialog

###############################################################################
# Dialog created from widget. In this case, no terminating buttons
# are visible. The behaviour is then similar to a main window without
# the minimize or maximize buttons
dialog_widget = ui("widget.ui", ui.TYPEDIALOG)
dialog_widget.show()

###############################################################################
# If the dialog should be created from a widget, you can automatically let
# itom place buttons at the right or bottom side of the widget. Define the
# title and the role of each button using a dictionary. The roles are taken
# from Qt (``QDialogButtonBox::ButtonRole``)
dialog_widget_buttonbar = ui(
    "widget.ui",
    ui.TYPEDIALOG,
    ui.BUTTONBAR_VERTICAL,
    {"AcceptRole": "OK", "RejectRole": "Cancel"},
)
print("Result of the dialog:", dialog_widget_buttonbar.show(1))
