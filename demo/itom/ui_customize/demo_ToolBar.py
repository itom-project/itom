"""Toolbar
===========

This demo shows how buttons are added and removed from the ``itom`` toolbar.
Frequently used methods are thus easier to access.
By clicking the button, these are executed. """

from functools import partial
from itom import addButton
from itom import removeButton
from itom import version as itomVersion

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoToolbar.png'


def method():
    """Sample callback method 1."""
    print("The method 'method' has been clicked")


def methodArgs(arg1, arg2):
    """Sample callback method 2."""
    print("The method 'methodArgs' has been clicked. Args:", arg1, arg2)


class Test:
    """Sample class."""

    def doit(self):
        """Sample member method of this class."""
        print("The member 'doit' of class 'Test' has been clicked.")


###############################################################################
# If this demo is executed multiple times, try to remove all
# existing toolbars ``demobar`` and ``otherbar``. This command is
# used and explained later.
try:
    removeButton("demobar")
except RuntimeError:
    pass

try:
    removeButton("otherbar")
except RuntimeError:
    pass

###############################################################################
# Add a single button without icon to a new toolbar with the name ``demobar``.
# The callback function is the unbounded method ``method`` without arguments.
addButton("demobar", "call method", method)

###############################################################################
# This is quite similar than the addButton above, however internally it
# makes a difference if a Python-scripted method is used as callback or
# a method from the itom module, implemented in C.
addButton("demobar", "call itom.version()", itomVersion)

###############################################################################
# Add another button with an icon to the same toolbar. This time,
# the unbounded method ``methodArgs`` should be triggered if the button is clicked.
# the name of the button is shown in the tooltip text of the button.
addButton("demobar", "call methodArgs", methodArgs, icon=":/arrows/icons/plus.png", argtuple=("arg1", 23))
# add another button to 'demobar' and use a lambda function as callback
addButton("demobar", "call lambda function", lambda: print("lambda func call"))

###############################################################################
# Call a partial method. This is a method, that wraps a base method with
# more arguments, but selected arguments are already preset.
addButton(
    "demobar", "call partial method", partial(lambda num, base: print(int(num, base)), base=2), argtuple=("10010",)
)

###############################################################################
# Add a button to the 'demobar' toolbar, that evaluates a Python code string.
addButton("demobar", "call code string", "print('code string')")

###############################################################################
# Add a button that triggers a member method of the object ``myTest``.
# .. hint:: If a button triggers such a member method, the button does not
#     explicitly keep a reference to the object, such that this object must
#     be kept by any other variable. Else, a RuntimeError is raised when the
#     button is triggered.

myTest = Test()
addButton("demobar", "call bounded method", code=myTest.doit, icon=":/classNavigator/icons/class.png")

###############################################################################
# Create a new button and get its handle
handle = addButton("demobar", "temp", method)

###############################################################################
# And remove the button again
removeButton(handle)

###############################################################################
# Next step: create some buttons in another toolbar ``otherbar`` and
# then remove the entire toolbar ``otherbar``:
for i in range(0, 5):
    addButton("otherbar", "btn%i" % i, method)

###############################################################################
# At first remove one button
removeButton("otherbar", "btn3")

###############################################################################
# Then remove all remaining buttons including the toolbar 'otherbar'.
removeButton("otherbar")

###############################################################################
# Following buttons/bar will be added to the ``itom`` toolbar.
# 
# .. image:: ../../_static/demoToolBar_1.png
#    :width: 75%