"""Toolbar
===========

This demo shows how buttons are added and removed from the ``itom`` toolbar.
Frequently used methods are thus easier to access.
By clicking the button, these are executed. """

###############################################################################
# Create a new application class.


class myToolBar:
    def __init__(self):
        # init method which creates a member test with value 1
        self.test = 1

    # Add the delete function, which is called from the python-garbage-collector randomly after the class was deleted
    def __del__(self):
        # clean up buttonbar entries after killing the class
        removeButton("helloBar", "HelloWorld")

###############################################################################
# Add a new function to the class ``myToolBar``.
    def printHelloWorld(self, test):
        # Try to print Hello-World and return test
        print("Hello World")
        return test

    def __len__(self):
        return 42

###############################################################################
# Create a new instance of the class ``myToolBar`` with the name app.


app = myToolBar()

# Add the button Hello with the function myToolBar.printHelloWorld(...) to the buttonbar amipola
addButton(
    "helloBar",
    "HelloWorld",
    "res = app.printHelloWorld(True)",
    ":/classNavigator/icons/global.png",
)

# For Debug test with single step
app.printHelloWorld(True)
