"""String input
==============

This demo shows how to get input from the user.
For more sophisticated inputs based on input boxes etc.
please see the methods in the module ``itom.ui``.

Inputs in this demo will force an input line in the command line
(green background). Put some text there and press return to continue."""

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoStringInput.png'


name = input("Please put your name after the colon:")
age = input("Please put your age:")

try:
    age_ = int(age)
    print("Hello %s. Your age is %i" % (name, age_))
except Exception:
    print("Your age could not be interpreted as integer")