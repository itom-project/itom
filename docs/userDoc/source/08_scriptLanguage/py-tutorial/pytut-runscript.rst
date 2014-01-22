.. include:: /../../include/global.inc

Creating a project folder and run a first program
-------------------------------------------------
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau


Create a project folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keeping track of more difficult programs is inconvenient when only using the |itom| terminal. That's why you will learn how to use script files in this chapter. Basically, script files are just regular text files that happen to have the extension *.py*.

We will store all our script files in a project folder within the |itom| installation directory called *python_tutorial*. This folder can be created within |itom| by right-clicking on the |itom| folder in the File System Panel and choosing *create new folder*. If you've used the default installation directory, you should end up with a folder like this: *C:\\Program Files\\iTOM\\python_tutorial*.


Creating a .py file and running it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now create a new python file in *C:\\Program Files\\iTOM\\python_tutorial* and rename the default file name (*New Script.py*) to *Hello World.py*. When double-clicking the file name, the Script Editor will pop up, which we will use for all upcoming examples. For a start, type the following line of code into the Script Editor and save the file.

.. code-block:: python
    :linenos:
    
    print("Hello World")

To run the code you've just inserted, press *F5* or click on the *run* button.

As a result, you will see the output 'Hello World' in the |itom| terminal. As you've probably figured out, the command :py:func:`print` is used to output certain characters and, hence, can be used to give the user some sort of feedback. Please note that the print command changed from earlier 2.X versions of |Python|.