.. include:: ../include/global.inc

.. _script-python-problems-solutions:

Python - common problems and solutions
========================================

The following list state some common problems or known issues concerning |Python| (partially in combination with |itom|):

Re-Assigning a variable
-------------------------

**Problem:** If you assign an object with a limited access to anything (like a single camera) to a variable in a script, an access error might occur if you try to re-run the script:

.. code-block:: python
    
    cam = dataIO("IDSuEye", camera_id = 0)
    
Once you re-execute the same command, an error like the following one might be raised::
    
    RuntimeError: Could not load plugin IDSuEye with error message:
    Camera (0) could not be opened
    
This is due to the fact, that the re-execution at first tries to create a new instance of class *dataIO* with the same camera than is already opened. Then, this instance
is assigned to the existing variable *cam*. In this moment, the recent content of *cam* is not longer in use and hence destroyed.

**Solution:** At first assign *None* or any other low-level value to *cam* and then re-assign the object that requires access to a limited structure. In some rare cases this
is even not enough, since |Python| uses the concept of a garbage collector. Therefore, an object is only marked for deleted if it is not longer in use. The garbage collector is
regularly called and finally deletes all marked objects. In this case, force the garbage collector to be executed:

.. code-block:: python
    
    import gc #import garbage collector
    
    cam = None
    gc.collect() #start the garbage collector
    cam = dataIO("IDSuEye", camera_id = 0)
    
Variable deleted but referenced object is not closed
-----------------------------------------------------

**Problem:** I delete a variable in |Python| but the value (e.g. a hardware instance - dataIO or actuator) is not closed.

**Solution:** At first, you should check if the variable you deleted is really the last variable that referenced to the underlying value. If you opened a hardware instance by
the GUI you need to know that the GUI also holds a reference to the hardware. Therefore, the hardware must additionally be closed via a mouse click in the GUI. If the value is nevertheless
not immediately destroyed, the last raised exception or the garbage collector of |Python| can be the reason.

.. code-block:: python
    
    class MyClass():
        def __init__(self):
            pass
            
        def __del__(self):
            print("MyClass destroyed")
            
    m = MyClass()
    raise RuntimeError(m)
    del m
    
In this example, an instance of class *MyClass* is created (variable *m*). Afterwards, a runtime error is raised with *m* as single argument. Finally, *m* is deleted, but the destructor
of *MyClass* is not called (no text *MyClass destroyed* is printed out). However, if you raise another runtime error::
    
    raise RuntimeError()
    
the class is destroyed and the text appears. This is due to the fact, that the last exception that has been raised is still in memory and holds a reference to the passed argument, here, the
instance *m* of class *MyClass*.

.. note::
    
    This behaviour changes in itom version > 1.4.0. Then, the last exception is not stored any more in the variables :data:`sys.last_type`,
   :data:`sys.last_value` and :data:`sys.last_traceback`.

Nevertheless, it might happen, that the object referenced by a variable (like a camera) is not immediately destroyed even if the last referencing variable is deleted. In |Python| objects are not directly deleted if they are not used any more, but they are only marked for deletion. Then, regularly, a garbage collector is executed that finally deletes all values marked for deletion. The reason is that deleting objects might be complicated and it is therefore better to execute this if the interpreter is idle or many objects have been marked. In order to directly force the garbage collector to delete marked objects, use:

.. code-block:: python
    
    import gc #import garbage collector
    gc.collect() #start the garbage collector
    
Codec error
------------

**Problem:** When executing a |Python| script, a syntax error with an error message similar to the following one appears:

.. code-block:: python
    
    File "C:\test.py", line 10
    SyntaxError: (unicode error) 'utf-8' codec can't decode byte oxe4 in position 0: unexpected end of data
    
**Solution:** You used any special character (even in comments) in your code. Per default and if not otherwise state, a document is always parsed using the 'utf-8' codec.
This codec does not support special characters (like German 'Umlaute' or the greek letter µ). If you want to use such characters, you need to indicate the codec of your file,
e.g. by adding::
    
    # -- coding: iso-8859-15 --
    
at the first line of your script file (iso-8859-15 represents the Western Europe charset). You can also choose **insert codec...** from the context menu or the edit menu of the editor.
Choose then the codec you want to use and the corresponding line is automatically prepended to your script. The list of default codec is only a small subset of
possible codecs, you can also insert another codec into the editable drop-down box. For more information see https://www.python.org/dev/peps/pep-0263/.

