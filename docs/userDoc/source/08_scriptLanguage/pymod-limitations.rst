.. include:: ../include/global.inc

.. _script-python-limitations:

Python Limitations in |itom|
****************************

You can use almost all methods, classes or modules provided by the core of |python| or any external modules, if you consider the following hints or rules:

* The embedded python interpreter in |itom| is based on |python| 3.2.
* |python| is executed in its own thread in |itom|, therefore you should not use any modules that create some form of GUI. Usually the creation of any GUI element needs to be executed in the main thread of the calling application. Therefore don't use the python package **PyQt**. Instead use the GUI-functionality which is directly provided by |itom| and allows to extend the GUI of |itom|.
* If you were used to use to python methods, that asked the user for some input in the original python command window, you cannot use these commands in |itom|. Instead use message or input boxes that are provided by the class **ui** of the module **itom** in order to get some basic information from the user.
* It can not be guaranteed that the use of the **threading** module of |python| is fully usable together with |itom|.

