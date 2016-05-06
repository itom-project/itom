.. include:: ../include/global.inc

.. _script-python-limitations:

Python Limitations in |itom|
****************************

You can use almost all methods, classes or modules provided by the core of |python| or any external modules, if you consider the following hints or rules:

* The embedded python interpreter in |itom| is based on |python| 3.x.
* |python| is executed in its own thread in |itom|, therefore you should not use any modules that create some form of GUI. Usually the creation of any GUI element needs to be executed in the main thread of the calling application. Therefore don't use the python package **PyQt**. Instead use the GUI-functionality which is directly provided by |itom| and allows to extend the GUI of |itom|.
* The |python| command :py:meth:`input` requires a string written in the command line and confirmed by the return key.
