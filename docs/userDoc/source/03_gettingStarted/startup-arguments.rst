.. include:: ../include/global.inc

.. _startup-and-arguments:

Startup and Arguments
***********************

To start |itom| call the executable **qitom**. Under Windows this is for instance the file **qitom.exe**.

It is furthermore possible to pass additional arguments to the **qitom** executable.
The following arguments are possible (the order of arguments is unimportant):

1. **<path-to-python-script.py>**: opens the indicated script in the script editor.
   This argument can be appended multiple times.
2. **log** or **log=<path-to-directory>**: The log is saved to the file itomlog.txt in
   the user directory ``C:\Users\<UserName>\AppData\Local\qitom``.
   The ``log=<path-to-directory>`` argument can be used to define a different path
   where the log files are saved.
3. **nolog**: Do not write a log file. Overwrites any **log** argument (see above).
   If neither **log** nor **nolog** is given, the behaviour is equal to **log** without
   user defined path.
4. **name=<usernameID>**: Pass an additional ID of an available user. If given,
   itom is started with this user and its corresponding setting files.
5. **run=<path-to-python-script.py>**: similar to 1. Runs the given script
   (if it exists) after possible autostart scripts, which are part of the
   active user settings. The path to the script can also be put into quotation marks,
   if it contains spaces or other special characters. This argument can be appended
   multiple times.
6. **pipManager**: If this argument is given, only opens the Python Package
   Manager instead of the main GUI of itom. This can for instance be used
   to update **Numpy**, since this cannot be updated if the main GUI is started,
   since itom directly uses Numpy, such that some of its files are blocked during
   the runtime of itom.


Logging
===========

All messages sent via qDebug, QWarning, via the ``itom.log`` function and python
error calls are written to the log. The log is written by default unless the ``nolog``
startup argument is given as described in the previous chapter.
The log is saved to the user directory ``C:\Users\<UserName>\AppData\Local\qitom``
or the directory given with the ``log`` argument.

A form of log rotation is used to save disk space and prevent extremely large log files.
The most recent log messages are appended to the file ``itomlog.txt``. When the file
size exceeds 5 MB on startup, the file will be backed up by renaming it to contain the
current date and a new file will be created. When there are more than two of these
backup files, all but the two newest ones will be deleted.
