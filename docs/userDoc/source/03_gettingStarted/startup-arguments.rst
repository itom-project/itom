.. include:: ../include/global.inc


Startup and Arguments
***********************

To start |itom| call the executable **qitom**. Under Windows this is for instance the file **qitom.exe**.

It is furthermore possible to pass additional arguments to the **qitom** executable.
The following arguments are possible (the order of arguments is unimportant):

1. **<path-to-python-script.py>**: opens the indicated script in the script editor. This argument can be appended multiple times.
2. **log**: If the argument **log** is contained in the list of arguments, all messages sent via qDebug, QWarning... are sent to the logfile
             **itomlog.txt** in the application directory of itom. This can be used for debugging.
3. **name=<usernameID>**: Pass an additional ID of an available user. If given, itom is started with this user and its corresponding setting files.
4. **run=<path-to-python-script.py>**: similar to 1. Runs the given script (if it exists) after possible autostart scripts, which are part of the 
             active user settings. The path to the script can also be put into quotion marks, if it contains spaces or other special characters.
             This argument can be appended multiple times.
5. **pipManager**: If this argument ist given, only opens the Python Package Manager instead of the main GUI of itom. This can for instance be used
             to update **Numpy**, since this cannot be updated if the main GUI is started, since itom directly uses Numpy, such that some of its files
             are blocked during the runtime of itom.