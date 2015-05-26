.. include:: ../include/global.inc

.. _script-language-reload-modules:

Reload modified modules
==========================

Usually, script files (hence *modules*) that are imported by another script in python are pre-compiled and cached for a faster execution once the script is loaded or imported another time.
These cached files are always stored in a sub-folder **__pycache__** (file suffix .pyc). The advantage of this feature is a faster code execution once the pre-compiled and cached file is
available. On the other hand, this feature can bring some drawbacks during the development process if the content of modules or packages may change. Then, these changes will not become active.

There are different possibilities to force Python to reload such a changed module:

1. Restart |itom|: The date of creation of all cached files (pyc) is compared with the change date of the corresponding py-files and they are recompiled if the scripts are newer.
2. The Python builtin-module **imp** provides mechanisms like the method **reload** to force Python to reload a specific module.
3. The mechanisms provided by the **imp** module are covered by the dialog **reload modules...** that is available in the menu **Script >> reload modules** of the main window of |itom|.
4. |itom| consists of a powerful auto-reload tool that can check all modules are their dependencies whether they have changed since the last check and reloads them. This tool is discussed in the
course of this section.

At first, let us denote several issues that may happen due to the caching mechanism of |itom|. Consider the following three script files:

.. code-block:: python
    
    #script1.py
    import mod2
    
    print("version 1")
    mod2.func2()
    
.. code-block:: python
    
    #mod2.py
    import mod3
    
    def func2():
        print("func2, version 1")
        mod3.func3()
    
.. code-block:: python
    
    #mod3.py
    
    def func3():
        print("func3, version 1")
        
If *script1.py* is now executed, the modules *mod2* and *mod3* are imported and cached. The output at the first run is::
    
    'version 1'
    'func2, version 1'
    'func3, version 1'

If we change now the strings 'version 1' to 'version 2' in all three files and execute *script1.py* again, the output will be as follows::
    
    'version 2'
    'func2, version 1'
    'func3, version 1'

Only the first line changed, the other two stayed unchanged since *mod2* and *mod3* are still cached. Of course a restart of |itom| would lead to the
right result::
    
    'version 2'
    'func2, version 2'
    'func3, version 2'

Another possibility would be to reload *mod2* using the Python builtin-module *imp*, since *mod2* is imported by *script1.py*::
    
    import imp
    imp.reload(mod2)

Another execution of *script1.py* will now lead to the following result::
    
    'version 2'
    'func2, version 2'
    'func3, version 1'

The last result, coming from *mod3* is still unchanged. This comes due to the fact that the *imp.reload* command does not resolve any dependencies but
only tries to reload one single module, corresponding to the content of one single py-file. Therefore, you always need to know where exactly code changes
have occurred and reload all related modules. To simplify this mechanism, you can use the dialog **Reload modules...** that is reachable via the |itom| menu
**Script >> reload modules >> reload modules...**. Sometimes, the reload may fail. Reasons for this and further limitations of the reload process are discussed later.

In order to provide an easy way to automatically reload all modules that have been changed since the last execution, |itom| provides an auto-reload tool.
This tool has been inspired by the autoreload module of *IPython* (http://ipython.org/ipython-doc/dev/config/extensions/autoreload.html) and is fully integrated into
|itom|. Enable the tool by the menu **Script >> reload modules >> autoreload modules**. Depending on further settings, the currently executed script file, code command or
function is checked (including all its dependencies) for changes are reloaded if necessary. You have full control in which cases you want that check being executed.
This is controlled by the further options in the submenu **Script >> reload modules**:

* autoreload before script execution: The check is executed whenever you run or debug a script file
* autoreload before single command: The check is executed before you execute a string command from the command line of |itom|
* autoreload before events and function calls: The check is executed if any python code or function is executed due to an event or signal (e.g. button click in a GUI)

Try to enable the autoreload tool and enable at least the option *autoreload before script execution*. Then change the version strings in all files to 'version 3' and execute script3::
    
    'version 3'
    'func2, version 3'
    'func3, version 3'

Using this tool, you do not need to worry about reloading any changed modules. This gives you a powerful tool for developing more complex scripts that are divided into multiple files.
The autoreload tool can also be enabled and configured using the command :py:func:`itom.autoReloader`.

Sometimes, you will notice that reloading a module using the *imp* module will fail or not work. Consider the following script::
    
    #mod4.py
    
    class MyRect():
        def __init__(self, height, width):
            self.sizes = [height, width]
        
        def getSizes(self):
            print("size of MyRect", self.sizes)

Now type into the command line::
    
    import mod4
    rect = mod4.MyRect(4,5)
    rect.getSizes()

You will obtain::
    
    'size of MyRect: [4,5]'

If you change now the print-command in the method 'getSizes' of class 'MyRect' to::
    
    print("width:", self.sizes[1], ",height:", self.sizes[0])

and call::
    
    imp.reload(mod4)

in order to reload *mod4.py* again, a call to::
    
    rect.getSizes()

in the command line will still lead to the old result. This is due to the fact, that the *imp* module cannot reload objects that are still referenced by another variable. In this case,
the global variable **rect** is an instance of the class **MyRect**. Therefore, it is not possible to reload this class before deleting the variable **rect**. However, if you enable the
autoreload tool and enable the option **autoreload before single command** before changing the print command, you will see that this still is also able to replace the code of a class method
even if there are already active instances of this class.

The autoreload tool is a way more powerful than the native *imp* implementation. However there are still some limitations:

Reloading Python modules in a reliable way is in general difficult, and unexpected things may occur. 'autoreload' tries to work 
around common pitfalls by replacing function code objects and parts of classes previously in the module with new versions. This makes the following things to work:

* Functions and classes imported via 'from xxx import foo' are upgraded to new versions when 'xxx' is reloaded.
* Methods and properties of classes are upgraded on reload, so that calling 'c.foo()' on an object 'c' created before the reload causes the new code for 'foo' to be executed.

Some of the known remaining caveats are:

* Replacing code objects does not always succeed: changing a @property in a class to an ordinary method or a method to a member variable can cause problems (but in old objects only).
* Functions that are removed (eg. via monkey-patching) from a module before it is reloaded are not upgraded.
* C extension modules cannot be reloaded, and so cannot be autoreloaded.

    (taken from http://ipython.org/ipython-doc/dev/config/extensions/autoreload.html)
