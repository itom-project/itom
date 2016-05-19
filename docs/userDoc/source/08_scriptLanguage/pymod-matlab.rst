.. include:: ../include/global.inc

.. _pymod-matlab:

Python-Module matlab
*************************

|itom| provides the python module **matlab** that can be used to establish a remote control to the Matlab software.
Using this module, the following features are available:

* Remotely access Matlab via the matlab engine
* Put and read variables from itom / python to and from the Matlab workspace.
* Execute arbitrary Matlab commands like having executed them in the Matlab command window, including Matlab functions.
* Integrate existing Matlab functions into a Python-scripted measurement system
* ...

In order to use this **matlab** module, the following requirements have to be fulfilled:

* Matlab has to be installed in 32bit or 64bit, equal to |itom|
* The path **{MatlabRoot}/bin/win64** (or similar), containing the libraries **libeng.dll** and **libmx.dll** (or **libeng.so** and **libmx.so** under linux) must be in the PATH variable such that itom can find Matlab engine libraries during runtime. Please make sure, that the Qt bin directory is located in the PATH before the Matlab directory (ONLY if you have a self-compiled version of |itom|, for setup versions this restriction does not hold; The reason is, that the Matlab directory contains older versions of Qt libraries, too, that badly interfer with the itom Qt dependencies).
* If the matlab libraries could be loaded but the session (see below) could not be started, you have to register the COM components of Matlab. See this link (for Windows users): http://de.mathworks.com/help/matlab/matlab_external/register-matlab-as-automation-server.html

The following example is also included in the demo folder (**demoMatlabEngine.py**) and shows how to remotely control Matlab via itom:

.. code-blocks:: python
    
    try:
        import matlab
    except Exception as ex:
        print("itom is possibly compiled without Matlab support. This demo is not working")
        raise ex

    session = matlab.MatlabSession() #a Matlab console is opened

    session.setString("myString", "test") #creates the string variable 'myString' in the Matlab workspace with the value 'test'
    print("myString:", session.getString("myString")) #returns 'test' as answer in itom
    session.setValue("myArray", dataObject.randN([2,3],'int16')) #creates a 2x3 random matrix in Matlab (name: myArray)
    arr = session.getValue("myArray") #returns the 2x3 array 'myArray' from Matlab as Numpy array
    print(arr) 

    #read the current working directory of matlab
    session.run("curDir = cd")
    print(session.getString("curDir"))

    #run directly executes the command (as string). This is the same than typing this command into the command line of Matlab.
    #use this to also execute functions in Matlab. At first, send all required variables to the Matlab workspace, then execute a function
    #that uses these variables.

    del session #closes the session and deletes the instance

    #session.close() only closes the session
    
.. info::
    
    If Matlab is not properly installed, the opening of the MatlabSession will fail.
    
.. note::

    If the command **matlab.MatlabSesseion()** returns the **RuntimeError: error loading matlab engine: Cannot load library libeng.dll: The specified procedure could not be found.**, there is a version conflict between librarys loaded by itom/MatlabSession.     
    One known conflict happens (Matlab version 2015), while the libraries **icuio54.dll, icule54.dll, icuuc54.dll, icuin54.dll** are loaded by the **libeng.dll**.     
    One possible workaround is to copy the library files **icuio54.dll, icule54.dll, icuuc54.dll, icuin54.dll** from the **{MatlabRoot}/bin/win64** folder into the **{itomRoot}/lib** folder. 