.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-itomAPI:

itom API
========

Usually, plugins only have access to the sources which are defined in the **SDK** of |itom|. These are for instance the header and
source files contained in the folder **common** (e.g. **addInInterface.h**) and in the folder **plot** (only relevant for developing designer
plugins (hence plots, figures...). Further, the **SDK** contains the header-files and corresponding libraries for the **dataObject** and
the **pointCloud** library.

However, it is not desired at all, that plugins also include header or source files from |itom| itself. The intention
is, that it should be possible to develop plugins without the need to compile |itom| from its sources or have the source
code available.

Therefore, |itom| provides an **application programming interface** (API). The API has methods defined, which give plugins the possibility
to use important functionality of |itom|. All available methods of the **API** are defined in the file **apiFunctionsInc.h**, that also lies
in the folder **common** of the SDK. For methods concerning plots and figures, there is an additional API definition 
file **apiFunctionsGraphInc.h**.

Initialization
--------------

If your plugin or designer plugin widget is derived from classes **AddInBase** or **AbstractFigure**, which is usually the case,
you can use the **API** methods without further steps or includes.

When you are programming a plugin, derived from **AddInBase**, you can access any **API**-method at any time in your plugin (even in the constructor).
When programming a designer plugin widget that is handled as plotting plugin (derived from **AbstractFigure**) the APIs only become valid after the
construction of your plugin. 

.. note::

    Due to the software structure, the valid pointer is transmitted by |itom| sending the event with type **QEvent::User+123**,
    that is caught by your plugin after the construction.

Usage
-----

You can call the methods, defined in **apiFunctionsInc.h** or **apiFunctionsGraphInc.h** like normal function calls. In the following example, the filter **saveRPM**, which is defined
in another plugin should be called. Since, the mandatory, optional and output parameters of this filter are unknown, we will first request their default
values, then change their values and finally we call the filter-method. If you have knownledge about the parameters, you just can implement the second part.

.. code-block:: c++
    
    ito::RetVal retVal;
    QVector<ito::ParamBase> mandParams, optParams, outParams; //define three, empty parameter vectors
    
    //now get default values
    retVal = apiFilterParamBase("saveRPM", &mandParams,&optParams,&outParams);
    //retVal is retOk if the filter 'saveRPM' could be found. If the external plugin does not exist, the return value is retError.
    
    //change the value of some values
    mandParams[0].setVal<void*>( yourDataObjectPtr ); //set data object to save as first mandatory parameter
    mandParams[1].setVal<char*>( "C:\\test.rpm" );
    
    //now call the filter
    retVal += apiFilterCall("saveRPM", &mandParams, &optParams, &outParams);
    
    if(!retVal.containsError())
    {
        //filter could be successfully executed
        //you can now evaluate the output parameters, if the filter provided some.
    }
