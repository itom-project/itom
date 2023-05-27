.. include:: ../../include/global.inc

.. _plugin-retVal:

**RetVal** - The return type of |itom| methods
==============================================

The class **RetVal** is used for creating, transmitting and handling return values of methods in |itom| and plugins.

For using this class, include the file *retVal.h* from the folder *include/common* of the |itom| SDK directory

.. code-block:: c++

    #include "retVal.h"

and link against the library **itomCommonLib**, that is also contained in the SDK.

Any return value consists of the following main components:

* error state (enumeration ito::tRetValue)
    * retOk
    * retWarning
    * retError
* error number (user-defined error number; this number has no further functionality yet)
* error message (NULL or the warning or error message of the return value)

You can create a variable of type **RetVal** using different constructors or the static method **RetVal::format**.

.. cpp:function:: ito::RetVal::RetVal(tRetValue retValue = retOk)

    creates a return value with the default state *retOk*, if not otherwise stated. (constructor)

.. cpp:function:: ito::RetVal::RetVal(int retValue)

    creates a return value with the given state. (constructor)

.. cpp:function:: ito::RetVal::RetVal(ito::tRetValue retValue, int retCode, const char *pRetMessage)

    creates a return value with a given state, error number and error message. If you don't want to indicate an error message, set that value to NULL. (constructor)

.. cpp:function:: static ito::RetVal ito::RetVal::format(ito::tRetValue retValue, int retCode, const char *pRetMessage, ...)

    Use this static method to create a new return value where the message string can contain placeholders, known by the ordinary methods **sprintf**, ... The values filled into these placeholders are then appended as additional parameters to this method.

.. cpp:function:: RetVal & operator = (const RetVal rhs)

    By this operator you can assign a new return value to this return value.

Additionally you can use the mathematical operators **+** and **+=** to add a return value to an existing instance of class **RetVal**. The resulting return value keeps unchanged if the state of the added return value is less critical than the internal state, e.g. a state retOk is less critical than retError as well as retWarning is less critical than retWarning. If both states are the same, the return value is unchanged, too. In any other cases the state, error number and error message of the added return value is used for the new return value.

By this mechanism you can create chains of return values in the following form:

.. code-block:: c++

    ito::RetVal retValue = ito::retOk
    retValue += method1()
    retValue += method2()
    retValue += method3()

    if(retValue.containsError())
    {
        print("error while executing some methods");
    }

The actual status of the return value can be obtained using the following methods:

.. cpp:function:: int containsWarning()

    Returns true if the error state is equal to retWarning, that means the worst error state which has been added to this return value has been retWarning.

.. cpp:function:: int containsError()

    Returns true if the error state is equal to retError, that means the worst error state which has been added to this return value has been retError.

.. cpp:function:: int containsWarningOrError()

    Returns true if the worst error state, which has been added to this return value has been unequal to retOk.

.. cpp:function:: char *errorMessage()

    Returns a zero-terminated string containing the actual error-message of this return value or a zero-terminated, empty string
    if no message has been set (Caution: in |itom| <= 1.1.0 this method returned NULL in the latter case).

Additionally you can use the comparison operators **==** or **!=** to compare the error state of two return values or the error state of one return value with a given error state.

.. note::

    For a full reference of the class **RetVal** see :ref:`plugin-RetVal-Ref`.

.. toctree::
   :hidden:

   plugin-RetVal-Ref.rst
