.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-ParamsValidation:

Parameters - Validation
=======================

If a default-parameter (or let's say template) is given in form of an instance of class :cpp:class:`ito::Param` and a new value in terms of an instance
of class :cpp:class:`ito::ParamBase` is given, you might be interested if the new value fits to the type and the optional restrictions given by meta information of
the template.

Therefore, the class **ParamHelper** (folder *helper*) provide method to validate ParamBase-instances with respect to a given meta information
struct or to compare the compatibility of two different parameters.

.. note::
    
    Since **ParamHelper** is not directly available for plugins, the most important methods are also made available by the **API** functions
    (see :ref:`plugin-itomAPI`). Therefore the API-call for the validator functions is also indicated below.

Validation with respect to given meta information
---------------------------------------------------

If you have access to any meta information instance (derived from class **ParamMeta**), you can check whether the value of an instance of class
**ParamBase** or its derived class **Param** fits to the given requirements. There are different validator functions depending on the type of
meta information.

.. c:function:: static ito::RetVal validateCharMeta(const ito::CharMeta *meta, double value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the char meta information 'meta'.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal **apiValidateCharMeta** (const ito::CharMeta *meta, double value)

.. c:function:: static ito::RetVal validateIntMeta(const ito::IntMeta *meta, int value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the integer meta information 'meta'.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal **apiValidateIntMeta** (const ito::IntMeta *meta, int value)

.. c:function:: static ito::RetVal validateDoubleMeta(const ito::DoubleMeta *meta, double value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the double meta information 'meta'.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal **apiValidateDoubleMeta** (const ito::DoubleMeta *meta, double value)
    
.. c:function:: static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, double &value, bool allowRounding = true)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the double meta information 'meta'.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*. In difference to :c:func:`validateDoubleMeta`,
    this method rounds the given value to the nearest value, that fits to the step size contraints (if a step size != 0.0 is given).
    
    The rounding is only done, if *allowRounding* is equal to *true*, else *value* will not be modified and can be considered to be constant.
    
    **API-Call:** Right now, there is no direct API-function for this function, however :c:func:`validateAndCastParam` uses this function for double-based parameters.

.. c:function:: static ito::RetVal validateDoubleMetaAndRoundToStepSize(const ito::DoubleMeta *meta, ito::ParamBase &doubleParam, bool allowRounding = true)
    
    This method is equal to the method described above. The single difference is that the value to test is not passed as single double value but as
    parameter that must have the type ito::ParamBase::Double.

.. c:function:: static ito::RetVal validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory = false)
    
    This methods checks whether the string, passed by argument *value*, fits to the requirements of the string meta information *meta*.
    If it does not fit, *retError* is returned with an appropriate error message. If argument *mandatory* is false, *retOk* is also returned
    if the string is not given, hence, value is an empty string.
    
    **API-Call:** ito::RetVal **apiValidateStringMeta** (const ito::StringMeta *meta, const char* value, bool mandatory = false)

.. c:function:: static ito::RetVal validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory = false)
    
    This method checks whether the plugin given by 'value' fits to the requirements possibly defined in the 'meta'-plugin meta
    information struct. If this is the case *retOk* is returned, else *retError* with an appropriate error message. If 'value'
    is NULL *retOk* is only returned if argument 'mandatory' is *false*.
    
    **API-Call:** ito::RetVal **apiValidateHWMeta** (const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory = false)
    
.. c:function:: static ito::RetVal validateCharArrayMeta(const ito::ParamMeta *meta, const char* values, size_t len)
    
    This methods checks whether the array of character values named 'values' (length 'len') fits to the requirements given 
    by the meta information of *meta*. This can be both restrictions with respect to every single values as well as the length of the array.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal **apiValidateCharArrayMeta** (const ito::ParamMeta *meta, const char* values, size_t len)
    
.. c:function:: static ito::RetVal validateIntArrayMeta(const ito::ParamMeta *meta, const int* values, size_t len)
    
    This methods checks whether the array of integer values named 'values' (length 'len') fits to the requirements given 
    by the meta information of *meta*. This can be both restrictions with respect to every single values as well as the length of the array.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    If the meta information indicates that an interval, range or rectangle is expected, the specific validations and tests are done, too.
    
    **API-Call:** ito::RetVal **apiValidateIntArrayMeta** (const ito::ParamMeta *meta, const int* values, size_t len)
    
.. c:function:: static ito::RetVal validateDoubleArrayMeta(const ito::ParamMeta *meta, const double* values, size_t len)
    
    This methods checks whether the array of double values named 'values' (length 'len') fits to the requirements given 
    by the meta information of *meta*. This can be both restrictions with respect to every single values as well as the length of the array.
    If this is not the case *retError* with an appropriate error message is returned, else *retOk*.
    
    If the meta information indicates that an interval is expected, the specific validations and tests are done, too.
    
    **API-Call:** ito::RetVal **apiValidateDoubleArrayMeta** (const ito::ParamMeta *meta, const double* values, size_t len)

Overall validation methods
----------------------------

The following functions are overall functions that use the methods described above, depending on the type of the given template and parameter.

.. c:function:: static ito::RetVal validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict = true, bool mandatory = false)
    
    This method uses the methods above to check whether the value of 'param' is valid with respect to the type and meta information of 'templateParam'.
    If 'strict' is *false*, the type is tried to be converted to type of 'templateParam' if possible. The 'mandatory' parameter is redirected to
    the corresponding validation methods above as therefore has the same meaning.
    
    **API-Call:** ito::RetVal **apiValidateParam**(-same arguments-)
    
.. c:function:: static ito::RetVal validateAndCastParam(const ito::Param &templateParam, ito::ParamBase &param, bool strict = true, bool mandatory = false, bool roundToSteps = false)
    
    This method uses the methods above to check whether the value of 'param' is valid with respect to the type and meta information of 'templateParam'.
    If 'strict' is *false*, the type is tried to be converted to type of 'templateParam' if possible. The 'mandatory' parameter is redirected to
    the corresponding validation methods above as therefore has the same meaning.
    
    The difference to :c:func:`validateParam` is that the given argument *param* can be changed to the same type than *templateParam* if they can be cast
    and if it is a double value, its real value can be adapted to fit any step size constraints (only if *roundToSteps = true*).
    
    **API-Call:** ito::RetVal **apiValidateAndCastParam** (-same arguments-)

