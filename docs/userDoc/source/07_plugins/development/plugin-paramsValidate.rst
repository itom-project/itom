.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-ParamsValidation:

Parameters - Validation
=======================

If a default-parameter (or let's say template) is given in form of an instance of class :cpp:class:`ito::Param` and have now a value in form of an instance
of class **ito::ParamBase**, you might be interested if the new value fits to the type and optionally the restrictions given by meta information of
the template.

Therefore, the class **ParamHelper** (folder *helper*) provide method to validate ParamBase-instances with respect to a given meta information
struct or to compare the compatibility of two different parameters.

.. note::
    
    Since **ParamHelper** is not directly available for plugins, the most important methods are also made available by the **API** functions
    (see :ref:`plugin-itomAPI`). Therefore the API-call for the validator functions is also indicated below.

Validate with meta information
------------------------------

If you have access to any meta information instance (derived from class **ParamMeta**), you can check whether the value of an instance of class
**ParamBase** or its derived class **Param** fits to the given requirements. There are different validator functions depending on the type of
meta information.

.. c:function:: static ito::RetVal validateStringMeta(const ito::StringMeta *meta, const char* value, bool mandatory = false)
    
    This methods checks whether the string, passed by argument *value*, fits to the requirements of the string meta information *meta*.
    If it does not fit, *retError* is returned with an appropriate error message. If argument *mandatory* is false, *retOk* is also returned
    if the string is not given, hence, value is an empty string.
    
    **API-Call:** ito::RetVal apiValidateStringMeta(-same arguments-)

.. c:function:: static ito::RetVal validateDoubleMeta(const ito::DoubleMeta *meta, double value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the double meta information 'meta'.
    If this is not the case *retError* with appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal apiValidateDoubleMeta(-same arguments-)

.. c:function:: static ito::RetVal validateIntMeta(const ito::IntMeta *meta, int value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the integer meta information 'meta'.
    If this is not the case *retError* with appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal apiValidateIntMeta(-same arguments-)

.. c:function:: static ito::RetVal validateCharMeta(const ito::CharMeta *meta, double value)
    
    This methods checks whether the number 'value' does not exceed the boundaries given by the char meta information 'meta'.
    If this is not the case *retError* with appropriate error message is returned, else *retOk*.
    
    **API-Call:** ito::RetVal apiValidateCharMeta(-same arguments-)

.. c:function:: static ito::RetVal validateHWMeta(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory = false)
    
    This method checks whether the plugin given by 'value' fits to the requirements possibily defined in the 'meta'-plugin meta
    information struct. If this is the case *retOk* is returned, else *retError* with an appropriate error message. If 'value'
    is NULL *retOk* is only returned if argument 'mandatory' is *false*.
    
    **API-Call:** ito::RetVal apiValidateHWMeta(-same arguments-)

.. c:function:: static ito::RetVal validateParam(const ito::Param &templateParam, const ito::ParamBase &param, bool strict = true, bool mandatory = false)
    
    This method uses the methods above to check whether the value of 'param' is valid with respect to the type and meta information of 'templateParam'.
    If 'strict' is *false*, the type is tried to be converted to type of 'templateParam' if possible. The 'mandatory' parameter is redirected to
    the corresponding validation methods above as therefore has the same meaning.
    
    **API-Call:** ito::RetVal apiValidateParam(-same arguments-)

