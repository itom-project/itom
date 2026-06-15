/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef APIFUNCTIONSINC_H
#define APIFUNCTIONSINC_H

#ifndef Q_MOC_RUN
namespace ito
{
    #if defined(ITOM_IMPORT_API) //&& !defined(ITOM_CORE)
        void **ITOM_API_FUNCS;
    #else
        extern void **ITOM_API_FUNCS;
    #endif

    /**
    * \defgroup ITOM_API itom Standard API
    *
    * \brief The itom standard API contains a bunch of functions that can be called by the core application itom as well as by
    * every plugin or designer plugin.
    *
    * Every function is given by a certain preprocessor definition that describes
    * the return value, parameters and function name of the method to call. Each preprocessor definition is redirected
    * to a function pointer that becomes valid at runtime. The initialization of these function pointers in any plugins
    * is an automatic process by itom, called when loading the plugin.
    *
    * How to read the following definitions?
    *
    * Basically the first word after the #define word is the method to call. After the first star the return value
    * follows (the star into brackets is not part of the return value). Then there is a list of different parameters
    * for this method.
    *
    * \{
    */

    //! looks for a given filter (algorithm) defined in any algo plugin.
    /*!
        \param name is the name of the desired filter or algorithm
        \param filterDef reference to a ito::AddInAlgo::FilterDef struct pointer.
               If the filter is found, this reference points to the struct defined by the plugin.
        \return ito::RetVal (ito::retOk if filter has been found, else ito::retError)
        \sa apiFilterCall, apiFilterParam
    */
    #define apiFilterGetFunc \
        (*(ito::RetVal (*)(const QString &name, ito::AddInAlgo::FilterDef *&filterDef)) ito::ITOM_API_FUNCS[0])

    //! calls a filter (algorithm) defined in any algo plugin.
    /*!
        This api method calls another filter given by its specific name in any other plugin.
        The call is executed in the calling thread. You need to make sure, that all parameters that
        you provide as vectors to this function fully correspond to the number and type of the desired
        parameters. You can get the default vectors using the api method apiFilterParamBase.

        \param name is the name of the desired filter or algorithm
        \param paramsMand is a pointer to a vector of type ParamBase containing all mandatory parameters.
        \param paramsOpt is a pointer to a vector of type ParamBase containing all optional parameters.
        \param paramsOut is a pointer to a vector of type ParamBase containing all output parameters.
        \return ito::RetVal (ito::retOk if filter has been found, else ito::retError)
        \sa apiFilterParamBase, apiFilterCallExt
    */
    #define apiFilterCall \
        (*(ito::RetVal (*)(const QString &name, QVector<ito::ParamBase> *paramsMand, \
        QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)) ito::ITOM_API_FUNCS[1])

    //! calls a filter (algorithm) with an extended interface defined in any algo plugin.
    /*!
        This api method calls another filter given by its specific name in any other plugin.
        The call is executed in the calling thread. You need to make sure, that all parameters that
        you provide as vectors to this function fully correspond to the number and type of the desired
        parameters. You can get the default vectors using the api method apiFilterParamBase.

        Filters, that are called by this method, must implement the FilterDefExt interface, that
        has an additional observer parameter for the progress observation and possible cancellation
        of the filter call.

        \param name is the name of the desired filter or algorithm
        \param paramsMand is a pointer to a vector of type ParamBase containing all mandatory parameters.
        \param paramsOpt is a pointer to a vector of type ParamBase containing all optional parameters.
        \param paramsOut is a pointer to a vector of type ParamBase containing all output parameters.
        \param observer is the initialized observer of class ito::FunctionCancellationAndObserver.
        \return ito::RetVal (ito::retOk if filter has been found, else ito::retError)
        \sa apiFilterParamBase, apiFilterCall
    */
    #define apiFilterCallExt \
        (*(ito::RetVal (*)(const QString &name, QVector<ito::ParamBase> *paramsMand, \
        QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut, \
        QSharedPointer<ito::FunctionCancellationAndObserver> observer)) ito::ITOM_API_FUNCS[38])

    #define apiFilterParam \
        (*(ito::RetVal (*)(const QString &name, QVector<ito::Param> *paramsMand, \
        QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)) ito::ITOM_API_FUNCS[2])

    #define apiFilterParamBase \
        (*(ito::RetVal (*)(const QString &name, QVector<ito::ParamBase> *paramsMand, \
        QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)) ito::ITOM_API_FUNCS[3])

    #define apiFilterVersion \
        (*(ito::RetVal (*)(const QString &name, int &version)) ito::ITOM_API_FUNCS[35])

    #define apiFilterAuthor \
        (*(ito::RetVal (*)(const QString &name, QString &author)) ito::ITOM_API_FUNCS[36])

    #define apiFilterPluginName \
        (*(ito::RetVal (*)(const QString &name, QString &pluginName)) ito::ITOM_API_FUNCS[37])

    //! checks if a specific plugin is available and returns its default mandatory and optional parameter set for the initialization
    /*!
        This method checks if a specific plugin from a given plugin type is available. If so, retOk is returned and the arguments
        pluginIdx, paramsMand and paramsOpt are filled with valid values. Else, retError is returned.

        \param name is the plugin name
        \param pluginType is the base type of the plugin (ito::tPluginType, e.g. ito::typeDataIO, ito::typeActuator...)
        \param pluginIdx is the index of the found plugin that can be passed apiAddInOpenDataIO, apiAddInActuator
        \param paramsMand are the templates for the mandatory parameters for initializing the plugin.
               Please copy all values to QVector<ito::ParamBase> before editing the values and passing them to the initialization.
        \param paramsOpt are the templates for the optional parameters for initializing the plugin.
               Please copy all values to QVector<ito::ParamBase> before editing the values and passing them to the initialization.
    */
    #define apiAddInGetInitParams \
        (*(ito::RetVal (*)(const QString &pluginName, const int pluginType, int *pluginIdx, \
        QVector<ito::Param> *&paramsMand, QVector<ito::Param> *&paramsOpt)) ito::ITOM_API_FUNCS[4])

    //! opens an instance of specific actuator plugin.
    /*!
        This method let itom create an instance of a specific plugin (given by pluginName AND pluginIdx) and sets instance to the pointer of the newly created plugin.

        \param pluginName is the name of the plugin to open
        \param pluginIdx is the index of the plugin to open in its internal index list (obtained by apiAddInGetInitParams)
        \param autoLoadParams indicates if plugin parameters from last session should be automatically loaded at restart (only if m_autoLoadPolicy is set to autoLoadKeywordDefined in plugin). If unsure, set it to false.
        \param paramsMand are the mandatory parameters passed to the initialization method of the plugin (get their templates from apiAddInGetInitParams)
        \param paramsOpt are the optional parameters (similiar to paramsMand)
        \param instance is the pointer to ito::AddActuator that contains the instance of the recently opened plugin (if successfully opened)
        \return ito::RetVal (retOk if plugin instance could be loaded, else retError)

        If you want to have a thread-safe and easier approach to other plugins, consider to use the helper class ActuatorThreadCtrl.
    */
    #define apiAddInOpenActuator \
        (*(ito::RetVal (*)(const QString &pluginName, const int pluginIdx, const bool autoLoadParams, \
        QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, \
        ito::AddInActuator *&instance)) ito::ITOM_API_FUNCS[5])

    //! opens an instance of specific dataIO plugin.
    /*!
        This method let itom create an instance of a specific plugin (given by pluginName AND pluginIdx) and sets instance to the pointer of the newly created plugin.

        \param pluginName is the name of the plugin to open
        \param pluginIdx is the index of the plugin to open in its internal index list (obtained by apiAddInGetInitParams)
        \param autoLoadParams indicates if plugin parameters from last session should be automatically loaded at restart (only if m_autoLoadPolicy is set to autoLoadKeywordDefined in plugin). If unsure, set it to false.
        \param paramsMand are the mandatory parameters passed to the initialization method of the plugin (get their templates from apiAddInGetInitParams)
        \param paramsOpt are the optional parameters (similiar to paramsMand)
        \param instance is the pointer to ito::AddInDataIO that contains the instance of the recently opened plugin (if successfully opened)
        \return ito::RetVal (retOk if plugin instance could be loaded, else retError)

        If you want to have a thread-safe and easier approach to other plugins, consider to use the helper class DataIOThreadCtrl.
    */
    #define apiAddInOpenDataIO \
        (*(ito::RetVal (*)(const QString &pluginName, const int pluginIdx, const bool autoLoadParams, \
        QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, \
        ito::AddInDataIO *&instance)) ito::ITOM_API_FUNCS[6])

    //! decrements reference of given plugin instance. If the reference drops to zero, the instance is savely closed and deleted.
    /*!
        This method does not wait for the plugin to be closed since this might cause deadlocks if called from the close method or destructor of another plugin.
        The AddIn manager will close the given instance as soon as possible.
    */
    #define apiAddInClose \
        (*(ito::RetVal (*)(ito::AddInBase *instance)) ito::ITOM_API_FUNCS[31])

    //! validates a zero-terminated string with respect to given ito::StringMeta instance.
    /*!
        \param meta pointer to a ito::StringMeta instance describing the requirements for the string
        \param value a zero-terminated string that should be verified
        \param mandatory if true, an error is returned if the string is empty, else an empty string returns ito::retOk
        \return ito::RetVal (ito::retOk if the given string fits the requirements, else ito::retError)
        \sa apiValidateDoubleMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, ito::StringMeta
    */
    #define apiValidateStringMeta \
        (*(ito::RetVal (*)(const ito::StringMeta *meta, const char* value, bool mandatory)) ito::ITOM_API_FUNCS[7])

    //! validates a double value with respect to given ito::DoubleMeta instance.
    /*!
        \param meta pointer to a ito::DoubleMeta instance describing the requirements for the number
        \param value a double value that should be verified
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, ito::DoubleMeta
    */
    #define apiValidateDoubleMeta \
        (*(ito::RetVal (*)(const ito::DoubleMeta *meta, double value)) ito::ITOM_API_FUNCS[8])

    //! validates an integer value with respect to given ito::IntMeta instance.
    /*!
        \param meta pointer to a ito::IntMeta instance describing the requirements for the number
        \param value an integer value that should be verified
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateCharMeta, apiValidateHWMeta, ito::IntMeta
    */
    #define apiValidateIntMeta \
        (*(ito::RetVal (*)(const ito::IntMeta *meta, int value)) ito::ITOM_API_FUNCS[9])

    //! validates a char value with respect to given ito::CharMeta instance.
    /*!
        \param meta pointer to a ito::CharMeta instance describing the requirements for the number
        \param value an integer value that should be verified
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateHWMeta, ito::CharMeta
    */
    #define apiValidateCharMeta \
        (*(ito::RetVal (*)(const ito::CharMeta *meta, char value)) ito::ITOM_API_FUNCS[10])

//! validates a double value with respect to given ito::DoubleMeta instance.
    /*!
        \param meta pointer to a ito::ParamMeta instance (must be DoubleArrayMeta, DoubleIntervalMeta) describing the requirements for the values
        \param values a double array that should be verified
        \param len is the length of the array
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, ito::DoubleArrayMeta
    */
    #define apiValidateDoubleArrayMeta \
        (*(ito::RetVal (*)(const ito::ParamMeta *meta, const double* values, size_t len)) ito::ITOM_API_FUNCS[28])

    //! validates an array of integer values with respect to given ito::ParamMeta instance.
    /*!
        \param meta pointer to a ito::ParamMeta instance (must be IntArrayMeta, IntervalMeta, RangeMeta, RectMeta) describing the requirements for the values
        \param values an integer array that should be verified
        \param len is the length of the array
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateCharMeta, apiValidateHWMeta, ito::IntArrayMeta, ito::RangeMeta, ito::IntervalMeta, ito::RectMeta
    */
    #define apiValidateIntArrayMeta \
        (*(ito::RetVal (*)(const ito::ParamMeta *meta, const int* values, size_t len)) ito::ITOM_API_FUNCS[26])

    //! validates an array of char values with respect to given ito::ParamMeta instance.
    /*!
        \param meta pointer to a ito::ParamMeta instance (must be CharArrayMeta) describing the requirements for the values
        \param values a character array that should be verified
        \param len is the length of the array
        \return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateHWMeta, ito::CharArrayMeta
    */
    #define apiValidateCharArrayMeta \
        (*(ito::RetVal (*)(const ito::ParamMeta *meta, const char* values, size_t len)) ito::ITOM_API_FUNCS[25])

    //! validates a plugin pointer with respect to given ito::HWMeta instance.
    /*!
        The hardware pointer is an instance of an opened hardware plugin (dataIO, actuator)

        \param meta pointer to a ito::HWMeta instance describing the requirements for the given plugin
        \param value an instance of a plugin (inherited from ito::AddInBase) that should be verified
        \return ito::RetVal (ito::retOk if the given instance fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateCharMeta, ito::HWMeta
    */
    #define apiValidateHWMeta \
        (*(ito::RetVal (*)(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory)) ito::ITOM_API_FUNCS[11])

    #define apiCompareParam \
        (*(ito::tCompareResult (*)(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret)) ito::ITOM_API_FUNCS[12])

    //! checks whether a given parameter fits to a template parameter.
    /*!
        This method checks whether a parameter param fits to the requirements of a template parameter.

        At first the types of param and templateParam are checked. If they are not equal or not compatible (in case of strict==false), an
        error with an appropriate error message is returned. After this, the value of param is checked with respect to the meta information,
        that are optionally given in templateParam. If param does not fit to these requirements, an error is returned, too.

        If templateParam is an array type and param is the corresponding non-array type, the validation succeeds if the name of param
        is an index-base name, e.g. myParam[0]. Then it is assumed, that param is the value at the first position of templateParam.

        \param templateParam is the template parameter. Its type as well as optionally available meta information is used for the check
        \param param is the parameter to check. Only the current value and its type is used.
        \param strict indicates whether the types of param and templateParam must exactly fit (true) or if compatible types (e.g. int vs. double) are allowed as well (false).
        \param mandatory is a boolean value indicating if the given parameter must contain a value (NULL as value for strings or dataObjects is no value).
        \return ito::RetVal (ito::retOk if the given instance fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, apiValidateAndCastParam
    */
    #define apiValidateParam \
        (*(ito::RetVal (*)(const ito::Param &templateParam, const ito::ParamBase &param, bool strict, bool mandatory)) ito::ITOM_API_FUNCS[13])

    //! checks whether a given parameter fits to a template parameter and possibly casts the param to the required type.
    /*!
        This method checks whether a parameter param fits to the requirements of a template parameter.

        At first the types of param and templateParam are checked. If they are not equal or not compatible (in case of strict==false), an
        error with an appropriate error message is returned. After this, the value of param is checked with respect to the meta information,
        that are optionally given in templateParam. If param does not fit to these requirements, an error is returned, too.

        If templateParam is an array type and param is the corresponding non-array type, the validation succeeds if the name of param
        is an index-base name, e.g. myParam[0]. Then it is assumed, that param is the value at the first position of templateParam.

        \param templateParam is the template parameter. Its type as well as optionally available meta information is used for the check
        \param param is the parameter to check. Only the current value and its type is used.
        \param strict indicates whether the types of param and templateParam must exactly fit (true) or if compatible types (e.g. int vs. double) are allowed as well (false).
        \param mandatory is a boolean value indicating if the given parameter must contain a value (NULL as value for strings or dataObjects is no value).
        \param roundToStep if true, double parameters are checked with respect to the given min,max and increment. If the given value does not fit to the increment, it is cast to the next allowed value.
        \return ito::RetVal (ito::retOk if the given instance fits the requirements, else ito::retError)
        \sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, apiValidateParam
    */
    #define apiValidateAndCastParam \
        (*(ito::RetVal (*)(const ito::Param &templateParam, const ito::ParamBase &param, bool strict, bool mandatory, bool roundToStep)) ito::ITOM_API_FUNCS[25])

    //! Finds reference to desired parameter in parameter map of any plugin
    /*!
        The parameters of all plugins are contained in the map ito::AddInBase::m_params that
        maps the parameter name to its value (ito::Param). This method tries to find a given parameter name
        in this map and if found, returns its reference by providing an iterator to the specific value.

        \param paramMap is the reference to th entire parameter map
        \param key is the parameter name that should be found (key-only, no index or additional tag)
        \param iterator if parameter has been found, this is an iterator to the specific parameter
        \param errorIfReadonly returns ito::retError if the found parameter is read-only (with specific error message). This can be used in the ito::AddInBase::setParam method of any plugin.
        \return ito::RetVal (ito::retOk if parameter has been found (and optionally is readable), else ito::retError)
    */
    #define apiGetParamFromMapByKey \
        (*(ito::RetVal (*)(QMap<QString,ito::Param> &paramMap, const QString &key, QMap<QString,ito::Param>::iterator &found, bool errorIfReadOnly)) ito::ITOM_API_FUNCS[14])

    //! parses a parameter name and splits it into specific parts
    /*!
        When reading or writing a parameter of any plugin instance, the parameter name must have a specific form:

        - paramName
        - paramName[index]
        - paramName:additionalTag
        - paramName[index]:additionalTag

        This method checks the given name and splits it into the parts. If the name does not fit to the stated
        form, ito::retError is returned.

        \param name is the full parameter name
        \param paramName is the simple parameter name
        \param hasIndex true if index is given, else false
        \param index is the provided integer index value
        \param additionalTag is any suffix provided after the colon-sign
        \return ito::RetVal (ito::retOk if parameter name has the desired form, else ito::retError)
    */
    #define apiParseParamName \
        (*(ito::RetVal (*)(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag)) ito::ITOM_API_FUNCS[15])

    #define apiGetItemFromParamArray \
        (*(ito::RetVal (*)(const ito::Param &arrayParam, const int index, ito::Param &itemParam)) ito::ITOM_API_FUNCS[16])

    #define apiGetParam \
        (*(ito::Param (*)(const ito::Param &param, const bool hasIndex, const int index, ito::RetVal &ret)) ito::ITOM_API_FUNCS[20])

    //! update all values in paramMap
    /*!
        For each value in vector values, their corresponding entry in paramMap is searched and if available set to the value given in the
        values vector. No validation besides a general type validation is done.

        \param paramMap is the map with parameters to set
        \param values are the new values, their name is used as keyword for paramMap
        \return ito::RetVal (ito::retOk if all parameters could be set, else ito::retError)
    */
    #define apiUpdateParameters \
        (*(ito::RetVal (*)(QMap<QString, ito::Param> &paramMap, const QVector<QSharedPointer<ito::ParamBase> > &values)) ito::ITOM_API_FUNCS[23])

    #define apiSaveQLIST2XML \
        (*(ito::RetVal (*)(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile)) ito::ITOM_API_FUNCS[17])

    #define apiLoadXML2QLIST \
        (*(ito::RetVal (*)(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile)) ito::ITOM_API_FUNCS[18])

    //! returns a shallow or deep copy of a given data object that fits to given requirements
    /*!
        Use this simple api method to test a given data object if it fits some requirements.
        If this is the case, a shallow copy of the input data object is returned. Else, it is
        tried to convert into the required object and a converted deep copy is returned. If the
        input object does not fit the given requirements, NULL is returned and the ito::RetVal
        parameter contains an error status including error message.

        \note In any case you need to delete the returned data object

        \param dObj the input data object
        \param nrDims the required number of dimensions
        \param type the required type of the returned data object
        \param sizeLimits can be NULL if the sizes should not be checked, else it is an array with length (2*nrDims). Every adjacent pair describes the minimum and maximum size of each dimension.
        \param retval can be a pointer to an instance of ito::RetVal or NULL. If given, the status of this method is added to this return value.
        \return shallow or deep copy of the input data object or NULL (in case of unsolvable incompatibility)
    */
    #define apiCreateFromDataObject \
        (* (ito::DataObject* (*)(const ito::DataObject *dObj, int nrDims, ito::tDataType type, int *sizeLimits, ito::RetVal *retval)) ito::ITOM_API_FUNCS[19])

    //! returns a shallow or deep copy of a given data object that fits to given requirements
    /*!
        Use this simple api method to test a given data object if it fits some requirements.
        If this is the case, a shallow copy of the input data object is returned. Else, it is
        tried to convert into the required object and a converted deep copy is returned. If the
        input object does not fit the given requirements, NULL is returned and the ito::RetVal
        parameter contains an error status including error message.

        \note In any case you need to delete the returned data object

        \param dObj the input data object
        \param nrDims the required number of dimensions
        \param type the required type of the returned data object
        \param name name of the data object for an improved error message (zero-terminated string) or NULL if no name is known.
        \param sizeLimits can be NULL if the sizes should not be checked, else it is an array with length (2*nrDims). Every adjacent pair describes the minimum and maximum size of each dimension.
        \param retval can be a pointer to an instance of ito::RetVal or NULL. If given, the status of this method is added to this return value.
        \return shallow or deep copy of the input data object or NULL (in case of unsolvable incompatibility)
    */
    #define apiCreateFromNamedDataObject \
        (* (ito::DataObject* (*)(const ito::DataObject *dObj, int nrDims, ito::tDataType type, const char *name, int *sizeLimits, ito::RetVal *retval)) ito::ITOM_API_FUNCS[24])

    //! returns the current working directory of itom
    /*!
        The current working directory is the current working directory of both python and itom itself. Its absolute path is returned as string.

        \return current working directory
    */
    #define apiGetCurrentWorkingDir \
        (* (QString (*)(void)) ito::ITOM_API_FUNCS[21])

    //! helper function to show and process a configuration dialog of a plugin
    /*!
        Use this simple api method in the method showConfDialog of a plugin to show and process
        the configuration dialog, whose instance is passed by configDialogInstance. This api function
        shows the dialog, passes the current parameters of the plugin and processes the changes. The given
        dialog instance is automatically deleted at the end.

        \param plugin the instance of the plugin itself
        \param configDialogInstance a new instance of the configuration dialog inherited from ito::AbstractAddInConfigDialog
    */
    #define apiShowConfigurationDialog \
        (* (ito::RetVal (*)(ito::AddInBase *plugin, ito::AbstractAddInConfigDialog *configDialogInstance)) ito::ITOM_API_FUNCS[22])


    // function moved to apiFunctionsGui
    //! sends the given ParamBase value to the global python workspace
    /*!
        This methods sends the given ParamBase value to the global python workspace using the indicated variable name. Existing
        values with the same name will be overwritten, unless they cover functions, methods, classes or types.

        Invalid variable name will also result in an error.

        \param varname is the variable name (must be a valid python variable name)
        \param value is the ParamBase value

        \return ito::retOk on success, else ito::retError
    */
//    #define apiSendParamToPyWorkspace \
//        (* (ito::RetVal (*)(const QString &varname, const QSharedPointer<ito::ParamBase> &value)) ito::ITOM_API_FUNCS[29])

    // function moved to apiFunctionsGui
    //! sends the given ParamBase value to the global python workspace
    /*!
        This methods sends the given ParamBase values to the global python workspace using the indicated variable names. Existing
        values with the same name will be overwritten, unless they cover functions, methods, classes or types.

        Invalid variable names will also result in an error.

        \param varnames are the variable name (must be valid python variable names)
        \param values are the ParamBase values

        \return ito::retOk on success, else ito::retError
    */
//    #define apiSendParamsToPyWorkspace \
//        (* (ito::RetVal (*)(const QStringList &varnames, const QVector<QSharedPointer<ito::ParamBase> > &values)) ito::ITOM_API_FUNCS[30])

    // function moved to apiFunctionsGui
    //! read a property from an QObject based instance.
    /*!
        \param object is the object
        \propName is the name of the property
        \value is a reference to the obtained value

        \return ito::retOk if property could be found and read, else ito::retError
    */
//    #define apiQObjectPropertyRead \
//        (* (ito::RetVal (*)(const QObject *object, const char* propName, QVariant &value)) ito::ITOM_API_FUNCS[32])

    // function moved to apiFunctionsGui
    //! write a property to an QObject based instance.
    /*!
    \param object is the object
    \propName is the name of the property
    \value is the value to set

    \return ito::retOk if property could be found and written, else ito::retError
    */
//    #define apiQObjectPropertyWrite \
//        (* (ito::RetVal (*)(QObject *object, const char* propName, const QVariant &value)) ito::ITOM_API_FUNCS[33])

    //! Get itom / user's settings file name.
    /*!

    \return QString settings file
    */
    #define apiGetSettingsFile \
            (* (QString (*)(void)) ito::ITOM_API_FUNCS[34])

    /** \} */


//#if defined(ITOM_IMPORT_API)
//static int importItomApi(void** apiArray)
//{
//    ito::ITOM_API_FUNCS = apiArray;
//    return 0;
//}
//#endif

};

#endif //Q_MOC_RUN

#endif
