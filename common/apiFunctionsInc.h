/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

namespace ito 
{

    #if defined(ITOM_IMPORT_API) && !defined(ITOM_CORE)
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
		\param filterDef reference to a ito::AddInAlgo::FilterDef struct pointer. If the filter is found, this reference points to the struct defined by the plugin.
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
		\sa apiFilterParamBase
	*/
	#define apiFilterCall \
		(*(ito::RetVal (*)(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)) ito::ITOM_API_FUNCS[1])

	#define apiFilterParam \
		(*(ito::RetVal (*)(const QString &name, QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)) ito::ITOM_API_FUNCS[2])

    #define apiFilterParamBase \
		(*(ito::RetVal (*)(const QString &name, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)) ito::ITOM_API_FUNCS[3])

	#define apiAddInGetInitParams \
		(*(ito::RetVal (*)(const QString &name, const int, int *, QVector<ito::Param> *&, QVector<ito::Param> *&)) ito::ITOM_API_FUNCS[4])

	#define apiAddInOpenActuator \
		(*(ito::RetVal (*)(const QString &name, const int, const bool, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt)) ito::ITOM_API_FUNCS[5])

	#define apiAddInOpenDataIO \
		(*(ito::RetVal (*)(const QString &name, const int, const bool, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt)) ito::ITOM_API_FUNCS[6])

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
		\param value a double value that sould be verified
		\return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
		\sa apiValidateStringMeta, apiValidateIntMeta, apiValidateCharMeta, apiValidateHWMeta, ito::DoubleMeta
	*/
    #define apiValidateDoubleMeta \
        (*(ito::RetVal (*)(const ito::DoubleMeta *meta, double value)) ito::ITOM_API_FUNCS[8])

	//! validates an integer value with respect to given ito::IntMeta instance.
	/*!
		\param meta pointer to a ito::IntMeta instance describing the requirements for the number
		\param value an integer value that sould be verified
		\return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
		\sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateCharMeta, apiValidateHWMeta, ito::IntMeta
	*/
    #define apiValidateIntMeta \
        (*(ito::RetVal (*)(const ito::IntMeta *meta, int value)) ito::ITOM_API_FUNCS[9])

	//! validates a char value with respect to given ito::CharMeta instance.
	/*!
		\param meta pointer to a ito::CharMeta instance describing the requirements for the number
		\param value an integer value that sould be verified
		\return ito::RetVal (ito::retOk if the given value fits the requirements, else ito::retError)
		\sa apiValidateStringMeta, apiValidateDoubleMeta, apiValidateIntMeta, apiValidateHWMeta, ito::CharMeta
	*/
    #define apiValidateCharMeta \
        (*(ito::RetVal (*)(const ito::CharMeta *meta, char value)) ito::ITOM_API_FUNCS[10])

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
    
    #define apiValidateParam \
        (*(ito::RetVal (*)(const ito::Param &templateParam, const ito::ParamBase &param, bool strict, bool mandatory)) ito::ITOM_API_FUNCS[13])
	
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

	/** \} */


#if defined(ITOM_IMPORT_API)
static int importItomApi(void** apiArray)
{
    ito::ITOM_API_FUNCS = apiArray;
    return 0;
}
#endif

};

#endif
