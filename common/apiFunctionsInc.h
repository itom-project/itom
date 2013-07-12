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

	#define apiFilterGetFunc \
		(*(ito::RetVal (*)(const QString &, ito::AddInAlgo::FilterDef *&)) ito::ITOM_API_FUNCS[0])

	#define apiFilterCall \
		(*(ito::RetVal (*)(const QString &, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *)) ito::ITOM_API_FUNCS[1])

	#define apiFilterParam \
		(*(ito::RetVal (*)(const QString &, QVector<ito::Param> *, QVector<ito::Param> *, QVector<ito::Param> *)) ito::ITOM_API_FUNCS[2])

    #define apiFilterParamBase \
		(*(ito::RetVal (*)(const QString &, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *)) ito::ITOM_API_FUNCS[3])

	#define apiAddInGetInitParams \
		(*(ito::RetVal (*)(const QString &, const int, int *, QVector<ito::Param> *&, QVector<ito::Param> *&)) ito::ITOM_API_FUNCS[4])

	#define apiAddInOpenActuator \
		(*(ito::RetVal (*)(const QString &, const int, const bool, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *)) ito::ITOM_API_FUNCS[5])

	#define apiAddInOpenDataIO \
		(*(ito::RetVal (*)(const QString &, const int, const bool, QVector<ito::ParamBase> *, QVector<ito::ParamBase> *)) ito::ITOM_API_FUNCS[6])

    #define apiValidateStringMeta \
        (*(ito::RetVal (*)(const ito::StringMeta *meta, const char* value, bool mandatory)) ito::ITOM_API_FUNCS[7])

    #define apiValidateDoubleMeta \
        (*(ito::RetVal (*)(const ito::DoubleMeta *meta, double value)) ito::ITOM_API_FUNCS[8])

    #define apiValidateIntMeta \
        (*(ito::RetVal (*)(const ito::IntMeta *meta, int value)) ito::ITOM_API_FUNCS[9])

    #define apiValidateCharMeta \
        (*(ito::RetVal (*)(const ito::CharMeta *meta, char value)) ito::ITOM_API_FUNCS[10])

    #define apiValidateHWMeta \
        (*(ito::RetVal (*)(const ito::HWMeta *meta, ito::AddInBase *value, bool mandatory)) ito::ITOM_API_FUNCS[11])

    #define apiCompareParam \
        (*(ito::tCompareResult (*)(const ito::Param &paramTemplate, const ito::Param &param, ito::RetVal &ret)) ito::ITOM_API_FUNCS[12])
    
    #define apiValidateParam \
        (*(ito::RetVal (*)(const ito::Param &templateParam, const ito::ParamBase &param, bool strict, bool mandatory)) ito::ITOM_API_FUNCS[13])

    #define apiGetParamFromMapByKey \
        (*(ito::RetVal (*)(QMap<QString,ito::Param> &paramMap, const QString &key, QMap<QString,ito::Param>::iterator &found, bool errorIfReadOnly)) ito::ITOM_API_FUNCS[14])
    
    #define apiParseParamName \
        (*(ito::RetVal (*)(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag)) ito::ITOM_API_FUNCS[15])

    #define apiGetItemFromParamArray \
        (*(ito::RetVal (*)(const ito::Param &arrayParam, const int index, ito::Param &itemParam)) ito::ITOM_API_FUNCS[16])

    #define apiSaveQLIST2XML \
        (*(ito::RetVal (*)(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile)) ito::ITOM_API_FUNCS[17])

    #define apiLoadXML2QLIST \
        (*(ito::RetVal (*)(QMap<QString, ito::Param> *paramList , QString id, QFile &paramFile)) ito::ITOM_API_FUNCS[18])

#if defined(ITOM_IMPORT_API)
static int importItomApi(void** apiArray)
{
    ito::ITOM_API_FUNCS = apiArray;
    return 0;
}
#endif

};

#endif
