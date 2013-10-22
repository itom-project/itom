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

#include "helperCommon.h"

#include <qobject.h>
#include <qmap.h>
#include <qstringlist.h>
#include <qsharedpointer.h>




namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
	//! checks param vector
	/*!
		\param [in] params is a pointer to QVector<ito::Param>. This pointer is checked.
		\return ito::RetVal, that contains an error if params is NULL
	*/
    ito::RetVal checkParamVector(QVector<ito::Param> *params)
    {
        if (params == NULL)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("parameter vector is not initialized").toAscii().data());
        }
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
	//! verifies that the three param vectors are not NULL
	/*!
		If any of the given input parameters of type QVector<ito::Param>* are NULL, a ito::RetVal is returned,
		that contains an error. Use this method in any algorithm-method in order to check the given input.

		\param [in] paramsMand is the first parameter vector
		\param [in] paramsOpt is the second parameter vector
		\param [in] paramsOut is the third parameter vector
		\return ito::RetVal, that contains an error if params is NULL
		\sa checkParamVector
	*/
    ito::RetVal checkParamVectors(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
    {
        if (paramsMand == NULL)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("mandatory parameter vector is not initialized").toAscii().data());
        }
        if (paramsOpt == NULL)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("optional parameter vector is not initialized").toAscii().data());
        }
        if (paramsOut == NULL)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("output parameter vector is not initialized").toAscii().data());
        }
        paramsMand->clear();
        paramsOpt->clear();
        paramsOut->clear();
        return ito::retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
	//! brief returns a parameter from the parameter-vector, that fits to a specific name
	/*!
		

		\param name description
		\return 
	*/
    ito::Param* getParamByName(QVector<ito::Param> *paramVec, const char* name, ito::RetVal *retval)
    {
        const char *temp;
        if (paramVec)
        {
            ito::Param* data = paramVec->data();

            for (int i = 0; i < paramVec->size(); ++i)
            {
                temp = data[i].getName();
                if (strcmp(temp,name) == 0)
                {
                    return &(data[i]);
                }
            }
        }
        if (retval) *retval += ito::RetVal::format(ito::retError, 0 , QObject::tr("parameter '%1' cannot be found in given parameter vector").arg(name).toAscii().data());
        return NULL;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::ParamBase* getParamByName(QVector<ito::ParamBase> *paramVec, const char* name, ito::RetVal *retval)
    {
        const char *temp;
        if (paramVec)
        {
            ito::ParamBase* data = paramVec->data();

            for (int i = 0; i < paramVec->size(); ++i)
            {
                temp = data[i].getName();
                if (strcmp(temp,name) == 0)
                {
                    return &(data[i]);
                }
            }
        }
        if (retval) *retval += ito::RetVal::format(ito::retError, 0 , QObject::tr("parameter '%1' cannot be found in given parameter vector").arg(name).toAscii().data());
        return NULL;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    QHash<QString, ito::Param*> createParamHashTable(QVector<ito::Param> *paramVec)
    {
        QHash<QString, ito::Param*> hashTable;
        if (paramVec)
        {
            ito::Param* data = paramVec->data();

            for (int i = 0; i < paramVec->size(); ++i)
            {
                hashTable.insert(data[i].getName() , &(data[i]));
            }
        }
        return hashTable;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    bool checkNumericParamRange(const ito::Param &param, double value, bool *ok)
    {
        bool done = false;
        bool result = false;

        if (param.isNumeric())
        {
            const ito::ParamMeta *meta = param.getMeta();
            if (meta)
            {
                done = true;
                switch(meta->getType())
                {
                case ito::Param::Char:
                    {
                        const ito::CharMeta *cMeta = (const ito::CharMeta*)meta;
                        if (value >= cMeta->getMin() && value <= cMeta->getMax()) result = true;
                    }
                    break;
                case ito::Param::Int:
                    {
                        const ito::IntMeta *iMeta = (const ito::IntMeta*)meta;
                        if (value >= iMeta->getMin() && value <= iMeta->getMax()) result = true;
                    }
                    break;
                case ito::Param::Double:
                    {
                        const ito::DoubleMeta *dMeta = (const ito::DoubleMeta*)meta;
                        if (value >= dMeta->getMin() && value <= dMeta->getMax()) result = true;
                    }
                    break;
                }
            }
            else
            {
                done = true;
                result = true;
            }
        }

        if (ok) *ok = done;
        return result;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal getParamValue(const QMap<QString, Param> *m_params, const QString &key, ito::Param &val, QString &pkey, int &index)
    {
        ito::RetVal retValue(ito::retOk);
        index = -1;
        pkey = key;

        QString paramName;
        bool hasIndex;
        QString additionalTag;

        if (key == "")
        {
            retValue += ito::RetVal(ito::retError, 0, QObject::tr("name of requested parameter is empty.").toAscii().data());
        }
        else
        {
            retValue += parseParamName(key, paramName, hasIndex, index, additionalTag);
            if (retValue.containsError() || paramName.isEmpty())
            {
                retValue = ito::RetVal::format(ito::retError, 0, QObject::tr("the parameter name '%1' is invald").arg(key).toAscii().data());
            }
            else
            {
                if (!hasIndex) index = -1;

                QMap<QString, ito::Param>::const_iterator paramIt =  m_params->find(paramName);

                if (paramIt != m_params->constEnd())
                {
                    pkey = paramName;
                    if ((paramIt.value().getType() == ito::Param::DoubleArray) || (paramIt.value().getType() == ito::Param::IntArray))
                    {
                        if (index < 0)
                        {
                            val = paramIt.value();
                        }
                        else if (index < paramIt.value().getLen())
                        {
                            val = paramIt.value()[index];
                        }
                        else
                        {
                            val = ito::Param();
                            retValue += ito::RetVal(ito::retError, 0, QObject::tr("array index of parameter out of bounds.").toAscii().data());
                        }
                    }
                    else
                    {
                        if (index >= 0)
                        {
                            retValue += ito::RetVal(ito::retWarning, 0, QObject::tr("given index of parameter name ignored since parameter is no array type").toAscii().data());
                        }
                        val = paramIt.value();
                    }
                }
                else
                {
                    retValue += ito::RetVal(ito::retError, 0, QObject::tr("parameter not found in m_params.").toAscii().data());
                }
            }
        }

        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    void paramHelperSetValue(ito::Param &param, ito::ParamBase value, const int pos)
    {
        void *dPtr = NULL;
        switch (param.getType() & ~ito::Param::Pointer)
        {
            case ito::Param::Char:
                dPtr = param.getVal<char *>();
                if (pos >= 0)
                {
                    ((char*)dPtr)[pos] = value.getVal<char>();
                }
                else
                {
                    char val = value.getVal<char>();
                    for (int num = 0; num < param.getLen(); num++)
                    {
                        ((char*)dPtr)[num] = val;
                    }
                }
            break;

            case ito::Param::Int:
                dPtr = param.getVal<int *>();
                if (pos >= 0)
                {
                    ((int*)dPtr)[pos] = value.getVal<int>();
                }
                else
                {
                    int *val = value.getVal<int *>();
                    int cntLimit = param.getLen() > value.getLen() ? value.getLen() : param.getLen();
                    for (int num = 0; num < cntLimit; num++)
                    {
                        ((int*)dPtr)[num] = val[num];
                    }
                }
            break;

            case ito::Param::Double:
                dPtr = param.getVal<double *>();
                if (pos >= 0)
                {
                    ((double*)dPtr)[pos] = value.getVal<double>();
                }
                else
                {
                    double *val = value.getVal<double *>();
                    int cntLimit = param.getLen() > value.getLen() ? value.getLen() : param.getLen();
                    for (int num = 0; num < cntLimit; num++)
                    {
                        ((double*)dPtr)[num] = val[num];
                    }
                }
            break;
        }
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    ito::RetVal setParamValue(const QMap<QString, Param> *m_params, const QString &key, const ito::ParamBase val, QString &pkey, int &index)
    {
        ito::RetVal retValue(ito::retOk);
        QStringList plkey;
        pkey = key;
        index = -1;

        QString paramName;
        bool hasIndex;
        QString additionalTag;

        if (key == "")
        {
            retValue += ito::RetVal(ito::retError, 0, QObject::tr("name of requested parameter is empty.").toAscii().data());
        }
        else
        {
            retValue += parseParamName(key, paramName, hasIndex, index, additionalTag);
            if (retValue.containsError() || paramName.isEmpty())
            {
                retValue = ito::RetVal::format(ito::retError, 0, QObject::tr("the parameter name '%1' is invald").arg(key).toAscii().data());
            }
            else
            {
                if (!hasIndex) index = -1;

                QMap<QString, ito::Param>::iterator paramIt = (QMap<QString, ito::Param>::iterator)(m_params->find(paramName)); //TODO: why do I need a cast here???

                if (paramIt != m_params->constEnd())
                {
                    pkey = paramName;
                    if ((paramIt.value().getType() == ito::Param::DoubleArray) || (paramIt.value().getType() == ito::Param::IntArray))
                    {
                        Param tempParam;
                        tempParam = paramIt.value();

                        if (index < 0)
                        {
                            paramHelperSetValue(paramIt.value(), val, -1);
                        }
                        else if (index < paramIt.value().getLen())
                        {
                            paramHelperSetValue(paramIt.value(), val, index);
                        }
                        else
                        {
                            retValue += ito::RetVal(ito::retError, 0, QObject::tr("array index out of bounds.").toAscii().data());
                        }
                    }
                    else
                    {
                        if (index >= 0)
                        {
                            retValue += ito::RetVal(ito::retWarning, 0, QObject::tr("given index of parameter name ignored since parameter is no array type").toAscii().data());
                        }
                        paramIt.value().copyValueFrom(&val);
                    }
                }
                else
                {
                    retValue += ito::RetVal(ito::retError, 0, QObject::tr("parameter not found in m_params.").toAscii().data());
                }
            }
        }

        return retValue;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    //! parses parameter name with respect to regular expression, assigned for parameter-communcation with plugins
    /*!
        This method parses any parameter-name with respect to the rules defined for possible names of plugin-parameters.

        The regular expression used for the check is "^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$"

        Then the components are:

        [0] full string
        [1] PARAMNAME
        [2] [INDEX] or empty-string if no index is given
        [3] INDEX or empty-string if no index is given
        [4] :ADDITIONALTAG or empty-string if no tag is given
        [5] ADDITIONALTAG or empty-string if no tag is given

        \param [in] name is the raw parameter name
        \param [out] paramName is the real parameter name (first part of name; part before the first opening bracket ('[') or if not available the first colon (':'))
        \param [out] hasIndex indicates whether the name contains an index part (defined by a number within two brackets (e.g. '[NUMBER]'), which has to be appended to the paramName
        \param [out] index is the fixed-point index value or -1 if hasIndex is false
        \param [out] additionalTag is the remaining string of name which is the part after the first colon (':'). If an index part exists, the first colon after the index part is taken.
    */
    ito::RetVal parseParamName(const QString &name, QString &paramName, bool &hasIndex, int &index, QString &additionalTag)
    {
        ito::RetVal retValue = ito::retOk;
        paramName = QString();
        hasIndex = false;
        index = -1;
        additionalTag = QString();

        QRegExp rx("^([a-zA-Z]+\\w*)(\\[(\\d+)\\]){0,1}(:(.*)){0,1}$");
        if (rx.indexIn(name) == -1)
        {
            retValue += ito::RetVal(ito::retError,0,QObject::tr("invalid parameter name").toAscii().data());
        }
        else
        {
            QStringList pname = rx.capturedTexts();
            paramName = pname[1];
            if (pname.size()>=4)
            {
                if (!pname[3].isEmpty())
                {
                    index = pname[3].toInt(&hasIndex);
                }
            }
            if (pname.size() >=6)
            {
                additionalTag = pname[5];
            }
        }


        return retValue;
    }

    

} //end namespace ito
