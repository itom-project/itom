/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#include "sharedFunctionsQt.h"

#include <iostream>

#include <qdir.h>
#include <QtWidgets/qapplication.h>
#include <qtextstream.h>
#include <qvariant.h>
#include <QDebug>
#include <QFileInfo>

#include <QtXml/qdom.h> /*!< this is for the plugin param save / load*/
#include <QXmlStreamWriter> /*!< this is for the dataobject save / load*/

namespace ito
{
    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @detail Compared the abs(dValue) with the 10^(3N) and according to the results mu p M ... are added to the unit
    *           Allowed units are SI-Unit except kg and mm. If % is given as input unit, values are multiplied by 100
    *   @param [in]  scaleThisUnitsOnly     List with scaleable units (e.g. mm, m)
    *   @param [in]  unitIn                 Old unit (e.g. mm, m, %)
    *   @param [in]  dVal                   Double value (e.g. mm, m, %)
    *   @param [out] dValOut                Scaled value
    *   @param [out] unitOut                Scaled unit m -> mm or \mu m
    *
    */
    RetVal formatDoubleWithUnit(QStringList scaleThisUnitsOnly, QString unitIn, double dVal, double &dValOut, QString &unitOut)
    {
        double aval = fabs(dVal);
        double factor = 1;
        RetVal retVal = retOk;

        unitOut.clear();

        if (!scaleThisUnitsOnly.contains(unitIn, Qt::CaseSensitive))
        {
            unitOut = QString(unitIn);
            dValOut = dVal;
            retVal = RetVal(retWarning, 1, QObject::tr("Tried to scale unscaleable unit").toLatin1().data());
        }
        else if (unitIn.isEmpty())
        {
            retVal = RetVal(retWarning, 1, QObject::tr("No unit specified").toLatin1().data());
        }
        else if (aval <= std::numeric_limits<double>::min())
        {
            unitOut = QString(unitIn);  // Do not scale unit for 0.0
        }
        else if (unitIn == "%")
        {
            dValOut = dVal*100;
            unitOut = unitIn;
        }
        else
        {
            QString tempUnit(unitIn);
            if (!unitIn.compare("mm"))
            {
                tempUnit = QString("m");
                factor = 1000;
                aval = aval / factor;
            }
            if (!unitIn.compare("kg"))
            {
                tempUnit = QString("g");
                factor = 0.001;
                aval = aval / factor;
            }

            if (aval < 1.0E-18 || aval >= 1.0E21)
            {    /* smaller than 1 atto */
                dValOut =  dVal / factor;
                unitOut = QString(tempUnit);
            }
            else if (aval < 1.0E-15)
            {
                dValOut = dVal / factor / 1.0E-18;
                unitOut.append("a");
            }
            else if (aval < 1.0E-12)
            {
                dValOut = dVal / factor / 1.0E-15;
                unitOut.append("f");
            }
            else if (aval < 1.0E-9)
            {
                dValOut = dVal / factor / 1.0E-12;
                unitOut.append("p");
            }
            else if (aval < 1.0E-6)
            {
                dValOut = dVal / factor / 1.0E-9;
                unitOut.append("n");
            }
            else if (aval < 1.0E-3)
            {
                dValOut = dVal / factor / 1.0E-6;
                unitOut = QChar(0x00, 0xB5); // \mu
            }
            else if (aval < 1.0)
            {
                dValOut = dVal / factor /  1.0E-3;
                unitOut.append("m");
            }
            else if (aval < 1000.0)
            {
                dValOut = dVal / factor;
            }
            else if (aval < 1.0E6)
            {
                dValOut = dVal / factor / 1.0E3;
                unitOut.append("k");
            }
            else if (aval < 1.0E9)
            {
                dValOut = dVal / factor / 1.0E6;
                unitOut.append("M");
            }
            else if (aval < 1.0E12)
            {
                dValOut = dVal / factor / 1.0E9;
                unitOut.append("G");
            }
            else if (aval < 1.0E15)
            {
                dValOut = dVal / factor / 1.0E12;
                unitOut.append("T");
            }
            else if (aval < 1.0E18)
            {
                dValOut = dVal / factor / 1.0E15;
                unitOut.append("P");
            }
            else if (aval < 1.0E21)
            {
                dValOut = dVal / factor / 1.0E18;
                unitOut.append("E");
            }
            unitOut.append(tempUnit);
        }
        return retVal;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   function for generates the plugin xml file handle
    *   @param [in]  fName          filename (is needed e.g. to get filename)
    *   @param [out] paramFile      reference to unopened parameter file
    *
    *   The function generates the xml parameter file name and returns the a QFile handle.
    *   The name has the same name as the plugin in the plugins directory.
    *   \sa loadXML2QLIST, saveQLIST2XML
    */
    RetVal generateAutoSaveParamFile(QString plugInName, QFile &paramFile)
    {
        QDir pluginsDir = QDir(qApp->applicationDirPath());
        RetVal ret = retOk;

#if defined(WIN32)
        if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
            pluginsDir.cdUp();
#elif defined(__APPLE__)
        if (pluginsDir.dirName() == "MacOS")
        {
            pluginsDir.cdUp();
            pluginsDir.cdUp();
            pluginsDir.cdUp();
        }
#endif
        pluginsDir.cd("plugins");

        if (plugInName == "")
        {
            return RetVal(retWarning, 0, QObject::tr("Pluginname undefined. No xml file loaded").toLatin1().data());
        }
        if (plugInName.endsWith(".dll", Qt::CaseInsensitive))
        {
            plugInName.chop(4);
        }

        plugInName.append(".xml");

        paramFile.setFileName(pluginsDir.absoluteFilePath(plugInName));

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @param [out] paramList  Empty List of Type QMap. If not empty this function will clear the list before reading
    *   @param [in]  id         Identifier of the plugin. Currently implemented as integer number only
    *   @param [in]  paramFile  Filename of the file. The file will be opened/closed in this function
    *
    *   \details This function reads the parameters for a plugin specified with id from an XML file. During initialisation
    *   an xml file with the same name as the plugin library in the plugin directory is used to load the plugin parameters. The xml file
    *   is checked for the current plugin-file version and type when opened. The parameters have in the calling function afterwards.
    */
    RetVal loadXML2QLIST(QMap<QString, Param> *paramList, QString id, QFile &paramFile)
    {
        RetVal ret = retOk;

        QDomDocument paramDomDoc;
        QString errorStr;
        int errorLine = 0;
        int errorColumn = 0;
        Param param;
        int idFound = 0;

        if (!paramList)
        {
            return RetVal(retWarning, 0, QObject::tr("ParamList not properly initialized").toLatin1().data());
        }

        if (!paramList->empty())
        {
            paramList->clear();
        }


        if (!paramFile.open(QIODevice::ReadOnly))
        {
            QString err = QString("Can't open xml file: %1").arg(paramFile.fileName());
            return RetVal(retWarning, 0, QObject::tr(err.toLatin1().data()).toLatin1().data());
        }

        if (!paramDomDoc.setContent((QIODevice *)&paramFile, true, &errorStr, &errorLine, &errorColumn))
        {
                //QMessageBox::information(window(), tr("DOM Bookmarks"),
                //    tr("Parse error at line %1, column %2:\n%3")
                //    .arg(errorLine)
                //    .arg(errorColumn)
                //    .arg(errorStr));
                QString errStr = QString(
                    "Parse error at line %1, column %2 error: %3").arg(errorLine).arg(errorColumn).arg(errorStr);
                return ito::RetVal(ito::retWarning, 0, QObject::tr(errorStr.toLatin1().data()).toLatin1().data());
        }

        QDomElement root = paramDomDoc.documentElement();
        if (root.tagName() != "xplugin")
        {
            //QMessageBox::information(window(), tr("DOM Bookmarks"),
            //                            tr("The file is not an XBEL file."));
            return retWarning;
        }
        else if (root.hasAttribute("version") && root.attribute("version") != "1.0")
        {
            //QMessageBox::information(window(), tr("DOM Bookmarks"),
            //                            tr("The file is not an XBEL version 1.0 "
            //                            "file."));
            return retWarning;
        }

        QDomElement child = root.firstChildElement("instance");
        while (!child.isNull())
        {
            if (child.attribute("id") == id)
            {

                QDomElement instParam = child.firstChildElement();
                while (!instParam.isNull())
                {
                    QString paramName;
                    if (instParam.attribute("namepref") != "")
                    {
                        paramName = QString(instParam.attribute("namepref") + ":" + instParam.nodeName());
                    }
                    else
                    {
                        paramName = instParam.nodeName();
                    }

                    if (instParam.attribute("type") == "number")
                    {
                        param = Param(paramName.toLatin1().data(), ParamBase::Double, instParam.text().toDouble(), NULL, NULL);
                        //param.setVal<double>(instParam.text().toDouble());
                        paramList->insert(param.getName(),param);
                    }
                    else if (instParam.attribute("type") == "string")
                    {
                        QByteArray cvalDecoded = QByteArray::fromPercentEncoding(instParam.text().toLatin1());
                        param = Param(paramName.toLatin1().data(), ParamBase::String, cvalDecoded.data(), NULL);
                        //param.setVal<char*>(cvalDecoded.data());
                        paramList->insert(param.getName(),param);
                    }
                    else if (instParam.attribute("type") == "numericVector")
                    {
                        // okay first try to decode the vector from a binary coding
                        QByteArray cvalDecoded = QByteArray::fromBase64(instParam.text().toLatin1());
                        unsigned int ptrlength = instParam.attribute("ptrlength").toInt();
                        bool isBinary = true;
                        bool asciiFailed = false;

                        if (instParam.attribute("ptrtype") == "uint8")
                        {
                            // check is the binary length is equal to the number of elements
                            if (ptrlength == (unsigned int)cvalDecoded.length())
                            {
                                param = Param(paramName.toLatin1().data(), ParamBase::CharArray, ptrlength, (int*)cvalDecoded.data(), NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::CharArray, 0, NULL);
                                //param.setVal<char*>(cvalDecoded.data(), ptrlength);
                                paramList->insert(param.getName(),param);
                                isBinary = true;
                            }
                            else
                            {
                                // length is not equal
                                isBinary = false;
                            }
                        }
                        else if (instParam.attribute("ptrtype") == "int32")
                        {
                            // check is the binary length is equal to the number of elements
                            if ((ptrlength * sizeof(int)) == (unsigned int)cvalDecoded.length())
                            {
                                param = Param(paramName.toLatin1().data(), ParamBase::IntArray, ptrlength, (int*)cvalDecoded.data(), NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::IntArray, 0, NULL, NULL);
                                //param.setVal<int*>((int*)cvalDecoded.data(), ptrlength);
                                paramList->insert(param.getName(),param);
                                isBinary = true;
                            }
                            else
                            {
                                // length is not equal
                                isBinary = false;
                            }
                        }
                        else if (instParam.attribute("ptrtype") == "float64")
                        {
                            // check is the binary length is equal to the number of elements
                            if ((ptrlength * sizeof(double)) == (unsigned int)cvalDecoded.length())
                            {
                                param = Param(paramName.toLatin1().data(), ParamBase::DoubleArray, ptrlength, (double*)cvalDecoded.data(), NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::DoubleArray, 0, NULL, NULL);
                                //param.setVal<double*>((double*)cvalDecoded.data(), ptrlength);
                                paramList->insert(param.getName(),param);
                                isBinary = true;
                            }
                            else
                            {
                                // length is not equal
                                isBinary = false;
                            }
                        }

                        // if decoding from binary failed, try ascii-coded ([value0;value1;valueN...])
                        if (isBinary == false)
                        {
                            asciiFailed = false;
                            cvalDecoded = instParam.text().toLatin1();
                            QList<QByteArray> tokes = cvalDecoded.split(';');

                            if (ptrlength != tokes.length())
                            {
                                asciiFailed = true;
                            }
                            else if (instParam.attribute("ptrtype") == "uint8")
                            {
                                char *cArray = (char *)calloc(ptrlength, sizeof(char));
                                for (unsigned int vCnt = 0; vCnt < ptrlength; vCnt++)
                                {
                                    cArray[vCnt] = cv::saturate_cast<char>(tokes[vCnt].toDouble());
                                }
                                param = Param(paramName.toLatin1().data(), ParamBase::CharArray, ptrlength, cArray, NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::CharArray, 0, NULL);
                                //param.setVal<char*>(cArray, ptrlength);
                                paramList->insert(param.getName(),param);
                                free(cArray);

                            }
                            else if (instParam.attribute("ptrtype") == "int32")
                            {
                                int *iArray = (int *)calloc(ptrlength, sizeof(int));
                                for (unsigned int vCnt = 0; vCnt < ptrlength; vCnt++)
                                {
                                    iArray[vCnt] = cv::saturate_cast<int>(tokes[vCnt].toDouble());
                                }
                                param = Param(paramName.toLatin1().data(), ParamBase::IntArray, ptrlength, iArray, NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::IntArray, 0, NULL, NULL);
                                //param.setVal<int*>(iArray, ptrlength);
                                paramList->insert(param.getName(),param);
                                free(iArray);
                            }
                            else if (instParam.attribute("ptrtype") == "float64")
                            {
                                double *dArray = (double *)calloc(ptrlength, sizeof(double));
                                for (unsigned int vCnt = 0; vCnt < ptrlength; vCnt++)
                                {
                                    dArray[vCnt] = cv::saturate_cast<double>(tokes[vCnt].toDouble());
                                }
                                param = Param(paramName.toLatin1().data(), ParamBase::DoubleArray, ptrlength, dArray, NULL);
                                //param = Param(instParam.nodeName().toLatin1().data(), ParamBase::DoubleArray,  0, NULL, NULL);
                                //param.setVal<double*>(dArray, ptrlength);
                                paramList->insert(param.getName(),param);
                                free(dArray);
                            }
                        }

                        if (asciiFailed == true)
                        {
                            QString errStr = QString("Decoding Error: %1 could not be decoded from numericVector").arg(instParam.nodeName());
                            ret += ito::RetVal(ito::retWarning, 0, QObject::tr(errStr.toLatin1().data()).toLatin1().data());
                        }
                    }
                    instParam = instParam.nextSiblingElement();
                }

                ret = retOk;
                idFound = 1;
                break;
            }
            child = child.nextSiblingElement("instance");
        }

        paramFile.close();

        if (!idFound)
        {
            QString idStr = QString("Id: %1 not found in xml file").arg(id);
            ret += ito::RetVal(ito::retWarning, 0, QObject::tr(idStr.toLatin1().data()).toLatin1().data());
        }
        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @param [in/out] paramList  List of Type QMap with the parameters to save. The parameters are deleted during writing.
    *   @param [in]  id            Identifier of the plugin. Currently implemented as integer number only
    *   @param [in]  paramFile     Filename of the file. The file will be opened/closed in this function
    *
    *   \details This function writes the parameters of a plugin to an XML file. During plugin closing this function is executed with
    *   a file name with same name as the plugin library in the plugin directory The xml file
    *   is checked for the current plugin-file version and type when opened. In case of a type conflict the parameter is currently not saved.
    */
    RetVal saveQLIST2XML(QMap<QString, Param> *paramList, QString id, QFile &paramFile)
    {
        RetVal ret = retOk;
        int created = 0;
        QDomDocument paramDomDoc;
        QString errorStr;
        int errorLine = 0;
        int errorColumn = 0;

        QFileInfo checkFile(paramFile);

        if (!checkFile.exists())
        {
            paramFile.open(QIODevice::ReadWrite);
            paramFile.close();
        }

        if (!paramFile.open(QIODevice::ReadOnly))
        {
            return RetVal(retWarning, 0, QObject::tr("Can't open xml file").toLatin1().data());
        }

        paramDomDoc.setContent((QIODevice *)&paramFile, true, &errorStr, &errorLine, &errorColumn);
        paramFile.close();

        QDomElement root = paramDomDoc.documentElement();
        if (!root.isNull())
        {
            if (root.tagName() != "xplugin")
            {
                //QMessageBox::information(window(), tr("DOM Bookmarks"),
                //                            tr("The file is not an XBEL file."));
                return retWarning;
            }
            else if (root.hasAttribute("version") && root.attribute("version") != "1.0")
            {
                //QMessageBox::information(window(), tr("DOM Bookmarks"),
                //                            tr("The file is not an XBEL version 1.0 "
                //                            "file."));
                return retWarning;
            }
        }
        else
        {
            root = paramDomDoc.createElement("xplugin");
            root.setAttribute("version", "1.0");
            paramDomDoc.appendChild(root);
        }

        QDomElement child = root.firstChildElement("instance");
        do
        {
            if ((!child.isNull()) && (child.attribute("id") == id)) //(child.attribute("id").compare(id) == 0))
            {

                QDomElement instParam = child.firstChildElement();
                while (!instParam.isNull())
                {
                    QString paramName;
                    if (instParam.attribute("namepref") != "")
                    {
                        paramName =  instParam.attribute("namepref") + ":" + instParam.nodeName();
                    }
                    else
                    {
                        paramName = instParam.nodeName();
                    }

                    QMap<QString, Param>::Iterator it = paramList->find(paramName);

                    if ((it != paramList->end()) && (it.value().getAutosave()))
                    {
                        QDomNode newVal;
                        if (it.value().isNumeric())
                        {
                            instParam.setAttribute("type", "number");
                            instParam.removeAttribute("ptrtype");
                            instParam.removeAttribute("ptrlength");
                            QVariant qvval = it.value().getVal<double>();
                            newVal = instParam.firstChild();
                            newVal.setNodeValue(qvval.toString());
                        }
                        else
                        {
                            switch (it.value().getType())
                            {
                                case ParamBase::CharArray :
                                case ParamBase::IntArray :
                                case ParamBase::DoubleArray :
                                {
                                    QByteArray cval;

                                    if (it->getType() ==  (ParamBase::CharArray))
                                    {
                                        instParam.setAttribute("ptrtype", "uint8");
                                        cval = QByteArray((char*) it->getVal<char*>(), it->getLen() * sizeof(char));
                                    }
                                    else if (it->getType() ==  (ParamBase::IntArray))
                                    {
                                        instParam.setAttribute("ptrtype", "int32");
                                        cval = QByteArray((char*) it->getVal<int*>(), it->getLen() * sizeof(int));
                                    //int testvar[5] = {55, 22, 1024, -10, 13};
                                    //instParam.setAttribute("ptrlength", 5);
                                    //QByteArray cval((char*) &testvar, 5 * 4);
                                    }
                                    else if (it->getType() ==  (ParamBase::DoubleArray))
                                    {
                                        instParam.setAttribute("ptrtype", "float64");
                                        cval = QByteArray((char*) it->getVal<double*>(), it->getLen() * sizeof(double));
                                    }
                                    else
                                    {
                                        break;
                                    }

                                    instParam.setAttribute("type", "numericVector");
                                    instParam.setAttribute("ptrlength", it->getLen());

                                    QString cvalEncoded(cval.toBase64());
                                    newVal = instParam.firstChild();
                                    newVal.setNodeValue(cvalEncoded);
                                }
                                break;

                                case ParamBase::String :
                                {
                                    instParam.setAttribute("type", "string");
                                    instParam.removeAttribute("ptrtype");
                                    instParam.removeAttribute("ptrlength");
                                    char * tbuf = it.value().getVal<char*>(); //borrowed reference
                                    QByteArray cval(tbuf);
                                    QString cvalEncoded(cval.toPercentEncoding());
                                    newVal = instParam.firstChild();
                                    newVal.setNodeValue(cvalEncoded);
                                }
                                break;
                            }
                        }
                    }
                    paramList->remove(paramName.toLatin1());
                    instParam = instParam.nextSiblingElement();
                }

                QMap<QString, Param>::Iterator it = paramList->begin();
                for (; it != paramList->end(); it++)
                {
                    if (it.value().getAutosave())
                    {
                        ito::Param tmpParam = it.value();
                        QString paramName = it.value().getName();
                        QString namepref;
                        if (paramName.contains(":"))
                        {
                            QStringList parts = paramName.split(":");
                            namepref = parts[0];
                            paramName = parts[1];
                        }
                        QDomElement newParam = paramDomDoc.createElement(paramName);
                        if (namepref != "")
                        {
                            newParam.setAttribute("namepref", namepref);
                        }
                        QDomText tvalue;
                        QVariant qvar;
                        if (it.value().isNumeric())
                        {
                            newParam.setAttribute("type", "number");
                            qvar = it.value().getVal<double>();
                            tvalue = paramDomDoc.createTextNode(qvar.toString());
                            newParam.appendChild(tvalue);
                            child.appendChild(newParam);
                        }
                        else
                        {
                            switch (it.value().getType())
                            {
                                case ParamBase::CharArray :
                                case ParamBase::IntArray :
                                case ParamBase::DoubleArray :
                                {
                                    QByteArray cval;

                                    if (it->getType() ==  (ParamBase::CharArray))
                                    {
                                        newParam.setAttribute("ptrtype", "uint8");
                                        cval = QByteArray((char*) it->getVal<char*>(), it->getLen() * sizeof(char));
                                    }
                                    else if (it->getType() ==  (ParamBase::IntArray))
                                    {
                                        newParam.setAttribute("ptrtype", "int32");
                                        cval = QByteArray((char*) it->getVal<int*>(), it->getLen() * sizeof(int));
                                    //int testvar[5] = {55, 22, 1024, -10, 13};
                                    //cval = QByteArray((char*) &testvar, 5 * 4);
                                    }
                                    else if (it->getType() ==  (ParamBase::DoubleArray))
                                    {
                                        newParam.setAttribute("ptrtype", "float64");
                                        cval = QByteArray((char*) it->getVal<double*>(), it->getLen() * sizeof(double));
                                        //double testvar[5] = {1.1, 0.2, 3.7, 6.6667, 22};
                                        //cval = QByteArray((char*) &testvar, 5 * 8);
                                    }
                                    else
                                    {
                                        break;
                                    }

                                    newParam.setAttribute("type", "numericVector");
                                    instParam.setAttribute("ptrlength", it->getLen());
                                    //newParam.setAttribute("ptrlength", 5);
                                    QString cvalEncoded(cval.toBase64());

                                    tvalue = paramDomDoc.createTextNode(cvalEncoded);
                                    newParam.appendChild(tvalue);
                                    child.appendChild(newParam);
                                }
                                break;

                                case ParamBase::String :
                                    newParam.setAttribute("type", "string");
                                    char * tbuf = it.value().getVal<char*>(); //borrowed reference
                                    QByteArray cval(tbuf);
                                    QString cvalEncoded(cval.toPercentEncoding());

                                    tvalue = paramDomDoc.createTextNode(cvalEncoded);
                                    newParam.appendChild(tvalue);
                                    child.appendChild(newParam);
                                break;
                            }
    //                        qDebug() << newParam.text();
                        }
                    }
                }

                if (!paramFile.open(QIODevice::WriteOnly))
                {
                    return RetVal(retWarning, 0, QObject::tr("Can't open xml file").toLatin1().data());
                }

                QTextStream out((QIODevice *)&paramFile);
                paramDomDoc.save(out, 4);
                paramFile.close();
                return ret;
            }
            child = child.nextSiblingElement("instance");

            if (child.isNull())     //!< passed all elements without success, so add new one
            {
                QDomElement newID = paramDomDoc.createElement("instance");
//                newID.setAttribute("id", QString::number(id.toInt()));
                newID.setAttribute("id", id);
                root.appendChild(newID);
                child = newID;
                created++;
            }
        }
        while (created < 2);

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @param [in/out] oldList  Paramlist with all plugin-parameters, which will contain the merged parameters in the end
    *   @param [in/out] newList  New parameter values to set
    *   @param [in]  checkAutoSave  Flag to enable / disable autosave control to avoid obsolet parameters to overwrite exisiting parameters.
    *
    *   \details This function compares the new list with the old list. If new list contains parameters which do not exist in the existing list,
    *   the paremeter is ignored and a warning is added to the errormessage stack.
    *   If the checkAutoSave parameter is true, parameters in oldList are not altered of the autosave is disabled. In this case a warning is returned.
    *   In case the paremters Type is not equal, a warning is returned and the paremeter is not altered.
    *   At the moment only parameters of numeric values and strings are merged.
    */
    RetVal mergeQLists(QMap<QString, Param> *oldList, QMap<QString, Param> *newList, bool checkAutoSave, bool deleteUnchangedParams)
    {
        RetVal ret = retOk;
        ParamBase paramTemp;

        char name[500]={0};
//        char errorbuf[501] = {0};
        QString msg;
        bool doNotIgnoreThisMissing = false;

        foreach (paramTemp, *oldList) // first check if newlist contains all (autosave) parameter!!
        {
            memset(name, 0, sizeof(name));
            _snprintf(name, 500, "%s", paramTemp.getName());
            if (!strlen(name))
            {
                continue;
            }
            QMap<QString, Param>::iterator paramIt = newList->find(name);

            if (checkAutoSave)
            {
                doNotIgnoreThisMissing = paramTemp.getAutosave();
            }
            else
            {
                doNotIgnoreThisMissing = false;
            }

            if ((paramIt == newList->end()))
            {
                if (doNotIgnoreThisMissing)
                {
                    if (ret.containsWarning())
                    {
                        msg = QObject::tr("%1\nAutosave parameter %2 not found").arg(QLatin1String(ret.errorMessage())).arg(name);
                    }
                    else
                    {
                        msg = QObject::tr("XML-Import warnings:\nAutosave parameter %1 not found").arg(name);
                    }
//                    ret += RetVal(retWarning, 0, errorbuf);
                    ret += RetVal(retWarning, 0, msg.toLatin1().data());
                }

                if (deleteUnchangedParams)
                {
                    oldList->remove(paramTemp.getName());
                }
            }
        }

        foreach (paramTemp, *newList) // now set all parameters from the new list to the old list, which already exist in the old
        {
            memset(name,0,sizeof(name));
            _snprintf(name,500,"%s", paramTemp.getName());
            if (!strlen(name))
            {
                continue;
            }

            QMap<QString, Param>::iterator paramIt = oldList->find(name);
            if (paramIt == oldList->end())
            {
                if (ret.containsWarning())
                {
                    msg = QObject::tr("%1\nObsolete parameter %2").arg(QLatin1String(ret.errorMessage())).arg(name);
                }
                else
                {
                    msg = QObject::tr("XML-Import warnings:\nObsolete parameter %1").arg(name);
                }
//                ret += RetVal(retWarning, 0, errorbuf);
                ret += RetVal(retWarning, 0, msg.toLatin1().data());
            }
            else if (paramIt.value().getAutosave() == 0)
            {
                if (ret.containsWarning())
                {
                    msg = QObject::tr("%1\nParameter %2 not autosave").arg(QLatin1String(ret.errorMessage())).arg(name);
                }
                else
                {
                    msg = QObject::tr("XML-Import warnings:\nParameter %1 not autosave").arg(name);
                }
//                ret += RetVal(retWarning, 0, errorbuf);
                ret += RetVal(retWarning, 0, msg.toLatin1().data());
            }
            else    // So parameters exist and now we have to check if they are similar types
            {
                if (paramIt.value().isNumeric() && paramTemp.isNumeric())
                {
                    paramIt.value().setVal<double>(paramTemp.getVal<double>());
                }
                else if (paramTemp.getType() == paramIt.value().getType())
                {
                    switch(paramTemp.getType())
                    {
                        case ParamBase::CharArray :
                        case ParamBase::IntArray :
                        case ParamBase::DoubleArray :
                        case ParamBase::String :
                            ret += paramIt.value().copyValueFrom(&paramTemp);
                            //paramIt.value().setVal<char *>(paramTemp.getVal<char *>());  // WARNING: param.getVal<char*>() has to be deleted via free!
                        break;
                        default:
                            if (ret.containsWarning())
                            {
                                msg = QObject::tr("%1\nParameter not loadable %2").arg(ret.errorMessage()).arg(name);
                            }
                            else
                            {
                                msg = QObject::tr("XML-Import warnings:\nParameter not loadable %1").arg(name);
                            }
                            ret += RetVal(retWarning, 0, msg.toLatin1().data());
                        break;
                    }
                }
                else
                {
                    if (ret.containsWarning())
                    {
                        msg = QObject::tr("%1\nType conflict for %2").arg(QLatin1String(ret.errorMessage())).arg(name);
                    }
                    else
                    {
                        msg = QObject::tr("XML-Import warnings:\nType conflict for %1").arg(name);
                    }
//                    ret += RetVal(retWarning, 0, errorbuf);
                    ret += RetVal(retWarning, 0, msg.toLatin1().data());
                }
            }
        }

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief  This helper function writes the header of the Object to the xml stream.
    *   \detail This helper function writes the header (dims, sizes, type) of an object and the metaData (complete DataObjectTags without tagsMap) from the Object to the xml stream.
    *           The values of the header are stored as string. The values of each axis-tag / value-tag / rotation matrix are in case of string-type directly written to the stream or in case of double converted
    *           to either strings directly (15 significat digits, >32Bit) or stored as lostfree binary (QByteArray::toBase64() to avoid XML-conflict).
    *           WARNING: Do not change the header (dims, sizes, type) information or the value of tags exported as binary (d2b).
    *
    *   @param [in|out] stream           outgoing xml-stream
    *   @param [in]     dObjOut          The allocated src-Object
    *   @param [in]     doubleAsBinary   Toggle binary export for double
    *
    *   \sa saveDOBJ2XML, saveDOBJSpecificData2XML, DataObjectTags
    */

    inline RetVal writeObjectHeaderToFileV1(QXmlStreamWriter &stream, DataObject *dObjOut, bool doubleAsBinary, int &elementsize)
    {
        QByteArray cvalEncoded("");
        QString type("");
        switch(dObjOut->getType())
        {
        case tUInt8:
            elementsize = 1;
            type = "tUInt8";
            break;
        case tUInt16:
            elementsize = 2;
            type = "tUInt16";
            break;
        case tUInt32:
            elementsize = 4;
            type = "tUInt32";
            break;
        case tInt8:
            elementsize = 1;
            type = "tInt8";
            break;
        case tInt16:
            elementsize = 2;
            type = "tInt16";
            break;
        case tInt32:
            elementsize = 4;
            type = "tInt32";
            break;
        case tFloat32:
            elementsize = 4;
            type = "tFloat32";
            break;
        case tFloat64:
            elementsize = 8;
            type = "tFloat64";
            break;
        case tComplex64:
            elementsize = 8;
            type = "tComplex64";
            break;
        case tComplex128:
            elementsize = 16;
            type = "tComplex128";
            break;
        case tRGBA32:
            elementsize = 4;
            type = "tRGBA32";
            break;
        default:
            return RetVal(retError, 0, QObject::tr("Save object failed: Type not supported").toLatin1().data());
        }

        // First add informations of the dataObject to element DataObject like FormatVersion of this file...

        stream.writeAttribute("FormatVersion", "1.0");
        stream.writeAttribute("dataType", type);
        //stream.writeAttribute("isTransposed",  QString::number(dObjOut->isT()));
        stream.writeAttribute("dims", QString::number(dObjOut->getDims()));

        int dim = dObjOut->getDims() - 2;

        for (int i = 0; i < dim; i++)
        {
            QString attrib = "dim";
            attrib.append(QString::number(i));
            stream.writeAttribute(attrib, QString::number(dObjOut->getSize(i)));
        }
        stream.writeAttribute("dimY", QString::number(dObjOut->getSize(dim)));
        dim++;
        stream.writeAttribute("dimX", QString::number(dObjOut->getSize(dim)));

        // Now add metaData to XML-Stream

        stream.writeStartElement("metaData");
        {
            bool valid = true;

            if (doubleAsBinary)
            {
                stream.writeAttribute("doubleExport", "d2b");
            }
            else
            {
                stream.writeAttribute("doubleExport", "d2s");
            }

            for (int i = 0; i < dObjOut->getDims()-2; i++)
            {
                QString element = "dim";
                element.append(QString::number(i));
                stream.writeStartElement(element);
                {
                    if (doubleAsBinary)
                    {
                        double dtVal = dObjOut->getAxisOffset(i);
                        stream.writeAttribute("offset", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                        dtVal = dObjOut->getAxisScale(i);
                        stream.writeAttribute("scale", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                    }
                    else
                    {
                        stream.writeAttribute("offset", QString::number(dObjOut->getAxisOffset(i), 'g', 15));
                        stream.writeAttribute("scale", QString::number(dObjOut->getAxisScale(i), 'g', 15));
                    }
                    cvalEncoded = QByteArray(dObjOut->getAxisUnit(i, valid).data()).toPercentEncoding();
                    stream.writeAttribute("unit", cvalEncoded);
                    cvalEncoded = QByteArray(dObjOut->getAxisDescription(i, valid).data()).toPercentEncoding();
                    stream.writeAttribute("description", cvalEncoded);
                    stream.writeCharacters(" ");
                }
                stream.writeEndElement(); // dimN
            }
            int dim = dObjOut->getDims()-1;

            stream.writeStartElement("dimX");
            {
                if (doubleAsBinary)
                {
                    double dtVal = dObjOut->getAxisOffset(dim);
                    stream.writeAttribute("offset", QByteArray((char*)&(dtVal), sizeof(double)).toBase64());
                    dtVal = dObjOut->getAxisScale(dim);
                    stream.writeAttribute("scale", QByteArray((char*)&(dtVal), sizeof(double)).toBase64());
                }
                else
                {
                    stream.writeAttribute("offset", QString::number(dObjOut->getAxisOffset(dim), 'g', 15));
                    stream.writeAttribute("scale", QString::number(dObjOut->getAxisScale(dim), 'g', 15));
                }
                cvalEncoded = QByteArray(dObjOut->getAxisUnit(dim, valid).data()).toPercentEncoding();
                stream.writeAttribute("unit", cvalEncoded);
                cvalEncoded = QByteArray(dObjOut->getAxisDescription(dim, valid).data()).toPercentEncoding();
                stream.writeAttribute("description", cvalEncoded);
                stream.writeCharacters(" ");
            }
            dim = dObjOut->getDims() - 2;
            stream.writeEndElement(); // dimX
            stream.writeStartElement("dimY");
            {
                if (doubleAsBinary)
                {
                    double dtVal = dObjOut->getAxisOffset(dim);
                    stream.writeAttribute("offset", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                    dtVal = dObjOut->getAxisScale(dim);
                    stream.writeAttribute("scale", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                }
                else
                {
                    stream.writeAttribute("offset", QString::number(dObjOut->getAxisOffset(dim), 'g', 15));
                    stream.writeAttribute("scale", QString::number(dObjOut->getAxisScale(dim), 'g', 15));
                }

                cvalEncoded = QByteArray(dObjOut->getAxisUnit(dim, valid).data()).toPercentEncoding();
                stream.writeAttribute("unit", cvalEncoded);
                cvalEncoded = QByteArray(dObjOut->getAxisDescription(dim, valid).data()).toPercentEncoding();
                stream.writeAttribute("description", cvalEncoded);

                stream.writeCharacters(" ");
            }
            stream.writeEndElement(); // dimY

            stream.writeStartElement("values");
            {
                if (doubleAsBinary)
                {
                    double dtVal = dObjOut->getValueOffset();
                    stream.writeAttribute("offset", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                    dtVal = dObjOut->getValueScale();
                    stream.writeAttribute("scale", QByteArray((char*)&(dtVal),sizeof(double)).toBase64());
                }
                else
                {
                    stream.writeAttribute("offset", QString::number(dObjOut->getValueOffset(), 'g', 15));
                    stream.writeAttribute("scale", QString::number(dObjOut->getValueScale(), 'g', 15));
                }

                cvalEncoded = QByteArray(dObjOut->getValueUnit().data()).toPercentEncoding();
                stream.writeAttribute("unit", cvalEncoded);

                cvalEncoded = QByteArray(dObjOut->getValueDescription().data()).toPercentEncoding();
                stream.writeAttribute("description", cvalEncoded);
                stream.writeCharacters(" ");
            }
            stream.writeEndElement(); // values

            stream.writeStartElement("Rotation_Matrix");
            {
                double dRotMat[9];
                dObjOut->getXYRotationalMatrix(dRotMat[0], dRotMat[1], dRotMat[2], dRotMat[3], dRotMat[4], dRotMat[5], dRotMat[6], dRotMat[7], dRotMat[8]);
                if (doubleAsBinary)
                {
                    stream.writeAttribute("r11", QByteArray((char*)&(dRotMat[0]), sizeof(double)).toBase64());
                    stream.writeAttribute("r12", QByteArray((char*)&(dRotMat[1]), sizeof(double)).toBase64());
                    stream.writeAttribute("r13", QByteArray((char*)&(dRotMat[2]), sizeof(double)).toBase64());
                    stream.writeAttribute("r21", QByteArray((char*)&(dRotMat[3]), sizeof(double)).toBase64());
                    stream.writeAttribute("r22", QByteArray((char*)&(dRotMat[4]), sizeof(double)).toBase64());
                    stream.writeAttribute("r23", QByteArray((char*)&(dRotMat[5]), sizeof(double)).toBase64());
                    stream.writeAttribute("r31", QByteArray((char*)&(dRotMat[6]), sizeof(double)).toBase64());
                    stream.writeAttribute("r32", QByteArray((char*)&(dRotMat[7]), sizeof(double)).toBase64());
                    stream.writeAttribute("r33", QByteArray((char*)&(dRotMat[8]), sizeof(double)).toBase64());
                }
                else
                {
                    stream.writeAttribute("r11", QString::number(dRotMat[0], 'g', 15));
                    stream.writeAttribute("r12", QString::number(dRotMat[1], 'g', 15));
                    stream.writeAttribute("r13", QString::number(dRotMat[2], 'g', 15));
                    stream.writeAttribute("r21", QString::number(dRotMat[3], 'g', 15));
                    stream.writeAttribute("r22", QString::number(dRotMat[4], 'g', 15));
                    stream.writeAttribute("r23", QString::number(dRotMat[5], 'g', 15));
                    stream.writeAttribute("r31", QString::number(dRotMat[6], 'g', 15));
                    stream.writeAttribute("r32", QString::number(dRotMat[7], 'g', 15));
                    stream.writeAttribute("r33", QString::number(dRotMat[8], 'g', 15));

                }
                stream.writeCharacters(" ");
            }
            stream.writeEndElement(); // Rotation Matrix
        }
        stream.writeEndElement(); // metaData
        return retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief  This helper function writes the tags defined in the tagMap (DataObjectTags) from the Object to the xml stream.
    *   \detail This helper function writes the tags defined in the tagMap (DataObjectTags) from the Object to the xml stream.
    *           Therefore the values of each tag are in case of string-type directly written to the stream or in case of double converted
    *           to either strings directly (15 significat digits, >32Bit) or stored as lostfree binary (QByteArray::toBase64() to avoid XML-conflict).
    *           WARNING: Do not change information or the value of tags exported as binary (d2b).
    *
    *   @param [in|out] stream           outgoing xml-stream
    *   @param [in]     dObjOut          The allocated src-Object
    *   @param [in]     doubleAsBinary   Toggle binary export for double
    *
    *   \sa saveDOBJ2XML, saveDOBJSpecificData2XML, DataObjectTags
    */
    inline RetVal writeObjectTagsToFileV1(QXmlStreamWriter &stream, DataObject *dObjOut, bool doubleAsBinary)
    {
        QByteArray cvalEncoded("");
        stream.writeStartElement("tagSpace");
        {
            int numberOfTags = dObjOut->getTagListSize();
            stream.writeAttribute("tagNums", QString::number(numberOfTags));
            for (int i = 0; i < numberOfTags; i++)
            {
                std::string key;
                DataObjectTagType value;
                dObjOut->getTagByIndex(i, key, value);
                cvalEncoded = QByteArray(key.data()).toPercentEncoding();
                stream.writeStartElement(cvalEncoded);
                if (value.getType() == DataObjectTagType::typeDouble)
                {
                    double dtval = value.getVal_ToDouble();
                    if (doubleAsBinary)
                    {
                        char ctval[sizeof(double)] = {0};
                        memcpy(ctval, &dtval, sizeof(double));
                        stream.writeAttribute("type", "d2b");
                        stream.writeCharacters(QByteArray(ctval, sizeof(double)).toBase64());
                    }
                    else
                    {
                        stream.writeAttribute("type", "d2s");
                        stream.writeCharacters(QString::number(dtval, 'g', 15));
                    }
                }
                else
                {
                    stream.writeAttribute("type", "s");
                    cvalEncoded = QByteArray(value.getVal_ToString().data()).toPercentEncoding();
                    stream.writeCharacters(cvalEncoded);
                }
                stream.writeEndElement(); // single tag
            }
        }
        stream.writeEndElement(); // tagSpace
        return retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This helper function writes the data(cv::mats) from the Object to the xml stream.
    *   \detail This helper function writes the data(cv::mats) from the Object to the xml stream.
    *           Therefore the data is converted using QByteArray::toBase64() to avoid XML-conflict with the binary data.
    *
    *   @param [in|out] stream      outgoing xml-stream
    *   @param [in]     dObjOut     The allocated src-Object
    *   @param [in]     elementsize Size of each matrix element in bytes
    *
    *   \sa saveDOBJ2XML, saveDOBJSpecificData2XML
    */
    inline RetVal writeObjectDataToFileV1(QXmlStreamWriter &stream, DataObject *dObjOut, int elementsize)
    {
        int z_length = dObjOut->calcNumMats();
        int lineSize = dObjOut->getSize(dObjOut->getDims()-1) * elementsize;
        int lines = dObjOut->getSize(dObjOut->getDims()-2);

        stream.writeStartElement("data");
        stream.writeAttribute("planes", QString::number(z_length));

        for (int i = 0; i < z_length; i++)
        {
            QString plane = "plane";
            plane.append(QString::number(i));
            stream.writeStartElement(plane);
            QByteArray cval("");
            cval.reserve(lineSize * lines);
            for (int y = 0; y < lines; y++)
            {
                char* dataptr = (char *)((cv::Mat *)dObjOut->get_mdata()[dObjOut->seekMat(i)])->ptr(y);
                cval.append(dataptr, lineSize);

            }
            //QByteArray temp = cval.toBase64();
            //cval.fromBase64(temp);
            stream.writeCDATA(cval.toBase64());
            stream.writeEndElement(); // planeN
        }
        stream.writeEndElement(); // data
        return retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @param [in] dObjOut           DataObject to save
    *   @param [in] folderFileName    Name of the folder and the filename e.g. c:\\bla.xml or c://bla.xml
    *   @param [in] doubleAsBinary    If true, double are exported as binary, by defaults they are saved as strings
    *
    *   \details This function writes data and meta data of a dataObject to the harddrive. The file format is based on xml.
    *            The data of the dataObject are converted to binary without XML-registed signs by QByteArray::toBase64()).
    *            The metaData and tagSpace are either saved as binary (QByteArray::toBase64()) or saves as strings with 15 signifiant digits (more than 32-Bit).
    *            So for most applications doubleAsBinary==false is enough.
    *
    *   \autor Lyda
    *   \date  04.2012
    *   \sa writeObjectHeaderToFileV1, writeObjectTagsToFileV1, writeObjectDataToFileV1
    *
    */
    RetVal saveDOBJ2XML(DataObject *dObjOut, QString folderFileName, bool onlyHeaderObjectFile, bool doubleAsBinary)
    {
        RetVal ret(retOk);

        if (!dObjOut)
        {
            return RetVal(retError, 0, QObject::tr("Save object failed: Invalid object handle").toLatin1().data());
        }

        if ((dObjOut->getDims() == 0) || (dObjOut->getTotal() == 0))
        {
            return RetVal(retError, 0, QObject::tr("Save object failed: Object seems empty").toLatin1().data());
        }

        /*ret += dObjOut->evaluateTransposeFlag();

        if (ret.containsError())
        {
            return RetVal(retError, 0, QObject::tr("Save object failed: evaluate transpose failed").toLatin1().data());
        }*/

        int elementsize = 1;
        QString type;

        QFile paramFile;
        QString fileName(folderFileName);
        QFileInfo checkFile(folderFileName);

        if (checkFile.suffix().isEmpty())
        {
            if (onlyHeaderObjectFile)
            {
                fileName.append(".idh");
            }
            else
            {
                fileName.append(".ido");
            }
        }

        paramFile.setFileName(fileName);
        checkFile.setFile(paramFile);
        fileName = checkFile.canonicalFilePath();

        if (!checkFile.isWritable() && checkFile.exists())
        {
            return RetVal(retError, 0, QObject::tr("Save object failed: File not writeable").toLatin1().data());
        }

        paramFile.open(QIODevice::WriteOnly);

        QXmlStreamWriter stream(&paramFile);
        QString attrname;

        // Qt5: UTF-8 is the default codec, Qt6: uses always UTF-8
        stream.setAutoFormatting(true);

        stream.writeStartDocument();
        if (onlyHeaderObjectFile)
        {
            stream.writeStartElement("itomDataObjectHeader");
        }
        else
        {
            stream.writeStartElement("itomDataObject");
        }
        {
            stream.writeAttribute("href", "http://www.ito.uni-stuttgart.de");

            if (!ret.containsError()) ret += writeObjectHeaderToFileV1(stream, dObjOut, doubleAsBinary, elementsize);
            if (!ret.containsError()) ret += writeObjectTagsToFileV1(stream, dObjOut, doubleAsBinary);
            if (!onlyHeaderObjectFile)
            {
                if (!ret.containsError()) ret += writeObjectDataToFileV1(stream, dObjOut, elementsize);
            }
        }
        stream.writeEndElement(); // itomDataObject or itomDataObjectHeader
        stream.writeEndDocument();

        paramFile.close();

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This helper function reads the stream till the next startElement.
    *   \detail The Qt-Function readNextStartElement sometimes stops at the end-element (:P). So the function tries to read until it reaches the next startelement but only for maxtimes trys
    *           the Function checks if the attribute exists and than tries to convert to the value of the attribute either from binary or with string to double functions.
    *
    *   @param [in]     stream      incomming xml-stream
    *   @param [in|out] times       Counts of iterations
    *   @param [in]     maxtimes    maximal number of iterations to perform
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ
    */
    inline bool readTillNext(QXmlStreamReader &stream, int &times, int maxtimes)
    {
        bool ret = true;
        for (times = 0; times <  maxtimes; times++)
        {
            ret = stream.readNextStartElement();
            if (!stream.isEndElement())
            {
                break;
            }
        }
        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This helper function extracts a double from the xml-Attributes of the Stream
    *   \detail This helper function extracts a double from the xml-Attributes of the Stream copied by the caller with the attrStream = attrStream = stream.attributes();.
    *           the Function checks if the attribute exists and than tries to convert to the value of the attribute either from binary or with string to double functions.
    *
    *   @param [in]     attrStream      incomming attribute-stream
    *   @param [in]     Element         name of the element (only for error msg)
    *   @param [in]     Attrib          name of the attribute to extract
    *   @param [in|out] val             Must be filled with default value and is filled with value from the XML-Stream
    *   @param [in]     isBinary        Must be true if attribute value was stored as binary else false
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ, loadObjectHeaderFromXMLV1
    */
    inline RetVal readDoubleFromXML(QXmlStreamAttributes &attrStream, QString &Element, QString &Attrib, double &val, bool isBinary)
    {
        if (attrStream.hasAttribute(Attrib))
        {
            if (!isBinary)
            {
                QByteArray cval = QByteArray::fromPercentEncoding(attrStream.value(Attrib).toString().toLatin1());
                val = cval.toDouble();
            }
            else
            {
                QByteArray cvalDecoded = QByteArray::fromBase64(attrStream.value(Attrib).toString().toLatin1());
                memcpy(&val, cvalDecoded.data(), sizeof(double));
            }
        }
        else
        {
            QString warning = QObject::tr("Load object warning: Metadata \" %1 \" for %2 missing").arg(Attrib).arg(Element);
            return RetVal(retWarning, 0, warning.toLatin1().data());
        }
        return retOk;
    }
    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This helper function extracts a std::string from the xml-Attributes of the Stream
    *   \detail This helper function extracts a std::string from the xml-Attributes of the Stream copied by the caller with the attrStream = attrStream = stream.attributes();.
    *           the Function checks if the attribute exists and than tries to convert to the value of the attribute from QString to std::string.
    *
    *   @param [in]     attrStream      incomming attribute-stream
    *   @param [in]     Element         name of the element (only for error msg)
    *   @param [in]     Attrib          name of the attribute to extract
    *   @param [in|out] val             Must be filled with default value and is filled with value from the XML-Stream
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ, loadObjectHeaderFromXMLV1
    */

    inline RetVal readStdStringFromXML(QXmlStreamAttributes &attrStream, QString &Element, QString &Attrib, std::string &val)
    {

        if (attrStream.hasAttribute(Attrib))
        {
            QByteArray cval = QByteArray::fromPercentEncoding(attrStream.value(Attrib).toString().toLatin1());
            val = cval.data();
        }
        else
        {
            return RetVal(retWarning, 0, QObject::tr("Load object warning: Metadata \" %1 \" for %2 missing").arg(Attrib).arg(Element).toLatin1().data());
        }
        return retOk;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This function creates an dataObject from the header of the xml-file
    *   \detail This function creates an dataObject from the header of the xml-file by parsing the XML-stream.
    *           The first start element, already read by the calling function must contain the attributes dims, dataType and dim0..dimn-2, dimX, dimY
    *
    *   @param [in|out] stream      The xml-Stream from the xml-file
    *   @param [out] dObjIn         Destination dataContainter of type dataObject with size / dims / type speficied in the input xml
    *   @param [out] elementsize    Byte-Size of the current dataObjekt
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ
    */
    RetVal createObjectFromXMLV1(QXmlStreamReader &stream, DataObject &dObjIn, int &elementsize)
    {
        RetVal ret(retOk);

        int *sizes= NULL;     /*!< Sizes of the new dataObject. Will be freed at end: */
        QString attrname;
        QString type("");
        QXmlStreamAttributes attrStream = stream.attributes();

        char ndims = 0;
        int objType = tUInt8;
        //char isTransposed = 0;

        if (attrStream.hasAttribute("dims") && attrStream.hasAttribute("dataType"))
        {
            type += attrStream.value("dataType");
            ndims = attrStream.value("dims").toString().toInt();
            if (ndims < 2)
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: Number of dims smaller 2").toLatin1().data());
            }
        }

        if (!ret.containsError())
        {
            if (NULL==(sizes = (int *)calloc(ndims, sizeof(int))))
            {
                ret += RetVal(retError, 0, QObject::tr("Not enough memory to alloc sizes vector").toLatin1().data());
            }
        }

        if (!ret.containsError())
        {
            for (unsigned char i = 0; i < ndims - 2; i++)
            {
                attrname = "dim";
                attrname.append(QString::number(i));
                if (attrStream.hasAttribute(attrname))
                {
                    sizes[i] = attrStream.value(attrname).toString().toInt();
                }
                else
                {
                    ret += RetVal(retError, 0, QObject::tr("Load object failed: dimension size missing").toLatin1().data());
                    break;
                }
            }
        }

        if (!ret.containsError())
        {
            if (attrStream.hasAttribute("dimX"))
            {
                sizes[ndims - 1] = attrStream.value("dimX").toString().toInt();
            }
            else
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: dimX not specified").toLatin1().data());
            }
        }

        if (!ret.containsError())
        {
            if (attrStream.hasAttribute("dimY"))
            {
                sizes[ndims - 2] = attrStream.value("dimY").toString().toInt();
            }
            else
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: dimY not specified").toLatin1().data());
            }
        }

        /*if (!ret.containsError())
        {
            if (attrStream.hasAttribute("isTransposed"))
            {
                isTransposed = attrStream.value("isTransposed").toString().toInt();
            }
            else
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: isTransposed not specified").toLatin1().data());
            }
        }*/

        if (!ret.containsError())
        {
            if (type.compare("tUInt8") == 0)
            {
                elementsize = 1;
                objType = tUInt8;
            }
            else if (type.compare("tUInt16") == 0)
            {
                elementsize = 2;
                objType = tUInt16;
            }
            else if (type.compare("tUInt32") == 0)
            {
                elementsize = 4;
                objType = tUInt32;
            }
            else if (type.compare("tInt8") == 0)
            {
                elementsize = 1;
                objType = tInt8;
            }
            else if (type.compare("tInt16") == 0)
            {
                elementsize = 2;
                objType = tInt16;
            }
            else if (type.compare("tInt32") == 0)
            {
                elementsize = 4;
                objType = tInt32;
            }
            else if (type.compare("tFloat32") == 0)
            {
                elementsize = 4;
                objType = tFloat32;
            }
            else if (type.compare("tFloat64") == 0)
            {
                elementsize = 8;
                objType = tFloat64;
            }
            else if (type.compare("tComplex64") == 0)
            {
                elementsize = 8;
                objType = tComplex64;
            }
            else if (type.compare("tComplex128") == 0)
            {
                elementsize = 16;
                objType = tComplex128;
            }
            else if (type.compare("tRGBA32") == 0)
            {
                elementsize = 4;
                objType = tRGBA32;
            }
            else
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: type not supported").toLatin1().data());
            }
        }

        if (!ret.containsError())
        {
            dObjIn = DataObject(ndims, sizes, objType);
            if (dObjIn.getDims() < 2)
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: Error during allocating memory").toLatin1().data());
            }
            /*else
            {
                dObjIn.setT(isTransposed);
            }*/
        }

        if (NULL != sizes)
        {
            free(sizes);
        }

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This function fills the MetaData (DataObjectTags) of an allocated dataObject from the values of an xml-file
    *   \detail This function fills the MetaData (DataObjectTags) of an allocated dataObject from the values of an xml-file.
    *           This includes the axis-Tags (offset, scale, unit, description), value-Tags ((offset), (scale), unit, description) and the rotation matrix.
    *           It does not include the tag-Space (std::map<std::string, DataObjectTagType> m_tags) (e.g. protocol ...)
    *
    *   @param [in|out] stream      The xml-Stream from the xml-file
    *   @param [in|out] dObjIn      allocated dataObject
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ, DataObjectTags, DataObject
    */
    inline RetVal loadObjectHeaderFromXMLV1(QXmlStreamReader &stream, DataObject &dObjIn)
    {
        RetVal ret(retOk);
        QXmlStreamAttributes attrStream;
        QString elementName("");
        bool doubleAsBinary = false;
        int ndims = dObjIn.getDims();

        bool done = true;
        int loops = 0;

        //stream.readNext();
        //if (stream.isEndElement()) stream.readNext();
        stream.readNextStartElement();

        if (stream.atEnd())
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: file corrupted at metaData (v1.0)").toLatin1().data());
        }

        QString test = stream.qualifiedName().toString();

        if (stream.qualifiedName().toString().compare("metaData") != 0)
        {
            return RetVal(retWarning, 0, QObject::tr("Load object warning: file has invalid metaData for v1.0").toLatin1().data());
        }

        attrStream = stream.attributes();
        if (attrStream.hasAttribute("doubleExport"))
        {
            QString type = attrStream.value("doubleExport").toString();

            if (type.compare("d2s") == 0)
            {
                doubleAsBinary = false;
            }
            else if (type.compare("d2b") == 0)
            {
                doubleAsBinary = true;
            }
            else
            {
                return RetVal(retWarning, 0, QObject::tr("Load object warning: DoubleExportType for v1.0 invalid").toLatin1().data());
            }
        }
        else
        {
            return RetVal(retWarning, 0, QObject::tr("Load object warning: DoubleExportType for v1.0 missing").toLatin1().data());
        }

        for (char i = 0; i < ndims - 2; i++)
        {
            elementName = ("dim");
            elementName.append(QString::number(i));

            done = readTillNext(stream, loops, 10);

            attrStream = stream.attributes();
            if (stream.qualifiedName().toString().compare(elementName) == 0)
            {
                double dVal = 0.0;
                QString attrName("offset");
                ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
                dObjIn.setAxisOffset(i, dVal);

                dVal = 1.0;
                attrName = "scale";
                ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
                dObjIn.setAxisScale(i, dVal);
                std::string strVal("");
                attrName = "unit";
                ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
                dObjIn.setAxisUnit(i, strVal);

                strVal = "";
                attrName = "description";
                ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
                dObjIn.setAxisDescription(i, strVal);
            }
            else
            {
                QString warning = QObject::tr("Load object warning: MetaData for %1 missing").arg(elementName);
                ret += RetVal(retWarning, 0, warning.toLatin1().data());
            }
        }

        done = readTillNext(stream, loops, 10);

        elementName = "dimX";
        if (stream.qualifiedName().toString().compare(elementName) == 0)
        {
            attrStream = stream.attributes();
            double dVal = 0.0;
            QString attrName("offset");
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            dObjIn.setAxisOffset(ndims-1, dVal);

            dVal = 1.0;
            attrName = "scale";
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            dObjIn.setAxisScale(ndims-1, dVal);

            std::string strVal("");
            attrName = "unit";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setAxisUnit(ndims-1, strVal);

            strVal = "";
            attrName = "description";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setAxisDescription(ndims-1, strVal);
        }
        else
        {
            ret += RetVal(retWarning, 0, QObject::tr("Load object warning: MetaData for dimX missing").toLatin1().data());
        }

        done = readTillNext(stream, loops, 10);
        elementName = "dimY";
        if (stream.qualifiedName().toString().compare(elementName) == 0)
        {
            attrStream = stream.attributes();
            double dVal = 0.0;
            QString attrName("offset");
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            dObjIn.setAxisOffset(ndims-2, dVal);

            dVal = 1.0;
            attrName = "scale";
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            dObjIn.setAxisScale(ndims-2, dVal);

            std::string strVal("");
            attrName = "unit";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setAxisUnit(ndims-2, strVal);

            strVal = "";
            attrName = "description";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setAxisDescription(ndims-2, strVal);
        }
        else
        {
            ret += RetVal(retWarning, 0, QObject::tr("Load object warning: MetaData for dimY missing").toLatin1().data());
        }

        done = readTillNext(stream, loops, 10);
        elementName = "values";
        if (stream.qualifiedName().toString().compare(elementName) == 0)
        {
            attrStream = stream.attributes();
            double dVal = 0.0;
            QString attrName("offset");
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            //dObjIn.setValueOffset(dVal);

            dVal = 1.0;
            attrName = "scale";
            ret += readDoubleFromXML(attrStream, elementName, attrName, dVal, doubleAsBinary);
            //dObjIn.setValueScale(dVal);

            std::string strVal("");
            attrName = "unit";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setValueUnit(strVal);

            strVal = "";
            attrName = "description";
            ret += readStdStringFromXML(attrStream, elementName, attrName, strVal);
            dObjIn.setValueDescription(strVal);
        }
        else
        {
            ret += RetVal(retWarning, 0, QObject::tr("Load object warning: MetaData for values missing").toLatin1().data());
        }

        done = readTillNext(stream, loops, 10);
        elementName = "Rotation_Matrix";
        if (stream.qualifiedName().toString().compare(elementName) == 0)
        {
            attrStream = stream.attributes();
            double RD[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
            RetVal ret2(retOk);
            QString attrName("r11");
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[0], doubleAsBinary);
            attrName = "r12";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[1], doubleAsBinary);
            attrName = "r13";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[2], doubleAsBinary);
            attrName = "r21";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[3], doubleAsBinary);
            attrName = "r22";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[4], doubleAsBinary);
            attrName = "r23";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[5], doubleAsBinary);
            attrName = "r31";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[6], doubleAsBinary);
            attrName = "r32";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[7], doubleAsBinary);
            attrName = "r33";
            ret2 += readDoubleFromXML(attrStream, elementName, attrName, RD[8], doubleAsBinary);
            if (!ret2.containsWarningOrError()) dObjIn.setXYRotationalMatrix(RD[0], RD[1], RD[2], RD[3], RD[4], RD[5], RD[6], RD[7], RD[8]);
            else ret += RetVal(retWarning, 0, QObject::tr("Load object warning: MetaData import for Rotation Matrix failed").toLatin1().data());
        }
        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This function fills the tagsSpace (DataObjectTags) of an allocated dataObject from the values of an xml-file
    *   \detail This function fills the tagsSpace (DataObjectTags) of an allocated dataObject from the values of an xml-file.
    *           This onlye includes the tag-Space (std::map<std::string, DataObjectTagType> m_tags) (e.g. protocol ...) and does not
    *            include the axis-Tags (offset, scale, unit, description), value-Tags ((offset), (scale), unit, description) and the rotation matrix.
    *
    *   @param [in|out] stream      The xml-Stream from the xml-file
    *   @param [in|out] dObjIn      allocated dataObject
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ, DataObjectTags, DataObject
    */
    inline RetVal loadTagSpaceFromXMLV1(QXmlStreamReader &stream, DataObject &dObjIn)
    {
        RetVal ret(retOk);
        QXmlStreamAttributes attrStream;
        QByteArray cvalDecoded("");
        std::string tagName;
        QString tagValue;
        int loops;
        bool done;
        int nTags = 0;
        QString error;

        done = readTillNext(stream, loops, 10);

        attrStream = stream.attributes();

        if (stream.atEnd())
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: file corrupted at tagSpace (v1.0)").toLatin1().data());
        }

        if (!stream.qualifiedName().toString().compare("tagSpace") == 0)
        {
            error = QObject::tr("Load object failed: tag space not at expected position. Got %1 instead").arg(stream.qualifiedName().toString());
            return RetVal(retError, 0, error.toLatin1().data());
        }

        //TODO: what should this look like? in case put brackets here!
        if (stream.isStartElement())
        {
            if (attrStream.hasAttribute("tagNums"))
            {
                nTags = attrStream.value("tagNums").toString().toInt();
            }
            else
            {
                return RetVal(retWarning, 0, QObject::tr("Load object failed: tags Space invalid").toLatin1().data());
            }

            for (int i = 0; i < nTags; i++)
            {
                done = readTillNext(stream, loops, 10);
                attrStream = stream.attributes();
                if (!stream.isEndElement())  // okay tag-space is not empty
                {
                    cvalDecoded = QByteArray::fromPercentEncoding(stream.qualifiedName().toString().toLatin1());
                    tagName = cvalDecoded.data();

                    tagValue = stream.readElementText();

                    if (attrStream.hasAttribute("type"))
                    {
                        QString tagType = attrStream.value("type").toString();

                        if (tagType.compare("d2s") == 0)
                        {
                            double dVal = tagValue.toDouble();
                            dObjIn.setTag(tagName, dVal);
                        }
                        else if (tagType.compare("d2b") == 0)
                        {
                            double dVal = 0.0;
                            QByteArray cvalDecoded = QByteArray::fromBase64(tagValue.toLatin1());
                            memcpy(&dVal, cvalDecoded.data(), sizeof(double));
                            dObjIn.setTag(tagName, dVal);
                        }
                        else if (tagType.compare("s") == 0)
                        {
                            cvalDecoded = QByteArray::fromPercentEncoding(tagValue.toLatin1());
                            std::string sVal = cvalDecoded.data();
                            dObjIn.setTag(tagName, sVal);
                        }
                        else
                        {
                            ret += RetVal(retWarning, 0, QObject::tr("Load object warning: invalid tagType found").toLatin1().data());
                            continue;
                        }
                    }
                    else
                    {
                        ret += RetVal(retWarning, 0, QObject::tr("Load object warning: invalid tagType found").toLatin1().data());
                        continue;
                    }
                }
                else
                {
                    return RetVal(retWarning, 0, QObject::tr("Load object warning: tagsSpace invalid").toLatin1().data());
                }
            }
        }

        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   \brief This function copies the CDATA from the xml-file to the allocated dataObject.
    *   \detail This function copies the CDATA from the xml-file to the allocated dataObject.
    *           The data was before packed (substitution of xml-registered characters) during saving and is unpacked here.
    *           The data is stored plane-wise. The function checks if the plane-size if the object is equal to the imported size.
    *
    *   @param [in|out] stream      The xml-Stream from the xml-file
    *   @param [in|out] dObjIn      allocated dataObject
    *   @param [in] elementsize     Size of the each matrix-element
    *
    *   \sa loadXML2DOBJ, loadXML2EmptyDOBJ, DataObjectTags, DataObject
    */
    inline RetVal loadDataFromXMLV1(QXmlStreamReader &stream, DataObject &dObjIn, int elementsize)
    {
        RetVal ret(retOk);
        stream.readNextStartElement();
        int numPlanes = dObjIn.calcNumMats();
        int dims = dObjIn.getDims();
        int lineSize = dObjIn.getSize(dims-1) * elementsize;
        int lines = dObjIn.getSize(dims-2);
        int planeSize = lines*lineSize;
        QString error;

        bool done = true;
        int loops = 0;

        done = readTillNext(stream, loops, 10);

        if (stream.atEnd())
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: dataSpace missing").toLatin1().data());
        }

        if (!stream.qualifiedName().toString().compare("data") == 0)
        {
            error = QObject::tr("Load object failed: dataSpace not at expected position. Got %1 instead").arg(stream.qualifiedName().toString());
            return RetVal(retError, 0, error.toLatin1().data());
        }

        QXmlStreamAttributes attrStream = stream.attributes();

        if (attrStream.hasAttribute("planes"))
        {
            if (attrStream.value("planes").toString().toInt() != numPlanes)
            {
                ret += RetVal(retWarning, 0, QObject::tr("Load object warning: dataSpace and dataObject are not equal").toLatin1().data());
            }
        }
        else
        {
            ret +=  RetVal(retWarning, 0, QObject::tr("Load object warning: dataSpace attributes corrupted").toLatin1().data());
        }

        for (int i = 0; i < numPlanes; i++)
        {
           done = readTillNext(stream, loops, 10);
             //done = stream.readNext();
            if (!stream.isEndElement() && !stream.atEnd())  // okay tag-space is not empty
            {
                attrStream = stream.attributes();
                QByteArray cval = QByteArray::fromBase64(stream.readElementText().toLatin1());
                if (cval.length() != planeSize)
                {
                    error = QObject::tr("Load object warning: dataSpace for a plane corrupted. Got %1 instead of %2 bytes").arg(cval.length()).arg(planeSize);
                    return RetVal(retError, 0, error.toLatin1().data());
                }
                else
                {
                    char* dataptr = (char *)((cv::Mat *)dObjIn.get_mdata()[dObjIn.seekMat(i)])->ptr(0);
                    memcpy(dataptr, cval.data(), planeSize);
                }
            }
            else
            {
                return RetVal(retError, 0, QObject::tr("Load object failed: dataStream ended before finished reading").toLatin1().data());
            }
        }
        return ret;
    }

    //----------------------------------------------------------------------------------------------------------------------------------
    /**
    *   @param [Out] dObjIn          Destination dataContainter of type dataObject
    *   @param [in]  folderFileName  Folder and Filename of the Sourcefile
    *
    *   \details This function loads data from a xml-file to a dataObject. The file must be compatible to the file-format describted before.
    *   There are to possilbe import methods:
    *        1. onlyHeaderObjectFile == false tries to import a ido file with a itoDataObject-note/startelement.
    *        1. onlyHeaderObjectFile == true tries to import a idh file with a itoDataObjectHeader-note/startelement. The dataSpace will be ignored
    *
    *   \sa createObjectFromXMLV1, loadObjectHeaderFromXMLV1, loadTagSpaceFromXMLV1, loadDataFromXMLV1
    */
    RetVal loadXML2DOBJ(DataObject *dObjIn, QString folderFileName, bool onlyHeaderObjectFile, bool appendEnding)
    {
        RetVal ret(retOk);    /*!< Returnvalue for the complete function */
        QFile paramFile;     /*!< Handle to the source data */
        QString readSigns;

        int elementsize = 1;

        DataObject tempObjIn;

        if (!dObjIn)
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: Invalid object handle").toLatin1().data());
        }

        // First start with the properties of the file, check if it ist readable ...

        QFileInfo checkFile(folderFileName);

//        if (appendEnding || checkFile.suffix().isEmpty())
        // should this be && here, because when we use || and already have an ending, we endup with 2 endings
        if (appendEnding && checkFile.suffix().isEmpty())
        {
            if (onlyHeaderObjectFile)
            {
                folderFileName.append(".idh");
            }
            else
            {
                folderFileName.append(".ido");
            }

            checkFile.setFile(folderFileName);
        }

        paramFile.setFileName(checkFile.canonicalFilePath());

        if (!checkFile.isReadable() && !checkFile.exists())
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: file not readable or does not exists").toLatin1().data());
        }

        // open the file

        paramFile.open(QIODevice::ReadOnly);

        // Get the XML-Stream

        QXmlStreamReader stream(&paramFile);
        QXmlStreamAttributes attrStream;

        if (stream.atEnd())
        {
            return RetVal(retError, 0, QObject::tr("Load object failed: file seems corrupt").toLatin1().data());
        }
        else
        {
            readSigns = stream.documentVersion().toString();
			const QString stringToComp = "1.0";

            if (readSigns.compare(stringToComp) == 0)
            {
                paramFile.close();
                return  RetVal(retError, 0, QObject::tr("Load object failed: wrong xml version").toLatin1().data());
            }
        }

        readSigns = stream.documentEncoding().toString();
		const QString stringToComp = "UTF-8";

        if (readSigns.compare(stringToComp) == 0)
        {
            paramFile.close();
            return RetVal(retError, 0, QObject::tr("Load object failed: wrong document encoding").toLatin1().data());
        }

        if (stream.atEnd())
        {
            paramFile.close();
            return RetVal(retError, 0, QObject::tr("Load object failed: unexpected file ending").toLatin1().data());
        }

        stream.readNextStartElement();

        QString startNoteName("");

        if (onlyHeaderObjectFile)
        {
            startNoteName.append("itoDataObjectHeader");
        }
        else
        {
            startNoteName.append("itoDataObject");
        }

        if (stream.qualifiedName().toString().compare(startNoteName) == 0)
        {
            paramFile.close();
            return  RetVal(retError, 0, QObject::tr("Load object failed: file is no itomDataObjectFile").toLatin1().data());
        }

        attrStream = stream.attributes();

        if (attrStream.hasAttribute("FormatVersion"))
        {
            if (attrStream.value("FormatVersion").toString().compare("1.0") == 0)
            {
                ret += createObjectFromXMLV1(stream, tempObjIn, elementsize);                               // Create the object by xml-parameter

                if (!ret.containsError())
                {
                    ret += loadObjectHeaderFromXMLV1(stream, tempObjIn);               // Fill meta data
                }

                if (!ret.containsError())
                {
                    ret += loadTagSpaceFromXMLV1(stream, tempObjIn);                   // Fill tag-Map data
                }

                if (!onlyHeaderObjectFile)
                {
                    if (!ret.containsError())
                    {
                        ret += loadDataFromXMLV1(stream, tempObjIn, elementsize);          // Fill dataSpace of the dataObject
                    }
                }
            }
            else
            {
                ret += RetVal(retError, 0, QObject::tr("Load object failed: illegal format version").toLatin1().data());
            }
        }
        else
        {
            ret += RetVal(retError, 0, QObject::tr("Load object failed: object header not valied").toLatin1().data());
        }

        // Get the data
        if (!ret.containsError())
        {
            (*dObjIn) = tempObjIn;
        }

        paramFile.close();
        return ret;
    }

}   // end namespace ito
