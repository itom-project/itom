/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom.
  
    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "helpSystem.h"

#include <qfileinfo.h>
#include <qfile.h>
#include <qtextstream.h>
#include <qprocess.h>
#include <qcoreapplication.h>
#include <qdiriterator.h>
#include <qdatetime.h>

#include <qxmlstream.h>
#include <qdebug.h>

using namespace ito;

namespace ito
{

HelpSystem* HelpSystem::m_pHelpSystem = NULL;

/*static*/ HelpSystem * HelpSystem::getInstance(void)
{
    static HelpSystemSingleton w;
    if (HelpSystem::m_pHelpSystem == NULL)
    {
        HelpSystem::m_pHelpSystem = new ito::HelpSystem();
    }
    return HelpSystem::m_pHelpSystem;
}

HelpSystem::HelpSystem() :
    m_upToDate(false)
{
    m_helpCollectionName = "itomHelpProject.qhc";
    m_helpCollectionProject = "itomHelpProject.qhcp";
    QDir appPath(QDir::cleanPath(QCoreApplication::applicationDirPath()));

    if(!appPath.exists("help"))
    {
        appPath.mkdir("help");
    }

    appPath.cd("help");
    m_helpDirectory = appPath;



}

HelpSystem::~HelpSystem()
{
    HelpSystem::m_pHelpSystem = NULL;
}

RetVal HelpSystem::rebuildHelpIfNotUpToDate()
{
    RetVal retValue(retOk);
    quint16 checksum1 = 0;
    quint16 checksum2 = 0;
    QStringList documentationFiles;
    getCheckSumOfBuild(m_helpDirectory, m_helpCollectionProject, checksum1);
    scanDocumentationFiles(documentationFiles,checksum2);

    if(checksum1 == checksum2 && checksum1 != 0)
    {
        m_upToDate = true;
    }
    else
    {
        m_upToDate = false;
        retValue += rebuildHelpCollection(documentationFiles, checksum2, m_helpDirectory);
        m_upToDate = true;

    }

    return retValue;
}


QString HelpSystem::getHelpCollectionAbsFileName() const
{
    if(m_upToDate)
    {
        return m_helpDirectory.filePath(m_helpCollectionName);
    }
    else
    {
        return QString();
    }
}



RetVal HelpSystem::scanDocumentationFiles(QStringList &qchFiles, quint16 &checksum)
{
    QStringList baseFolders;

    QDir appPath;
    QDir folder;
    QString temp;
    QString checksumString;
    QFileInfo fileInfo;
    int i;

    //documentation folder
    appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
    i=1;
    while(appPath.exists("docs/userDoc") == false && i > 0)
    {
        appPath.cdUp();
        --i;
    }

    if(appPath.exists("docs/userDoc"))
    {
        baseFolders << appPath.filePath("docs/userDoc");
    }
    
    //plugin base folder
    appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
    i=1;
    while(appPath.exists("plugins") == false && i > 0)
    {
        appPath.cdUp();
        --i;
    }

    if(appPath.exists("plugins"))
    {
        baseFolders << appPath.filePath("plugins");
    }

    qchFiles.clear();
    checksum = 0;

    foreach(const QString &baseFolder, baseFolders)
    {
        QDirIterator it(baseFolder, QStringList("*.qch"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
        while(it.hasNext())
        {
            temp = it.next();
            fileInfo.setFile(temp);
            checksumString.append(fileInfo.fileName()).append(fileInfo.lastModified().toString());
            qchFiles.append(temp);
        }

    }

    checksum = qChecksum(checksumString.toAscii().data(), checksumString.size());

    return RetVal(retOk);
}

RetVal HelpSystem::getCheckSumOfBuild(QDir &helpDir, QString &projectFileName, quint16 &checksum)
{
    if(helpDir.exists(projectFileName) == false)
    {
        checksum = 0;
    }
    else
    {
        checksum = 0;
        QFile file(helpDir.filePath(projectFileName));
        if(!file.open(QIODevice::ReadOnly))
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("file could not be opened.").toAscii().data());
        }

        QXmlStreamReader stream(&file);
        QStringRef ReadSigns;
        QXmlStreamAttributes attr;

        if(stream.atEnd())
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed: file seems corrupt").toAscii().data());
        }

        ReadSigns = stream.documentVersion();
        if(!ReadSigns.compare("1.0"))
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed:  wrong xml version").toAscii().data());
        }

        ReadSigns = stream.documentEncoding();
        if(!ReadSigns.compare("UTF-8"))
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed: wrong document encoding").toAscii().data());
        }

        while(stream.readNextStartElement())
        {
            qDebug() << stream.name();
            if(stream.name() == "QHelpCollectionProject")
            {
                attr = stream.attributes();
                ReadSigns = attr.value("itomChecksum");
                bool ok = false;
                checksum = ReadSigns.toString().toUInt(&ok);
                if(!ok)
                {
                    file.close();
                    return RetVal(retError, 0, QObject::tr("Load XML file failed: could not intepret checksum content as uint").toAscii().data());
                }
                else
                {
                    break;
                }
            }
            else
            {
                stream.skipCurrentElement();
            }

            /*if(stream.qualifiedName().compare("itomChecksum"))
            {
                bool ok = false;
                checksum = stream.readElementText().toUInt(&ok);
                if(!ok)
                {
                    file.close();
                    return RetVal(retError, 0, "Load XML file failed: could not intepret checksum content as uint");
                }
                break;
            }*/
        }
        file.close();
    }

    return RetVal(retOk);
}


RetVal HelpSystem::rebuildHelpCollection(QStringList &qchFiles, quint16 checksum, QDir &helpDir)
{
    helpDir.setNameFilters(QStringList("*.qch"));
    helpDir.setFilter(QDir::Files | QDir::NoDotAndDotDot);
    QFile file;
    QFileInfo fileInfo;
    QStringList oldFiles = helpDir.entryList();
    QStringList baseFileNames;
    QString destination;
    foreach(const QString &fileName, oldFiles)
    {
        m_helpDirectory.remove(fileName);
    }

    if(m_helpDirectory.exists(m_helpCollectionName))
    {
        m_helpDirectory.remove(m_helpCollectionName);
    }
    if(m_helpDirectory.exists(m_helpCollectionProject))
    {
        m_helpDirectory.remove(m_helpCollectionProject);
    }

    //copy all qchFiles to m_helpDirectory
    
    foreach(const QString &fileName, qchFiles)
    {
        file.setFileName(fileName);
        fileInfo.setFile(file);
        qDebug() << helpDir;
        qDebug() << fileInfo.fileName();
        destination = helpDir.filePath(fileInfo.fileName());
        qDebug() << file.copy( destination );

        baseFileNames << QString(fileInfo.baseName());

    }

    //create helpCollectionProject
    file.setFileName(m_helpDirectory.filePath(m_helpCollectionProject));
    if(!file.open(QIODevice::WriteOnly))
    {
        file.close();
        return RetVal(retError, 0, QObject::tr("collection project file could not be opened").toAscii().data());
    }

    QXmlStreamWriter stream(&file);
    stream.setCodec("UTF-8");       // Set text codec
    stream.setAutoFormatting(true);

    stream.writeStartDocument();

    stream.writeStartElement("QHelpCollectionProject");
    stream.writeAttribute("version", "1.0");
    stream.writeAttribute("itomChecksum", QString::number(checksum));

    stream.writeStartElement("assistant");
    stream.writeTextElement("title", "Documentation ITOM");
    stream.writeTextElement("homePage", "qthelp://org.sphinx.itom.0.0/doc/index.html");
    stream.writeTextElement("startPage", "qthelp://org.sphinx.itom.0.0/doc/index.html");
    stream.writeEndElement(); //assistant

    stream.writeStartElement("docFiles");

    //stream.writeStartElement("generate");
    //foreach(const QString &elem, baseFileNames)
    //{
    //    stream.writeStartElement("file");
    //    stream.writeTextElement("input", QString(elem).append(".qhp"));
    //    stream.writeTextElement("output", QString(elem).append(".qch"));
    //    stream.writeEndElement(); //file
    //}
    //stream.writeEndElement(); //generate

    stream.writeStartElement("register");
    foreach(const QString &elem, baseFileNames)
    {
        stream.writeTextElement("file", QString(elem).append(".qch"));
    }
    stream.writeEndElement(); //register

    stream.writeEndElement(); //docFiles

    stream.writeEndElement(); // QHelpCollectionProject
    stream.writeEndDocument();

    file.close();

    QProcess process;
    QStringList args;
    args << file.fileName();
    qDebug() << args;
    process.start("qcollectiongenerator", args);
    process.waitForFinished(60000);


    return RetVal(retOk);
}



} //end namespace ito