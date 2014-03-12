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

#include "../organizer/processOrganizer.h"

using namespace ito;

namespace ito
{

//-----------------------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------------------
HelpSystem::HelpSystem() :
    m_upToDate(false),
    m_upToDatePlugins(false)
{
    m_helpCollectionName = "itomHelpProject.qhc";
    m_helpCollectionProject = "itomHelpProject.qhcp";
    m_pluginHelpCollectionName = "pluginHelpProject.qhc";
    m_pluginHelpCollectionProject = "pluginHelpProject.qhcp";
    QDir appPath(QDir::cleanPath(QCoreApplication::applicationDirPath()));

    if(!appPath.exists("help"))
    {
        appPath.mkdir("help");
    }

    appPath.cd("help");
    m_helpDirectory = appPath;


    //buildPluginHelp();
}

//-----------------------------------------------------------------------------------------
HelpSystem::~HelpSystem()
{
    HelpSystem::m_pHelpSystem = NULL;
}

//-----------------------------------------------------------------------------------------
RetVal HelpSystem::rebuildHelpIfNotUpToDate()
{
    RetVal retValue(retOk);
    quint16 checksum1 = 0;
    quint16 checksum2 = 0;
    QStringList documentationFiles;

    //check itom documentation
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

    //check plugin documentation
    checksum1 = 0;
    checksum2 = 0;
    getCheckSumOfBuild(m_helpDirectory, m_pluginHelpCollectionProject, checksum1);
    scanPluginQhpFiles(checksum2);

    if(checksum1 == checksum2 && checksum1 != 0)
    {
        m_upToDatePlugins = true;
    }
    else
    {
        m_upToDatePlugins = false;
        retValue += rebuildHelpCollection(documentationFiles, checksum2, m_helpDirectory);
        m_upToDatePlugins = true;
    }
    return retValue;
}

//-----------------------------------------------------------------------------------------
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


//-----------------------------------------------------------------------------------------
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
    
    ////plugin base folder
    //appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
    //i=1;
    //while(appPath.exists("plugins") == false && i > 0)
    //{
    //    appPath.cdUp();
    //    --i;
    //}

    //if(appPath.exists("plugins"))
    //{
    //    baseFolders << appPath.filePath("plugins");
    //}

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

    checksum = qChecksum(checksumString.toLatin1().data(), checksumString.size());

    return RetVal(retOk);
}

//-----------------------------------------------------------------------------------------
RetVal HelpSystem::scanPluginQhpFiles(quint16 &checksum)
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

    checksum = 0;

    foreach(const QString &baseFolder, baseFolders)
    {
        QDirIterator it(baseFolder, QStringList("*.qhp"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
        while(it.hasNext())
        {
            temp = it.next();
            fileInfo.setFile(temp);
            checksumString.append(fileInfo.fileName()).append(fileInfo.lastModified().toString());
        }
    }

    checksum = qChecksum(checksumString.toLatin1().data(), checksumString.size());

    return RetVal(retOk);
}

//-----------------------------------------------------------------------------------------
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
            return RetVal(retError, 0, QObject::tr("file could not be opened.").toLatin1().data());
        }

        QXmlStreamReader stream(&file);
        QStringRef ReadSigns;
        QXmlStreamAttributes attr;

        if(stream.atEnd())
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed: file seems corrupt").toLatin1().data());
        }

        ReadSigns = stream.documentVersion();
        if(!ReadSigns.compare("1.0"))
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed:  wrong xml version").toLatin1().data());
        }

        ReadSigns = stream.documentEncoding();
        if(!ReadSigns.compare("UTF-8"))
        {
            file.close();
            return RetVal(retError, 0, QObject::tr("Load XML file failed: wrong document encoding").toLatin1().data());
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
                    return RetVal(retError, 0, QObject::tr("Load XML file failed: could not intepret checksum content as uint").toLatin1().data());
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

//-----------------------------------------------------------------------------------------
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
        return RetVal(retError, 0, QObject::tr("collection project file could not be opened").toLatin1().data());
    }

    QXmlStreamWriter stream(&file);
    stream.setCodec("UTF-8");       // Set text codec
    stream.setAutoFormatting(true);

    stream.writeStartDocument();

    stream.writeStartElement("QHelpCollectionProject");
    stream.writeAttribute("version", "1.0");
    stream.writeAttribute("itomChecksum", QString::number(checksum));

    stream.writeStartElement("assistant");
    stream.writeTextElement("title", "Documentation itom");
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

    QString app = ProcessOrganizer::getAbsQtToolPath( "qcollectiongenerator" );

    process.start(app.toLatin1().data() , args);
    process.waitForFinished(60000);


    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------------
RetVal HelpSystem::buildPluginHelp()
{
    RetVal retval;
    QDir templateDir;
    QDir buildDir;
    QDir pluginDir;

    templateDir = QDir(QCoreApplication::applicationDirPath());
    if (!templateDir.cd("docs/pluginDoc/template"))
    {
        retval += ito::RetVal(ito::retError,0,"templates for plugin documentation not found. Directory 'docs/pluginDoc/template' not available");
    }

    buildDir = QDir(QCoreApplication::applicationDirPath());
    buildDir.cd("docs/pluginDoc");

    if (!buildDir.exists("build"))
    {
        //create empty build folder
        if (buildDir.mkdir("build"))
        {
            buildDir.cd("build");
        }
        else
        {
            retval += ito::RetVal(ito::retError,0,"folder 'build' as subfolder of 'docs/pluginDoc' could not be created");
        }
    }
    else
    {
        buildDir.cd("build");

        //clear content of build folder
        if (!HelpSystem::removeDir(buildDir))
        {
            retval += ito::RetVal(ito::retError,0,"could not clear folder 'docs/pluginDoc/build'");
        }
        
    }

    pluginDir = QDir(QCoreApplication::applicationDirPath());
    if (!pluginDir.cd("plugins"))
    {
        retval += ito::RetVal(ito::retWarning,0,"no plugin directory available. No plugin documentation will be built");
    }

    if (!retval.containsError())
    {
        //copy content of _static folder of template folder to build/_static
        if (!copyDir(templateDir.filePath("_static"), buildDir.filePath("_static")))
        {
            retval += ito::RetVal(ito::retError,0,"could not copy folder 'docs/pluginDoc/template/_static' to 'docs/pluginDoc/build/_static'");
        }
    }

    if (!retval.containsError())
    {
        QDir thisPluginDir;
        QDir thisPluginDocsDir;
        QDir thisPluginBuildDir;

        QString tocs;
        QString keywords;
        QString files;

        //scan all folders in pluginDir and check if they have a docs subfolder containing a qhp-file
        foreach(QFileInfo info, pluginDir.entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot))
        {
            if (info.isDir())
            {
                qDebug() << info.absoluteFilePath();
                thisPluginDir = info.absoluteFilePath();
                thisPluginDocsDir = thisPluginDir;
                if (thisPluginDocsDir.cd("docs"))
                {
                    if (thisPluginDocsDir.entryInfoList(QStringList() << "*.qhp", QDir::Files).size() > 0)
                    {
                        if (buildDir.mkdir(thisPluginDir.dirName()))
                        {
                            thisPluginBuildDir = buildDir;
                            thisPluginBuildDir.cd(thisPluginDir.dirName());
                            retval += buildSinglePluginHelp(thisPluginBuildDir, thisPluginDocsDir, tocs, keywords, files);
                        }
                    }
                }
            }
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------
RetVal HelpSystem::buildSinglePluginHelp(QDir &buildDir, QDir &sourceDir, QString &tocs, QString &keywords, QString &files)
{
    ito::RetVal retval;

    //check if there is an _image folder in sourceDir and copy it to buildDir
    if (sourceDir.exists("_image"))
    {
        copyDir(sourceDir.absoluteFilePath("_image"), buildDir.absoluteFilePath("_image"));
    }

    foreach(QFileInfo info, sourceDir.entryInfoList(QDir::Files))
    {
        if (QString::compare(info.suffix(),"qhp",Qt::CaseInsensitive) == 0)
        {
            //analyze qhp file and return content in
            retval += analyzeQhpFile(QFile(info.absoluteFilePath()), tocs, keywords, files);
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------
RetVal HelpSystem::analyzeQhpFile(QFile &qhpFile, QString &tocs, QString &keywords, QString &files)
{
    if (qhpFile.open(QIODevice::ReadOnly))
    {
        QByteArray content = qhpFile.readAll();
        int start = content.indexOf("<toc>",0) + qstrlen("<toc>");
        int end = content.indexOf("</toc>",start);

        if (end > start)
        {
            tocs += content.mid(start,end-start);
        }

        start = content.indexOf("<keywords>",0) + qstrlen("<keywords>");
        end = content.indexOf("</keywords>",start);

        if (end > start)
        {
            keywords += content.mid(start,end-start);
        }

        start = content.indexOf("<files>",0) + qstrlen("<files>");
        end = content.indexOf("</files>",start);

        if (end > start)
        {
            files += content.mid(start,end-start);
        }
        qhpFile.close();
    }

    return ito::retOk;
}


//-------------------------------------------------------------------------------------------
/*static*/ bool HelpSystem::removeDir(const QDir &directory)
{
    bool result = true;

    Q_FOREACH(QFileInfo info, directory.entryInfoList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst)) 
    {
        if (info.isDir()) 
        {
            result = removeDir(info.absoluteFilePath());
            if (result)
            {
                result = info.absoluteDir().rmdir(info.fileName());
            }
        }
        else
        {
            result = QFile::remove(info.absoluteFilePath());
        }

        if (!result) 
        {
            return result;
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------------
/*static*/ bool HelpSystem::copyDir(const QDir &src, const QDir &dst)
{
    removeDir(dst);

    if (!dst.exists())
    {
        QDir dstTmp = dst;
        if (dstTmp.cdUp())
        {
            dstTmp.mkdir(dst.dirName());
        }
        else
        {
            return false;
        }
    }
    
    foreach(const QFileInfo &info, src.entryInfoList(QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot)) 
    {
        QString srcItemPath = src.filePath(info.fileName());
        QString dstItemPath = dst.filePath(info.fileName());
        if (info.isDir()) 
        {
            if (!copyDir(srcItemPath, dstItemPath)) 
            {
                return false;
            }
        } else if (info.isFile()) 
        {
            if (!QFile::copy(srcItemPath, dstItemPath)) 
            {
                return false;
            }
        } else 
        {
            qDebug() << "Unhandled item" << info.filePath() << "in cpDir";
        }
    }
    return true;
}



} //end namespace ito