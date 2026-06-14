/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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
#include <qregularexpression.h>

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
    m_upToDate(false)
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
}

//-----------------------------------------------------------------------------------------
HelpSystem::~HelpSystem()
{
    HelpSystem::m_pHelpSystem = NULL;
}

//-----------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \return RetVal
*/
RetVal HelpSystem::rebuildHelpIfNotUpToDate()
{
    RetVal retValue(retOk);
    quint16 checksum1 = 0;
    quint16 checksum2 = 0;
    QStringList documentationFiles;

    //check if qch file of plugins need to be build or rebuild
    retValue += getCheckSumOfPluginBuild(checksum1); //the checksum of the latest built plugin documentation is in the file .checksum in itom/docs/pluginDoc/build
    retValue += scanPluginQhpFiles(checksum2); //this checksum is built of the change-date and the filename of all qhp-files found in itom/plugins directory

    if (checksum1 != checksum2)
    {
        retValue += buildPluginHelp(checksum2);
    }

    //check itom documentation
    checksum1 = 0;
    checksum2 = 0;
    retValue += getCheckSumOfBuild(m_helpDirectory, m_helpCollectionProject, checksum1);
    retValue += scanDocumentationFiles(documentationFiles,checksum2);

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

//-----------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \return QString
*/
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
//! shortdesc
/*! longdesc

    \param qchFiles
    \param checksum
    \return RetVal
*/
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
    appPath.setPath(QDir::cleanPath(QCoreApplication::applicationDirPath()));
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

    if (appPath.exists("docs/pluginDoc/build"))
    {
        baseFolders << appPath.filePath("docs/pluginDoc/build");
    }

    if (appPath.exists("docs/additionalDocs"))
    {
        baseFolders << appPath.filePath("docs/additionalDocs");
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

    checksum = qChecksum(checksumString.toLatin1().data(), checksumString.size());

    return RetVal(retOk);
}

//-----------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param checksum
    \return RetVal
*/
RetVal HelpSystem::scanPluginQhpFiles(quint16 &checksum)
{
    QStringList baseFolders;

    QDir appPath;
    QDir folder;
    QString temp;
    QString checksumString;
    QFileInfo fileInfo;
    int i;

    //plugin base folder
    appPath.setPath(QDir::cleanPath(QCoreApplication::applicationDirPath()));
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

//---------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param checksum
    \return RetVal
*/
RetVal HelpSystem::getCheckSumOfPluginBuild(quint16 &checksum)
{
    ito::RetVal retval;
    checksum = 0; //not valid
    //documentation folder
    QDir appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());

    if (appPath.exists("docs/pluginDoc/build"))
    {
        appPath.cd("docs/pluginDoc/build");

        if (appPath.exists(".checksum"))
        {
            QFile file(appPath.absoluteFilePath(".checksum"));
            if (file.open(QIODevice::ReadOnly))
            {
                QString checksumStr = file.readAll();
                bool ok;
                checksum = checksumStr.toInt(&ok);

                if (!ok)
                {
                    checksum = 0;
                }

                file.close();
            }
        }

    }

    return retval;
}

//-----------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param helpDir
    \param projectFileName
    \param checksum
    \return RetVal
*/
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
            return RetVal(retWarning, 0, QObject::tr("File could not be opened.").toLatin1().data());
        }

        QXmlStreamReader stream(&file);
        QString readSigns;
        QXmlStreamAttributes attr;

        if(stream.atEnd())
        {
            file.close();
            return RetVal(retWarning, 0, QObject::tr("Load XML file failed: file seems corrupt").toLatin1().data());
        }

        readSigns = stream.documentVersion().toString();
		const QString stringToComp = "1.0";

        if(!readSigns.compare(stringToComp))
        {
            file.close();
            return RetVal(retWarning, 0, QObject::tr("Load XML file failed:  wrong xml version").toLatin1().data());
        }

        readSigns = stream.documentEncoding().toString();
		const QString stringToCompUTF = "UTF-8";

        if(!readSigns.compare(stringToCompUTF))
        {
            file.close();
            return RetVal(retWarning, 0, QObject::tr("Load XML file failed: wrong document encoding").toLatin1().data());
        }

        while(stream.readNextStartElement())
        {
            qDebug() << stream.name();
            if (stream.name().toString().contains(QString("QHelpCollectionProject")))
            {
                attr = stream.attributes();
                readSigns = attr.value("itomChecksum").toString();
                bool ok = false;
                checksum = readSigns.toUInt(&ok);
                if(!ok)
                {
                    checksum = 0;
                    file.close();
                    return RetVal(retWarning, 0, QObject::tr("Load XML file failed: could not interpret checksum content as uint").toLatin1().data());
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
        }
        file.close();
    }

    return RetVal(retOk);
}

//-----------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param qchFiles
    \param checksum
    \param helpDir
    \return RetVal
*/
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
        return RetVal(retError, 0, QObject::tr("Collection project file could not be opened").toLatin1().data());
    }

    QXmlStreamWriter stream(&file);
    // Qt5: UTF-8 is the default codec, Qt6: uses always UTF-8
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

#if (QT_VERSION >= QT_VERSION_CHECK(5, 12, 0))
    // https://blog.qt.io/blog/2018/11/02/whats-new-qt-help/
	QString app = ProcessOrganizer::getAbsQtToolPath("qhelpgenerator");
#else
    QString app = ProcessOrganizer::getAbsQtToolPath("qcollectiongenerator");
#endif

    process.start(app.toLatin1().data() , args);

    if (!process.waitForFinished(30000))
    {
#if (QT_VERSION >= QT_VERSION_CHECK(5, 12, 0))
        // https://blog.qt.io/blog/2018/11/02/whats-new-qt-help/
        return RetVal(
            retError, 0, QObject::tr("Error calling qhelpgenerator").toLatin1().data());
#else
        return RetVal(
            retError, 0, QObject::tr("Error calling qcollectiongenerator").toLatin1().data());
#endif

    }



    return RetVal(retOk);
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param checksum
    \return RetVal
*/
RetVal HelpSystem::buildPluginHelp(quint16 checksum)
{
    RetVal retval;
    QDir templateDir;
    QDir buildDir;
    QDir pluginDir;

    QString tocs;
    QString keywords;
    QString files;
    QPair<QString,QString> mainFileInfo;
    QList< QPair<QString, QString> > mainFileInfos;

    templateDir = QDir(QCoreApplication::applicationDirPath());
    if (!templateDir.cd("docs/pluginDoc/template"))
    {
        retval += ito::RetVal(ito::retWarning,0,QObject::tr("Templates for plugin documentation not found. Directory 'docs/pluginDoc/template' not available. Plugin documentation will not be built.").toLatin1().data());
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
            retval += ito::RetVal(ito::retWarning,0,QObject::tr("Folder 'build' as subfolder of 'docs/pluginDoc' could not be created. Plugin documentation will not be built.").toLatin1().data());
        }
    }
    else
    {
        buildDir.cd("build");

        //clear content of build folder
        if (!HelpSystem::removeDir(buildDir))
        {
            retval += ito::RetVal(ito::retWarning,0,QObject::tr("Could not clear folder 'docs/pluginDoc/build'. Plugin documentation will not be built.").toLatin1().data());
        }

    }

    pluginDir = QDir(QCoreApplication::applicationDirPath());
    if (!pluginDir.cd("plugins"))
    {
        retval += ito::RetVal(ito::retWarning,0,QObject::tr("No plugin directory available. No plugin documentation will be built.").toLatin1().data());
    }

    if (!retval.containsWarningOrError())
    {
        QDir thisPluginDir;
        QDir thisPluginDocsDir;
        QDir thisPluginBuildDir;
        QString warnings;

        //scan all folders in pluginDir and check if they have a docs subfolder containing a qhp-file
        foreach(QFileInfo info, pluginDir.entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot))
        {
            if (info.isDir())
            {
                qDebug() << info.absoluteFilePath();
                thisPluginDir.setPath(info.absoluteFilePath());
                thisPluginDocsDir = thisPluginDir;
                if (thisPluginDocsDir.cd("docs"))
                {
                    if (thisPluginDocsDir.entryInfoList(QStringList() << "*.qhp", QDir::Files).size() > 0)
                    {
                        //create subfolder for the plugin and copy/modify necessary files inside of this folder
                        if (buildDir.mkdir(thisPluginDir.dirName()))
                        {
                            thisPluginBuildDir = buildDir;
                            thisPluginBuildDir.cd(thisPluginDir.dirName());
                            ito::RetVal pluginRetVal = buildSinglePluginHelp(thisPluginDir.dirName(), thisPluginBuildDir, thisPluginDocsDir, tocs, keywords, files, mainFileInfo);

                            if (pluginRetVal == ito::retOk)
                            {
                                mainFileInfos.append(mainFileInfo);
                            }
                            else if (pluginRetVal == ito::retError)
                            {
                                retval += pluginRetVal;
                                break;
                            }
                            else //warning
                            {
                                warnings += "\nPlugin " + thisPluginDir.dirName() + ": " + pluginRetVal.errorMessage();
                            }
                        }
                    }
                }
            }
        }

        if (warnings.isEmpty() == false)
        {
            retval += ito::RetVal(ito::retWarning,0,warnings.toLatin1().data());
        }
    }

    if (mainFileInfos.size() > 0)
    {

        QByteArray mainIndexFile;
        QByteArray mainQhpFile;
        QFile file;

        if (!retval.containsWarningOrError())
        {

            //index.html
            file.setFileName(templateDir.absoluteFilePath("index.html"));
            if (file.open(QIODevice::ReadOnly))
            {
                mainIndexFile = file.readAll();
                file.close();
            }
            else
            {
                retval += ito::RetVal(ito::retWarning,0,QObject::tr("Error opening index.html of template folder").toLatin1().data());
            }

            //itomPluginDoc.qhp
            file.setFileName(templateDir.absoluteFilePath("itomPluginDoc.qhp"));
            if (file.open(QIODevice::ReadOnly))
            {
                mainQhpFile = file.readAll();
                file.close();
            }
            else
            {
                retval += ito::RetVal(ito::retWarning,0,QObject::tr("Error opening itomPluginDoc.qhp of template folder").toLatin1().data());
            }
        }

        if (!retval.containsWarningOrError())
        {
            mainQhpFile.replace("$tocInsert$", tocs.toLatin1());
            mainQhpFile.replace("$keywordsInsert$", keywords.toLatin1());
            mainQhpFile.replace("$filesInsert$", files.toLatin1());

            mainIndexFile.replace("$firstDocTitle$", mainFileInfos[0].first.toLatin1());
            mainIndexFile.replace("$firstDocHref$", mainFileInfos[0].second.toLatin1());

			QString currentYear = QDateTime::currentDateTime().toString("yyyy");
			mainIndexFile.replace("$currentYear$", currentYear.toLatin1());

			QString currentDate = QDateTime::currentDateTime().toString("MMM dd yyyy");
			mainIndexFile.replace("$currentDate$", currentDate.toLatin1());

            QString sectionEntry;
            int start = mainIndexFile.indexOf("<!--$toctree_item begin$");
            int end = mainIndexFile.indexOf("$toctree_item end$-->");

            if (end > start)
            {
                start += qstrlen("<!--$toctree_item begin$");
                sectionEntry = mainIndexFile.mid(start, end-start);
                QString sectionEntries;

                for (int i = 0; i < mainFileInfos.size(); ++i)
                {
                    QString temp = sectionEntry;
                    temp.replace("$toctreeItemHref$", mainFileInfos[i].second.toLatin1());
                    temp.replace("$toctreeItemTitle$", mainFileInfos[i].first.toLatin1());
                    sectionEntries += temp;
                    sectionEntries += "\n";
                }

                mainIndexFile.replace("<!--$toctree_item insert$-->", sectionEntries.toLatin1());
            }
        }

        if (!retval.containsWarningOrError())
        {
            //index.html
            file.setFileName(buildDir.absoluteFilePath("index.html"));
            if (file.open(QIODevice::WriteOnly))
            {
                file.write(mainIndexFile);
                file.close();
            }
            else
            {
                retval += ito::RetVal(ito::retWarning,0,QObject::tr("Error writing index.html of template folder").toLatin1().data());
            }

            //itomPluginDoc.qhp
            file.setFileName(buildDir.absoluteFilePath("itomPluginDoc.qhp"));
            if (file.open(QIODevice::WriteOnly))
            {
                file.write(mainQhpFile);
                file.close();
            }
            else
            {
                retval += ito::RetVal(ito::retWarning,0,QObject::tr("Error writing itomPluginDoc.qhp of template folder").toLatin1().data());
            }
        }

        if (!retval.containsWarningOrError())
        {
            //copy content of _static folder of template folder to build/_static
            if (!copyDir(templateDir.filePath("_static"), buildDir.filePath("_static")))
            {
                retval += ito::RetVal(ito::retWarning,0,QObject::tr("Could not copy folder 'docs/pluginDoc/template/_static' to 'docs/pluginDoc/build/_static'").toLatin1().data());
            }
        }


        if (!retval.containsWarningOrError())
        {
            QProcess process;
            QStringList args;
            args << buildDir.absoluteFilePath("itomPluginDoc.qhp");

            QString app = ProcessOrganizer::getAbsQtToolPath( "qhelpgenerator" );

            process.start(app.toLatin1().data() , args);
            if (!process.waitForFinished(30000))
            {
                retval += RetVal(retWarning,0,QObject::tr("Error calling qhelpgenerator for creating the plugin documentation.").toLatin1().data());
            }

        }
    }

    if (!retval.containsWarningOrError())
    {
        QFile checksumFile(buildDir.absoluteFilePath(".checksum"));
        if (checksumFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            checksumFile.write(QString::number(checksum).toLatin1());
            checksumFile.close();
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param pluginFolder
    \param buildDir
    \param sourceDir
    \param tocs
    \param keywords
    \param files
    \param mainFileInfo
    \return RetVal
*/
RetVal HelpSystem::buildSinglePluginHelp(const QString &pluginFolder, QDir &buildDir, QDir &sourceDir, QString &tocs, QString &keywords, QString &files, QPair<QString,QString> &mainFileInfo)
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
            QFile qhpFile(info.absoluteFilePath());
            QStringList filesToCopy;
            QString _tocs;
            QString _kwds;
            QString _files;
            QPair<QString,QString> _mainFileInfo;
            retval += analyzeQhpFile(pluginFolder,qhpFile, _tocs, _kwds, _files, filesToCopy, _mainFileInfo);

            if (retval == ito::retOk)
            {
                tocs += _tocs;
                keywords += _kwds;
                files += _files;
                mainFileInfo = _mainFileInfo;
                foreach(const QString &fileToCopy, filesToCopy)
                {
                    if (copyFile( QFileInfo(sourceDir.absoluteFilePath(fileToCopy)), buildDir))
                    {
                        if (fileToCopy.endsWith(".html"))
                        {
                            modifyHrefInHtmlFile(buildDir.absoluteFilePath(fileToCopy), ".."); //_static folder is one directory up
                        }
                    }
                }
            }
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param pluginFolder
    \param qhpFile
    \param tocs
    \param keywords
    \param files
    \param filesToCopy
    \param mainFileInfo
    \return RetVal
*/
RetVal HelpSystem::analyzeQhpFile(const QString &pluginFolder, QFile &qhpFile, QString &tocs, QString &keywords, QString &files, QStringList &filesToCopy, QPair<QString,QString> &mainFileInfo)
{
    QRegularExpression regExp(
        "^.*<toc>(.*)<\/toc>.*<keywords>(.*)<\/keywords>.*<files>(.*)<\/files>.*$",
        QRegularExpression::DotMatchesEverythingOption |
            QRegularExpression::MultilineOption);
    if (qhpFile.open(QIODevice::ReadOnly))
    {
        QByteArray content = qhpFile.readAll();

        QRegularExpressionMatch match = regExp.match(content);
        if (match.hasMatch())
        {
            QString mainFile;
            tocs += modifyTocs(match.captured(1), pluginFolder, mainFile);
            mainFileInfo.first = pluginFolder;
            mainFileInfo.second = mainFile;
            keywords += modifyKeywords(match.captured(2), pluginFolder);
            files += modifyFiles(
                match.captured(3),
                pluginFolder,
                QStringList() << "search.html"
                              << "_static",
                filesToCopy);
        }
        qhpFile.close();
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param in
    \param hrefPrefix
    \param mainFile
    \return QString
*/
QString HelpSystem::modifyTocs(const QString &in, const QString &hrefPrefix, QString &mainFile)
{

    //this file searches all ref="..." substrings and replaces ... by hrefPrefix/...
    QString tocs;
    QString href;
    int start = 0;
    int end = 0;
    QString startstr = "ref=\"";
    QString endstr = "\"";

    while ((start = in.indexOf(startstr, start)) != -1)
    {
        start += startstr.size();
        tocs += in.mid(end,start-end); //from last end
        end = in.indexOf("\"",start);
        href = in.mid(start, end -  start);
        href.prepend(hrefPrefix + "/");
        if (mainFile.isEmpty())
        {
            mainFile = href;
        }
        tocs += href;
    }

    tocs += in.mid(end);
    return tocs.trimmed();
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param in
    \param hrefPrefix
    \return QString
*/
QString HelpSystem::modifyKeywords(const QString &in, const QString &hrefPrefix)
{

    //this file searches all ref="..." substrings and replaces ... by hrefPrefix\...
    QString keywords;
    QString href;
    int start = 0;
    int end = 0;
    QString startstr = "ref=\"";
    QString endstr = "\"";

    while ((start = in.indexOf(startstr, start)) != -1)
    {
        start += startstr.size();
        keywords += in.mid(end,start-end); //from last end
        end = in.indexOf("\"",start);
        href = in.mid(start, end -  start);
        href.prepend(hrefPrefix + "/");
        keywords += href;
    }

    keywords += in.mid(end);
    return keywords.trimmed();
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param in
    \param hrefPrefix
    \param excludeContent
    \param filesToCopy
    \return QString
*/
QString HelpSystem::modifyFiles(const QString &in, const QString &hrefPrefix, const QStringList &excludeContent, QStringList &filesToCopy)
{
    //this file searches all ref="..." substrings and replaces ... by hrefPrefix\...
    QString files;
    QRegularExpression regExp("<file>([a-zA-Z0-9#.?%&]*)</file>");
    QString cap;

#ifndef WIN32
    QString delimiter = "/";
#else
    QString delimiter = "\\";
#endif

    int pos = 0;
    bool take;

    QRegularExpressionMatch match = regExp.match(in);
    if (match.hasMatch())
    {
        take = true;
        cap = match.captured(1);
        foreach(const QString &str, excludeContent)
        {
            if (cap.contains(str))
            {
                take = false;
                break;
            }
        }

        if (take)
        {
            filesToCopy.append(cap);
            files += "<file>" + hrefPrefix + delimiter + cap+ "</file>\n";
        }

        pos += match.capturedLength();
    }

    return files;
}

//-----------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param htmlFile
    \param prefix
    \return RetVal
*/
RetVal HelpSystem::modifyHrefInHtmlFile(const QString &htmlFile, const QString &prefix)
{
    ito::RetVal retval;
    QFile file(htmlFile);
    QByteArray searchString = "href=\"_static";
    QByteArray replaceString = "href=\"" + prefix.toLatin1() + "/_static";

    if (file.open(QIODevice::ReadOnly))
    {
        QByteArray ba = file.readAll();
        ba.replace(searchString,replaceString);
        file.close();

        if (file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        {
            file.write(ba);
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param directory
    \return bool
*/
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
//! shortdesc
/*! longdesc

    \param src
    \param dst
    \return bool
*/
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

//-------------------------------------------------------------------------------------------
//! shortdesc
/*! longdesc

    \param srcFileInfo
    \param dstFolder
    \return bool
*/
/*static*/ bool HelpSystem::copyFile(const QFileInfo &srcFileInfo, QDir &dstFolder)
{
    //check if dstFolder exists
    QDir tempFolder = dstFolder;

    if (tempFolder.exists() == false)
    {
        while(!tempFolder.cdUp())
        {
        }

        tempFolder.mkdir(dstFolder.absolutePath());
    }

    return QFile::copy( srcFileInfo.canonicalFilePath(), dstFolder.absoluteFilePath(srcFileInfo.fileName()));
}



} //end namespace ito
