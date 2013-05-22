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

#include "qsciApiManager.h"

#include "../global.h"

#include <qdebug.h>
#include <qfileinfo.h>
#include <qsettings.h>
#include <qfile.h>
#include <qtextstream.h>
#include <qcoreapplication.h>
#include <qdir.h>

#include "../AppManagement.h"


ito::QsciApiManager* ito::QsciApiManager::m_pQsciApiManager = NULL;

namespace ito
{

/*static*/ QsciApiManager * QsciApiManager::getInstance(void)
{
    static QsciApiSingleton waechter;
    if (QsciApiManager::m_pQsciApiManager == NULL)
    {
        QsciApiManager::m_pQsciApiManager = new QsciApiManager();
    }
    return QsciApiManager::m_pQsciApiManager;
}

QsciApiManager::QsciApiManager() :
    m_pApi(NULL),
    m_qSciLex(NULL),
    m_isPreparing(false),
    m_loaded(false)
{
    m_qSciLex = new QsciLexerPython();
    m_pApi = new QsciAPIs(m_qSciLex);
    connect(m_pApi, SIGNAL(apiPreparationFinished()), this, SLOT(apiPreparationFinished()));
    connect(m_pApi, SIGNAL(apiPreparationCancelled()), this, SLOT(apiPreparationCancelled()));
    connect(m_pApi, SIGNAL(apiPreparationStarted()), this, SLOT(apiPreparationStarted()));
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");
    QStringList apiList;
    QString apiFile;
    int size = settings.beginReadArray("apiFiles");
    for (int i = 0; i < size; ++i)
    {
        settings.setArrayIndex(i);
        apiList.append( settings.value("file",QString()).toString() );
    }
    settings.endArray();
    settings.endGroup();
    try
    {
        updateAPI(apiList);
    }
    catch(...){}
}

QsciApiManager::~QsciApiManager()
{
    delete m_pApi;
    m_pApi = NULL;
    delete m_qSciLex;
    m_qSciLex = NULL;
}


int QsciApiManager::updateAPI(QStringList files, bool forcePreparation)
{
    //read checksum of all desired files
    QList<APIFileInfo> apiFiles;
    APIFileInfo temp;

    bool calcChecksum = false; //do not calculate checksums (needs some time - every file must be opened and read)
    bool calcModified = true;

    QFileInfo apiFileInfo;
    QFile apiFile;
    QByteArray fileContent;
    QString absFile;

    QDir appBaseDir = QCoreApplication::applicationDirPath();

    foreach(const QString &file, files)
    {
        absFile = appBaseDir.absoluteFilePath(file);
        apiFileInfo.setFile( absFile );
        temp.absoluteFilename = apiFileInfo.canonicalFilePath();
        temp.exists = apiFileInfo.exists();
        if(calcChecksum)
        {
            apiFile.setFileName( temp.absoluteFilename );
            if(apiFile.open(QIODevice::ReadOnly))
            {
                fileContent = apiFile.readAll();
                temp.checksum = qChecksum(fileContent.data(), fileContent.length());
                apiFile.close();
            }
        }
        else
        {
            temp.checksum = 0;
        }

        if(calcModified)
        {
            temp.lastModified = QDateTime::fromString( apiFileInfo.lastModified().toString(Qt::ISODate), Qt::ISODate );
        }
        else
        {
            temp.lastModified = QDateTime::fromTime_t(0);
        }

        apiFiles.append( temp );
    }

    if(m_isPreparing)
    {
        m_pApi->cancelPreparation();
    }

    //read info file with information about content of parsed api files
    QString compileFile = m_pApi->defaultPreparedName();
    QString compileInfoFile = compileFile;
    QList<APIFileInfo> compiledApiFiles;
    QString b;
    QStringList bl;
    bool ok;
    int lastPoint = compileFile.lastIndexOf(".");

    if(lastPoint >= 0)
    {
        compileInfoFile.insert(lastPoint,"_info");
    }
    else
    {
        compileInfoFile.append("_info");
    }

    QFile infoFile(compileInfoFile);

    if(!infoFile.exists())
    {
        forcePreparation = true;
    }
    else
    {
        if(infoFile.open(QIODevice::ReadOnly))
        {
            while(!infoFile.atEnd())
            {
                b = infoFile.readLine().simplified();
                bl = b.split(";");
                temp.absoluteFilename = "";
                temp.checksum = 0;
                temp.lastModified = QDateTime::fromTime_t(0);
                temp.exists = true;
                if(bl.size() > 0)
                {
                    temp.absoluteFilename = bl[0];
                }
                if(bl.size() > 1)
                {
                    temp.checksum = bl[1].toInt(&ok);
                    if(!ok) temp.checksum = 0;
                }
                if(bl.size() > 2)
                {
                    temp.lastModified = QDateTime::fromString( bl[2], Qt::ISODate );
                }

                if(temp.absoluteFilename != "")
                {
                    compiledApiFiles.append( temp );
                }
            }
            infoFile.close();
            
            if(apiFiles.size() != compiledApiFiles.size())
            {
                forcePreparation = true;
            }
            else
            {
                qSort( apiFiles.begin(), apiFiles.end() );
                qSort( compiledApiFiles.begin(), compiledApiFiles.end() );

                for(int i = 0; i<apiFiles.size(); i++)
                {
                    if(apiFiles[i].absoluteFilename != compiledApiFiles[i].absoluteFilename)
                    {
                        forcePreparation = true;
                        break;
                    }
                    if(calcChecksum && (apiFiles[i].checksum != compiledApiFiles[i].checksum) )
                    {
                        forcePreparation = true;
                        break;
                    }
                    qDebug() << apiFiles[i].lastModified << " - " << compiledApiFiles[i].lastModified;
                    if(calcModified && (apiFiles[i].lastModified != compiledApiFiles[i].lastModified) )
                    {
                        forcePreparation = true;
                        break;
                    }
                }
            }
        }
        else
        {
            qDebug() << "could not load info file: " << compileInfoFile;
            forcePreparation = true;
        }
        if(!m_pApi->isPrepared())
        {
            forcePreparation = true;
        }
    }
    if(forcePreparation)
    {
        m_loaded = false;
        m_preparingFileInfo = compileInfoFile;
        m_preparingAPIFiles = apiFiles;
        m_isPreparing = true;
        QString file;
        m_pApi->clear();
        foreach(const APIFileInfo &file, apiFiles)
        {
            m_pApi->load( file.absoluteFilename );
        }
        m_pApi->prepare();
    }
    else
    {
        if(!m_loaded)
        {
            qDebug() <<"QsciApiManager::updateAPI -> try to load api file from filename: " << m_pApi->defaultPreparedName();

            if(!m_pApi->loadPrepared())
            {
                qDebug() << "api preparation file could not be loaded";
            }
            else
            {
                m_loaded = true;
            }
        }
        return 1;
    }

    return 0;

}

void QsciApiManager::apiPreparationFinished()
{
    QObject* mainWin = AppManagement::getMainWindow();
    if(mainWin)
    {
        QString text = tr("The python syntax documents have changed. The API has been updated.");
        QMetaObject::invokeMethod(mainWin, "showInfoMessageLine", Q_ARG(QString, text), Q_ARG(QString, "QSciApiManager") );
    }

    qDebug() << "API Preparation Finished";
    if(m_isPreparing)
    {
        m_loaded = true;
        QFile infoFile(m_preparingFileInfo);
        QString temp;

        if(infoFile.open(QIODevice::WriteOnly))
        {
            foreach(const APIFileInfo &file, m_preparingAPIFiles)
            {
                temp = QString("%1;%2;%3").arg( file.absoluteFilename ).arg( file.checksum ).arg( file.lastModified.toString( Qt::ISODate ) );
                infoFile.write(temp.toAscii().data(),temp.length());
                infoFile.write("\n\0");
            }
            infoFile.close();
        }
        else
        {
            qDebug() << "cannot open file " << m_preparingFileInfo << " for writing";
        }

        if(!m_pApi->savePrepared())
        {
            qDebug() << "cannot save prepared API file";
        }

        m_preparingFileInfo = "";
        m_preparingAPIFiles.clear();
        m_isPreparing = false;
    }
}

void QsciApiManager::apiPreparationCancelled()
{
    QObject* mainWin = AppManagement::getMainWindow();
    if(mainWin)
    {
        QString text = tr("The generation of the python syntax API has been cancelled.");
        QMetaObject::invokeMethod(mainWin, "showInfoMessageLine", Q_ARG(QString, text), Q_ARG(QString, "QSciApiManager") );
    }

    qDebug() << "API Preparation Cancelled";
    m_preparingFileInfo = "";
    m_preparingAPIFiles.clear();
    m_isPreparing = false;
    m_loaded = false;
}

void QsciApiManager::apiPreparationStarted()
{
    QObject* mainWin = AppManagement::getMainWindow();
    if(mainWin)
    {
        QString text = tr("The python syntax documents have changed. The API is being updated...");
        QMetaObject::invokeMethod(mainWin, "showInfoMessageLine", Q_ARG(QString, text), Q_ARG(QString, "QSciApiManager") );
    }

    qDebug() << "API Preparation Started";
    m_isPreparing = true;
}

}; //namespace ito
