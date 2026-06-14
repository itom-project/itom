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

#ifndef HELPSYSTEM_H
#define HELPSYSTEM_H

#include "../../common/sharedStructures.h"

#include <qdir.h>
#include <qstring.h>
#include <qmap.h>
#include <qpair.h>

namespace ito
{

class HelpSystem
{
public:
    static HelpSystem* getInstance();

    QString getHelpCollectionAbsFileName() const;
    inline bool isUpToDate() const { return m_upToDate; };
    RetVal rebuildHelpIfNotUpToDate();

protected:

private:
    HelpSystem(void);
    HelpSystem(HelpSystem  &/*copyConstr*/) {}
    ~HelpSystem(void);

    RetVal scanDocumentationFiles(QStringList &qchFiles, quint16 &checksum);
    RetVal scanPluginQhpFiles(quint16 &checksum);
    RetVal getCheckSumOfBuild(QDir &helpDir, QString &projectFileName, quint16 &checksum);
    RetVal getCheckSumOfPluginBuild(quint16 &checksum);

    RetVal rebuildHelpCollection(QStringList &qchFiles, quint16 checksum, QDir &helpDir);

    RetVal buildPluginHelp(quint16 checksum);
    RetVal buildSinglePluginHelp(const QString &pluginFolder, QDir &buildDir, QDir &sourceDir, QString &tocs, QString &keywords, QString &files, QPair<QString,QString> &mainFileInfo);
    RetVal analyzeQhpFile(const QString &pluginFolder, QFile &qhpFile, QString &tocs, QString &keywords, QString &files, QStringList &filesToCopy, QPair<QString,QString> &mainFileInfo);
    QString modifyTocs(const QString &in, const QString &hrefPrefix, QString &mainFile);
    QString modifyKeywords(const QString &in, const QString &hrefPrefix);
    QString modifyFiles(const QString &in, const QString &hrefPrefix, const QStringList &excludeContent, QStringList &filesToCopy);
    RetVal modifyHrefInHtmlFile(const QString &htmlFile, const QString &prefix);

    static bool removeDir(const QDir &directory);
    static bool copyDir(const QDir &src, const QDir &dst);
    static bool copyFile(const QFileInfo &srcFileInfo, QDir &dstFolder);


    QDir m_helpDirectory;
    QMap<QString, quint16> m_registeredFilesQCH;
    QString m_helpCollectionName;
    QString m_helpCollectionProject;
    QString m_pluginHelpCollectionName;
    QString m_pluginHelpCollectionProject;
    bool m_upToDate;




    static HelpSystem *m_pHelpSystem;

    //!< singleton nach: http://www.oop-trainer.de/Themen/Singleton.html
    class HelpSystemSingleton
    {
        public:
            ~HelpSystemSingleton()
            {
                #pragma omp critical
                {
                    if( HelpSystem::m_pHelpSystem != NULL)
                    {
                        delete HelpSystem::m_pHelpSystem;
                        HelpSystem::m_pHelpSystem = NULL;
                    }
                }
            }
    };
    friend class HelpSystemSingleton;

};

}; //namespace ito

#endif
