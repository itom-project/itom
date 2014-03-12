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

#ifndef HELPSYSTEM_H
#define HELPSYSTEM_H

#include "../../common/sharedStructures.h"

#include <qdir.h>
#include <qstring.h>
#include <qmap.h>

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

    RetVal rebuildHelpCollection(QStringList &qchFiles, quint16 checksum, QDir &helpDir);


    QDir m_helpDirectory;
    QMap<QString, quint16> m_registeredFilesQCH;
    QString m_helpCollectionName;
    QString m_helpCollectionProject;
    QString m_pluginHelpCollectionName;
    QString m_pluginHelpCollectionProject;
    bool m_upToDate;
    bool m_upToDatePlugins;


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