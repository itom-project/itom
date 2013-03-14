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

#ifndef APPMANAGEMENT_H
#define APPMANAGEMENT_H

#include <qstring.h>
#include <qobject.h>
#include <qmutex.h>

#include "./organizer/addInManager.h"

/* content */

//!< AppManagement (in order to provide access to basic organizers, managers and other main components to every instance
//   without causing many circular includes.
class AppManagement
{
    public:
        inline static QString& getSettingsFile() { QMutexLocker locker(&m_mutex); return m_settingsFile; };

        inline static QObject* getScriptEditorOrganizer() { QMutexLocker locker(&m_mutex); return m_sew; }; /*!< returns static pointer to ScriptEditorOrganizer instance */
        inline static QObject* getPythonEngine() { QMutexLocker locker(&m_mutex); return m_pe; };           /*!< returns static pointer to PythonEngine instance */
    
        inline static QObject* getPaletteOrganizer() { QMutexLocker locker(&m_mutex); return m_plo; };        /*!< returns static pointer to PaletteOrganizer instance */
        inline static QObject* getDesignerWidgetOrganizer() { QMutexLocker locker(&m_mutex); return m_dwo; };        /*!< returns static pointer to DesignerWidgetOrganizer instance */
    
        inline static QObject* getMainApplication() { QMutexLocker locker (&m_mutex); return m_app; };
        inline static QObject* getAddinManager() { QMutexLocker locker (&m_mutex); return qobject_cast<QObject*>(ito::AddInManager::getInstance()); };
	    inline static QObject* getMainWindow() { QMutexLocker locker (&m_mutex); return m_mainWin; };
        inline static QObject* getUiOrganizer() { QMutexLocker locker (&m_mutex); return m_uiOrganizer; };  /*!< returns static pointer to UiOrganizer instance */
        inline static QObject* getProcessOrganizer() { QMutexLocker locker (&m_mutex); return m_processOrganizer; };  /*!< returns static pointer to ProcessOrganizer instance */

        static void setScriptEditorOrganizer(QObject* scriptEditorOrganizer)                                /*!< sets ScriptEditorOrganizer instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_sew = scriptEditorOrganizer;
        }

        static void setPythonEngine(QObject* pythonEngine)                                                  /*!< sets PythonEngine instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_pe = pythonEngine;
        }

        static void setPaletteOrganizer(QObject* paletteOrganizer)                                                  /*!< sets PythonEngine instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_plo = paletteOrganizer;
        }

        static void setDesignerWidgetOrganizer(QObject* designerWidgetOrganizer)                                                  /*!< sets PythonEngine instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_dwo = designerWidgetOrganizer;
        }

        static void setMainApplication(QObject* mainApplication)
        {
            QMutexLocker locker(&m_mutex);
            m_app = mainApplication;
        }

        static void setMainWindow(QObject* mainWindow)
        {
            QMutexLocker locker(&m_mutex);
            m_mainWin = mainWindow;
        }	

        static void setUiOrganizer(QObject* uiOrganizer)
        {
            QMutexLocker locker(&m_mutex);
            m_uiOrganizer = uiOrganizer;
        }

        static void setSettingsFile(QString &settingsFile)
        {
            QMutexLocker locker(&m_mutex);
            m_settingsFile = settingsFile;
        }

        static void setProcessOrganizer(QObject* processOrganizer)
        {
            QMutexLocker locker(&m_mutex);
            m_processOrganizer = processOrganizer;
        }

        inline static void setUserName(const QString userName) { m_userName = userName; }
        inline static const QString getUserName() { return m_userName; }
        inline static void setUserRole(const int role) { m_userRole = role; }
        static void setUserRole(const QString role) 
        { 
            if (role == "user")
                m_userRole = 0;
            else if (role == "admin")
                m_userRole = 1;
            else
                m_userRole = 2;
        }
        inline static const int getUserRole() { return m_userRole; }

    private:
        static QObject* m_sew;  /*!< static pointer to ScriptEditorOrganizer (default: NULL) */
        static QObject* m_pe;   /*!< static pointer to PythonEngine (default: NULL) */
        static QObject* m_dwo;   /*!< static pointer to DesignerWidgetOrganizer (default: NULL) */
        static QObject* m_plo;   /*!< static pointer to FigureOrganizer (default: NULL) */
        static QObject* m_app;  /*!< static pointer to MainApplication (default: NULL) */
	    static QObject* m_mainWin;
        static QObject* m_uiOrganizer; /*!< static pointer to UiOrganizer (default: NULL) */
        static QObject* m_processOrganizer; /*!< static pointer to ProcessOrganizer (default: NULL) */
        static QMutex m_mutex;  /*!< static mutex, protecting every read and write operation in class AppManagement */

        static QString m_settingsFile;
        static int m_userRole;  /*< type of user: 0: "dumb" user, 1: admin user, 2: developer */
        static QString m_userName;  /*< id of current user */
};

#endif // APPMANAGEMENT_H