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

#ifndef APPMANAGEMENT_H
#define APPMANAGEMENT_H

//#include <qstring.h>
#include <qobject.h>
#include <qmutex.h>

namespace ito
{

/* content */

//!< AppManagement (in order to provide access to basic organizers, managers and other main components to every instance
//   without causing many circular includes.
class AppManagement
{
    public:
        static QString getSettingsFile();

        inline static QObject* getScriptEditorOrganizer() { QMutexLocker locker(&m_mutex); return m_sew; } /*!< returns static pointer to ScriptEditorOrganizer instance */
        inline static QObject* getPythonEngine() { QMutexLocker locker(&m_mutex); return m_pe; }           /*!< returns static pointer to PythonEngine instance */
        inline static QObject* getPythonStatePublisher() { QMutexLocker locker(&m_mutex); return m_psp; }           /*!< returns static pointer to PythonStatePublisher instance */

        inline static QObject* getPaletteOrganizer() { QMutexLocker locker(&m_mutex); return m_plo; }        /*!< returns static pointer to PaletteOrganizer instance */
        inline static QObject* getDesignerWidgetOrganizer() { QMutexLocker locker(&m_mutex); return m_dwo; }        /*!< returns static pointer to DesignerWidgetOrganizer instance */

        inline static QObject* getMainApplication() { QMutexLocker locker (&m_mutex); return m_app; }
        inline static QObject* getAddInManager() { QMutexLocker locker(&m_mutex); return m_addInManager; }
        inline static QObject* getMainWindow() { QMutexLocker locker (&m_mutex); return m_mainWin; }
        inline static QObject* getUiOrganizer() { QMutexLocker locker (&m_mutex); return m_uiOrganizer; }  /*!< returns static pointer to UiOrganizer instance */
        inline static QObject* getProcessOrganizer() { QMutexLocker locker (&m_mutex); return m_processOrganizer; }  /*!< returns static pointer to ProcessOrganizer instance */
        inline static QObject* getUserOrganizer() { QMutexLocker locker (&m_mutex); return m_userOrganizer; }
        inline static QObject* getCoutStream() { QMutexLocker locker (&m_mutex); return m_coutStream; }
        inline static QObject* getCerrStream() { QMutexLocker locker (&m_mutex); return m_cerrStream; }
        inline static QObject* getLogger() { QMutexLocker locker (&m_mutex); return m_logger; }


        static void setScriptEditorOrganizer(QObject* scriptEditorOrganizer)     /*!< sets ScriptEditorOrganizer instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_sew = scriptEditorOrganizer;
        }

        static void setAddInManager(QObject* addInManager)                      /*!< sets AddInManager instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_addInManager = addInManager;
        }

        static void setPythonEngine(QObject* pythonEngine)                       /*!< sets PythonEngine instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_pe = pythonEngine;
        }

        static void setPythonStatePublisher(QObject* pythonStatePublisher)       /*!< sets PythonStatePublisher instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_psp = pythonStatePublisher;
        }

        static void setPaletteOrganizer(QObject* paletteOrganizer)               /*!< sets PythonEngine instance pointer */
        {
            QMutexLocker locker(&m_mutex);
            m_plo = paletteOrganizer;
        }

        static void setDesignerWidgetOrganizer(QObject* designerWidgetOrganizer) /*!< sets PythonEngine instance pointer */
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

        static void setProcessOrganizer(QObject* processOrganizer)
        {
            QMutexLocker locker(&m_mutex);
            m_processOrganizer = processOrganizer;
        }

        static void setUserOrganizer(QObject* userOrganizer)
        {
            QMutexLocker locker(&m_mutex);
            m_userOrganizer = userOrganizer;
        }

        static void setStdCoutCerrStreamRedirections(QObject* coutStream, QObject* cerrStream)
        {
            QMutexLocker locker(&m_mutex);
            m_coutStream = coutStream;
            m_cerrStream = cerrStream;
        }

        static void setLogger(QObject* logger)
        {
            QMutexLocker locker(&m_mutex);
            m_logger = logger;
        }

        struct Timeouts
        {
            int pluginInitClose;
            int pluginGeneral;
            int pluginFileSaveLoad;
        };

        static Timeouts timeouts;

    private:
        static QObject* m_sew;  /*!< static pointer to ScriptEditorOrganizer (default: NULL) */
        static QObject* m_pe;   /*!< static pointer to PythonEngine (default: NULL) */
        static QObject* m_psp;  /*!< static pointer to PythonStatePublisher (default: NULL) */
        static QObject* m_dwo;   /*!< static pointer to DesignerWidgetOrganizer (default: NULL) */
        static QObject* m_plo;   /*!< static pointer to FigureOrganizer (default: NULL) */
        static QObject* m_app;  /*!< static pointer to MainApplication (default: NULL) */
        static QObject* m_mainWin;
        static QObject* m_addInManager;
        static QObject* m_uiOrganizer; /*!< static pointer to UiOrganizer (default: NULL) */
        static QObject* m_processOrganizer; /*!< static pointer to ProcessOrganizer (default: NULL) */
        static QObject *m_userOrganizer;    /*!< static pointer to UserOrganizer (default: NULL) */
        static QObject* m_coutStream; /*!< static pointer to QDebugStream for std::cout redirection */
        static QObject* m_cerrStream; /*!< static pointer to QDebugStream for std::cerr redirection */
        static QObject* m_logger; /*!< static pointer to Logger (default: nullptr) */

        static QMutex m_mutex;  /*!< static mutex, protecting every read and write operation in class AppManagement */

};

} //end namespace ito

#endif // APPMANAGEMENT_H
