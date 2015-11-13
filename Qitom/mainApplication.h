/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef MAINAPPLICATION_H
#define MAINAPPLICATION_H

#include "python/pythonEngineInc.h"
#include "organizer/scriptEditorOrganizer.h"
#include "organizer/paletteOrganizer.h"
#include "organizer/uiOrganizer.h"
#include "organizer/processOrganizer.h"
#include "organizer/designerWidgetOrganizer.h"
//
#include "widgets/mainWindow.h"

#include <qtranslator.h>

//using namespace ito;
class QSplashScreen;

namespace ito
{

class MainApplication : public QObject
{
    Q_OBJECT

    public:
        enum tGuiType { standard, console, none };          /*!< enumeration for gui-type (standard, console, none) */

        MainApplication(tGuiType guiType = standard);
        ~MainApplication();

        int loadSettings(const QString userName = "");
        void setupApplication();
        void finalizeApplication();

        int exec();
        int execPipManagerOnly();

        inline ScriptEditorOrganizer* getScriptEditorOrganizer() { return m_scriptEditorOrganizer; } /*!< returns member m_scriptEditorOrganizer */

        static MainApplication* instance();

    protected:
        void registerMetaObjects();

    private:
        tGuiType m_guiType;                                   /*!<  member for the desired gui-type */

        QThread* m_pyThread;                                  /*!<  Thread, where python engine is executed */
        PythonEngine* m_pyEngine;                             /*!<  pointer to the python engine */

        ScriptEditorOrganizer* m_scriptEditorOrganizer;       /*!<  pointer to scriptEditorOrganizer, organizing every existing script window or docking widget */
        MainWindow* m_mainWin;                                /*!<  pointer to the main window */

        static MainApplication* mainApplicationInstance;      /*!<  static pointer to the (singleton) instance of MainApplication */
        PaletteOrganizer* m_paletteOrganizer;                 /*!<  pointer to figureOrganizer */
        UiOrganizer* m_uiOrganizer;                           /*!<  pointer to uiOrganizer */
        DesignerWidgetOrganizer* m_designerWidgetOrganizer;   /*!< designerWidgetOrganizer */

        ito::ProcessOrganizer* m_processOrganizer;            /*!< pointer to processOrganizer */
//        bool m_hasGIL;                                        /*!< python global interpreter lock GIL */

        QTranslator m_Translator;                             /*!< pointer to a language-translation, different than the standard language (en) */
        QTranslator m_qtTranslator;
        QTranslator m_commonQtTranslator;

        QSplashScreen *m_splashScreen;

    signals:
        void propertiesChanged();

    private slots:

    public slots:
        void _propertiesChanged() { emit propertiesChanged(); }
        void mainWindowCloseRequest();
};

} //end namespace ito

#endif
