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

#include "mainApplication.h"
#include "global.h"
#include "version.h"
#include "AppManagement.h"

#include "widgets/abstractDockWidget.h"
#include "organizer/addInManager.h"
#include "./models/UserModel.h"
#include "organizer/userOrganizer.h"
#include "./ui/dialogSelectUser.h"

#include <qsettings.h>
#include <qstringlist.h>

#include <qdir.h>
#include <qtextcodec.h>
#include <qsplashscreen.h>

/*!
    \class MainApplication
    \brief The MainApplication class is the basic management class for the entire application
*/

//! static instance pointer initialization
MainApplication* MainApplication::mainApplicationInstance = NULL;

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    getter-method for static instance pointer
*/
MainApplication* MainApplication::instance()
{
    Q_ASSERT(MainApplication::mainApplicationInstance);
    return MainApplication::mainApplicationInstance;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!

    \param guiType Type of the desired GUI (normal, console, no)
    \sa tGuiType
*/
MainApplication::MainApplication(tGuiType guiType) : 
    m_pyThread(NULL), 
    m_pyEngine(NULL), 
    m_scriptEditorOrganizer(NULL), 
    m_mainWin(NULL), 
    m_paletteOrganizer(NULL),
    m_uiOrganizer(NULL), 
    m_processOrganizer(NULL),
    m_splashScreen(NULL)
{
    m_guiType = guiType;
    MainApplication::mainApplicationInstance = this;

    //qDebug() << QLibraryInfo::location( QLibraryInfo::BinariesPath );

    AppManagement::setMainApplication(qobject_cast<QObject*>(this));

    //global settings: the settings file will be stored in itomSettings/{organization}/{applicationName}.ini
    QCoreApplication::setOrganizationName("ito");
    QCoreApplication::setApplicationName("itom");
    QCoreApplication::setApplicationVersion( ITOM_VERSION_STR );
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    shutdown of whole application, including PythonEngine

    \sa PythonEngine
*/
MainApplication::~MainApplication()
{
    AppManagement::setMainApplication(NULL);
    MainApplication::mainApplicationInstance = NULL;
}


//----------------------------------------------------------------------------------------------------------------------------------
//! setup of application
/*!
    starts PythonEngine, MainWindow (dependent on gui-type) and all necessary managers and organizers.
    Builds import connections between MainWindow and PythonEngine as well as ScriptEditorOrganizer.

    \sa PythonEngine, MainWindow, ScriptEditorOrganizer
*/
void MainApplication::setupApplication()
{
    RetVal retValue = retOk;
	RetVal pyRetValue;
    QStringList startupScripts;

    QPixmap pixmap(":/application/icons/itomicon/splashScreen.png");

#if QT_POINTER_SIZE == 8
    QString text = QString(tr("Version %1\n%2")).arg(ITOM_VERSION_STR).arg(tr("64 bit (x64)"));
#else
    QString text = QString(tr("Version %1\n%2")).arg(ITOM_VERSION_STR).arg(tr("32 bit (x86)"));
#endif

#if USING_GIT == 1
    text.append( QString("\nRev. %1").arg(GIT_HASHTAG_ABBREV) );
#endif
    QPainter p;
    p.begin(&pixmap);
    p.setPen(Qt::black);
    QRectF rect(311,115,200,100); //position of the version text within the image
    p.drawText(rect, Qt::AlignRight, text);
    p.end();

    m_splashScreen = new QSplashScreen(pixmap);
    
    m_splashScreen->show();
    QCoreApplication::processEvents();

    QSettings *settings = new QSettings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    settings->beginGroup("Language");
    QString language = settings->value("language", "en").toString();
    QByteArray codec =  settings->value("codec", "UTF-8" ).toByteArray();
    settings->endGroup();
    settings->sync();

    QLocale local = QLocale(language); //language can be "language[_territory][.codeset][@modifier]"
    QString itomTranslationFolder = QCoreApplication::applicationDirPath() + "/translation";

    //load translation files
    m_splashScreen->showMessage( tr("load translations...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    //1. try to load qt-translations from qt-folder
    m_qtTranslator.load("qt_" + local.name(), QLibraryInfo::location(QLibraryInfo::TranslationsPath));
    if (m_qtTranslator.isEmpty())
    {
        //qt-folder is not available, then try itom translation folder
        m_qtTranslator.load("qt_" + local.name(), itomTranslationFolder);
    }
    QCoreApplication::instance()->installTranslator(&m_qtTranslator);

    //2. load itom-specific translation file
    m_Translator.load("qitom_" + local.name(), itomTranslationFolder);
    QCoreApplication::instance()->installTranslator(&m_Translator);

    //3. set default encoding codec
    QTextCodec *textCodec = QTextCodec::codecForName(codec);
    if (textCodec == NULL)
    {
        textCodec = QTextCodec::codecForName("UTF-8");
    }

    QTextCodec::setCodecForCStrings( textCodec );
    QTextCodec::setCodecForLocale( textCodec );

    settings->beginGroup("CurrentStatus");
    QDir::setCurrent(settings->value("currentDir",QDir::currentPath()).toString());
    settings->endGroup();
    settings->sync();

    if (m_guiType == standard || m_guiType == console)
    {
        m_splashScreen->showMessage( tr("load style...") , Qt::AlignRight | Qt::AlignBottom);
        QCoreApplication::processEvents();

        //set styles (if available)
        settings->beginGroup("ApplicationStyle");
        QString styleName = settings->value("style", "").toString();
        QString cssFile = settings->value("cssFile", "").toString();
        settings->endGroup();

        if (styleName != "")
        {
            QApplication::setStyle(styleName);
        }

        if (cssFile != "")
        {
            QDir styleFolder = QCoreApplication::applicationDirPath();
            if (styleFolder.exists(cssFile))
            {
                QFile css(styleFolder.filePath(cssFile));
                if (css.open(QFile::ReadOnly))
                {
                    QString cssContent(css.readAll());
                    qApp->setStyleSheet(cssContent);
                    css.close();
                }
            }
        }
    }

    //starting ProcessOrganizer for external processes like QtDesigner, QtAssistant, ...
    m_splashScreen->showMessage( tr("load process organizer...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    m_processOrganizer = new ito::ProcessOrganizer();
    AppManagement::setProcessOrganizer(qobject_cast<QObject*>(m_processOrganizer));

    qDebug("MainApplication::setupApplication");

   // starting AddInManager
    m_splashScreen->showMessage( tr("scan and load plugins...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    ito::AddInManager *AIM = ito::AddInManager::getInstance();
    connect(AIM, SIGNAL(splashLoadMessage(const QString&, int, const QColor &)), m_splashScreen, SLOT(showMessage(const QString&, int, const QColor &)));
    retValue += AIM->scanAddInDir("");

    qDebug("..plugins loaded");

    m_splashScreen->showMessage( tr("start python...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    m_pyEngine = new PythonEngine();
    AppManagement::setPythonEngine(qobject_cast<QObject*>(m_pyEngine));

    qDebug("..python engine started");

    //retValue += m_pyEngine->pythonSetup();
    m_pyThread = new QThread();
    m_pyEngine->moveToThread(m_pyThread);
    m_pyThread->start();
    QMetaObject::invokeMethod(m_pyEngine, "pythonSetup", Qt::BlockingQueuedConnection, Q_ARG(ito::RetVal*, &pyRetValue));

    qDebug("..python engine moved to new thread");

	retValue += pyRetValue;
	if(pyRetValue.containsError())
	{
		DELETE_AND_SET_NULL(m_pyEngine);
		AppManagement::setPythonEngine(NULL);
	}

    if (m_guiType == standard || m_guiType == console)
    {
        m_splashScreen->showMessage( tr("load main window...") , Qt::AlignRight | Qt::AlignBottom);
        QCoreApplication::processEvents();

        m_mainWin = new MainWindow();
        AppManagement::setMainWindow(qobject_cast<QObject*>(m_mainWin));

        m_splashScreen->showMessage( tr("load ui organizer...") , Qt::AlignRight | Qt::AlignBottom);
        QCoreApplication::processEvents();

        m_uiOrganizer = new UiOrganizer();
        AppManagement::setUiOrganizer(qobject_cast<QObject*>(m_uiOrganizer));

        m_splashScreen->showMessage( tr("scan and load designer widgets...") , Qt::AlignRight | Qt::AlignBottom);
        QCoreApplication::processEvents();

        m_designerWidgetOrganizer = new DesignerWidgetOrganizer(retValue);
        AppManagement::setDesignerWidgetOrganizer(qobject_cast<QObject*>(m_designerWidgetOrganizer));
    }
    else
    {
        m_mainWin = NULL;
    }

    qDebug("..main window started");

    m_paletteOrganizer = new PaletteOrganizer();
    AppManagement::setPaletteOrganizer(qobject_cast<QObject*>(m_paletteOrganizer));

    qDebug("..palette organizer started");

    m_splashScreen->showMessage( tr("load script editor organizer...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    m_scriptEditorOrganizer = new ScriptEditorOrganizer(m_mainWin != NULL);
    AppManagement::setScriptEditorOrganizer(m_scriptEditorOrganizer); //qobject_cast<QObject*>(scriptEditorOrganizer);

    qDebug("..script editor started");

    if (m_mainWin != NULL)
    {
        connect(m_scriptEditorOrganizer, SIGNAL(addScriptDockWidgetToMainWindow(AbstractDockWidget*,Qt::DockWidgetArea)), m_mainWin, SLOT(addAbstractDock(AbstractDockWidget*,Qt::DockWidgetArea)));
        connect(m_scriptEditorOrganizer, SIGNAL(removeScriptDockWidgetFromMainWindow(AbstractDockWidget*)), m_mainWin, SLOT(removeAbstractDock(AbstractDockWidget*)));
        connect(m_mainWin, SIGNAL(mainWindowCloseRequest()), this, SLOT(mainWindowCloseRequest()));
    }

    qDebug("..starting load settings");

    //try to execute startup-python scripts
    m_splashScreen->showMessage( tr("execute startup scripts...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    settings->beginGroup("Python");

    int size = settings->beginReadArray("startupFiles");
    for (int i = 0; i < size; ++i)
    {
        settings->setArrayIndex(i);
        startupScripts.append(settings->value("file",QString()).toString());
    }

    settings->endArray();
    settings->endGroup();
    settings->sync();

    if (startupScripts.count()>0)
    {
        QMetaObject::invokeMethod(m_pyEngine, "pythonRunFile", Q_ARG(QString, startupScripts.join(";")));
    }

    settings->beginGroup("CurrentStatus");
    QString currentDir = (settings->value("currentDir",QDir::currentPath()).toString());
    settings->endGroup();
    delete settings;

    m_splashScreen->showMessage( tr("scan and run scripts in autostart folder...") , Qt::AlignRight | Qt::AlignBottom);
    QCoreApplication::processEvents();

    //force python to scan and run files in autostart folder in itom-packages folder
    QMetaObject::invokeMethod(m_pyEngine, "scanAndRunAutostartFolder", Q_ARG(QString, currentDir) );

    ////since autostart-files could have changed current directory, re-change it to the value of the settings-file
    //settings.beginGroup("CurrentStatus");
    //QDir::setCurrent(settings.value("currentDir",QDir::currentPath()).toString());
    //settings.endGroup();

    if (retValue.containsError())
    {
		if (retValue.errorMessage())
		{
			std::cerr << "Error when starting the application: \n" << retValue.errorMessage() << std::endl;
		}
		else
		{
			std::cerr << "An unspecified error occurred when starting the application." << std::endl;
		}
    }
	else if (retValue.containsWarning())
	{
		if (retValue.errorMessage())
		{
			std::cout << "Warning when starting the application: \n" << retValue.errorMessage() << std::endl;
		}
		else
		{
			std::cout << "An unspecified warning occurred when starting the application." << std::endl;
		}
	}

    qDebug("..load settings done");
    qDebug("MainApplication::setupApplication .. done");

    std::cout << "\n\tWelcome to itom program!\n\n";
//    std::cout << "THIS ITOM-COPY IS A PREPUPLISHED ALPHA VERSION\nGIVEN TO ZEISS MICROSCOPY FOR INTERNAL USE WITHIN\nZEISS-ITO-COOPERATION.\nDO NOT DISTRIBUTE TO THIRD PARTY.\n !!! CONFIDENTIAL !!! \n\n";
    std::cout << "\tPlease report bugs under:\n\t\thttps://bitbucket.org/itom/itom/issues\n\tCheers your itom team\n" << std::endl;

    if(m_mainWin)
    {
        m_splashScreen->finish(m_mainWin);
    }
    else
    {
        m_splashScreen->close();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! setup of application
/*!
    stops PythonEngine, MainWindow (dependent on gui-type) and all necessary managers and organizers.
    Closes import connections between MainWindow and PythonEngine as well as ScriptEditorOrganizer.

    \sa PythonEngine, MainWindow, ScriptEditorOrganizer
*/
void MainApplication::finalizeApplication()
{
    DELETE_AND_SET_NULL(m_splashScreen);

    DELETE_AND_SET_NULL(m_scriptEditorOrganizer);
    AppManagement::setScriptEditorOrganizer(NULL);

    DELETE_AND_SET_NULL(m_paletteOrganizer);
    AppManagement::setPaletteOrganizer(NULL);

    DELETE_AND_SET_NULL(m_designerWidgetOrganizer);
    AppManagement::setDesignerWidgetOrganizer(NULL);

    DELETE_AND_SET_NULL(m_uiOrganizer);
    AppManagement::setUiOrganizer(NULL);

    DELETE_AND_SET_NULL(m_mainWin);
    AppManagement::setMainWindow(NULL);

	if(m_pyEngine)
	{
		ItomSharedSemaphore *waitCond = new ItomSharedSemaphore();
		QMetaObject::invokeMethod(m_pyEngine, "pythonShutdown", Q_ARG(ItomSharedSemaphore*, waitCond));
		waitCond->waitAndProcessEvents(-1);

		//call further objects, which have been marked by "deleteLater" during this finalize method (partI)
		QCoreApplication::sendPostedEvents ();
		QCoreApplication::sendPostedEvents (NULL,QEvent::DeferredDelete); //these events are not sent by the line above, since the event-loop already has been stopped.
		QCoreApplication::processEvents();

        waitCond->deleteSemaphore();
        waitCond = NULL;
    }

    DELETE_AND_SET_NULL(m_pyEngine);
    AppManagement::setPythonEngine(NULL);

    m_pyThread->quit();
    m_pyThread->wait();
    DELETE_AND_SET_NULL(m_pyThread);

    ito::AddInManager::closeInstance();

    DELETE_AND_SET_NULL(m_processOrganizer);
    AppManagement::setProcessOrganizer(NULL);

    //call further objects, which have been marked by "deleteLater" during this finalize method (partII)
    QCoreApplication::sendPostedEvents ();
    QCoreApplication::sendPostedEvents (NULL,QEvent::DeferredDelete); //these events are not sent by the line above, since the event-loop already has been stopped.
    QCoreApplication::processEvents();

    QString settingsName(AppManagement::getSettingsFile());
    QSettings *settings = new QSettings(settingsName, QSettings::IniFormat);
    settings->beginGroup("CurrentStatus");
    settings->setValue("currentDir",QDir::currentPath());
    settings->endGroup();
    delete settings;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked if user wants to close application
/*!
    \sa MainWindow
*/
void MainApplication::mainWindowCloseRequest()
{
    RetVal retValue(retOk);

    QSettings *settings = new QSettings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings->beginGroup("MainWindow");
    if (settings->value("askBeforeClose", false).toBool())
    {
        QMessageBox msgBox;
            msgBox.setText("Do you really want to exit the application?");
            msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel);
            msgBox.setDefaultButton(QMessageBox::Ok);
            msgBox.setIcon(QMessageBox::Question);
            int ret = msgBox.exec();

            if (ret == QMessageBox::Cancel)
            {
                settings->endGroup();
		delete settings;
                return;
            }  
    }
    settings->endGroup();
    delete settings;

    if (m_pyEngine != NULL)
    {
        if (m_pyEngine->isPythonBusy())
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Python is still running. Please close it first before shutting down this application"));
            msgBox.exec();
            retValue += RetVal(retError);
        }
    }

    if (retValue.containsError()) return;

    retValue += m_scriptEditorOrganizer->closeAllScripts(true);

    if (!retValue.containsError())
    {
        if (m_mainWin)
        {
            m_mainWin->hide();
        }
        QApplication::instance()->quit();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! exececution of the main event loop
/*!
    \return value passed to exit() method, which finishes the exec()-loop, 0 if finished by quit()
*/
int MainApplication::exec()
{

    if (m_guiType == standard)
    {
        m_mainWin->show();
        return QApplication::instance()->exec();
    }
    else if (m_guiType == console)
    {
        m_mainWin->show();
        return QApplication::instance()->exec();
    }
    else
    {
        return QApplication::instance()->exec();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
