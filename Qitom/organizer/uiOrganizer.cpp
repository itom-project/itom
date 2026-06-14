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
#include "common/typeDefs.h" //contains nullptr implementation
#include "../python/pythonEngineInc.h"
#include "uiOrganizer.h"

//#include "../../AddInManager/apiFunctions.h"
#include "../common/apiFunctionsInc.h"
#include "../api/apiFunctionsGraph.h"
#include "common/helperCommon.h"
#include "common/addInInterface.h"
#include "common/abstractApiWidget.h"
#include "../AppManagement.h"
#include "plot/AbstractFigure.h"
#include "plot/AbstractDObjFigure.h"
#include "plot/AbstractDObjPCLFigure.h"
#include "plot/AbstractItomDesignerPlugin.h"
#include "designerWidgetOrganizer.h"
#include "../helper/qpropertyHelper.h"
#include "../models/timerModel.h"

#include "widgetWrapper.h"
#include "userInteractionWatcher.h"
#include "../python/pythonQtConversion.h"

#include "widgets/mainWindow.h"

#include <qinputdialog.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qfiledialog.h>
#include <qcoreapplication.h>
#include <qpluginloader.h>
#include <QtUiTools/quiloader.h>
#include <qclipboard.h>

#include <QtUiPlugin/QDesignerCustomWidgetInterface>
#include <qsettings.h>
#include <qcoreapplication.h>
#include <qmainwindow.h>


namespace ito
{


//! destructor
/*!
    If the widget, observed by the UiDialogSet-instance is still valid, it is registered for deletion by the Qt-system.
*/
UiContainer::~UiContainer()
{
    if (!m_weakDialog.isNull())
    {
        if (m_type == UiContainer::uiTypeFigure)
        {
            MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());
            if (mainWin)
            {
                mainWin->removeAbstractDock(qobject_cast<ito::AbstractDockWidget*>(m_weakDialog.data()));
            }
        }
        else if (m_type == UiContainer::uiTypeQDockWidget)
        {
            MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());
            if (mainWin)
            {
                mainWin->removeDockWidget(qobject_cast<QDockWidget*>(m_weakDialog.data()));
            }
        }

        m_weakDialog.data()->deleteLater();
    }
}


/*!
    \class UiOrganizer
    \brief The UiOrganizer is started as singleton instance within itom and organizes all main windows, dialogs, widgets,...
            which are currently loaded at runtime from any ui-file or from a widget, provided by any algorithm plugin
*/


unsigned int UiOrganizer::autoIncUiDialogCounter = 99; //such that itom-plots have numbers >= 100. The matplotlib figure manager always distributes figure numbers >= 1. In order to avoid conflicts, this separation is set.
unsigned int UiOrganizer::autoIncObjectCounter = 1;



//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    creates the singleton instance of WidgetWrapper. The garbage collection timer is not started yet, since
    this is done if the first user interface becomes being organized by this class.
*/
UiOrganizer::UiOrganizer(ito::RetVal &retval) :
    m_garbageCollectorTimer(0),
    m_widgetWrapper(nullptr),
    m_pUiLoader(nullptr),
    m_pTimerModel(new TimerModel())
{
    m_dialogList.clear();
    m_objectList.clear();

    m_widgetWrapper = new WidgetWrapper(this);

    qRegisterMetaType<ito::UiDataContainer>("ito::UiDataContainer");
    qRegisterMetaType<ito::UiDataContainer>("ito::UiDataContainer&");
    qRegisterMetaType<ito::UiOrganizer::ClassInfoContainerList*>("ito::UiOrganizer::ClassInfoContainerList*");
    qRegisterMetaType<ito::UiOrganizer::ClassInfoContainerList*>("ito::UiOrganizer::ClassInfoContainerList&");
    qRegisterMetaType<QWeakPointer<QTimer> >("QWeakPointer<QTimer>");
    qRegisterMetaType<QWidget*>("QWidget*");


    if (QEvent::registerEventType(QEvent::User+123) != QEvent::User+123)
    {
        retval += ito::RetVal(ito::retWarning, 0, "The user defined event id 123 could not been registered for use in UiOrganizer since it is already in use.");
    }

    m_pUiLoader = new QUiLoader(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    Deletes all remaining user interfaces, by deleting the corresponding instances of UiDialogSet.
    Stops a possible garbage collection timer.
*/
UiOrganizer::~UiOrganizer()
{
    DELETE_AND_SET_NULL(m_pUiLoader);

    QHash<unsigned int, UiContainerItem>::const_iterator i = m_dialogList.constBegin();
    while (i != m_dialogList.constEnd())
    {
        delete i->container;
        ++i;
    }
    m_dialogList.clear();
    m_objectList.clear();

    QHash<QString, QTranslator*>::const_iterator qtransIter = m_transFiles.constBegin();
    while (qtransIter != m_transFiles.constEnd())
    {
        delete qtransIter.value();
        ++qtransIter;
    }
    m_transFiles.clear();

    if (m_garbageCollectorTimer > 0)
    {
        killTimer(m_garbageCollectorTimer);
        m_garbageCollectorTimer = 0;
    }

    DELETE_AND_SET_NULL(m_widgetWrapper);
    DELETE_AND_SET_NULL(m_pTimerModel);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! executes the garbage collection process
/*!
    both m_dialogList and m_objectList contain weak references to dialog, main windows or even widgets, which are
    contained in a dialog or something else. Since these widgets or windows can also be destroyed by the user, the
    UiOrganizer will not directly informed about this. Therefore, the garbage collection process will check both
    m_dialogList and m_objectList for objects, which already have been destroyed and delete these entries. Of course,
    we could also have established a connection to every widgets destroy-signal and delete the corresponding entry, when
    the destroy event will be fired. But these needs lots of connections in the Qt system, which is slower than checking
    for destroyed objects every 5 seconds or less. Due to the use of weak references, it is not dangerous to access an object
    which already has been destroyed.
*/
void UiOrganizer::execGarbageCollection()
{
    //check m_dialogList
    QMutableHashIterator<unsigned int, UiContainerItem > i(m_dialogList);
    while (i.hasNext())
    {
        i.next();
        if (i.value().container->getUiWidget() == NULL)
        {
            delete i.value().container;
            i.remove();
        }
    }

    //check m_objectList
    QMutableHashIterator<unsigned int, QPointer<QObject> > j(m_objectList);
    while (j.hasNext())
    {
        j.next();
        if (j.value().isNull())
        {
                j.remove();
        }
    }

    if (m_dialogList.size() == 0 && m_objectList.size() == 0 &&  m_garbageCollectorTimer > 0)
    {
        killTimer(m_garbageCollectorTimer);
        m_garbageCollectorTimer = 0;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! adds a given QObject* pointer to m_objectList if not yet available.
/*!
    If objPtr already exists in m_objectList, the existing key (handle) is returned, else objPtr is
    added as new value in the hash table and an new, auto incremented handle is created and returned.

    The hash table m_objectList is available, in order to increase the access to objects, which already have been used.

    \param[in] objPtr is the pointer to an existing instance, which inherits QObject
    \return the handle (key) of this QObject* in m_objectList
*/
unsigned int UiOrganizer::addObjectToList(QObject* objPtr)
{
    QHash<unsigned int, QPointer<QObject> >::const_iterator i = m_objectList.constBegin();

    while (i != m_objectList.constEnd())
    {
        if (i.value().data() == objPtr)
        {
            return i.key();
        }

        ++i;
    }

    m_objectList.insert(++UiOrganizer::autoIncObjectCounter, QPointer<QObject>(objPtr));

    return UiOrganizer::autoIncObjectCounter;
}

//----------------------------------------------------------------------------------------------------------------------------------
QObject* UiOrganizer::getWeakObjectReference(unsigned int objectID)
{
    QPointer<QObject> temp = m_objectList.value(objectID);
    if (temp.isNull() == false)
    {
        return temp.data();
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! method, called by garbage collection timer, in order to regularly start the garbage collection process
/*!
    \sa execGarbageCollection
*/
void UiOrganizer::timerEvent(QTimerEvent * /*event*/)
{
    execGarbageCollection();
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::startGarbageCollectorTimer()
{
    if (m_garbageCollectorTimer == 0)
    {
        //starts the timer to regularly check for remaining, non-deleted UIs
        m_garbageCollectorTimer = startTimer(5000);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
UiContainer* UiOrganizer::getUiDialogByHandle(unsigned int uiHandle)
{
    if (m_dialogList.contains(uiHandle))
    {
        return m_dialogList[uiHandle].container;
    }
    else
    {
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::loadPluginWidget(
        void* algoWidgetFunc,
        int uiDescription,
        const StringMap &dialogButtons,
        QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt,
        QSharedPointer<unsigned int>dialogHandle,
        QSharedPointer<unsigned int> objectID,
        QSharedPointer<QByteArray> className,
        ItomSharedSemaphore *semaphore)
{
    ito::RetVal retValue = ito::retOk;
    ito::AddInAlgo::t_algoWidget func = reinterpret_cast<ito::AddInAlgo::t_algoWidget>(algoWidgetFunc);
    QWidget *widget = func(paramsMand, paramsOpt, retValue);
    *objectID = 0;

    if (widget == NULL)
    {
        retValue += RetVal(retError, 0, tr("the plugin did not return a valid widget pointer.").toLatin1().data());
    }
    else
    {
        int winType;
        bool deleteOnClose = false;
        bool childOfMainWindow = false;
        int dockWidgetArea = Qt::TopDockWidgetArea;
        int buttonBarType = UserUiDialog::bbTypeNo;

        UiOrganizer::parseUiDescription(uiDescription, &winType, &buttonBarType, &childOfMainWindow, &deleteOnClose, &dockWidgetArea);

        if (winType == 0xff)
        {
            //guess windows type
            if (widget->inherits("QMainWindow"))
            {
                winType = typeMainWindow;
            }
            else if (widget->inherits("QDialog"))
            {
                winType = typeDialog;
            }
            else if (widget->inherits("QDockWidget"))
            {
                winType = typeDockWidget;
            }
            else
            {
                winType = typeDialog;
            }
        }

        uiDescription = UiOrganizer::createUiDescription(winType, buttonBarType, childOfMainWindow, deleteOnClose, dockWidgetArea);

        retValue += addWidgetToOrganizer(widget, uiDescription, dialogButtons, dialogHandle, objectID, className);

        if (retValue.containsError())
        {
            DELETE_AND_SET_NULL(widget);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::addWidgetToOrganizer(
        QWidget *widget,
        int uiDescription,
        const StringMap &dialogButtons,
        QSharedPointer<unsigned int>dialogHandle,
        QSharedPointer<unsigned int> objectID,
        QSharedPointer<QByteArray> className)
{
    ito::RetVal retValue;

    int type;
    int buttonBarType;
    bool childOfMainWindow;
    int dockWidgetArea;
    bool deleteOnClose = false;
    UiContainer *set = NULL;
    UiContainerItem containerItem;

    UiOrganizer::parseUiDescription(uiDescription, &type, &buttonBarType, &childOfMainWindow, &deleteOnClose, &dockWidgetArea);

    if ((dockWidgetArea & Qt::AllDockWidgetAreas) == 0)
    {
        retValue += ito::RetVal(ito::retError, 0, tr("dockWidgetArea is invalid").toLatin1().data());
    }
    else
    {

        if (type == ito::UiOrganizer::typeDialog)
        {
            //load the file and check whether it is inherited from qdialog. If so, directly load it, else stack it into a UserUiDialog
            if (widget->inherits("QDialog"))
            {
                //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
                setApiPointersToWidgetAndChildren(widget);

                startGarbageCollectorTimer();

                if (deleteOnClose)
                {
                    widget->setAttribute(Qt::WA_DeleteOnClose, true);
                }

                set = new UiContainer(qobject_cast<QDialog*>(widget));
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *objectID = addObjectToList(widget);
                *className = widget->metaObject()->className();
            }
            else
            {
                //int type, int buttonBarType, StringMap dialogButtons, bool childOfMainWindow
                UserUiDialog::tButtonBarType bbBarType = UserUiDialog::bbTypeNo;
                if (buttonBarType == UserUiDialog::bbTypeHorizontal) bbBarType = UserUiDialog::bbTypeHorizontal;
                if (buttonBarType == UserUiDialog::bbTypeVertical) bbBarType = UserUiDialog::bbTypeVertical;

                QMainWindow *mainWin = childOfMainWindow ? qobject_cast<QMainWindow*>(AppManagement::getMainWindow()) : NULL;
                UserUiDialog *dialog = new UserUiDialog(widget, bbBarType, dialogButtons, retValue, mainWin);

                if (dialog == NULL)
                {
                    retValue += RetVal(retError, 1020, tr("dialog could not be created").toLatin1().data());
                    widget->deleteLater();
                }
                else if (!retValue.containsError())
                {
                    //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
                    setApiPointersToWidgetAndChildren(dialog);

                    startGarbageCollectorTimer();

                    if (deleteOnClose)
                    {
                        dialog->setAttribute(Qt::WA_DeleteOnClose, true);
                    }

                    set = new UiContainer(dialog);
                    *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                    containerItem.container = set;
                    m_dialogList[*dialogHandle] = containerItem;
                    *objectID = addObjectToList(dialog);
                    *className = dialog->metaObject()->className();
                }
                else
                {
                    DELETE_AND_SET_NULL(dialog);
                }
            }
        }
        else if (type == ito::UiOrganizer::typeMainWindow)
        {
            //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
            setApiPointersToWidgetAndChildren(widget);

            startGarbageCollectorTimer();

            QMainWindow *win = qobject_cast<QMainWindow*>(widget);
            if (win)
            {
                if (deleteOnClose)
                {
                    win->setAttribute(Qt::WA_DeleteOnClose, true);
                }

                set = new UiContainer(win);
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *objectID = addObjectToList(win);
                *className = win->metaObject()->className();
            }
            else
            {
                widget->setWindowFlags(Qt::Window);

                if (deleteOnClose)
                {
                    widget->setAttribute(Qt::WA_DeleteOnClose, true);
                }

                set = new UiContainer(widget, UiContainer::uiTypeQMainWindow);
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *objectID = addObjectToList(widget);
                *className = widget->metaObject()->className();
            }
        }
        else if (type == ito::UiOrganizer::typeDockWidget) //dock widget
        {
            //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
            setApiPointersToWidgetAndChildren(widget);

            if (widget->inherits("QDialog"))
            {
                retValue += RetVal(retError, 0, tr("A widget inherited from QDialog cannot be docked into the main window").toLatin1().data());
                widget->deleteLater();
                widget = NULL;
            }
            else
            {
                QMainWindow *mainWin = qobject_cast<QMainWindow*>(AppManagement::getMainWindow());
                if (!mainWin)
                {
                    retValue += RetVal(retError, 0, tr("Main window not available for docking the user interface.").toLatin1().data());
                    widget->deleteLater();
                    widget = NULL;
                }
                else
                {
                    Qt::DockWidgetArea dwa = Qt::TopDockWidgetArea;
                    if (dockWidgetArea == Qt::LeftDockWidgetArea) dwa = Qt::LeftDockWidgetArea;
                    else if (dockWidgetArea == Qt::RightDockWidgetArea) dwa = Qt::RightDockWidgetArea;
                    else if (dockWidgetArea == Qt::BottomDockWidgetArea) dwa = Qt::BottomDockWidgetArea;

                    QDockWidget *dockWidget = NULL;

                    if (widget->inherits("QDockWidget"))
                    {
                        dockWidget = qobject_cast<QDockWidget*>(widget);
                    }
                    else
                    {
                        dockWidget = new QDockWidget(widget->windowTitle(), mainWin);
                        dockWidget->setWidget(widget);
                    }

                    mainWin->addDockWidget(dwa, dockWidget);
                    set = new UiContainer(dockWidget);
                    *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                    containerItem.container = set;
                    m_dialogList[*dialogHandle] = containerItem;
                    *objectID = addObjectToList(widget);
                    *className = widget->metaObject()->className();
                }
            }
        }
        else /* typeCentralWidget*/
        {
            //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
            setApiPointersToWidgetAndChildren(widget);

            if (widget->inherits("QDialog"))
            {
                retValue += RetVal(retError, 0, tr("A widget inherited from QDialog cannot be inserted into the main window").toLatin1().data());
                widget->deleteLater();
                widget = NULL;
            }
            else
            {
                MainWindow *mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());
                if (!mainWin)
                {
                    retValue += RetVal(retError, 0, tr("Main window not available for inserting the user interface.").toLatin1().data());
                    widget->deleteLater();
                    widget = NULL;
                }
                else
                {
                    retValue += mainWin->addCentralWidget(widget);
                    if (retValue.containsError())
                    {
                        widget->deleteLater();
                        widget = NULL;
                    }
                    else
                    {
                        set = new UiContainer(widget, UiContainer::uiTypeWidget);
                        *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                        containerItem.container = set;
                        m_dialogList[*dialogHandle] = containerItem;
                        *objectID = addObjectToList(widget);
                        *className = widget->metaObject()->className();
                    }
                }
            }
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getNewPluginWindow(
        const QString &pluginName,
        unsigned int &objectID,
        QWidget** newWidget,
        QWidget *parent /*= NULL*/)
{
    RetVal retValue = retOk;
    UiContainer *set = NULL;
    QMainWindow *win = NULL;
    QDialog *dlg = NULL;
    int dialogHandle;
    UiContainerItem cItem;


    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (dwo && dwo->figureClassExists(pluginName))
    {
        QWidget *mainParent = parent ? parent : qobject_cast<QWidget*>(AppManagement::getMainWindow());
        //mainParent will be set as parent-widget, this however is finally defined by the top-level mode of class AbstractFigure.
        *newWidget = loadDesignerPluginWidget(pluginName,retValue,AbstractFigure::ModeStandaloneWindow,mainParent);
        win = qobject_cast<QMainWindow*>(*newWidget);
        if (win)
        {
            if (win->inherits("ito::AbstractFigure"))
            {
                //((ito::AbstractFigure*)win)->setWindowMode(ito::AbstractFigure::ModeWindow);
            }
            else
            {
                win->setWindowFlags(Qt::Window);
                win->setAttribute(Qt::WA_DeleteOnClose, true);
            }
//            found = true;
        }

    }
    else
    {
        QString errorMsg = tr("plugin with name '%1' could be found.").arg(pluginName);
        retValue += ito::RetVal(retError, 0, errorMsg.toLatin1().data());
    }

    if (!retValue.containsError())
    {
        startGarbageCollectorTimer();

        if (win)
        {
            set = new UiContainer(win);
            dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
            cItem.container = set;
            m_dialogList[dialogHandle] = cItem;
            objectID = addObjectToList(win);
        }
        else
        {
            set = new UiContainer(dlg);
            dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
            cItem.container = set;
            m_dialogList[dialogHandle] = cItem;
            objectID = addObjectToList(dlg);
        }
    }
    else
    {
        DELETE_AND_SET_NULL(dlg);
        DELETE_AND_SET_NULL(win);
        *newWidget = NULL;
        objectID = -1;
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
QWidget* UiOrganizer::loadUiFile(const QString &filename, RetVal &retValue, QWidget *parent /*= NULL*/, const QString &objectNameSuffix /*= QString()*/)
{
    QFile file(QDir::cleanPath(filename));
    QWidget *wid = NULL;

    if (file.exists())
    {
        // set the working directory if QLoader to the directory where the ui-file is stored.
        // Then icons, assigned to the user-interface may be properly loaded, since their
        // path is always saved relatively to the ui-file,too.
        file.open(QFile::ReadOnly);
        QFileInfo fileinfo(filename);
        QDir workingDirectory = fileinfo.absoluteDir();

        // try to load translation file with the same basename than the ui-file and the suffix .qm.
        // After the basename the location string can be added using _ as delimiter.
        QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

        settings.beginGroup("Language");
        QString language = settings.value("language", "en").toString();
        settings.endGroup();

        QLocale local = QLocale(language); //language can be "language[_territory][.codeset][@modifier]"

        QTranslator *qtrans = new QTranslator();
        bool couldLoad = qtrans->load(local, fileinfo.baseName(), "_", fileinfo.path());
        if (couldLoad)
        {
            QString canonicalFilePath = fileinfo.canonicalFilePath();
            if (m_transFiles.contains(canonicalFilePath))
            {
                delete m_transFiles.value(canonicalFilePath);
                m_transFiles.remove(canonicalFilePath);
            }

            QCoreApplication::instance()->installTranslator(qtrans);
            m_transFiles.insert(canonicalFilePath, qtrans);
        }
        else
        {
            DELETE_AND_SET_NULL(qtrans);
        }

        m_pUiLoader->setWorkingDirectory(workingDirectory);

        if (objectNameSuffix == "")
        {
            wid = m_pUiLoader->load(&file, parent);
        }
        else
        {
            wid = m_pUiLoader->load(&file, NULL);
        }

        file.close();

        if (wid == NULL)
        {
            QString err = m_pUiLoader->errorString();
            retValue += RetVal(retError, 1007,
                tr("ui-file '%1' could not be loaded. Reason: %2.").arg(filename).arg(err).toLatin1().data());
        }
        else
        {
            if (objectNameSuffix != "")
            {
                QList<QWidget*> childWidgets = wid->findChildren<QWidget*>();
                QList<QLayout*> childLayouts = wid->findChildren<QLayout*>();

                foreach(QWidget *w, childWidgets)
                {
                    w->setObjectName(w->objectName() + objectNameSuffix);
                }

                foreach(QLayout *l, childLayouts)
                {
                    l->setObjectName(l->objectName() + objectNameSuffix);
                }

                // rename the widget itself
                wid->setObjectName(wid->objectName() + objectNameSuffix);

                wid->setParent(parent);
            }
        }
    }
    else
    {
        wid = NULL;
        retValue += RetVal(retError, 1006, tr("filename '%1' does not exist").arg(filename).toLatin1().data());
    }

    return wid;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::createNewDialog(
        const QString &filename,
        int uiDescription,
        const StringMap &dialogButtons,
        QSharedPointer<unsigned int> dialogHandle,
        QSharedPointer<unsigned int> objectID,
        QSharedPointer<QByteArray> className,
        ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;

    QWidget *wid = NULL;
    QString pluginClassName;

    if (filename.indexOf("itom://") == 0)
    {
        if (filename.toLower() == "itom://matplotlib" ||
            filename.toLower() == "itom://matplotlibfigure" ||
            filename.toLower() == "itom://matplotlibplot")
        {
            pluginClassName = "MatplotlibPlot";

            FigureWidget *fig = NULL;

            //create new figure and gives it its own reference, since no instance is keeping track of it
            QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
            QSharedPointer<unsigned int> figObjectID(new unsigned int);
            QSharedPointer<int> row(new int);
            *row = 1;
            QSharedPointer<int> col(new int);
            *col = 1;
            retValue += createFigure(guardedFigHandle, figObjectID, row, col, QPoint(), QSize(), NULL);
            if (!retValue.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
            {
                *dialogHandle = *(*guardedFigHandle);
                if (m_dialogList.contains(*dialogHandle))
                {
                    fig = qobject_cast<FigureWidget*>(m_dialogList[*dialogHandle].container->getUiWidget());
                    if (fig)
                    {
                        fig->setFigHandle(*guardedFigHandle);

                        fig->setAttribute(Qt::WA_DeleteOnClose, true);

                        QWidget *destWidget;

                        retValue += fig->loadDesignerWidget(0, 0, pluginClassName, &destWidget);

                        if (destWidget)
                        {
                            *objectID = addObjectToList(destWidget);

                            startGarbageCollectorTimer();

                            //destWidget->dumpObjectTree();
                            *className = destWidget->metaObject()->className();
                        }
                    }
                    else
                    {
                        retValue += RetVal::format(retError, 0,
                            tr("figHandle %i is no handle for a figure window.").toLatin1().data(), *dialogHandle);
                    }
                }
                else
                {
                    retValue += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), *dialogHandle);
                }
            }
        }
        else
        {
            QString errorMsg = tr("No internal dialog or window with name '%1' could be found.").arg(filename);
            retValue += ito::RetVal(retError, 0, errorMsg.toLatin1().data());
        }
    }
    else
    {
        bool childOfMainWindow;
        UiOrganizer::parseUiDescription(uiDescription, NULL, NULL, &childOfMainWindow, NULL, NULL);
        QMainWindow *mainWin = childOfMainWindow ? qobject_cast<QMainWindow*>(AppManagement::getMainWindow()) : NULL;

        wid = loadUiFile(filename, retValue, mainWin, "");

        if (!retValue.containsError())
        {
            retValue += addWidgetToOrganizer(wid, uiDescription, dialogButtons, dialogHandle, objectID, className);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::setApiPointersToWidgetAndChildren(QWidget *widget)
{
    //this method is also implemented in designerWidgetOrganizer!

    if (widget)
    {
        if (widget->inherits("ito::AbstractFigure"))
        {
            ((ito::AbstractFigure*)widget)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
            ((ito::AbstractFigure*)widget)->setApiFunctionBasePtr(ITOM_API_FUNCS);

            //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
            //transmitted to the plugin. This assignment cannot be done directly, since
            //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
            //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
            //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
            //also is necessary if any methods of the plugin are directly called from itom).
            QEvent evt((QEvent::Type)(QEvent::User+123));
            QCoreApplication::sendEvent(widget, &evt);
        }
        else if (widget->inherits("ito::AbstractApiWidget"))
        {
            ((ito::AbstractApiWidget*)widget)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
            ((ito::AbstractApiWidget*)widget)->setApiFunctionBasePtr(ITOM_API_FUNCS);

            //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
            //transmitted to the plugin. This assignment cannot be done directly, since
            //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
            //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
            //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
            //also is necessary if any methods of the plugin are directly called from itom).
            QEvent evt((QEvent::Type)(QEvent::User+123));
            QCoreApplication::sendEvent(widget, &evt);
        }

        QObjectList list = widget->children();
        foreach(QObject *obj, list)
        {
            setApiPointersToWidgetAndChildren(qobject_cast<QWidget*>(obj));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
QWidget* UiOrganizer::loadDesignerPluginWidget(
        const QString &className,
        RetVal &retValue,
        AbstractFigure::WindowMode winMode,
        QWidget *parent)
{
    QString tempClassName = className;
    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());

//    QUiLoader loader; //since designerWidgetOrganizer has been loaded earlier, all figure factories are loaded and correctly initialized!
    QWidget* widget = NULL;

    QStringList availableWidgets = m_pUiLoader->availableWidgets();

    bool found = false;
    foreach(const QString &name, availableWidgets)
    {
        if (QString::compare(name, tempClassName, Qt::CaseInsensitive) == 0)
        {
            found = true;
            break;
        }
    }

    if (!found)
    {
        //check for obsolete plot names
        if (QString::compare(className, "matplotlibfigure", Qt::CaseInsensitive) == 0)
        {
            tempClassName = "matplotlibplot";
        }
        else if (QString::compare(className, "itom1dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            tempClassName = "itom1dqwtplot";
        }
        else if (QString::compare(className, "itom2dqwtfigure", Qt::CaseInsensitive) == 0)
        {
            tempClassName = "itom2dqwtplot";
        }
        else
        {
            tempClassName = "";
        }

        if (tempClassName != "")
        {
            foreach(const QString &name, availableWidgets)
            {
                if (QString::compare(name, tempClassName, Qt::CaseInsensitive) == 0)
                {
                    found = true;
                    break;
                }
            }
        }
    }

    if (found)
    {
        widget = dwo->createWidget(tempClassName,parent, winMode); //loader.createWidget(className, parent);

        if (widget == NULL)
        {
            widget = m_pUiLoader->createWidget(tempClassName, parent);
        }

        if (widget == NULL)
        {
            retValue += RetVal(retError, 0, tr("designer plugin widget ('%1') could not be created").arg(tempClassName).toLatin1().data());
        }
        else
        {
            setApiPointersToWidgetAndChildren(widget);
        }
    }
    else
    {
        retValue += RetVal::format(retError, 0, tr("No designer plugin with className '%s' could be found. Please make sure that this plugin is compiled and the corresponding DLL and header files are in the designer folder").toLatin1().data(),className.toLatin1().data());
    }
    return widget;
}

//----------------------------------------------------------------------------------------------------------------------------------
/* creates a widget with a given className (case insensitive) and returns it. */
QWidget* UiOrganizer::createWidget(const QString &className, RetVal &retValue, QWidget *parent /*= NULL*/, const QString &objectName /*= QString()*/)
{
    QStringList availableWidgets = m_pUiLoader->availableWidgets();

    QString className_;

    foreach(const QString &availableWidget, availableWidgets)
    {
        if (className.compare(availableWidget, Qt::CaseInsensitive) == 0)
        {
            className_ = availableWidget;
            break;
        }
    }

    if (className_ == "")
    {
        retValue += ito::RetVal::format(ito::retError, 0, tr("Cannot find a widget with class name '%1'").arg(className).toLatin1().data());
        return NULL;
    }

    QWidget *w = m_pUiLoader->createWidget(className, parent, objectName);

    return w;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::deleteDialog(unsigned int handle, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    UiContainer *ptr = getUiDialogByHandle(handle);
    if (ptr)
    {
        delete ptr;
        m_dialogList.remove(handle);
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("dialog handle does not exist").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showDialog(
        unsigned int handle,
        int modalLevel,
        QSharedPointer<int> retCodeIfModal,
        ItomSharedSemaphore *semaphore)
{
    RetVal retValue;
    *retCodeIfModal = -1;

    UiContainer *ptr = getUiDialogByHandle(handle);
    if (ptr)
    {
        switch(ptr->getType())
        {
            case UiContainer::uiTypeUiDialog:
            case UiContainer::uiTypeQDialog:
            {
                QDialog *dlg = qobject_cast<QDialog*>(ptr->getUiWidget());
                if (dlg)
                {
                    if (modalLevel == 1) //blocking modal
                    {
                        dlg->setModal(true);
                        *retCodeIfModal = dlg->exec();
                    }
                    else if (modalLevel == 0) //non-modal
                    {
                        dlg->setModal(false);
                        dlg->show();
                    }
                    else //non-blocking modal
                    {
                        dlg->setModal(true);
                        dlg->show();
                    }
                }
            }
            break;
            case UiContainer::uiTypeQMainWindow:
            case UiContainer::uiTypeFigure:
            {
                QWidget *wid = ptr->getUiWidget();
                if (wid)
                {
                    if (modalLevel == 0) //non-modal
                    {
                        wid->setWindowModality(Qt::NonModal);
                        wid->show();
                    }
                    else if (modalLevel == 1) //blocking-modal
                    {
                        wid->setWindowModality(Qt::ApplicationModal);
                        wid->show();
                        //wait until window is hidden again
                        while(wid->isVisible())
                        {
                            QCoreApplication::processEvents();
                        }
                    }
                    else //non-blocking modal
                    {
                        wid->setWindowModality(Qt::ApplicationModal);
                        wid->show();
                    }
                }
            }
            break;
            case UiContainer::uiTypeQDockWidget:
            case UiContainer::uiTypeWidget:
            {
                QWidget *dockWidget = ptr->getUiWidget();
                if (dockWidget)
                {
                    dockWidget->show();
                }
            }
            break;
            default:
                retValue += RetVal(retError, 0, tr("Invalid widget type.").toLatin1().data());
                break;

        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::hideDialog(unsigned int handle, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        widget->hide();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getDockedStatus(
        unsigned int uiHandle,
        QSharedPointer<bool> docked,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        if (widget->inherits("ito::AbstractDockWidget"))
        {
            AbstractDockWidget *adw = (AbstractDockWidget*)widget;
            *docked = adw->docked();
        }
        else if (widget->inherits("QDockWidget"))
        {
            QDockWidget *dw = (QDockWidget*)widget;
            *docked = !dw->isFloating();
        }
        else
        {
            retValue += RetVal(retError, 0, tr("dialog cannot be docked").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setDockedStatus(
        unsigned int uiHandle,
        bool docked,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        if (widget->inherits("ito::AbstractDockWidget"))
        {
            AbstractDockWidget *adw = (AbstractDockWidget*)widget;

            if (docked)
            {
                adw->dockWidget();
            }
            else
            {
                adw->undockWidget();
            }
        }
        else if (widget->inherits("QDockWidget"))
        {
            QDockWidget *dw = (QDockWidget*)widget;
            dw->setFloating(!docked);
        }
        else
        {
            retValue += RetVal(retError, 0, tr("dialog cannot be docked or undocked").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setAttribute(
        unsigned int handle,
        Qt::WidgetAttribute attribute,
        bool on,
        ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        widget->setAttribute(attribute,on);
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::isVisible(unsigned int handle, QSharedPointer<bool> visible, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        *visible = widget->isVisible();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::handleExist(unsigned int handle, QSharedPointer<bool> exist, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);

    *exist = (ptr != NULL);

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::exists(unsigned int objectID, QSharedPointer<bool> exists, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        *exists = true;
    }
    else
    {
        *exists = false;
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetDouble(unsigned int objectID, const QString &title, const QString &label, double defaultValue, QSharedPointer<bool> ok, QSharedPointer<double> value, double min, double max, int decimals, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }


    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getDouble(parent, title, label, defaultValue, min, max, decimals, &tempOk);

    *ok = tempOk;

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetInt(unsigned int objectID, const QString &title, const QString &label, int defaultValue, QSharedPointer<bool> ok, QSharedPointer<int> value, int min, int max, int step, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }


    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getInt(parent, title, label, defaultValue, min, max, step, &tempOk);

    *ok = tempOk;

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetItem(unsigned int objectID, const QString &title, const QString &label, const QStringList &stringList, QSharedPointer<bool> ok, QSharedPointer<QString> value, int currentIndex, bool editable, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }


    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getItem(parent, title, label, stringList, currentIndex, editable, &tempOk);

    *ok = tempOk;

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetText(unsigned int objectID, const QString &title, const QString &label, const QString &defaultString, QSharedPointer<bool> ok, QSharedPointer<QString> value, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }

    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getText(parent, title, label, QLineEdit::Normal, defaultString, &tempOk);

    *ok = tempOk;

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showMessageBox(unsigned int objectID, int type, const QString &title, const QString &text, int buttons, int defaultButton, QSharedPointer<int> retButton, QSharedPointer<QString> retButtonText, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }

    QMessageBox::StandardButton stdBtn;
    QMetaObject metaObject = QMessageBox::staticMetaObject;
    QMetaEnum metaEnum = metaObject.enumerator(metaObject.indexOfEnumerator("StandardButtons"));
    QMessageBox::StandardButton stdDefaultButton;

    //check defaultButton:
    if (metaEnum.valueToKey(defaultButton) == NULL)
    {
        retValue += RetVal(retError, 1001, tr("defaultButton must be within enum QMessageBox::StandardButton").toLatin1().data());
    }
    else
    {
        stdDefaultButton = (QMessageBox::StandardButton)defaultButton;
    }

    QByteArray ba = metaEnum.valueToKeys(buttons);
    QMessageBox::StandardButtons stdButtons = static_cast<QMessageBox::StandardButtons>(metaEnum.keysToValue(ba.data()));

    if (stdButtons == -1)
    {
        retValue += RetVal(retError, 1001, tr("buttons must be within enum QMessageBox::StandardButtons").toLatin1().data());
    }

    if (defaultButton != 0 && stdButtons.testFlag(stdDefaultButton) == false)
    {
        retValue += RetVal(retError, 1001, tr("defaultButton must appear in buttons, too.").toLatin1().data());
    }

    if (!retValue.containsWarningOrError())
    {
        //if parent is inherited from AbstractDockWidget, the currently visible component
        //of parent is either the dock widget or the main widget. This is obtained
        //by getActiveInstance. This is necessary, since an invisible parent is ignored.
        if (parent && parent->inherits("ito::AbstractDockWidget"))
        {
            parent = ((ito::AbstractDockWidget*)parent)->getActiveInstance();
        }

        switch(type)
        {
        case 1:
            stdBtn = QMessageBox::information(parent, title, text, stdButtons, stdDefaultButton);
            break;
        case 2:
            stdBtn = QMessageBox::question(parent, title, text, stdButtons, stdDefaultButton);
            break;
        case 3:
            stdBtn = QMessageBox::warning(parent, title, text, stdButtons, stdDefaultButton);
            break;
        default:
            stdBtn = QMessageBox::critical(parent, title, text, stdButtons, stdDefaultButton);
            break;
        }
    }

    *retButton = (int)stdBtn;
    *retButtonText = QString(metaEnum.valueToKey((int)stdBtn));

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileDialogExistingDir(unsigned int objectID, const QString &caption, QSharedPointer<QString> directory, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }
    //if parent is inherited from AbstractDockWidget, the currently visible component
    //of parent is either the dock widget or the main widget. This is obtained
    //by getActiveInstance. This is necessary, since an invisible parent is ignored.
    else if (parent && parent->inherits("ito::AbstractDockWidget"))
    {
        parent = ((ito::AbstractDockWidget*)parent)->getActiveInstance();
    }

    QFlags<QFileDialog::Option> opt(options);
    QString result = QFileDialog::getExistingDirectory(parent, caption, *directory, opt);
    *directory = result;

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFilesOpenDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QStringList> files, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }
    //if parent is inherited from AbstractDockWidget, the currently visible component
    //of parent is either the dock widget or the main widget. This is obtained
    //by getActiveInstance. This is necessary, since an invisible parent is ignored.
    else if (parent && parent->inherits("ito::AbstractDockWidget"))
    {
        parent = ((ito::AbstractDockWidget*)parent)->getActiveInstance();
    }

    QFlags<QFileDialog::Option> opt(options);
    QStringList filters = filter.split(";;");
    QString *selectedFilter = NULL;
    if (selectedFilterIndex >= 0 && selectedFilterIndex < filters.size())
    {
        selectedFilter = new QString(filters[selectedFilterIndex]);
    }

    QStringList result = QFileDialog::getOpenFileNames(parent, caption, directory, filter, selectedFilter, opt);
    *files = result;

    DELETE_AND_SET_NULL(selectedFilter);

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileOpenDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }
    //if parent is inherited from AbstractDockWidget, the currently visible component
    //of parent is either the dock widget or the main widget. This is obtained
    //by getActiveInstance. This is necessary, since an invisible parent is ignored.
    else if (parent && parent->inherits("ito::AbstractDockWidget"))
    {
        parent = ((ito::AbstractDockWidget*)parent)->getActiveInstance();
    }

    QFlags<QFileDialog::Option> opt(options);
    QStringList filters = filter.split(";;");
    QString *selectedFilter = NULL;
    if (selectedFilterIndex >= 0 && selectedFilterIndex < filters.size())
    {
        selectedFilter = new QString(filters[selectedFilterIndex]);
    }

    QString result = QFileDialog::getOpenFileName(parent, caption, directory, filter, selectedFilter, opt);
    *file = result;

    DELETE_AND_SET_NULL(selectedFilter);

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileSaveDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if (objectID > 0)
    {
        parent = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    }
    if (parent == NULL)
    {
        parent = qobject_cast<QWidget*>(AppManagement::getMainWindow());
    }
    //if parent is inherited from AbstractDockWidget, the currently visible component
    //of parent is either the dock widget or the main widget. This is obtained
    //by getActiveInstance. This is necessary, since an invisible parent is ignored.
    else if (parent && parent->inherits("ito::AbstractDockWidget"))
    {
        parent = ((ito::AbstractDockWidget*)parent)->getActiveInstance();
    }

    QFlags<QFileDialog::Option> opt(options);
    QStringList filters = filter.split(";;");
    QString *selectedFilter = NULL;
    if (selectedFilterIndex >= 0 && selectedFilterIndex < filters.size())
    {
        selectedFilter = new QString(filters[selectedFilterIndex]);
    }

    QString result = QFileDialog::getSaveFileName(parent, caption, directory, filter, selectedFilter, opt);
    *file = result;

    DELETE_AND_SET_NULL(selectedFilter);

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getPropertyInfos(unsigned int objectID, QSharedPointer<QVariantMap> retPropertyMap, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    *retPropertyMap = QVariantMap();

    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        const QMetaObject *mo = obj->metaObject();
        QMetaProperty prop;
        int flags;

        for (int i = 0 ; i < mo->propertyCount() ; i++)
        {
            prop = mo->property(i);
            flags = 0;
            if (prop.isValid()) flags |= UiOrganizer::propValid;
            if (prop.isReadable()) flags |= UiOrganizer::propReadable;
            if (prop.isWritable()) flags |= UiOrganizer::propWritable;
            if (prop.isResettable()) flags |= UiOrganizer::propResettable;
            if (prop.isFinal()) flags |= UiOrganizer::propFinal;
            if (prop.isConstant()) flags |= UiOrganizer::propConstant;
            (*retPropertyMap)[prop.name()] = flags;
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::readProperties(unsigned int handle, const QString &widgetName, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore)
{
    unsigned int objectHandle = 0;
    UiContainer* ptr = getUiDialogByHandle(handle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        QWidget* child = widget->findChild<QWidget*>(widgetName);

        if (child)
        {
            objectHandle = this->addObjectToList(child);

        }
    }
    else
    {
        return RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    return readProperties(objectHandle, properties, semaphore);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::writeProperties(unsigned int handle, const QString &widgetName, const QVariantMap &properties, ItomSharedSemaphore *semaphore)
{
    unsigned int objectHandle = 0;
    UiContainer* ptr = getUiDialogByHandle(handle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if (widget)
    {
        QWidget* child = widget->findChild<QWidget*>(widgetName);
        if (child)
        {
            objectHandle = this->addObjectToList(child);

        }
    }
    else
    {
        return RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
    }

    return writeProperties(objectHandle, properties, semaphore);
}



//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::readProperties(unsigned int objectID, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);


    if (obj)
    {
        QStringList errString;

        QMap<QString, QVariant>::iterator i = properties->begin();
        while (i !=  properties->end())
        {
            i.value() = obj->property(i.key().toLatin1().data());
            if (!i.value().isValid())
            {
                QObject *newObj = NULL;
                QMetaProperty prop = m_widgetWrapper->fakeProperty(obj, i.key(), &newObj);
                if (prop.isValid() == false)
                {
                    errString.append(tr("property '%1' does not exist").arg(i.key()));
                }
                else
                {
                    i.value() = prop.read(newObj);
                    if (!i.value().isValid())
                    {
                        errString.append(tr("property '%1' could not be read").arg(i.key()));
                    }
                }
            }
            ++i;
        }

        if (errString.count() > 0)
        {
            retValue += RetVal(retError, errorObjPropRead, errString.join("\n").toLatin1().data());
    }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::writeProperties(unsigned int objectID, const QVariantMap &properties, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        QStringList errString;
        const QMetaObject *mo = obj->metaObject();
        QMetaProperty prop;
        int index;
        QMap<QString, QVariant>::const_iterator i = properties.constBegin();
        while (i !=  properties.constEnd())
        {
            index = mo->indexOfProperty(i.key().toLatin1().data());
            if (index == -1)
            {
                QObject *newObj = NULL;
                prop = m_widgetWrapper->fakeProperty(obj, i.key(), &newObj);
                if (prop.isValid() == false || !newObj)
                {
                    errString.append(tr("property '%1' does not exist").arg(i.key()));
                }
                else
                {
                    //check whether types need to be casted
                    //e.g. QVariantList can sometimes be casted to QPointF...
                    RetVal tempRet;
                    QVariant item;

                    if (prop.isEnumType())
                    {
                        item = QPropertyHelper::QVariantToEnumCast(i.value(), prop.enumerator(), tempRet);
                    }
                    else
                    {
                        item = QPropertyHelper::QVariantCast(i.value(), prop.userType(), tempRet);
                    }

                    if (tempRet.containsError())
                    {
                        retValue += tempRet;
                    }
                    else if (prop.write(newObj, item) == false)
                    {
                        errString.append(tr("property '%1' could not be written").arg(i.key()));
                }
            }
            }
            else
            {
                prop = mo->property(index);

                //check whether types need to be casted
                //e.g. QVariantList can sometimes be casted to QPointF...
                //bool ok;
                RetVal tempRet;
                QVariant item;

                if (prop.isEnumType())
                {
                    item = QPropertyHelper::QVariantToEnumCast(i.value(), prop.enumerator(), tempRet);
                }
                else
                {
                    item = QPropertyHelper::QVariantCast(i.value(), prop.userType(), tempRet);
                }

                if (tempRet.containsError())
                {
                    retValue += tempRet;
                }
                else if (prop.write(obj, item) == false)
                {
                    errString.append(tr("property '%1' could not be written").arg(i.key()));
            }

            }
            ++i;
        }

        if (errString.count() > 0)
        {
            retValue += RetVal(retError, errorObjPropRead, errString.join("\n").toLatin1().data());
    }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getAttribute(unsigned int objectID, int attributeNumber, QSharedPointer<bool> value, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    *value = false;
    RetVal retval;
    if (widget)
    {
        if (attributeNumber < 0 || attributeNumber >= Qt::WA_AttributeCount)
        {
            retval += RetVal(retError, 0, tr("The attribute number is out of range.").toLatin1().data());
        }
        else
        {
            *value = widget->testAttribute((Qt::WidgetAttribute)attributeNumber);
        }
    }
   else
    {
        retval += RetVal(retError, 0, tr("the objectID cannot be cast to a widget").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setAttribute(unsigned int objectID, int attributeNumber, bool value, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if (widget)
    {
        if (attributeNumber < 0 || attributeNumber >= Qt::WA_AttributeCount)
        {
            retval += RetVal(retError, 0, tr("The attribute number is out of range.").toLatin1().data());
        }
        else
        {
            widget->setAttribute((Qt::WidgetAttribute)attributeNumber, value);
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("the objectID cannot be cast to a widget").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getWindowFlags(unsigned int objectID, QSharedPointer<int> flags, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    *flags = 0;
    RetVal retval;
    if (widget)
    {
        *flags = widget->windowFlags();
    }
    else
    {
        retval += RetVal(retError, 0, tr("the objectID cannot be cast to a widget").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setWindowFlags(unsigned int objectID, int flags, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if (widget)
    {
        widget->setWindowFlags(Qt::WindowFlags(flags));
    }
    else
    {
        retval += RetVal(retError, 0, tr("the objectID cannot be cast to a widget").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::widgetMetaObjectCounts(unsigned int objectID, QSharedPointer<int> classInfoCount, QSharedPointer<int> enumeratorCount, QSharedPointer<int> methodCount, QSharedPointer<int> propertyCount, ItomSharedSemaphore *semaphore)
{
    *classInfoCount = -1;
    *enumeratorCount = -1;
    *methodCount = -1;
    *propertyCount = -1;
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        const QMetaObject *mo = obj->metaObject();
        *classInfoCount = mo->classInfoCount();
        *enumeratorCount = mo->enumeratorCount();
        *methodCount = mo->methodCount();
        *propertyCount = mo->propertyCount();
    }
    else
    {
        //the error code 'errorObjDoesNotExist' is checked in pythonUi::PyUiItem_mappingLength!
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject(unsigned int uiHandle, const QString &objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);

    if (ptr)
    {
        if (objectName != "")
        {
            QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

            if (widget)
            {
                QObject* obj = widget->findChild<QObject*>(objectName);
                if (obj)
                {
                    *objectID = addObjectToList(obj);
                }
                else
                {
                    retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
                }
            }
            else
            {
                retValue += RetVal(retError, 1001, tr("Dialog or plot handle does not (longer) exist. Maybe it has been closed before.").toLatin1().data());
            }
        }
        else //return reference to dialog or windows itself
        {
            QWidget* obj = ptr->getUiWidget();
            if (obj)
            {
                *objectID = addObjectToList(obj);
            }
            else
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("could not get reference to main dialog or window").toLatin1().data());
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("uiHandle is invalid").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject2(unsigned int parentObjectID, const QString &objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    if (m_objectList.contains(parentObjectID))
    {
        QObject* ptr = m_objectList[parentObjectID].data();

        if (ptr)
        {
            if (objectName != "")
            {
                QObject* obj = ptr->findChild<QObject*>(objectName);
                if (obj)
                {
                    *objectID = addObjectToList(obj);
                }
                else
                {
                    retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
                }
            }
            else //return reference to dialog or windows itself
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("no object name given.").toLatin1().data());
            }
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is invalid.").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is unknown.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject3(
    unsigned int parentObjectID,
    const QString &objectName,
    QSharedPointer<unsigned int> objectID,
    QSharedPointer<QByteArray> widgetClassName,
    ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    if (m_objectList.contains(parentObjectID))
    {
        QObject* ptr = m_objectList[parentObjectID].data();

        if (ptr)
        {
            if (objectName != "")
            {
                QObject* obj = ptr->findChild<QObject*>(objectName);
                if (obj)
                {
                    *objectID = addObjectToList(obj);
                    *widgetClassName = obj->metaObject()->className();
                }
                else
                {
                    //ptr->dumpObjectInfo();
                    //ptr->dumpObjectTree();
                    retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
                }
            }
            else //return reference to dialog or windows itself
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("no object name given.").toLatin1().data());
            }
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is invalid.").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("The parent widget is either unknown or does not exist any more.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getLayout(unsigned int objectID, QSharedPointer<unsigned int> layoutObjectID, QSharedPointer<QByteArray> layoutClassName, QSharedPointer<QString> layoutObjectName, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);

    if (m_objectList.contains(objectID))
    {
        QWidget* ptr = qobject_cast<QWidget*>(m_objectList[objectID].data());

        if (ptr)
        {
            if (ptr->layout())
            {
                QLayout* obj = ptr->layout();

                *layoutObjectID = addObjectToList(obj);
                *layoutClassName = obj->metaObject()->className();
                *layoutObjectName = obj->objectName();
            }
            else //return reference to dialog or windows itself
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("This uiItem has no layout.").toLatin1().data());
            }
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("This uiItem is no widet.").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("This widget is either unknown or does not exist any more.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal getObjectChildrenInfoRecursive(const QObject *obj, bool recursive, QSharedPointer<QStringList> &objectNames, QSharedPointer<QStringList> &classNames)
{
    ito::RetVal retval;

    if (obj)
    {
        QList<QObject*> children = obj->children();

        foreach (const QObject* child, children)
        {
            if (child->inherits("QWidget") || child->inherits("QLayout"))
            {
                if (child->objectName() != "")
                {
                    objectNames->append(child->objectName());
                    classNames->append(child->metaObject()->className());

                    if (recursive)
                    {
                        retval += getObjectChildrenInfoRecursive(child, recursive, objectNames, classNames);
                    }
                }
            }
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectChildrenInfo(unsigned int objectID, bool recursive, QSharedPointer<QStringList> objectNames, QSharedPointer<QStringList> classNames, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);
    objectNames->clear();
    classNames->clear();

    if (m_objectList.contains(objectID))
    {
        QObject* ptr = m_objectList[objectID].data();

        if (ptr)
        {
            retValue += getObjectChildrenInfoRecursive(ptr, recursive, objectNames, classNames);
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID is invalid.").toLatin1().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("The given object ID is unknown.").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
// returns the Qt internal signal index of the given signal signature and the widget instance.
/*
Every class, that has the Q_OBJECT macro defined and is derived from QObject (like any widget), can define several signals.
The Qt moc process turns every signal into an auto-incremented number, the so called signal index. This method tries to find
out the corresponding signal index of this signal.

\param objectID is the indentifier, that references the emitting object
\param signalSignature is the original signature of the signal, e.g. 'clicked(bool)'
\param signalIndex is the returned signal index, or -1 if the signal could not be found
\param objPtr is the object pointer that belongs to objectID. Hint: only use this pointer within another thread as long as you are sure that the object still exists
\param argTypes returned list of Qt meta type ids of the different arguments of the found signal

*/
RetVal UiOrganizer::getSignalIndex(unsigned int objectID, const QByteArray &signalSignature, QSharedPointer<int> signalIndex, QSharedPointer<QObject*> objPtr, QSharedPointer<IntList> argTypes, ItomSharedSemaphore *semaphore)
{
    *signalIndex = -1;
    argTypes->clear();
    RetVal retValue(retOk);
    int tempType;

    QObject *obj = getWeakObjectReference(objectID);
    *objPtr = obj;

    if (obj)
    {
        const QMetaObject *mo = obj->metaObject(); //the QMetaObject object of obj is able to return the signal index by the following method
        *signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature.data()));

        //every argument that can be used in signal-slot connections has to be registered to the Qt meta type system, hence,
        //every type gets a unique identifier of the meta type system. The following block tries to find out all type ids
        //of all arguments of the desired signal.
        QMetaMethod metaMethod = mo->method(*signalIndex);
        QList<QByteArray> names = metaMethod.parameterTypes();
        foreach (const QByteArray& name, names)
        {
            tempType = QMetaType::type(name.constData());
            if (tempType > 0)
            {
                argTypes->append(tempType);
            }
            else
            {
                QString msg = tr("parameter type %1 is unknown").arg(name.constData());
                retValue += RetVal(retError, errorUnregisteredType, msg.toLatin1().data());
                *signalIndex = -1;
                break;
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::connectWithKeyboardInterrupt(unsigned int objectID, const QByteArray &signalSignature, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    int signalIndex = -1;
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        const QMetaObject *mo = obj->metaObject();
        signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature.data()));

        if (signalIndex < 0)
        {
            retValue += RetVal(retError, errorSignalDoesNotExist, tr("signal does not exist").toLatin1().data());
        }
        else
        {
            if (!QMetaObject::connect(obj, signalIndex, this, this->metaObject()->indexOfSlot(QMetaObject::normalizedSignature("pythonKeyboardInterrupt(bool)"))))
            {
                retValue += RetVal(retError, errorConnectionError, tr("signal could not be connected to slot throwing a python keyboard interrupt.").toLatin1().data());
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::connectProgressObserverInterrupt(unsigned int objectID, const QByteArray &signalSignature, QPointer<QObject> progressObserver, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    int signalIndex = -1;
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if (progressObserver.isNull())
    {
        retValue += ito::RetVal(ito::retError, 0, "The given progress observer is not valid anymore.");
    }
    else if (obj)
    {
        const QMetaObject *mo = obj->metaObject();
        signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature.data()));

        if (signalIndex < 0)
        {
            retValue += RetVal(retError, errorSignalDoesNotExist, tr("signal does not exist").toLatin1().data());
        }
        else
        {
            //it is important to make a direct connection, since the progressObserver can also be created in the Python
            //thread, however we want the cancellation flag to be set immediately if the signal is emitted, hence, in
            //the thread of the caller (e.g. a button)
            if (!QMetaObject::connect(obj, signalIndex, progressObserver,
                progressObserver->metaObject()->indexOfSlot(QMetaObject::normalizedSignature("requestCancellation()")), Qt::DirectConnection))
            {
                retValue += RetVal(retError, errorConnectionError, tr("signal could not be connected to slot requesting the cancellation of the observed function call.").toLatin1().data());
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::callSlotOrMethod(
    bool slotNotMethod,
    unsigned int objectID,
    int slotOrMethodIndex,
    QSharedPointer<FctCallParamContainer> args,
    ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        //TODO: parse parameters and check whether there is a type 'ito::PythonQObjectMarshal':
        // if so, get object from objectID, destroy the arg, replace it by QObject*-type and give the object-pointer, casted to void*.

        if (slotNotMethod)
        {
            //if return value is set in qt_metacall, this is available in args->args()[0].
            //the metacall returns negative if slot has been handled/was found(not some bool...)
            if (obj->qt_metacall(QMetaObject::InvokeMetaMethod, slotOrMethodIndex, args->args()) > 0)
            {
                retValue += RetVal(retError, errorSlotDoesNotExist, tr("slot could not be found").toLatin1().data());
            }
        }
        else
        {
            // ck 07.03.17
            // changed call of widgetWrapper to already return ito::RetVal with
            // calls obj functions without changing threads
            retValue += m_widgetWrapper->call(obj, slotOrMethodIndex, args->args());
        }

        //check if arguments have to be marshalled (e.g. QObject* must be transformed to objectID
        //before passed to python in other thread)
        if (args->getRetType() == QMetaType::type("ito::PythonQObjectMarshal"))
        {
            //add m_object to weakObject-List and pass its ID to python. TODO: right now,
            //we do not check if the object is a child of obj
            ito::PythonQObjectMarshal* m = (ito::PythonQObjectMarshal*)args->args()[0];
            if (m->m_object)
            {
                m->m_objectID = addObjectToList((QObject*)(m->m_object));
                m->m_object = NULL;
            }
        }
        else if (args->getRetType() == QMetaType::type("QWidget*"))
        {
            QWidget *w = *((QWidget**)args->args()[0]);
            ito::PythonQObjectMarshal m;
            m.m_className = w->metaObject()->className();
            m.m_object = NULL;
            m.m_objectID = addObjectToList(w);
            m.m_objName = w->objectName().toLatin1();
            args->initRetArg(QMetaType::type("ito::PythonQObjectMarshal"));
            *((ito::PythonQObjectMarshal*)args->args()[0]) = m;
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getMethodDescriptions(unsigned int objectID,
    QSharedPointer<MethodDescriptionList> methodList,
    ItomSharedSemaphore *semaphore
)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);
    methodList->clear();

    if (obj)
    {
        const QMetaObject *mo = obj->metaObject();
        QMetaMethod metaMethod;
        QList<QByteArray> paramTypes;
        bool ok = false;

        for (int i = 0; i < mo->methodCount(); ++i)
        {
            metaMethod = mo->method(i);
            ok = true;

            if (metaMethod.access() == QMetaMethod::Public && (metaMethod.methodType() == QMetaMethod::Slot || metaMethod.methodType() == QMetaMethod::Method))
            {
                //check if args can be interpreted by QMetaType:
                if (strcmp(metaMethod.typeName(), "") != 0)
                {
                    if (QMetaType::type(metaMethod.typeName()) == 0)
                    {
                        //qDebug() << "unsupported return type: " << metaMethod.typeName();
                        ok = false;
                    }
                }

                if (ok)
                {
                    paramTypes = metaMethod.parameterTypes();

                    for (int j = 0; j < paramTypes.size(); ++j)
                    {
                        if (QMetaType::type(paramTypes[j].data()) == 0)
                        {
                            //qDebug() << "unsupported arg type: " << paramTypes[j].data();
                            ok = false;
                            break;
                        }
                    }
                }

                if (ok)
                {
                    methodList->append(MethodDescription(metaMethod));
                }
            }
        }

        methodList->append(m_widgetWrapper->getMethodList(obj));
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The widget is not available (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::pythonKeyboardInterrupt(bool /*checked*/)
{
    PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (pyEngine)
    {
        bool interruptActuatorsAndTimers = false;
        pyEngine->pythonInterruptExecutionThreadSafe(&interruptActuatorsAndTimers);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< the following map translates Qt/C++ datatypes into their Python representations. This is for instance used in the info()-method in Python to show the user the Python syntax.
struct PyCMap {
    const char *ctype;
    const char *pytype;
} pyCMap[] = {
    { "QString", "str" },
    { "QByteArray", "bytearray" },
    { "ito::Shape", "shape" },
    { "QVector<ito::Shape>", "seq. of shape" },
    { "QSharedPointer<ito::DataObject>", "dataObject" },
    { "QVector<double>", "seq. of float" },
    { "double", "float" },
    { "QVector<float>", "seq. of float" },
    { "QFont", "font" },
    { "QColor", "color str, rgba or hex" },
    { "QPointer<ito::AddInDataIO>", "dataIO" },
    { "QPointer<ito::AddInActuator>", "actuator" },
    { "QPointer<ito::AddInBase>", "dataIO or actuator" },
    { "QStringList", "seq. of str" },
    { "ito::AutoInterval", "autoInterval" },
    { "ito::ItomPlotHandle", "uiItem" },
    { "QRegion", "region" },
    { "QTime", "datetime.time" },
    { "QDate", "datetime.date" },
    { "QDateTime", "datetime.datetime" },
    { "QVector<int>", "seq. of int" },
    { "QVector3D", "seq. of 3 floats" },
    { 0, 0 }
};

//----------------------------------------------------------------------------------------------------------------------------------
QByteArray UiOrganizer::getReadableMethodSignature(const QMetaMethod &method, bool pythonNotCStyle, QByteArray *methodName /*= NULL*/, bool *valid /*= NULL*/)
{
    QByteArray name = method.name();

    if (methodName)
        *methodName = name;

    QList<QByteArray> parameters;
    QList<QByteArray> names = method.parameterNames();
    QList<QByteArray> types = method.parameterTypes();

    if (valid)
    {
        *valid = true;
    }

    const PyCMap *e;
    bool found = false;

    for (int i = 0; i < names.count(); ++i)
    {
        if (pythonNotCStyle)
        {
            if (valid && types[i].contains("*"))
            {
                *valid = false;
            }
            else
            {
                found = false;
                e = pyCMap;
                while (e->ctype && !found)
                {
                    if (types[i] == e->ctype)
                    {
                        parameters << (names[i] + " {" + e->pytype + "}");
                        found = true;
                    }
                    ++e;
                }
            }

            if (!found)
            {
                parameters << (names[i] + " {" + types[i] + "}");
            }
        }
        else
        {
            parameters << (types[i] + " " + names[i]);
        }
    }


    return (name + "(" + parameters.join(", ") + ")");
}

//----------------------------------------------------------------------------------------------------------------------------------
QByteArray UiOrganizer::getReadableParameter(const QByteArray &parameter, bool pythonNotCStyle, bool *valid /*= NULL*/)
{
    if (valid)
    {
        *valid = true;
    }

    if (pythonNotCStyle)
    {
        if (valid && parameter.contains("*"))
        {
            *valid = false;
            return parameter;
        }
        else
        {
            const PyCMap *e = pyCMap;
            while (e->ctype)
            {
                if (parameter == e->ctype)
                {
                    return e->pytype;
                }
                ++e;
            }
        }
    }

    return parameter;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::UiOrganizer::ClassInfoContainerList::Iterator UiOrganizer::parseMetaPropertyForEnumerationTypes(const QMetaProperty &prop, ClassInfoContainerList &currentPropList)
{
    if (prop.isEnumType() || prop.isFlagType())
    {
        QMetaEnum e = prop.enumerator();
        QByteArray listName;
        ClassInfoContainer::Type type = prop.isFlagType() ? ClassInfoContainer::TypeFlag : ClassInfoContainer::TypeEnum;

        if (strlen(e.scope()) > 0)
        {
            listName = QByteArray(e.scope()) + "::" + QByteArray(e.name());
        }
        else
        {
            listName = QByteArray(e.name());
        }

        bool found = false;
        ClassInfoContainerList::Iterator it = currentPropList.begin();
        while (it != currentPropList.end())
        {
            if (it->m_type == type && it->m_name == listName)
            {
                found = true;
                break;
            }
            it++;
        }

        if (!found)
        {
            QByteArray values;
            for (int i = 0; i < e.keyCount(); ++i)
            {
                if (i > 0)
                {
                    values += QByteArray(";");
                }
                values += QByteArray("'") + e.key(i) + QByteArray("' (") + QByteArray::number(e.value(i)) + QByteArray(")");
            }
            return currentPropList.insert(currentPropList.end(), ClassInfoContainer(type, listName, values, values));
        }
        else
        {
            return it;
        }
    }

    return currentPropList.end();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectAndWidgetName(unsigned int objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        *objectName = obj->objectName().toLatin1();
        *widgetClassName = obj->metaObject()->className();
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object ID is not available").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectInfo(const QString &classname, bool pythonNotCStyle, ito::UiOrganizer::ClassInfoContainerList *objectInfo, ItomSharedSemaphore *semaphore)
{
    ito::RetVal retval;
    QWidget* newWidget = loadDesignerPluginWidget(classname, retval, ito::AbstractFigure::ModeStandaloneWindow, NULL);
    if (newWidget)
    {
        retval += getObjectInfo((QObject*)newWidget, UiOrganizer::infoShowItomInheritance, pythonNotCStyle, objectInfo);
    }

    DELETE_AND_SET_NULL(newWidget)

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}


//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectInfo(const QObject *obj, int type, bool pythonNotCStyle, ito::UiOrganizer::ClassInfoContainerList *objectInfo, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);
    ClassInfoContainerList tmpObjectInfo;

    if (obj)
    {
        QMap<QByteArray, QByteArray> propInfoMap, signalInfoMap, slotInfoMap;

        const QMetaObject *mo = obj->metaObject();
        QByteArray className = mo->className();
        QByteArray firstClassName = className;
        QStringList qtBaseClasses = QStringList() << "QWidget" << "QMainWindow" << "QFrame";
        QByteArray methodName;
        QByteArray signature;
        bool readonly;
        bool valid;
        QString description, shortDescription;

        while (mo != NULL)
        {
            if ((type & infoShowInheritanceUpToWidget) && qtBaseClasses.contains(className, Qt::CaseInsensitive) == 0)
            {
                break;
            }

            if (className.startsWith('Q') && (type & infoShowItomInheritance))
            {
                break;
            }

            if (mo != obj->metaObject()) //mo is already any base class
            {
                tmpObjectInfo.append(ClassInfoContainer(ClassInfoContainer::TypeInheritance, QLatin1String(className), "", ""));
            }

            for (int i = mo->classInfoCount() - 1; i >= 0; i--)
            {
                QMetaClassInfo ci = mo->classInfo(i);
                if (i >= mo->classInfoOffset())
                {
                    if (strstr(ci.name(), "prop://") == ci.name())
                    {
                        QByteArray prop = QByteArray(&(ci.name()[7]));
                        if (!propInfoMap.contains(prop))
                        {
                            //prefer the documentation from derived classes instead of the one from base classes, due to possible function overload
                            propInfoMap[prop] = ci.value();
                        }
                    }
                    else if (strstr(ci.name(), "signal://") == ci.name())
                    {
                        QByteArray prop = QByteArray(&(ci.name()[9]));
                        if (!signalInfoMap.contains(prop))
                        {
                            //prefer the documentation from derived classes instead of the one from base classes, due to possible function overload
                            signalInfoMap[prop] = ci.value();
                        }
                    }
                    else if (strstr(ci.name(), "slot://") == ci.name())
                    {
                        QByteArray prop = QByteArray(&(ci.name()[7]));
                        if (!slotInfoMap.contains(prop))
                        {
                            //prefer the documentation from derived classes instead of the one from base classes, due to possible function overload
                            slotInfoMap[prop] = ci.value();
                        }
                    }
                    else
                    {
                        tmpObjectInfo.append(ClassInfoContainer(ClassInfoContainer::TypeClassInfo, QLatin1String(ci.name()), QLatin1String(ci.value())));
                    }
                }
            }

            for (int i = mo->propertyCount() - 1; i >= 0; i--)
            {
                QMetaProperty prop = mo->property(i);
                signature = getReadableParameter(prop.typeName(), pythonNotCStyle, &valid);
                readonly = (prop.isWritable() == false);

                if (i >= mo->propertyOffset() && valid)
                {
                    ClassInfoContainerList::Iterator enumIterator = parseMetaPropertyForEnumerationTypes(prop, tmpObjectInfo);
                    QString str1("prop_");
                    str1 += QString(prop.name());
                    QString enumString;
                    QString enumStringShort;

                    if (enumIterator != tmpObjectInfo.end())
                    {
                        if (enumIterator->m_type == ClassInfoContainer::TypeEnum)
                        {
                            enumString = QString("\n\nThe type '%1' is an enumeration that can have one of the following values (str or int):\n\n* ").arg(QLatin1String(signature));
                            enumStringShort = QLatin1String(" Enumeration values: ");
                        }
                        else
                        {
                            enumString = QString("\n\nThe type '%1' is a flag mask that can be a combination of one or several of the following values (or-combination number values or semicolon separated strings):\n\n* ").arg(QLatin1String(signature));
                            enumStringShort = QLatin1String(" Flag values (combination possible): ");
                        }

                        enumString += enumIterator->m_shortDescription.split(";").join("\n* ");
                        enumStringShort += enumIterator->m_shortDescription;
                    }


                    if (propInfoMap.contains(prop.name()))
                    {
                        int idx = propInfoMap[prop.name()].indexOf("\n");
                        if (idx < 0)
                        {
                            shortDescription = QString("%1 {%2} -> %3%4").arg(prop.name()).arg(QString(signature)).arg(QString(propInfoMap[prop.name()])).arg(readonly ? " (readonly)" : "");
                        }
                        else
                        {
                            shortDescription = QString("%1 {%2} -> %3 ...%4").arg(prop.name()).arg(QString(signature)).arg(QString(propInfoMap[prop.name()].left(idx))).arg(readonly ? " (readonly)" : "");
                        }

                        description = QString("{%1} -> %2%3%4").arg(QString(signature)).arg(QString(propInfoMap[prop.name()])).arg(readonly ? " (readonly)" : "").arg(enumString);

                        if (enumStringShort.isEmpty() == false)
                        {
                            shortDescription += enumStringShort;
                        }
                    }
                    else
                    {
                        shortDescription = QString("%1 {%2}%3").arg(prop.name()).arg(QString(signature)).arg(readonly ? " (readonly)" : "");
                        description = QString("{%1} -> %3%4").arg(QString(signature)).arg(readonly ? " (readonly)" : "").arg(enumString);

                        if (enumStringShort.isEmpty() == false)
                        {
                            shortDescription += (QString(" ->") + enumStringShort);
                        }
                    }



                    tmpObjectInfo.append(ClassInfoContainer(ClassInfoContainer::TypeProperty, QLatin1String(prop.name()), shortDescription, description));
                }
            }

            for (int i = mo->methodCount() - 1; i >= 0; i--)
            {
                QMetaMethod meth = mo->method(i);
                QByteArray methodSignature = meth.methodSignature();

                if (meth.methodType() == QMetaMethod::Signal)
                {
                    //if the info is rendered for python style, exclude all signals or slots that have the tag ITOM_PYNOTACCESSIBLE before their definition.
                    if (i >= mo->methodOffset() && (!pythonNotCStyle || strcmp(meth.tag(), "ITOM_PYNOTACCESSIBLE") != 0))
                    {
                        signature = getReadableMethodSignature(meth, pythonNotCStyle, &methodName, &valid);
                        if (valid)
                        {
                            QString str2(signature);

                            if (pythonNotCStyle)
                            {
                                if (signalInfoMap.contains(methodName) && !signalInfoMap[methodName].isEmpty())
                                {
                                    int idx = signalInfoMap[methodName].indexOf("\n");
                                    if (idx < 0)
                                    {
                                        shortDescription = QString("%1 -> %2 (connect signature: %3)").arg(QLatin1String(signature)).arg(QLatin1String(signalInfoMap[methodName])).arg(QLatin1String(methodSignature));
                                    }
                                    else
                                    {
                                        shortDescription = QString("%1 -> %2 ... (connect signature: %3)").arg(QLatin1String(signature)).arg(QLatin1String(signalInfoMap[methodName].left(idx))).arg(QLatin1String(methodSignature));
                                    }

                                    description = QString("%1 -> %2\n\nNotes\n-----\n\nTo connect to this signal use the following signature::\n\n    yourItem.connect('%3', yourMethod)\n"). \
                                        arg(QLatin1String(signature)).arg(QLatin1String(signalInfoMap[methodName])).arg(QLatin1String(methodSignature));
                                }
                                else
                                {
                                    description = QString("%1 -> signature for connection to this signal: %2").arg(QLatin1String(signature)).arg(QLatin1String(methodSignature));
                                    shortDescription = QString("%1 -> signature for connection to this signal: %2").arg(QLatin1String(signature)).arg(QLatin1String(methodSignature));
                                }
                            }
                            else
                            {
                                if (signalInfoMap.contains(methodName) && !signalInfoMap[methodName].isEmpty())
                                {
                                    int idx = signalInfoMap[methodName].indexOf("\n");
                                    if (idx < 0)
                                    {
                                        shortDescription = QString("%1 -> %2").arg(QLatin1String(signature)).arg(QLatin1String(signalInfoMap[methodName]));
                                    }
                                    else
                                    {
                                        shortDescription = QString("%1 -> %2 ...").arg(QLatin1String(signature)).arg(QLatin1String(signalInfoMap[methodName].left(idx)));
                                    }

                                    description = QString("%1 -> %2").arg(QLatin1String(signature)).arg(QLatin1String(slotInfoMap[methodName]));
                                }
                                else
                                {
                                    description = shortDescription = QLatin1String(signature);
                                }
                            }

                            tmpObjectInfo.append(ClassInfoContainer(ClassInfoContainer::TypeSignal, QLatin1String(methodName), shortDescription, description));
                        }
                    }
                }
                else if (meth.methodType() == QMetaMethod::Slot && meth.access() == QMetaMethod::Public)
                {
                    //if the info is rendered for python style, exclude all signals or slots that have the tag ITOM_PYNOTACCESSIBLE before their definition.
                    if (i >= mo->methodOffset() && (!pythonNotCStyle || strcmp(meth.tag(), "ITOM_PYNOTACCESSIBLE") != 0))
                    {
                        signature = getReadableMethodSignature(meth, pythonNotCStyle, &methodName, &valid);
                        if (valid)
                        {
                            QString str2(signature);


                            if (slotInfoMap.contains(methodName) && !slotInfoMap[methodName].isEmpty())
                            {
                                int idx = slotInfoMap[methodName].indexOf("\n");
                                if (idx < 0)
                                {
                                    shortDescription = QString("%1 -> %2").arg(QLatin1String(signature)).arg(QLatin1String(slotInfoMap[methodName]));
                                }
                                else
                                {
                                    shortDescription = QString("%1 -> %2...").arg(QLatin1String(signature)).arg(QLatin1String(slotInfoMap[methodName].left(idx)));
                                }

                                description = QString("%1 -> %2").arg(QLatin1String(signature)).arg(QLatin1String(slotInfoMap[methodName]));
                            }
                            else
                            {
                                shortDescription = description = QLatin1String(signature);
                            }

                            tmpObjectInfo.append(ClassInfoContainer(ClassInfoContainer::TypeSlot, QLatin1String(methodName), shortDescription, description));
                        }
                    }
                }
            }

            if (type & (infoShowItomInheritance | infoShowInheritanceUpToWidget | infoShowAllInheritance))
            {
                mo = mo->superClass();
                if (mo)
                {
                    className = mo->className();
                }
            }
            else
            {
                mo = NULL;
            }
        }

        if (!objectInfo)
        {
            if (tmpObjectInfo.length() > 0)
            {
                std::sort(
                    tmpObjectInfo.begin(),
                    tmpObjectInfo.end(),
                    [](ito::ClassInfoContainer t1, ito::ClassInfoContainer t2) {
                        return t1.m_name < t2.m_name;
                    });
            }

            std::cout << "Widget '" << firstClassName.data() << "'\n--------------------------\n" << std::endl;
            valid = false;
            foreach(const ClassInfoContainer &c, tmpObjectInfo)
            {
                if (c.m_type == ClassInfoContainer::TypeClassInfo)
                {
                    if (!valid)
                    {
                        std::cout << "\nClass Info\n---------------\n";
                        valid = true;
                    }

                    std::cout << " " << c.m_shortDescription.toLatin1().data() << "\n";
                    std::cout << "\n" << std::endl;
                }


            }

            valid = false;
            foreach(const ClassInfoContainer &c, tmpObjectInfo)
            {
                if (c.m_type == ClassInfoContainer::TypeProperty)
                {
                    if (!valid)
                    {
                        std::cout << "\nProperties\n---------------\n";
                        valid = true;
                    }

                    std::cout << " " << c.m_shortDescription.toLatin1().data() << "\n";
                }

            }

            valid = false;
            foreach(const ClassInfoContainer &c, tmpObjectInfo)
            {
                if (c.m_type == ClassInfoContainer::TypeSignal)
                {
                    if (!valid)
                    {
                        std::cout << "\nSignals\n---------------\n";
                        valid = true;
                    }

                    std::cout << " " << c.m_shortDescription.toLatin1().data() << "\n";
                }

            }

            valid = false;
            foreach(const ClassInfoContainer &c, tmpObjectInfo)
            {
                if (c.m_type == ClassInfoContainer::TypeSlot)
                {
                    if (!valid)
                    {
                        std::cout << "\nSlots\n---------------\n";
                        valid = true;
                    }

                    std::cout << " " << c.m_shortDescription.toLatin1().data() << "\n";
                }

            }
        }
        else
        {
            *objectInfo = tmpObjectInfo;
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("The requested widget does not exist (any more).").toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectID(const QObject *obj, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);

    *objectID = 0;

    QHash<unsigned int, QPointer<QObject> >::iterator elem = m_objectList.begin();
    while (elem != m_objectList.end())
    {
        if(elem.value() == obj)
        {
            *objectID = elem.key();
            break;
        }
        ++elem;
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void  UiOrganizer::figureAssureMinimalSize(ito::FigureWidget* fig)
{
    //minimum size of a new figure window (see also apiFunctionsGraph::mgetFigure)
    QSize minimumFigureSize = fig->defaultSize();
    QSize sz = fig->sizeHint();
    sz.rwidth() = qMax(minimumFigureSize.width(), sz.width());
    sz.rheight() = qMax(minimumFigureSize.height(), sz.height());
    fig->resize(sz);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UiOrganizer::figurePlot(
        ito::UiDataContainer &dataCont,
        ito::UiDataContainer &xAxisCont,
        QSharedPointer<unsigned int> figHandle,
        QSharedPointer<unsigned int> objectID,
        int areaRow, int areaCol,
        QString className,
        QVariantMap properties,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, figObjectID, row, col, QPoint(), QSize(), NULL);
        if (!retval.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
        {
            *figHandle = *(*guardedFigHandle);
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            fig->setFigHandle(*guardedFigHandle);
        }
    }

    if (!retval.containsError())
    {
        if (m_dialogList.contains(*figHandle))
        {
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            if (fig)
            {
                QWidget *destWidget = NULL;
#if ITOM_POINTCLOUDLIBRARY > 0
                if (dataCont.getType() == ito::ParamBase::PointCloudPtr)
                {
                    retval += fig->plot(dataCont.getPointCloud(), areaRow, areaCol, className, &destWidget);
                }
                else if (dataCont.getType() == ito::ParamBase::PolygonMeshPtr)
                {
                    retval += fig->plot(dataCont.getPolygonMesh(), areaRow, areaCol, className, &destWidget);
                }
                else if (dataCont.getType() == ito::ParamBase::DObjPtr)
#else
                if (dataCont.getType() == ito::ParamBase::DObjPtr)
#endif
                {
                    retval += fig->plot(dataCont.getDataObject(), xAxisCont.getDataObject() ,areaRow, areaCol, className, &destWidget);
                }
                else
                {
                    retval += ito::RetVal(ito::retError, 0, tr("unsupported data type").toLatin1().data());
                }

                if (!retval.containsError())
                {
                    *objectID = addObjectToList(destWidget);

                    if (properties.size() > 0)
                    {
                        retval += writeProperties(*objectID, properties, NULL);
                    }

                    figureAssureMinimalSize(fig);
                }
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("figHandle %i is not handle for a figure window.").toLatin1().data(), *figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), *figHandle);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureLiveImage(
        AddInDataIO* dataIO,
        QSharedPointer<unsigned int> figHandle,
        QSharedPointer<unsigned int> objectID,
        int areaRow, int areaCol,
        QString className,
        QVariantMap properties,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, figObjectID, row, col, QPoint(), QSize(), NULL);
        if (!retval.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
        {
            *figHandle = *(*guardedFigHandle);
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            fig->setFigHandle(*guardedFigHandle);
        }
    }

    if (!retval.containsError())
    {

        if (m_dialogList.contains(*figHandle))
        {
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            if (fig)
            {
                QWidget *destWidget;
                retval += fig->liveImage(dataIO, areaRow, areaCol, className, &destWidget);

                *objectID = addObjectToList(destWidget);

                if (properties.size() > 0)
                {
                    retval += writeProperties(*objectID, properties, NULL);
                }

                figureAssureMinimalSize(fig);
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("figHandle %i is not handle for a figure window.").toLatin1().data(), *figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), *figHandle);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureDesignerWidget(
        QSharedPointer<unsigned int> figHandle,
        QSharedPointer<unsigned int> objectID,
        int areaRow, int areaCol,
        QString className,
        QVariantMap properties,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, figObjectID, row, col, QPoint(), QSize(), NULL);
        if (!retval.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
        {
            *figHandle = *(*guardedFigHandle);
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            fig->setFigHandle(*guardedFigHandle);
        }
    }

    if (!retval.containsError())
    {

        if (m_dialogList.contains(*figHandle))
        {
            fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
            if (fig)
            {
                QWidget *destWidget;
                retval += fig->loadDesignerWidget(areaRow, areaCol, className, &destWidget);

                *objectID = addObjectToList(destWidget);

                if (properties.size() > 0)
                {
                    retval += writeProperties(*objectID, properties, NULL);
                }

                figureAssureMinimalSize(fig);
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("figHandle %i is not handle for a figure window.").toLatin1().data(), *figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), *figHandle);
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::createFigure(
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigureHandle,
        QSharedPointer<unsigned int> objectID,
        QSharedPointer<int> rows, QSharedPointer<int> cols,
        QPoint offset /*= QPoint()*/, QSize size /*= QSize()*/,
        ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    unsigned int h;
    UiContainerItem containerItem;
    UiContainer *set = NULL;
    MainWindow *mainWin = NULL;
    unsigned int forcedHandle = 0;
    bool found = false;

    *objectID = 0;

    //you can pass a figure handle by guardedFigureHandle.
    //if this is the case, all available figures are checked for this handle
    //and if it exists and is a figure, its guardedHandle will be returned.
    if (!guardedFigureHandle.isNull() && !(*guardedFigureHandle).isNull())
    {
        h = *(*guardedFigureHandle);

        if (m_dialogList.contains(h))
        {
            containerItem = m_dialogList[h];
            if (containerItem.container->getType() == UiContainer::uiTypeFigure)
            {
                FigureWidget *fig = qobject_cast<FigureWidget*>(containerItem.container->getUiWidget());
                if (fig)
                {
                    if (size.isValid())
                    {
                        fig->resize(size);
                    }

                    if (offset.isNull() == false)
                    {
                        fig->move(offset);
                    }
                    *rows = fig->rows();
                    *cols = fig->cols();
                    *guardedFigureHandle = (containerItem.guardedHandle).toStrongRef();
                    *objectID = addObjectToList(const_cast<FigureWidget*>(fig));
                    found = true;
                }
                else
                {
                    retValue += ito::RetVal(ito::retError, 0, tr("figure window is not available any more").toLatin1().data());
                }
            }
            else
            {
                retValue += ito::RetVal::format(ito::retError, 0, tr("handle '%1' is no figure.").toLatin1().data(), h);
            }
        }
        else
        {
            if (h == 0)
            {
                retValue += ito::RetVal(ito::retError, 0, tr("handle '0' cannot be assigned.").toLatin1().data());
            }
            else
            {
                forcedHandle = h;
            }
        }
    }

    if (!found)
    {
        startGarbageCollectorTimer();

        unsigned int *handle = new unsigned int; //will be guarded and destroyed by guardedFigureHandle below
        if (forcedHandle == 0)
        {
            *handle = ++UiOrganizer::autoIncUiDialogCounter;
        }
        else
        {
            *handle = forcedHandle; //does not exist any more!
            if (UiOrganizer::autoIncObjectCounter < forcedHandle)
            {
                UiOrganizer::autoIncObjectCounter = forcedHandle; //the next figure must always get a really new and unique handle number
            }
        }

        QString title = tr("Figure %1").arg(*handle);
        FigureWidget *fig2 = new FigureWidget(title, false, true, *rows, *cols, NULL);
        fig2->setAttribute(Qt::WA_DeleteOnClose); //always delete figure window, if user closes it
        QObject::connect(fig2,SIGNAL(destroyed(QObject*)),this,SLOT(figureDestroyed(QObject*)));

        mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());
        if (mainWin)
        {
            mainWin->addAbstractDock(fig2, Qt::TopDockWidgetArea);
        }

        if (size.isValid())
        {
            fig2->resize(size);
        }

        if (offset.isNull() == false)
        {
            fig2->move(offset);
        }

        set = new UiContainer(fig2);

        *guardedFigureHandle = QSharedPointer<unsigned int>(handle); //, threadSafeDeleteUi);
        *objectID = addObjectToList(fig2);
        containerItem.container = set;
        containerItem.guardedHandle = (*guardedFigureHandle).toWeakRef();
        m_dialogList[*handle] = containerItem;
    }

    if (semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::figureDestroyed(QObject *obj)
{
    QHash<unsigned int, ito::UiContainerItem>::iterator i = m_dialogList.begin();
    while (i != m_dialogList.end())
    {
        if (i.value().container->getUiWidget() == obj)
        {
            delete i.value().container;
            m_dialogList.erase(i);
            break;
        }
        ++i;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getSubplot(
        QSharedPointer<unsigned int> figHandle,
        unsigned int subplotIndex,
        QSharedPointer<unsigned int> objectID,
        QSharedPointer<QByteArray> objectName,
        QSharedPointer<QByteArray> widgetClassName,
        ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;

    if (m_dialogList.contains(*figHandle))
    {
        fig = qobject_cast<FigureWidget*>(m_dialogList[*figHandle].container->getUiWidget());
        if (fig)
        {
            QObject* obj = fig->getSubplot(subplotIndex);
            if (obj)
            {
                *objectID = addObjectToList(obj);
                *widgetClassName = obj->metaObject()->className();
                *objectName = obj->objectName().toLatin1();
            }
            else
            {
                retval += RetVal::format(retError, errorObjDoesNotExist, tr("subplot at indexed position %i is not available").toLatin1().data(), subplotIndex);
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("figHandle %i is not a handle for a figure window.").toLatin1().data(), *figHandle);
        }
    }
    else
    {
        retval += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), *figHandle);
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::closeAllFloatableFigures(ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;
    QHash<unsigned int, ito::UiContainerItem>::iterator i = m_dialogList.begin();
    while (i != m_dialogList.end())
    {
        fig = qobject_cast<FigureWidget*>(i.value().container->getUiWidget());
        if (fig && !fig->docked())
        {
            fig->setFigHandle(empty);
            delete i.value().container;
            i = m_dialogList.erase(i);
        }
        else
        {
            ++i;
        }
    }
    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureClose(unsigned int figHandle, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;

    if (figHandle > 0) //delete one single figure
    {
        if (m_dialogList.contains(figHandle))
        {
            fig = qobject_cast<FigureWidget*>(m_dialogList[figHandle].container->getUiWidget());
            if (fig)
            {
                fig->setFigHandle(empty);
                delete m_dialogList[figHandle].container;
                m_dialogList.remove(figHandle);
            }
            else
            {
                retval += RetVal::format(retError, 0, tr("figHandle %i is not a handle for a figure window.").toLatin1().data(), figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError, 0, tr("figHandle %i not available.").toLatin1().data(), figHandle);
        }
    }
    else //delete all figures
    {
        QHash<unsigned int, ito::UiContainerItem>::iterator i = m_dialogList.begin();
        FigureWidget *fig;
        while (i != m_dialogList.end())
        {
            fig = qobject_cast<FigureWidget*>(i.value().container->getUiWidget());
            if (fig)
            {
                fig->setFigHandle(empty);
                delete i.value().container;
                i = m_dialogList.erase(i);
            }
            else
            {
                ++i;
            }
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureShow(const unsigned int& handle/*=0*/,ItomSharedSemaphore *semaphore /*=NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;


    if (handle == 0)//all figures
    {
        QHash<unsigned int, ito::UiContainerItem>::iterator i = m_dialogList.begin();
        while (i != m_dialogList.end())
        {
            fig = qobject_cast<FigureWidget*>(i.value().container->getUiWidget());
            if (fig)
            {
                fig->raiseAndActivate();

            }
            ++i;
        }
    }
    else
    {
       if (m_dialogList.contains(handle))
       {
           fig = qobject_cast<FigureWidget*>(m_dialogList[handle].container->getUiWidget());
           if (fig)
           {
               fig->raiseAndActivate();
           }
       }
       else
       {
           retval += RetVal::format(retError, 0, tr("could not get figure with handle %i.").toLatin1().data(), handle);
       }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureMinimizeAll(ItomSharedSemaphore *semaphore /*=NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;

    QHash<unsigned int, ito::UiContainerItem>::iterator i = m_dialogList.begin();

    while (i != m_dialogList.end())
    {
        fig = qobject_cast<FigureWidget*>(i.value().container->getUiWidget());
        if (fig)
        {
            fig->mini();

        }
        ++i;
    }


    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figurePickPoints(unsigned int objectID, QSharedPointer<QVector<ito::Shape> > shapes, int maxNrPoints, ItomSharedSemaphore *semaphore)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if (widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1 ||metaObject->indexOfSignal("userInteractionDone(int,bool,QVector<ito::Shape>)") == -1)
        {
            retval += RetVal(retError, 0, tr("The desired widget has no signals/slots defined that enable the pick points interaction").toLatin1().data());
        }
        else
        {
            UserInteractionWatcher *watcher = new UserInteractionWatcher(widget, ito::Shape::MultiPointPick, maxNrPoints, shapes, semaphore, this);
            connect(watcher, SIGNAL(finished()), this, SLOT(watcherThreadFinished()));
            QThread *watcherThread = new QThread();
            watcher->moveToThread(watcherThread);
            watcherThread->start();

            m_watcherThreads[watcher] = watcherThread;
        }
    }
   else
    {
        retval += RetVal(retError, 0, tr("the required widget does not exist (any more)").toLatin1().data());
    }

    if (semaphore && retval.containsError()) //else the semaphore is released by the userInteractionWatcher-instance, executed in a separate thread and monitoring the widget
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureDrawGeometricShapes(
        unsigned int objectID,
        QSharedPointer<QVector<ito::Shape> > shapes,
        int shapeType, int maxNrPoints,
        ItomSharedSemaphore *semaphore)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if (widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1 ||metaObject->indexOfSignal("userInteractionDone(int,bool,QVector<ito::Shape>)") == -1)
        {
            retval += RetVal(retError, 0, tr("The desired widget has no signals/slots defined that enable the pick points interaction").toLatin1().data());
        }
        else
        {
            UserInteractionWatcher *watcher = new UserInteractionWatcher(widget, (ito::Shape::ShapeType)shapeType, maxNrPoints, shapes, semaphore, this);
            connect(watcher, SIGNAL(finished()), this, SLOT(watcherThreadFinished()));
            QThread *watcherThread = new QThread();
            watcher->moveToThread(watcherThread);
            watcherThread->start();

            m_watcherThreads[watcher] = watcherThread;
        }
    }
   else
    {
        retval += RetVal(retError, 0, tr("the required widget does not exist (any more)").toLatin1().data());
    }

    if (semaphore && retval.containsError()) //else the semaphore is released by the userInteractionWatcher-instance, executed in a separate thread and monitoring the widget
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figurePickPointsInterrupt(unsigned int objectID)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if (widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1)
        {
            retval += RetVal(retError, 0, tr("The desired widget has no signals/slots defined that enable the pick points interaction").toLatin1().data());
        }
        else
        {
            QMetaObject::invokeMethod(obj, "userInteractionStart", Q_ARG(int,1), Q_ARG(bool,false), Q_ARG(int,0));
        }
    }
   else
    {
        retval += RetVal(retError, 0, tr("the required widget does not exist (any more)").toLatin1().data());
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::isFigureItem(
        unsigned int objectID,
        QSharedPointer<unsigned int> isFigureItem,
        ItomSharedSemaphore *semaphore)
{
    QWidget *widget = qobject_cast<QWidget*>(getWeakObjectReference(objectID));
    RetVal retval;

    if (widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1 || metaObject->indexOfSignal("userInteractionDone(int,bool,QVector<ito::Shape>)") == -1)
        {
            *isFigureItem = 0;
        }
        else
        {
            *isFigureItem = 1;
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("the required widget does not exist (any more)").toLatin1().data());
        *isFigureItem = 0;
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getAvailableWidgetNames(
        QSharedPointer<QStringList> widgetNames,
        ItomSharedSemaphore *semaphore)
{
    ito::RetVal retval;

    *widgetNames = m_pUiLoader->availableWidgets();

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::connectWidgetsToProgressObserver(bool hasProgressBar, unsigned int progressBarObjectID, bool hasLabel, unsigned int labelObjectID, QSharedPointer<ito::FunctionCancellationAndObserver> progressObserver, ItomSharedSemaphore *semaphore)
{
    ito::RetVal retval;

    if (progressObserver.isNull())
    {
        retval += ito::RetVal(ito::retError, 0, "progressObserver is invalid");
    }
    else
    {
        if (hasProgressBar)
        {
            QObject *progressBar = getWeakObjectReference(progressBarObjectID);

            if (!progressBar)
            {
                retval += ito::RetVal(ito::retError, 0, "progressBar widget does not exist.");
            }
            else
            {
                bool conn = QObject::connect(progressObserver.data(), SIGNAL(progressValueChanged(int)), progressBar, SLOT(setValue(int)));
                if (!conn)
                {
                    retval += ito::RetVal(ito::retError, 0, "Could not connect with 'setValue(int)' slot of progressBar. Probably the progressBar does not have such a slot.");
                }
            }
        }

        if (hasLabel)
        {
            QObject *label = getWeakObjectReference(labelObjectID);

            if (!label)
            {
                retval += ito::RetVal(ito::retError, 0, "label widget does not exist.");
            }
            else
            {
                bool conn = QObject::connect(progressObserver.data(), SIGNAL(progressTextChanged(QString)), label, SLOT(setText(QString)));
                if (!conn)
                {
                    retval += ito::RetVal(ito::retError, 0, "Could not connect with 'setText(QString)' slot of label. Probably the label does not have such a slot.");
                }
            }
        }
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ //void UiOrganizer::threadSafeDeleteUi(unsigned int *handle)
//{
    //UiOrganizer *orga = qobject_cast<UiOrganizer*>(AppManagement::getUiOrganizer());
    //if (orga)
    //{
    //    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
    //    QMetaObject::invokeMethod(orga, "deleteDialog", Q_ARG(uint, *handle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())); //'unsigned int' leads to overhead and is automatically transformed to uint in invokeMethod command
    //    //question: do we need locker here?
    //    locker.getSemaphore()->wait(-1);
    //}
    //delete handle;
//}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::watcherThreadFinished()
{
    QObject *sender = QObject::sender();
    if (sender)
    {
        if (m_watcherThreads.contains(sender))
        {
            sender->deleteLater();

            QThread *thread = m_watcherThreads[sender];
            thread->quit();
            thread->wait();
            DELETE_AND_SET_NULL(thread);
            m_watcherThreads.remove(sender);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! registerActiveTimer can be used to register a running timer instance for the 'active timer dialog' of the main window
/*!
This method is usually invoked by the Python class pyTimer such that the timer instance can be shown in the active timer
dialog of the main window where it can also be stopped by the user.

\param timer is the weak pointer to the active QTimer instance
\param name is a name that describes the timer (e.g. its timer id)
\return ito::retOk if active timer was valid and could be registered (e.g. for active timer dialog), else ito::retError
\sa unregisterActiveTimer
*/
RetVal UiOrganizer::registerActiveTimer(const QWeakPointer<QTimer>& timer, const QString &name)
{
    ito::RetVal retval;

    if (!timer.isNull())
    {
        m_pTimerModel->registerNewTimer(timer, name);
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("timer is invalid").toLatin1().data());
    }

    return retval;
}


//-------------------------------------------------------------------------------------
//!  returns the timer model that organizes all registered QTimer objects
/*!
\return TimerModel*
*/
TimerModel* UiOrganizer::getTimerModel() const
{
    return m_pTimerModel;
}

//-------------------------------------------------------------------------------------
//! getAllAvailableHandles ruturns all available figure handles
/*!
This method is usually invoked by mainWindow.

\param list is a shared pointer to to a QList of type unsigned int. Here the values will be placed in
\param semaphore is the optional semaphore for thread-based calls (or NULL)
\return ito::retOk if a the list could be filled
*/
RetVal UiOrganizer::getAllAvailableHandles(QSharedPointer<QList<unsigned int> > list, ItomSharedSemaphore * semaphore /*=NULL*/)
{
    ito::RetVal retval;
    list->clear();
    auto it = m_dialogList.constBegin();
    FigureWidget *fig = nullptr;

    while (it != m_dialogList.constEnd())
    {
        fig = qobject_cast<FigureWidget*>(it.value().container->getUiWidget());

        if (fig)
        {
            list->append(it.key());
        }

        ++it;
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
RetVal UiOrganizer::getPlotWindowTitlebyHandle(const unsigned int& objectID, QSharedPointer<QString> title, ItomSharedSemaphore * semaphore /*=NULL*/)
{
    ito::RetVal retval;
    FigureWidget *fig = nullptr;

    if (m_dialogList.contains(objectID))
    {
        fig = qobject_cast<FigureWidget*>(m_dialogList[objectID].container->getUiWidget());
        if (fig)
        {
            *title = fig->windowTitle();
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("could not find figure with given handle %1").arg(objectID).toLatin1().data());
    }

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}

//-------------------------------------------------------------------------------------
RetVal UiOrganizer::copyStringToClipboard(const QString &text, ItomSharedSemaphore *semaphore)
{
    RetVal retval;
    auto clipboard = QApplication::clipboard();
    clipboard->setText(text, QClipboard::Clipboard);

    if (semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
    }

    return retval;
}


} //end namespace ito
