/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#if (QT_VERSION < QT_VERSION_CHECK(5, 5, 0))
#include <QtDesigner/QDesignerCustomWidgetInterface>
#else
#include <QtUiPlugin/QDesignerCustomWidgetInterface>
#endif
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
    m_widgetWrapper(NULL),
    m_pUiLoader(NULL)
{
    m_dialogList.clear();
    m_objectList.clear();

    m_widgetWrapper = new WidgetWrapper();

    qRegisterMetaType<ito::UiDataContainer>("ito::UiDataContainer");
    qRegisterMetaType<ito::UiDataContainer>("ito::UiDataContainer&");
    qRegisterMetaType<ito::UiOrganizer::ClassInfoContainerList*>("ito::UiOrganizer::ClassInfoContainerList*");
    qRegisterMetaType<ito::UiOrganizer::ClassInfoContainerList*>("ito::UiOrganizer::ClassInfoContainerList&");
	qRegisterMetaType<QPointer<QTimer> >("QPointer<QTimer>");
	

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
RetVal UiOrganizer::loadPluginWidget(void* algoWidgetFunc, int uiDescription, const StringMap &dialogButtons, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore)
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

        retValue += addWidgetToOrganizer(widget, uiDescription, dialogButtons, dialogHandle, initSlotCount, objectID, className);

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
RetVal UiOrganizer::addWidgetToOrganizer(QWidget *widget, int uiDescription, const StringMap &dialogButtons, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className)
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
                *initSlotCount = widget->metaObject()->methodOffset();
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
                    *initSlotCount = dialog->metaObject()->methodOffset();
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
                *initSlotCount = win->metaObject()->methodOffset();
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
                *initSlotCount = widget->metaObject()->methodOffset();
                *objectID = addObjectToList(widget);
                *className = widget->metaObject()->className();
            }
        }
        else //dock widget
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
                    *initSlotCount = widget->metaObject()->methodOffset();
                    *objectID = addObjectToList(widget);
                    *className = widget->metaObject()->className();
                }
            }
        }
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getNewPluginWindow(const QString &pluginName, unsigned int &objectID, QWidget** newWidget, QWidget *parent /*= NULL*/)
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

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::createNewDialog(const QString &filename, int uiDescription, const StringMap &dialogButtons, QSharedPointer<unsigned int> dialogHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    
    QWidget *wid = NULL;
    QString pluginClassName;

    if (filename.indexOf("itom://") == 0)
    {
        if (filename.toLower() == "itom://matplotlib" || filename.toLower() == "itom://matplotlibfigure" || filename.toLower() == "itom://matplotlibplot")
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
            retValue += createFigure(guardedFigHandle, initSlotCount, figObjectID, row, col, QPoint(), QSize(), NULL);
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
                        retValue += RetVal::format(retError, 0, tr("figHandle %i is no handle for a figure window.").toLatin1().data(), *dialogHandle);
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
        QFile file(QDir::cleanPath(filename));
        if (file.exists())
        {
            bool childOfMainWindow;
            UiOrganizer::parseUiDescription(uiDescription, NULL, NULL, &childOfMainWindow, NULL, NULL);
            QMainWindow *mainWin = childOfMainWindow ? qobject_cast<QMainWindow*>(AppManagement::getMainWindow()) : NULL;

            //set the working directory if QLoader to the directory where the ui-file is stored. Then icons, assigned to the user-interface may be properly loaded, since their path is always saved relatively to the ui-file,too.
            file.open(QFile::ReadOnly);
            QFileInfo fileinfo(filename);
            QDir workingDirectory = fileinfo.absoluteDir();

            //try to load translation file with the same basename than the ui-file and the suffix .qm. After the basename the location string can be added using _ as delimiter.
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
                delete qtrans;
            }

            m_pUiLoader->setWorkingDirectory(workingDirectory);
            wid = m_pUiLoader->load(&file, mainWin);
            file.close();

            if (wid == NULL)
            {
                retValue += RetVal(retError, 1007, tr("ui-file '%1' could not be correctly parsed.").arg(filename).toLatin1().data());
            }
        }
        else
        {
            wid = NULL;
            retValue += RetVal(retError, 1006, tr("filename '%1' does not exist").arg(filename).toLatin1().data());
        }

        if (!retValue.containsError())
        {
            retValue += addWidgetToOrganizer(wid, uiDescription, dialogButtons, dialogHandle, initSlotCount, objectID, className);
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
QWidget* UiOrganizer::loadDesignerPluginWidget(const QString &className, RetVal &retValue, AbstractFigure::WindowMode winMode, QWidget *parent)
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
RetVal UiOrganizer::showDialog(unsigned int handle, int modalLevel, QSharedPointer<int> retCodeIfModal, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

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
                        *retCodeIfModal = -1;
                        dlg->show();
                    }
                    else //non-blocking modal
                    {
                        dlg->setModal(true);
                        *retCodeIfModal = -1;
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
            {
                QWidget *dockWidget = ptr->getUiWidget();
                if (dockWidget)
                {
                    dockWidget->show();
                }
            }
            break;

        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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

    if (ptr)
    {
        ptr->getUiWidget()->hide();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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
RetVal UiOrganizer::getDockedStatus(unsigned int uiHandle, QSharedPointer<bool> docked, ItomSharedSemaphore *semaphore /*= NULL*/)
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
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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
RetVal UiOrganizer::setDockedStatus(unsigned int uiHandle, bool docked, ItomSharedSemaphore *semaphore /*= NULL*/)
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
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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
RetVal UiOrganizer::setAttribute(unsigned int handle, Qt::WidgetAttribute attribute, bool on, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);

    if (ptr)
    {
        ptr->getUiWidget()->setAttribute(attribute,on);
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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

    if (ptr)
    {
        *visible = ptr->getUiWidget()->isVisible();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("Dialog or plot handle does (not longer) exist. Maybe it has been closed before.").toLatin1().data());
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

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
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

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
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

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
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

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
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
    UiContainer* set = getUiDialogByHandle(handle);
    if (set)
    {
        QWidget* widget = set->getUiWidget()->findChild<QWidget*>(widgetName);
        if (widget)
        {
            objectHandle = this->addObjectToList(widget);

        }
    }

    return readProperties(objectHandle, properties, semaphore);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::writeProperties(unsigned int handle, const QString &widgetName, const QVariantMap &properties, ItomSharedSemaphore *semaphore)
{
    unsigned int objectHandle = 0;
    UiContainer* set = getUiDialogByHandle(handle);
    if (set)
    {
        QWidget* widget = set->getUiWidget()->findChild<QWidget*>(widgetName);
        if (widget)
        {
            objectHandle = this->addObjectToList(widget);

        }
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
            QObject* obj = ptr->getUiWidget()->findChild<QObject*>(objectName);
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
RetVal UiOrganizer::getChildObject3(unsigned int parentObjectID, const QString &objectName, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore)
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
        const QMetaObject *mo = obj->metaObject();
        *signalIndex = mo->indexOfSignal(QMetaObject::normalizedSignature(signalSignature.data()));

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
RetVal UiOrganizer::connectWithKeyboardInterrupt(unsigned int objectID, const QByteArray &signalSignature, ItomSharedSemaphore *semaphore)
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
RetVal UiOrganizer::callSlotOrMethod(bool slotNotMethod, unsigned int objectID, int slotOrMethodIndex, QSharedPointer<FctCallParamContainer> args, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if (obj)
    {
        //TODO: parse parameters and check whether there is a type 'ito::PythonQObjectMarshal':
        // if so, get object from objectID, destroy the arg, replace it by QObject*-type and give the object-pointer, casted to void*.
        
        bool success;
        if (slotNotMethod)
        {
            success = obj->qt_metacall(QMetaObject::InvokeMetaMethod, slotOrMethodIndex, args->args());
            //if return value is set in qt_metacall, this is available in args->args()[0].
            if (success == false)
            {
                retValue += RetVal(retError, errorSlotDoesNotExist, tr("slot could not be found").toLatin1().data());
            }
        }
        else
        {
            // ck 07.03.17
            // changed call of widgetWrapper to already return ito::RetVal with more detailed information about the failure reason 
            retValue += m_widgetWrapper->call(obj, slotOrMethodIndex, args->args());
        }

        //check if arguments have to be marshalled (e.g. QObject* must be transformed to objectID before passed to python in other thread)
        if (args->getRetType() == QMetaType::type("ito::PythonQObjectMarshal"))
        {
            //add m_object to weakObject-List and pass its ID to python. TODO: right now, we do not check if the object is a child of obj
            ito::PythonQObjectMarshal* m = (ito::PythonQObjectMarshal*)args->args()[0];
            if (m->m_object)
            {
                m->m_objectID = addObjectToList((QObject*)(m->m_object));
                m->m_object = NULL;
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
RetVal UiOrganizer::getMethodDescriptions(unsigned int objectID, QSharedPointer<MethodDescriptionList> methodList, ItomSharedSemaphore *semaphore)
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
        for (int i = 0; i<mo->methodCount(); ++i)
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
    PyErr_SetInterrupt();
}

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
    QByteArray name;
#if QT_VERSION >= 0x050000
    name = method.name();
#else
    QString methName = method.signature();
    methName.chop(methName.length() - methName.indexOf('('));
    name = methName.toLatin1();
#endif

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

#if QT_VERSION >= 0x050400
    return (name + "(" + parameters.join(", ") + ")");
#else
    QByteArray p;
    for (int i = 0; i < parameters.size(); ++i)
    {
        if (i == 0)
        {
            p += parameters[i];
        }
        else
        {
            p += (QByteArray(", ") + parameters[i]);
        }
    }

    return (name + "(" + p + ")");
#endif
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
#if QT_VERSION >= 0x050000                
                QByteArray methodSignature = meth.methodSignature();
#else
                QByteArray methodSignature = meth.signature();
#endif

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

                                    description = QString("%1 -> %2\n\nNotes\n--------------\n\nTo connect to this signal use the following signature::\n\n    yourItem.connect('%3', yourMethod)\n"). \
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
                qSort(tmpObjectInfo);
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
////----------------------------------------------------------------------------------------------------------------------------------
////----------------------------------------------------------------------------------------------------------------------------------
//RetVal UiOrganizer::plotImage(QSharedPointer<ito::DataObject> dataObj, QSharedPointer<unsigned int> plotHandle, QString plotClassName /*= ""*/, ItomSharedSemaphore *semaphore)
//{
//    ItomSharedSemaphoreLocker locker(semaphore);
//
//    ito::RetVal retval;
//    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
//    if (dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, tr("DesignerWidgetOrganizer is not available").toLatin1().data());
//    }
//    else
//    {
//        int dims = dataObj->getDims();
//        int sizex = dataObj->getSize(dims - 1);
//        int sizey = dataObj->getSize(dims - 2);
//        if ((dims == 1) || ((dims > 1) && ((sizex == 1) || (sizey == 1))))
//        {
//            plotClassName = dwo->getFigureClass("DObjStaticLine", plotClassName, retval);
//            
//        }
//        else
//        {
//            plotClassName = dwo->getFigureClass("DObjStaticImage", plotClassName, retval);
//            //not 1D so try 2D ;-) new 2dknoten()
//        }
//
//        if (!retval.containsError())
//        {
//            QObject *window;
//            retval += getNewPluginWindow(plotClassName, *plotHandle, &window);
//
//            if (!retval.containsError() && window)
//            {
//                ito::AbstractDObjFigure *mainWin;
//
//                if (window->inherits("ito::AbstractDObjFigure"))
//                    mainWin = (ito::AbstractDObjFigure*)(window);
//                else
//                    mainWin = NULL;
//
//                if (mainWin)
//                {
//                    mainWin->setWindowMode(ito::AbstractFigure::ModeWindow);
//                    /*mainWin->setWindowFlags(Qt::Window);
//                    mainWin->setAttribute(Qt::WA_DeleteOnClose, true);*/
//
//                    mainWin->setSource(dataObj);
//
//                    mainWin->show();
//                }
//                else
//                {
//                    retval += ito::RetVal(ito::retError,0, tr("Plot window could not be loaded").toLatin1().data());
//                }
//            }
//        }
//    }
//
//    if (semaphore)
//    {
//        semaphore->returnValue = retval;
//        semaphore->release();
////        ItomSharedSemaphore::deleteSemaphore(semaphore);
////        semaphore = NULL;
//    }
//
//    return RetVal(retOk);
//}
//
////----------------------------------------------------------------------------------------------------------------------------------
//RetVal UiOrganizer::liveData(AddInDataIO* dataIO, QString widgetName,  QObject **window, ItomSharedSemaphore *semaphore)
//{
//    RetVal retVal(retOk);
//    unsigned int plotHandle;
//    ito::AbstractDObjFigure *mainWin = NULL;
//
//    retVal += getNewPluginWindow(widgetName, plotHandle, window);
//
//    if (!retVal.containsError() && (*window))
//    {
//        if ((*window)->inherits("ito::AbstractDObjFigure"))
//            mainWin = (ito::AbstractDObjFigure*)(*window);
//        else
//            mainWin = NULL;
//
//        if (mainWin)
//        {
//            mainWin->setWindowMode(ito::AbstractFigure::ModeWindow);
//            //mainWin->setWindowFlags(Qt::Window);
//            //mainWin->setAttribute(Qt::WA_DeleteOnClose, true);
//            mainWin->show();
//        }
//        else
//        {
//            retVal += ito::RetVal(ito::retError,0, tr("Plot window could not be loaded").toLatin1().data());
//        }
//    }
//
//    mainWin->setCamera(QPointer<ito::AddInDataIO>(dataIO));
//    /*mainWin->setProperty("liveSource", v);
//    
//    if (dataIO)
//    {
//        AddInManager *aim = AddInManager::getInstance();
//        retVal += aim->incRef((ito::AddInBase*)dataIO);
//        mainWin->getInputParam("liveData")->setVal<void*>(dataIO);
//    }
//
//    QMetaObject::invokeMethod(dataIO, "startDeviceAndRegisterListener", Q_ARG(QObject*, *window), Q_ARG(ItomSharedSemaphore*, NULL));*/
//
//    if (semaphore)
//    {
//        semaphore->returnValue = retVal;
//        semaphore->release();
//        ItomSharedSemaphore::deleteSemaphore(semaphore);
//        semaphore = NULL;
//    }
//
//    return retVal;
//}
//
////----------------------------------------------------------------------------------------------------------------------------------
//RetVal UiOrganizer::liveImage(AddInDataIO* dataIO, QString plotClassName /*= ""*/, ItomSharedSemaphore *semaphore)
//{
//    ito::RetVal retval;
//    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
//    if (dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, tr("DesignerWidgetOrganizer is not available").toLatin1().data());
//    }
//    else
//    {
//        plotClassName = dwo->getFigureClass("DObjLiveImage", plotClassName, retval);
//
//        if (!retval.containsError())
//        {
//            QObject *window = NULL;
//            retval = liveData(dataIO, plotClassName, &window, NULL);
//        //    ito::RetVal retval = liveData(dataIO, "itom2DGVFigure", &window, NULL);
//
//            if (dataIO && !retval.containsError())
//            {
//                long maxInt = 1;
//                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//                ito::Param param = dataIO->getParamRec("bpp");
//                if (param.getName() != NULL)   // Parameter is defined
//                {
//                    QSharedPointer<ito::Param> qsParam(new ito::Param(param));
//                    QMetaObject::invokeMethod(dataIO, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
//                    while (!locker.getSemaphore()->wait(PLUGINWAIT))
//                    {
//                        if (!dataIO->isAlive())
//                        {
//                            break;
//                        }
//                    }
//                    if (!locker.getSemaphore()->returnValue.containsError())
//                    {
//                        int bpp = (*qsParam).getVal<int>();
//
//                        if (bpp < 17)    // scale only for int8, uint8, int16, uint16
//                        {
//                            maxInt = (maxInt << bpp) - 1;
//                            if (window)
//                            {
//                                ((ito::AbstractDObjFigure*)window)->setZAxisInterval(QPointF(0.0, maxInt));
//                                ((ito::AbstractDObjFigure*)window)->setColorPalette("grayMarked");
//                            }
//                        }
//                        else if (bpp == 24)
//                        {
//                            maxInt = (maxInt << bpp) - 1;
//                            if (window)
//                            {
//                                ((ito::AbstractDObjFigure*)window)->setZAxisInterval(QPointF(0.0, maxInt));
//                                ((ito::AbstractDObjFigure*)window)->setColorPalette("RGB24");
//                            }                    
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    if (semaphore)
//    {
//        semaphore->returnValue = retval;
//        semaphore->release();
//        ItomSharedSemaphore::deleteSemaphore(semaphore);
//        semaphore = NULL;
//    }
//
//    return retval;
//}
//
////----------------------------------------------------------------------------------------------------------------------------------
//RetVal UiOrganizer::liveLine(AddInDataIO* dataIO, QString plotClassName /*= ""*/, ItomSharedSemaphore *semaphore)
//{
//    ito::RetVal retval;
//    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
//    if (dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, tr("DesignerWidgetOrganizer is not available").toLatin1().data());
//    }
//    else
//    {
//        plotClassName = dwo->getFigureClass("DObjLiveLine", plotClassName, retval);
//
//        if (!retval.containsError())
//        {
//
//            QObject *window = NULL;
//            retval = liveData(dataIO, plotClassName, &window, NULL);
//
//            if (dataIO && !retval.containsError())
//            {
//                long maxInt = 1;
//                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//                ito::Param param = dataIO->getParamRec("bpp");
//                if (param.getName() != NULL)   // Parameter is defined
//                {
//                    QSharedPointer<ito::Param> qsParam(new ito::Param(param));
//                    QMetaObject::invokeMethod(dataIO, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
//                    while (!locker.getSemaphore()->wait(PLUGINWAIT))
//                    {
//                        if (!dataIO->isAlive())
//                        {
//                            break;
//                        }
//                    }
//                    if (!locker.getSemaphore()->returnValue.containsError())
//                    {
//                        int bpp = (*qsParam).getVal<int>();
//
//                        if (bpp < 16)    // scale only for int8, uint8, int16, uint16
//                        {
//                            maxInt = (maxInt << bpp) - 1;
//                            if (window)
//                            {
//                                ((ito::AbstractDObjFigure*)window)->setYAxisInterval(QPointF(0.0, maxInt));
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    if (semaphore)
//    {
//        semaphore->returnValue = retval;
//        semaphore->release();
//        ItomSharedSemaphore::deleteSemaphore(semaphore);
//        semaphore = NULL;
//    }
//
//    return retval;
//}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal UiOrganizer::figurePlot(ito::UiDataContainer &dataCont, ito::UiDataContainer &xAxisCont, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> initSlotCount(new unsigned int);
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, initSlotCount, figObjectID, row, col, QPoint(), QSize(), NULL);
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

                    //minimum size of a new figure window (see also apiFunctionsGraph::mgetFigure)
                    QSize minimumFigureSize(700, 400);
                    QSize sz = fig->sizeHint();
                    sz.rwidth() = qMax(minimumFigureSize.width(), sz.width());
                    sz.rheight() = qMax(minimumFigureSize.height(), sz.height());
                    fig->resize(sz);
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
RetVal UiOrganizer::figureLiveImage(AddInDataIO* dataIO, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> initSlotCount(new unsigned int);
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, initSlotCount, figObjectID, row, col, QPoint(), QSize(), NULL);
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
RetVal UiOrganizer::figureDesignerWidget(QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if (*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>());
        QSharedPointer<unsigned int> initSlotCount(new unsigned int);
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure(guardedFigHandle, initSlotCount, figObjectID, row, col, QPoint(), QSize(), NULL);
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
RetVal UiOrganizer::createFigure(QSharedPointer< QSharedPointer<unsigned int> > guardedFigureHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<int> rows, QSharedPointer<int> cols, QPoint offset /*= QPoint()*/, QSize size /*= QSize()*/, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    unsigned int h;
    UiContainerItem containerItem;
    UiContainer *set = NULL;
    MainWindow *mainWin = NULL;
    unsigned int forcedHandle = 0;
    bool found = false;

    *initSlotCount = 0;
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
                    *initSlotCount = fig->metaObject()->methodOffset();
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
                retValue += ito::RetVal::format(ito::retError, 0, tr("handle '%i' is no figure.").toLatin1().data(), h);
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
        *initSlotCount = fig2->metaObject()->methodOffset();
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
RetVal UiOrganizer::getSubplot(QSharedPointer<unsigned int> figHandle, unsigned int subplotIndex, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore /*= NULL*/)
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
RetVal UiOrganizer::figureDrawGeometricShapes(unsigned int objectID, QSharedPointer<QVector<ito::Shape> > shapes, int shapeType, int maxNrPoints, ItomSharedSemaphore *semaphore)
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
RetVal UiOrganizer::isFigureItem(unsigned int objectID,  QSharedPointer<unsigned int> isFigureItem, ItomSharedSemaphore *semaphore)
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
RetVal UiOrganizer::getAvailableWidgetNames(QSharedPointer<QStringList> widgetNames, ItomSharedSemaphore *semaphore)
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
\param semaphore is the optional semaphore for thread-based calls (or NULL)
\return ito::retOk if active timer was valid and could be registered (e.g. for active timer dialog), else ito::retError
\sa unregisterActiveTimer
*/
RetVal UiOrganizer::registerActiveTimer(const QPointer<QTimer>& timer, const QString &name, ItomSharedSemaphore *semaphore /*= NULL*/)
{
	ito::RetVal retval;
	if (timer.data())
	{
		TimerContainer timerContainer;
		timerContainer.timer = timer;
		timerContainer.name = name;
		m_timers.append(timerContainer);
		connect(timer.data(), SIGNAL(destroyed()), this, SLOT(unregisterActiveTimer()));
	}
	else
	{
        retval += ito::RetVal(ito::retError, 0, tr("timer is invalid").toLatin1().data());
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
//!  returns a QList with the TimerContainers inside. This function is for instance by 'timer dialog'
/*!
This method is usually called by the Python 'active timer dialog'  and is rquired for updating the sialog

\return QList<TimerContainer> 
*/
QList<TimerContainer> UiOrganizer::getRegisteredTimers()
{
	return m_timers;
}

//! unregisterActiveTimer scans the m_timers qList for NULL pointers 
/*!
This private slot is usually connected to the destroyed signal of a timer registered in m_timers. mTimers provides the 
active timers which are needed for the timer manager. If a timer is deleted the pointer in m_timers will be NULL. In this
slot the list is scanned for pointers equal to NULL.


\param semaphore is the optional semaphore for thread-based calls (or NULL)
\return ito::retOk if active timer was valid and could be registered (e.g. for active timer dialog), else ito::retError
\sa registerActiveTimer
*/
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::unregisterActiveTimer(ItomSharedSemaphore *semaphore  /*= NULL*/)
{
	ito::RetVal retval;
	int i;
	for (i = m_timers.length() - 1; i >= 0; --i)
	{
		if (m_timers.at(i).timer == NULL)
		{
			m_timers.removeAt(i);
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

} //end namespace ito
