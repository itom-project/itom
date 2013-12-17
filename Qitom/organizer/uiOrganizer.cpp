/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut f�r Technische Optik (ITO),
    Universit�t Stuttgart, Germany

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

#include "../api/apiFunctions.h"
#include "../api/apiFunctionsGraph.h"

#include "common/helperCommon.h"
#include "common/addInInterface.h"
#include "../AppManagement.h"
#include "plot/AbstractFigure.h"
#include "plot/AbstractDObjFigure.h"
#include "plot/AbstractItomDesignerPlugin.h"
#include "designerWidgetOrganizer.h"

#include "widgetWrapper.h"
#include "userInteractionWatcher.h"
#include "../python/pythonQtConversion.h"

#include "common/sharedStructuresPrimitives.h"

#include "qmainwindow.h"
#include "widgets/mainWindow.h"

#include <qinputdialog.h>
#include <qmessagebox.h>
#include <qmetaobject.h>
#include <qfiledialog.h>
//#include <QtUiTools/quiloader.h>
#include <qcoreapplication.h>
#include <qpluginloader.h>
#include <QtDesigner/QDesignerCustomWidgetInterface>
#include <qsettings.h>
#include <qcoreapplication.h>





//! destructor
/*!
	If the widget, observed by the UiDialogSet-instance is still valid, it is registered for deletion by the Qt-system.
*/
UiContainer::~UiContainer()
{
    if(!m_weakDialog.isNull())
    {
        if(m_type == UiContainer::uiTypeFigure)
        {
            MainWindow *mainWin = qobject_cast<MainWindow*>( AppManagement::getMainWindow() );
            if(mainWin)
            {
                mainWin->removeAbstractDock( qobject_cast<ito::AbstractDockWidget*>(m_weakDialog.data()) );
            }
        }
		else if(m_type == UiContainer::uiTypeQDockWidget)
		{
			MainWindow *mainWin = qobject_cast<MainWindow*>( AppManagement::getMainWindow() );
            if(mainWin)
            {
                mainWin->removeDockWidget( qobject_cast<QDockWidget*>(m_weakDialog.data()) );
            }
		}

        m_weakDialog.data()->deleteLater();
        m_weakDialog.clear();
    }
}





/*!
    \class UiOrganizer
    \brief The UiOrganizer is started as singleton instance within itom and organizes all main windows, dialogs, widgets,...
			which are currently loaded at runtime from any ui-file or from a widget, provided by any algorithm plugin
*/


unsigned int UiOrganizer::autoIncUiDialogCounter = 1;
unsigned int UiOrganizer::autoIncObjectCounter = 1;

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    creates the singleton instance of WidgetWrapper. The garbage collection timer is not started yet, since
	this is done if the first user interface becomes being organized by this class.
*/
UiOrganizer::UiOrganizer() :
    m_garbageCollectorTimer(0)
{
    m_dialogList.clear();
	m_objectList.clear();

    m_widgetWrapper = new WidgetWrapper();

    /*QUiLoader loader;
    ito::RetVal retval = scanPlugins(qApp->applicationDirPath().append("/designer"), m_pluginInfoList);*/
    /*QSharedPointer<unsigned int> test3;
    {
        QSharedPointer<unsigned int> test(new unsigned int, threadSafeDeleteUi);
        QWeakPointer<unsigned int> test2 = test;
        test3 = test2;
    }*/
}

//----------------------------------------------------------------------------------------------------------------------------------
//! destructor
/*!
    Deletes all remaining user interfaces, by deleting the corresponding instances of UiDialogSet.
	Stops a possible garbage collection timer.
*/
UiOrganizer::~UiOrganizer()
{
    QHash<unsigned int, UiContainerItem>::const_iterator i = m_dialogList.constBegin();
    while (i != m_dialogList.constEnd())
    {
        delete i->container;
        ++i;
    }
    m_dialogList.clear();
    m_objectList.clear();


    if(m_garbageCollectorTimer > 0)
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
        if(i.value().container->getUiWidget() == NULL)
        {
            delete i.value().container;
            i.remove();
        }
    }

    //check m_objectList
    QMutableHashIterator<unsigned int, QWeakPointer<QObject> > j(m_objectList);
    while (j.hasNext())
    {
        j.next();
        if(j.value().isNull())
        {
                j.remove();
        }
    }

    if(m_dialogList.size() == 0 && m_objectList.size() == 0 &&  m_garbageCollectorTimer > 0)
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
    QHash<unsigned int, QWeakPointer<QObject> >::const_iterator i = m_objectList.constBegin();
    while (i != m_objectList.constEnd())
    {
        if(i.value().data() == objPtr)
        {
            return i.key();
        }
        ++i;
    }
    m_objectList.insert(++UiOrganizer::autoIncObjectCounter, QWeakPointer<QObject>(objPtr));

    return UiOrganizer::autoIncObjectCounter;
}

//----------------------------------------------------------------------------------------------------------------------------------
QObject* UiOrganizer::getWeakObjectReference(unsigned int objectID)
{
    QWeakPointer<QObject> temp = m_objectList.value(objectID);
    if(temp.isNull() == false)
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
//ito::RetVal UiOrganizer::scanPlugins(QString pluginPath, QHash<QString, PluginInfo> &pluginInfoList)
//{
////    foreach (const QString &path, pluginPath) 
//    {
//        const QDir dir(pluginPath);
//        const QStringList candidates = dir.entryList(QDir::Files);
//
//        foreach (const QString &plugin, candidates) 
//        {
//            if (!QLibrary::isLibrary(plugin))
//                continue;
//
//            QString loaderPath = pluginPath;
//            loaderPath += QLatin1Char('/');
//            loaderPath += plugin;
//
//            QPluginLoader loader(loaderPath);
//            if (loader.load())
//            {
//                QDesignerCustomWidgetInterface *iface = NULL;
//                // try with a normal plugin, we do not support collections
//                if ((iface = qobject_cast<QDesignerCustomWidgetInterface *>(loader.instance())) && 
//                    ((loader.instance())->inherits("ito::AbstractItomDesignerPlugin")))
//                {
//                    ito::AbstractItomDesignerPlugin *absIDP = (ito::AbstractItomDesignerPlugin *)loader.instance();
//                    pluginInfoList.insert(iface->name(), PluginInfo(absIDP->getPlotDataTypes(), absIDP->getPlotDataFormats(), absIDP->getPlotFeatures()));
//                }
//            }
//            else
//            {
//                QString err = loader.errorString();
//            }
//
//        }
//    }
//
//    return ito::retOk;
//}

//----------------------------------------------------------------------------------------------------------------------------------
UiContainer* UiOrganizer::getUiDialogByHandle(unsigned int uiHandle)
{
    if(m_dialogList.contains(uiHandle))
    {
        return m_dialogList[uiHandle].container;
    }
    else
    {
        return NULL;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::loadPluginWidget(void* algoWidgetFunc, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore)
{
    ito::RetVal retValue = ito::retOk;
    ito::AddInAlgo::t_algoWidget func = reinterpret_cast<ito::AddInAlgo::t_algoWidget>(algoWidgetFunc);
    QWidget *widget = func(paramsMand, paramsOpt, retValue);
    *objectID = 0;

    if(widget == NULL)
    {
        retValue += RetVal(retError, 0, tr("the plugin did not return a valid widget pointer.").toAscii().data());
    }
    else
    {
        retValue += addWidgetToOrganizer(widget, dialogHandle, initSlotCount, NULL, NULL);

        if(retValue.containsError())
        {
            DELETE_AND_SET_NULL(widget);
        }
        else
        {
            *objectID = addObjectToList(widget);
            *className = widget->metaObject()->className();
        }
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::addWidgetToOrganizer(QWidget *widget, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QWidget *parent, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    *initSlotCount = 0;
    *dialogHandle = 0;
    UiContainer::tUiType widgetType = UiContainer::uiTypeUiDialog; //default widget is of type QWidget and should be display within UiDialog
    QString className;
    UiContainer *set = NULL;
    UiContainerItem containerItem;

    if(parent == NULL) //take mainWindow of itom as parent
    {
        parent = qobject_cast<QWidget*>( AppManagement::getMainWindow() );
    }

    
    if(widget == NULL)
    {
        retValue += RetVal(retError, 0, tr("widget is NULL").toAscii().data());
    }
    else
    {
        //auto-check widget-type
        const QMetaObject *metaObject = widget->metaObject();
        while(metaObject != NULL)
        {
            className = metaObject->className();
            if(QString::compare(className, "QMainWindow", Qt::CaseInsensitive) == 0)
            {
                widgetType = UiContainer::uiTypeQMainWindow;
                break;
            }
            else if(QString::compare(className, "QDialog", Qt::CaseInsensitive) == 0)
            {
                widgetType = UiContainer::uiTypeQDialog;
                break;
            }
            else if(QString::compare(className, "QDockWidget", Qt::CaseInsensitive) == 0)
            {
                widgetType = UiContainer::uiTypeQDockWidget;
                break;
            }
            else if(QString::compare(className, "QWidget", Qt::CaseInsensitive) == 0)
            {
                widgetType = UiContainer::uiTypeUiDialog; //default widget is of type QWidget and should be display within UiDialog
                break;
            }
            
            metaObject = metaObject->superClass();
        }


        if(m_garbageCollectorTimer == 0)
        {
            m_garbageCollectorTimer = startTimer(5000);
        }

        switch(widgetType)
        {
            case UiContainer::uiTypeQMainWindow:
            {
                set = new UiContainer(qobject_cast<QMainWindow*>(widget));
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *initSlotCount = widget->metaObject()->methodOffset();
                break;
            }
            case UiContainer::uiTypeQDialog:
            {
                set = new UiContainer(qobject_cast<QDialog*>(widget));
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *initSlotCount = widget->metaObject()->methodOffset();
                break;
            }
            case UiContainer::uiTypeQDockWidget:
            {
                retValue += RetVal(retError, 0, tr("widgets of type QDockWidget are not yet implemented").toAscii().data());
                break;
            }
            default: //widget packed into UiDialog
            {
                QMap<QString, QString> buttons;
                UserUiDialog *dialog = new UserUiDialog(widget,UserUiDialog::bbTypeNo,buttons, retValue, parent);
                set = new UiContainer(dialog);
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;

                *initSlotCount = dialog->metaObject()->methodOffset();
                break;
            }
        }
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getNewPluginWindow(QString pluginName, unsigned int &objectID, QWidget** newWidget, QWidget *parent /*= NULL*/)
{
    RetVal retValue = retOk;
    UiContainer *set = NULL;
    QMainWindow *win = NULL;
    QDialog *dlg = NULL;
    bool found = false;
    int dialogHandle;
    UiContainerItem cItem;


    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if(dwo && dwo->figureClassExists(pluginName) )
    {
        QWidget *mainParent = parent ? parent : qobject_cast<QWidget*>(AppManagement::getMainWindow());
        //mainParent will be set as parent-widget, this however is finally defined by the top-level mode of class AbstractFigure.
        *newWidget = loadDesignerPluginWidget(pluginName,retValue,AbstractFigure::ModeStandaloneWindow,mainParent);
        win = qobject_cast<QMainWindow*>(*newWidget);
        if(win)
        {
            if(win->inherits("ito::AbstractFigure"))
            {
                //((ito::AbstractFigure*)win)->setWindowMode(ito::AbstractFigure::ModeWindow);
            }
            else
            {
                win->setWindowFlags(Qt::Window);
                win->setAttribute(Qt::WA_DeleteOnClose, true);
            }
            found = true;
        }
        
    }
    else
    {
        QString errorMsg = tr("plugin with name '%1' could be found.").arg(pluginName);
        retValue += ito::RetVal(retError, 0, errorMsg.toAscii().data());
    }

    if(!retValue.containsError())
    {
        if(m_garbageCollectorTimer == 0)
        {
            m_garbageCollectorTimer = startTimer(5000);
        }

        if(win)
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

    /*if (!retValue.containsError())
        qobject_cast<QMainWindow*>(*newWidget)->show();*/

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::createNewDialog(QString filename, int uiDescription, StringMap dialogButtons, QSharedPointer<unsigned int> dialogHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    UiContainer *set = NULL;
    UiContainerItem containerItem;
    QMainWindow *win = NULL;
    QDialog *dlg = NULL;
    QWidget *wid = NULL;
    bool found = false;
    bool deleteOnClose = false;
    QString pluginClassName;

    int type, buttonBarType;
    bool childOfMainWindow;
    UiOrganizer::parseUiDescription(uiDescription, &type, &buttonBarType, &childOfMainWindow, &deleteOnClose);

    QMainWindow *mainWin = childOfMainWindow ? qobject_cast<QMainWindow*>( AppManagement::getMainWindow() ) : NULL;

    if(filename.indexOf("itom://") == 0)
    {
        filename = filename.mid(7); //cut "itom://"
        if(filename == "matplotlib" || filename == "MatplotlibFigure" || filename == "MatplotlibPlot")
        {
            pluginClassName = "MatplotlibPlot";
            win = qobject_cast<QMainWindow*>(loadDesignerPluginWidget(pluginClassName,retValue,AbstractFigure::ModeStandaloneWindow, NULL));
            if(win)
            {
                win->setWindowFlags(Qt::Window);
                win->setAttribute(Qt::WA_DeleteOnClose, true);

                //win = new WinMatplotlib();
                found = true;
            }
        }
        else
        {
            QString errorMsg = tr("No internal dialog or window with name '%1' could be found.").arg(filename);
            retValue += ito::RetVal(retError, 0, errorMsg.toAscii().data());
        }

        if(!retValue.containsError())
        {
            if(m_garbageCollectorTimer == 0)
            {
                m_garbageCollectorTimer = startTimer(5000);
            }

            if(win)
            {
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
                set = new UiContainer(dlg);
                *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                containerItem.container = set;
                m_dialogList[*dialogHandle] = containerItem;
                *initSlotCount = dlg->metaObject()->methodOffset();
                *objectID = addObjectToList(dlg);
                *className = dlg->metaObject()->className();
            }
        }
        else
        {
            DELETE_AND_SET_NULL(dlg);
            DELETE_AND_SET_NULL(win);
        }
    }
    else
    {
        QFile file(QDir::cleanPath(filename));
        if (file.exists())
        {
            //set the working directory if QLoader to the directory where the ui-file is stored. Then icons, assigned to the user-interface may be properly loaded, since their path is always saved relatively to the ui-file,too.
            file.open(QFile::ReadOnly);
		    QFileInfo fileinfo(filename);
		    QDir workingDirectory = fileinfo.absoluteDir();
            m_uiLoader.setWorkingDirectory(workingDirectory);
            wid = m_uiLoader.load(&file, mainWin);
            file.close();

            if (wid == NULL)
            {
                retValue += RetVal(retError, 1007, tr("ui-file '%1' could not be correctly parsed.").arg(filename).toAscii().data());
            }
        }
        else
        {
            wid = NULL;
            retValue += RetVal(retError, 1006, tr("filename '%1' does not exist").arg(filename).toAscii().data());
        }

        if (!retValue.containsError())
        {
            if(type == ito::UiOrganizer::typeDialog)
            {
                //load the file and check whether it is inherited from qdialog. If so, directly load it, else stack it into a UserUiDialog
                if (wid->inherits("QDialog"))
                {
                    //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
                    setApiPointersToWidgetAndChildren(wid);

                    if(m_garbageCollectorTimer == 0)
                    {
                        m_garbageCollectorTimer = startTimer(5000);
                    }

                    if(deleteOnClose)
                    {
                        wid->setAttribute(Qt::WA_DeleteOnClose, true);
                    }

                    set = new UiContainer(qobject_cast<QDialog*>(wid));
                    *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                    containerItem.container = set;
                    m_dialogList[*dialogHandle] = containerItem;
                    *initSlotCount = wid->metaObject()->methodOffset();
                    *objectID = addObjectToList(wid);
                    *className = wid->metaObject()->className();
                }
                else
                {
                    //int type, int buttonBarType, StringMap dialogButtons, bool childOfMainWindow
                    UserUiDialog::tButtonBarType bbBarType = UserUiDialog::bbTypeNo;
                    if(buttonBarType == UserUiDialog::bbTypeHorizontal) bbBarType = UserUiDialog::bbTypeHorizontal;
                    if(buttonBarType == UserUiDialog::bbTypeVertical) bbBarType = UserUiDialog::bbTypeVertical;

                    UserUiDialog *dialog = new UserUiDialog(wid, bbBarType, dialogButtons, retValue, mainWin);

                    if(dialog == NULL)
                    {
                        retValue += RetVal(retError, 1020, tr("dialog could not be created").toAscii().data());
						wid->deleteLater();
                    }
                    else if(!retValue.containsError())
                    {
                        //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
                        setApiPointersToWidgetAndChildren(dialog);

                        if(m_garbageCollectorTimer == 0)
                        {
                            m_garbageCollectorTimer = startTimer(5000);
                        }

                        if(deleteOnClose)
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
            else if(type == ito::UiOrganizer::typeMainWindow)
            {
                //check whether any child of dialog is of type AbstractFigure and if so setApiFunctionPointers to it
                setApiPointersToWidgetAndChildren(wid);

                if(m_garbageCollectorTimer == 0)
                {
                    m_garbageCollectorTimer = startTimer(5000);
                }

                win = qobject_cast<QMainWindow*>(wid);
                if(win)
                {
                    if(deleteOnClose)
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
                    wid->setWindowFlags(Qt::Window);

                    if(deleteOnClose)
                    {
                        wid->setAttribute(Qt::WA_DeleteOnClose, true);
                    }

                    set = new UiContainer(wid,UiContainer::uiTypeQMainWindow);
                    *dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
                    containerItem.container = set;
                    m_dialogList[*dialogHandle] = containerItem;
                    *initSlotCount = wid->metaObject()->methodOffset();
                    *objectID = addObjectToList(wid);
                    *className = wid->metaObject()->className();
                }
            }
            else //dock widget
            {
				if (wid->inherits("QDialog"))
				{
					retValue += RetVal(retError, 0, "A widget inherited from QDialog cannot be docked into the main window");
					wid->deleteLater();
					wid = NULL;
				}
				else
				{
					QMainWindow *mainWin = qobject_cast<QMainWindow*>( AppManagement::getMainWindow() );
					if (!mainWin)
					{
						retValue += RetVal(retError, 0, "Main window not available for docking the user interface.");
						wid->deleteLater();
						wid = NULL;
					}
					else
					{
						QDockWidget *dockWidget = new QDockWidget(wid->windowTitle(), mainWin);
						dockWidget->setWidget(wid);
						mainWin->addDockWidget(Qt::TopDockWidgetArea, dockWidget);

						set = new UiContainer(dockWidget);
						*dialogHandle = ++UiOrganizer::autoIncUiDialogCounter;
						containerItem.container = set;
						m_dialogList[*dialogHandle] = containerItem;
						*initSlotCount = wid->metaObject()->methodOffset();
						*objectID = addObjectToList(wid);
						*className = wid->metaObject()->className();
					}

				}
            }
        }

        
    }



    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::setApiPointersToWidgetAndChildren(QWidget *widget)
{
    if(widget)
    {
        if(widget->inherits("ito::AbstractFigure"))
        {
            ((ito::AbstractFigure*)widget)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
            ((ito::AbstractFigure*)widget)->setApiFunctionBasePtr(ITOM_API_FUNCS);

            //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
            //transmitted to the plugin. This assignment cannot be done directly, since 
            //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
            //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
            //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
            //also is necessary if any methods of the plugin are directly called from itom).
            QEvent evt( (QEvent::Type)(QEvent::User+123) );
            QCoreApplication::sendEvent(widget, &evt);
        }

        QObjectList list = widget->children();
        foreach(QObject *obj, list)
        {
            setApiPointersToWidgetAndChildren( qobject_cast<QWidget*>(obj) );
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

    QStringList availableWidgets = m_uiLoader.availableWidgets();

    bool found = false;
    foreach( const QString &name, availableWidgets )
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
            foreach( const QString &name, availableWidgets )
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
        widget = dwo->createWidget(tempClassName,parent,QString(),winMode); //loader.createWidget(className, parent);

        if(widget == NULL)
        {
            widget = m_uiLoader.createWidget(tempClassName, parent);
        }

        if(widget == NULL)
        {
            retValue += RetVal(retError,0,tr("designer plugin widget ('%1') could not be created").arg(tempClassName).toAscii().data());
        }
        else
        {
            setApiPointersToWidgetAndChildren(widget);
        }
    }
    else
    {
        retValue += RetVal::format(retError,0,tr("No designer plugin with className '%s' could be found. Please make sure that this plugin is compiled and the corresponding DLL and header files are in the designer folder").toAscii().data(),className.toAscii().data());
    }
    return widget;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::deleteDialog(unsigned int handle, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    UiContainer *ptr = getUiDialogByHandle(handle);
    if(ptr)
    {
        delete ptr;
        m_dialogList.remove(handle);
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showDialog(unsigned int handle, int modalLevel, QSharedPointer<int> retCodeIfModal, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    UiContainer *ptr = getUiDialogByHandle(handle);
    if(ptr)
    {
        switch(ptr->getType())
        {
            case UiContainer::uiTypeUiDialog:
            case UiContainer::uiTypeQDialog:
            {
                QDialog *dlg = qobject_cast<QDialog*>(ptr->getUiWidget());
                if(dlg)
                {
                    if(modalLevel == 1) //blocking modal
                    {
                        dlg->setModal(true);
                        *retCodeIfModal = dlg->exec();
                    }
                    else if(modalLevel == 0) //non-modal
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
                if(wid)
                {
                    if(modalLevel == 0) //non-modal
                    {
                        wid->setWindowModality( Qt::NonModal );
                        wid->show();
                    }
                    else if(modalLevel == 1) //blocking-modal
                    {
                        wid->setWindowModality( Qt::ApplicationModal );
                        wid->show();
                        //wait until window is hidden again
                        while(wid->isVisible())
                        {
                            QCoreApplication::processEvents();
                        }
                    }
                    else //non-blocking modal
                    {
                        wid->setWindowModality( Qt::ApplicationModal );
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
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::hideDialog(unsigned int handle, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);

    if(ptr)
    {
        ptr->getUiWidget()->hide();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;

}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getDockedStatus(unsigned int uiHandle, QSharedPointer<bool> docked, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if(widget)
    {
        if(widget->inherits("ito::AbstractDockWidget"))
        {
            AbstractDockWidget *adw = (AbstractDockWidget*)widget;
            *docked = adw->docked();
        }
		else if(widget->inherits("QDockWidget"))
		{
			QDockWidget *dw = (QDockWidget*)widget;
			*docked = !dw->isFloating();
		}
        else
        {
            retValue += RetVal(retError, 0, tr("dialog cannot be docked").toAscii().data());
        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setDockedStatus(unsigned int uiHandle, bool docked, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);
    QWidget *widget = ptr ? ptr->getUiWidget() : NULL;

    if(widget)
    {
        if(widget->inherits("ito::AbstractDockWidget"))
        {
            AbstractDockWidget *adw = (AbstractDockWidget*)widget;

            if(docked)
            {
                adw->dockWidget();
            }
            else
            {
                adw->undockWidget();
            }
        }
		else if(widget->inherits("QDockWidget"))
		{
			QDockWidget *dw = (QDockWidget*)widget;
			dw->setFloating(!docked);
		}
        else
        {
            retValue += RetVal(retError, 0, tr("dialog cannot be docked or undocked").toAscii().data());
        }
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setAttribute(unsigned int handle, Qt::WidgetAttribute attribute, bool on, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);

    if(ptr)
    {
        ptr->getUiWidget()->setAttribute(attribute,on);
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::isVisible(unsigned int handle, QSharedPointer<bool> visible, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);
    UiContainer *ptr = getUiDialogByHandle(handle);

    if(ptr)
    {
        *visible = ptr->getUiWidget()->isVisible();
    }
    else
    {
        retValue += RetVal(retError, 1001, tr("dialog handle does not exist").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetDouble(QString title, QString label, double defaultValue, QSharedPointer<bool> ok, QSharedPointer<double> value, double min, double max, int decimals, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QMainWindow *mainWin = qobject_cast<QMainWindow*>( AppManagement::getMainWindow() );

    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getDouble(mainWin, title, label, defaultValue, min, max, decimals, &tempOk);

    *ok = tempOk;

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetInt(QString title, QString label, int defaultValue, QSharedPointer<bool> ok, QSharedPointer<int> value, int min, int max, int step, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QMainWindow *mainWin = qobject_cast<QMainWindow*>( AppManagement::getMainWindow() );

    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getInt(mainWin, title, label, defaultValue, min, max, step, &tempOk);

    *ok = tempOk;

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetItem(QString title, QString label, QStringList stringList, QSharedPointer<bool> ok, QSharedPointer<QString> value, int currentIndex, bool editable, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QMainWindow *mainWin = qobject_cast<QMainWindow*>( AppManagement::getMainWindow() );

    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getItem(mainWin, title, label, stringList, currentIndex, editable, &tempOk);

    *ok = tempOk;

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showInputDialogGetText(QString title, QString label, QString defaultString, QSharedPointer<bool> ok, QSharedPointer<QString> value, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QMainWindow *mainWin = qobject_cast<QMainWindow*>( AppManagement::getMainWindow() );

    bool tempOk = false;
    *ok = false;

    *value = QInputDialog::getText(mainWin, title, label, QLineEdit::Normal, defaultString, &tempOk);

    *ok = tempOk;

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showMessageBox(unsigned int uiHandle, int type, QString title, QString text, int buttons, int defaultButton, QSharedPointer<int> retButton, QSharedPointer<QString> retButtonText, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = RetVal(retOk);

    QWidget *parent = NULL;
    if(uiHandle > 0)
    {
        UiContainer *ptr = getUiDialogByHandle(uiHandle);
        if(ptr) parent = ptr->getUiWidget();
    }
    if(parent == NULL)
    {
        parent = qobject_cast<QWidget*>( AppManagement::getMainWindow() );
    }

    QMessageBox::StandardButton stdBtn;
    QMetaObject metaObject = QMessageBox::staticMetaObject;
    QMetaEnum metaEnum = metaObject.enumerator( metaObject.indexOfEnumerator( "StandardButtons" ));
    QMessageBox::StandardButton stdDefaultButton;

    //check defaultButton:
    if(metaEnum.valueToKey(defaultButton) == NULL)
    {
        retValue += RetVal(retError, 1001, tr("defaultButton must be within enum QMessageBox::StandardButton").toAscii().data());
    }
    else
    {
        stdDefaultButton = (QMessageBox::StandardButton)defaultButton;
    }

    QByteArray ba = metaEnum.valueToKeys(buttons);
    QMessageBox::StandardButtons stdButtons = static_cast<QMessageBox::StandardButtons>(metaEnum.keysToValue(ba.data()));

    if(stdButtons == -1)
    {
        retValue += RetVal(retError, 1001, tr("buttons must be within enum QMessageBox::StandardButtons").toAscii().data());
    }

    if(defaultButton != 0 && stdButtons.testFlag(stdDefaultButton) == false)
    {
        retValue += RetVal(retError, 1001, tr("defaultButton must appear in buttons, too.").toAscii().data());
    }

    if(!retValue.containsWarningOrError())
    {

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

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileDialogExistingDir(unsigned int uiHandle, QString caption, QSharedPointer<QString> directory, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if(uiHandle > 0)
    {
        UiContainer *ptr = getUiDialogByHandle(uiHandle);
        if(ptr) parent = ptr->getUiWidget();
    }
    if(parent == NULL)
    {
        parent = qobject_cast<QWidget*>( AppManagement::getMainWindow() );
    }

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
    QString result = QFileDialog::getExistingDirectory(parent, caption, *directory, opt);
    *directory = result;

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileOpenDialog(unsigned int uiHandle, QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if(uiHandle > 0)
    {
        UiContainer *ptr = getUiDialogByHandle(uiHandle);
        if(ptr) parent = ptr->getUiWidget();
    }
    if(parent == NULL)
    {
        parent = qobject_cast<QWidget*>( AppManagement::getMainWindow() );
    }

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
    QStringList filters = filter.split(";;");
    QString *selectedFilter = NULL;
    if(selectedFilterIndex >= 0 && selectedFilterIndex < filters.size())
    {
        selectedFilter = new QString(filters[selectedFilterIndex]);
    }

    QString result = QFileDialog::getOpenFileName(parent, caption, directory, filter, selectedFilter, opt);
    *file = result;

    DELETE_AND_SET_NULL(selectedFilter);

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::showFileSaveDialog(unsigned int uiHandle, QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex, int options, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    QWidget *parent = NULL;
    if(uiHandle > 0)
    {
        UiContainer *ptr = getUiDialogByHandle(uiHandle);
        if(ptr) parent = ptr->getUiWidget();
    }
    if(parent == NULL)
    {
        parent = qobject_cast<QWidget*>( AppManagement::getMainWindow() );
    }

    QFileDialog::Options opt = 0;
    opt = (~opt) & options;
    QStringList filters = filter.split(";;");
    QString *selectedFilter = NULL;
    if(selectedFilterIndex >= 0 && selectedFilterIndex < filters.size())
    {
        selectedFilter = new QString(filters[selectedFilterIndex]);
    }

    QString result = QFileDialog::getSaveFileName(parent, caption, directory, filter, selectedFilter, opt);
    *file = result;

    DELETE_AND_SET_NULL(selectedFilter);

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getPropertyInfos(unsigned int objectID, QSharedPointer<QVariantMap> retPropertyMap, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    *retPropertyMap = QVariantMap();

    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
        const QMetaObject *mo = obj->metaObject();
        QMetaProperty prop;
        int flags;

        for(int i = 0 ; i < mo->propertyCount() ; i++)
        {
            prop = mo->property(i);
            flags = 0;
            if(prop.isValid()) flags |= UiOrganizer::propValid;
            if(prop.isReadable()) flags |= UiOrganizer::propReadable;
            if(prop.isWritable()) flags |= UiOrganizer::propWritable;
            if(prop.isResettable()) flags |= UiOrganizer::propResettable;
            if(prop.isFinal()) flags |= UiOrganizer::propFinal;
            if(prop.isConstant()) flags |= UiOrganizer::propConstant;
            (*retPropertyMap)[prop.name()] = flags;
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::readProperties(unsigned int handle, QString widgetName, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore)
{
    unsigned int objectHandle = 0;
    UiContainer* set = getUiDialogByHandle(handle);
    if(set)
    {
        QWidget* widget = set->getUiWidget()->findChild<QWidget*>(widgetName);
        if(widget)
        {
            objectHandle = this->addObjectToList(widget);

        }
    }

    return readProperties(objectHandle, properties, semaphore);
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::writeProperties(unsigned int handle, QString widgetName, QVariantMap properties, ItomSharedSemaphore *semaphore)
{
    unsigned int objectHandle = 0;
    UiContainer* set = getUiDialogByHandle(handle);
    if(set)
    {
        QWidget* widget = set->getUiWidget()->findChild<QWidget*>(widgetName);
        if(widget)
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


    if(obj)
    {
        QStringList errString;

        QMap<QString, QVariant>::iterator i = properties->begin();
        while (i !=  properties->end())
        {
            i.value() = obj->property(i.key().toAscii().data());
            if(!i.value().isValid())
            {
                QObject *newObj = NULL;
                QMetaProperty prop = m_widgetWrapper->fakeProperty(obj, i.key(), &newObj);
                if(prop.isValid() == false)
                {
                    errString.append( tr("property '%1' does not exist").arg(i.key()) );
                }
                else
                {
                    i.value() = prop.read(newObj);
                    if(!i.value().isValid())
                    {
                        errString.append( tr("property '%1' could not be read").arg(i.key()) );
                    }
                }
            }
            ++i;
        }

        if (errString.count() > 0)
        {
            retValue += RetVal(retError, errorObjPropRead, errString.join("\n").toAscii().data());
    }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::writeProperties(unsigned int objectID, QVariantMap properties, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
        QStringList errString;
        const QMetaObject *mo = obj->metaObject();
        QMetaProperty prop;
        int index;
        QMap<QString, QVariant>::const_iterator i = properties.constBegin();
        while (i !=  properties.constEnd())
        {
            index = mo->indexOfProperty( i.key().toAscii().data() );
            if(index == -1)
            {
                QObject *newObj = NULL;
                prop = m_widgetWrapper->fakeProperty(obj, i.key(), &newObj);
                if(prop.isValid() == false)
                {
                    errString.append( tr("property '%1' does not exist").arg(i.key()) );
                }
                else
                {
					//check whether types need to be casted
					//e.g. QVariantList can sometimes be casted to QPointF...
                    RetVal tempRet;
                    QVariant item;

                    if (prop.isEnumType())
                    {
                        item = PythonQtConversion::QVariantToEnumCast(i.value(), prop.enumerator(), tempRet);
                    }
                    else
                    {
				        item = PythonQtConversion::QVariantCast(i.value(), prop.type(), tempRet);
                    }

                    if(tempRet.containsError())
                    {
                        retValue += tempRet;
                    }
                    else if(prop.write(obj, item) == false)
                    {
                        errString.append( tr("property '%1' could not be written").arg(i.key()) );
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
                    item = PythonQtConversion::QVariantToEnumCast(i.value(), prop.enumerator(), tempRet);
                }
                else
                {
				    item = PythonQtConversion::QVariantCast(i.value(), prop.type(), tempRet);
                }

                if(tempRet.containsError())
                {
                    retValue += tempRet;
                }
                else if(prop.write(obj, item) == false)
                {
                    errString.append( tr("property '%1' could not be written").arg(i.key()) );
            }
                
            }
            ++i;
        }

        if (errString.count() > 0)
        {
            retValue += RetVal(retError, errorObjPropRead, errString.join("\n").toAscii().data());
    }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
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
    if(widget)
    {
        if(attributeNumber < 0 || attributeNumber >= Qt::WA_AttributeCount)
        {
            retval += RetVal(retError,0,tr("The attribute number is out of range.").toAscii().data() );
        }
        else
        {
            *value = widget->testAttribute( (Qt::WidgetAttribute)attributeNumber);
        }
    }
   else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setAttribute(unsigned int objectID, int attributeNumber, bool value, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if(widget)
    {
        if(attributeNumber < 0 || attributeNumber >= Qt::WA_AttributeCount)
        {
            retval += RetVal(retError,0,tr("The attribute number is out of range.").toAscii().data() );
        }
        else
        {
            widget->setAttribute( (Qt::WidgetAttribute)attributeNumber, value );
        }
    }
    else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
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
    if(widget)
    {
        *flags = widget->windowFlags();
    }
    else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::setWindowFlags(unsigned int objectID, int flags, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if(widget)
    {
        widget->setWindowFlags( Qt::WindowFlags(flags) );
    }
    else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
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

    if(obj)
    {
        const QMetaObject *mo = obj->metaObject();
        *classInfoCount = mo->classInfoCount();
        *enumeratorCount = mo->enumeratorCount();
        *methodCount = mo->methodCount();
        *propertyCount = mo->propertyCount();
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectInfo(unsigned int objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
        *objectName = obj->objectName().toAscii();
        *widgetClassName = obj->metaObject()->className();
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object ID is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject(unsigned int uiHandle, QString objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    UiContainer *ptr = getUiDialogByHandle(uiHandle);

    if(ptr)
    {
        if(objectName != "")
        {
            QObject* obj = ptr->getUiWidget()->findChild<QObject*>(objectName);
            if(obj)
            {
                *objectID = addObjectToList(obj);
            }
            else
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
            }
        }
        else //return reference to dialog or windows itself
        {
            QWidget* obj = ptr->getUiWidget();
            if(obj)
            {
                *objectID = addObjectToList(obj);
            }
            else
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("could not get reference to main dialog or window").toAscii().data());
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("uiHandle is invalid").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject2(unsigned int parentObjectID, QString objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    if(m_objectList.contains(parentObjectID))
    {
        QObject* ptr = m_objectList[parentObjectID].data();

        if(ptr)
        {
            if(objectName != "")
            {
                QObject* obj = ptr->findChild<QObject*>(objectName);
                if(obj)
                {
                    *objectID = addObjectToList(obj);
                }
                else
                {
                    retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
                }
            }
            else //return reference to dialog or windows itself
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("no object name given.").toAscii().data());
            }
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is invalid.").toAscii().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is unknown.").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getChildObject3(unsigned int parentObjectID, QString objectName, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);

    if(m_objectList.contains(parentObjectID))
    {
        QObject* ptr = m_objectList[parentObjectID].data();

        if(ptr)
        {
            if(objectName != "")
            {
                QObject* obj = ptr->findChild<QObject*>(objectName);
                if(obj)
                {
                    *objectID = addObjectToList(obj);
                    *widgetClassName = obj->metaObject()->className();
                }
                else
                {
                    retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
                }
            }
            else //return reference to dialog or windows itself
            {
                retValue += RetVal(retError, errorObjDoesNotExist, tr("no object name given.").toAscii().data());
            }
        }
        else
        {
            retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is invalid.").toAscii().data());
        }
    }
    else
    {
        retValue += RetVal(retError, errorUiHandleInvalid, tr("The object ID of the parent widget is unknown.").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getSignalIndex(unsigned int objectID, QString signalSignature, QSharedPointer<int> signalIndex, QSharedPointer<QObject*> objPtr, QSharedPointer<IntList> argTypes, ItomSharedSemaphore *semaphore)
{
    *signalIndex = -1;
    argTypes->clear();
    RetVal retValue(retOk);
    int tempType;

    QObject *obj = getWeakObjectReference(objectID);
    *objPtr = obj;

    if(obj)
    {
        const QMetaObject *mo = obj->metaObject();
        *signalIndex = mo->indexOfSignal( QMetaObject::normalizedSignature( signalSignature.toAscii().data() ) );

        QMetaMethod metaMethod = mo->method(*signalIndex);
        QList<QByteArray> names = metaMethod.parameterTypes();
        foreach (const QByteArray& name, names)
        {
            tempType =QMetaType::type( name.constData() );
            if(tempType > 0)
            {
                argTypes->append(tempType);
            }
            else
            {
                QString msg = tr("parameter type %1 is unknown").arg(name.constData());
                retValue += RetVal(retError, errorUnregisteredType, msg.toAscii().data());
                *signalIndex = -1;
                break;
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::connectWithKeyboardInterrupt(unsigned int objectID, QString signalSignature, ItomSharedSemaphore *semaphore)
{
    int signalIndex = -1;
    RetVal retValue(retOk);

    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
        const QMetaObject *mo = obj->metaObject();
        signalIndex = mo->indexOfSignal( QMetaObject::normalizedSignature( signalSignature.toAscii().data() ) );

        if(signalIndex < 0)
        {
            retValue += RetVal(retError, errorSignalDoesNotExist, tr("signal does not exist").toAscii().data());
        }
        else
        {
            if(!QMetaObject::connect(obj, signalIndex, this, this->metaObject()->indexOfSlot( QMetaObject::normalizedSignature("pythonKeyboardInterrupt(bool)"))))
            {
                retValue += RetVal(retError, errorConnectionError, tr("signal could not be connected to slot throwing a python keyboard interrupt.").toAscii().data());
            }
        }
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::callSlotOrMethod(bool slotNotMethod, unsigned int objectID, int slotOrMethodIndex, QSharedPointer<FctCallParamContainer> args, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
        //TODO: parse parameters and check whether there is a type 'ito::PythonQObjectMarshal':
        // if so, get object from objectID, destroy the arg, replace it by QObject*-type and give the object-pointer, casted to void*.
        
        bool success;
        if(slotNotMethod)
        {
            success = obj->qt_metacall(QMetaObject::InvokeMetaMethod, slotOrMethodIndex, args->args());
        }
        else
        {
            success = m_widgetWrapper->call(obj, slotOrMethodIndex, args->args());
        }

        //check if arguments have to be marshalled (e.g. QObject* must be transformed to objectID before passed to python in other thread)
        if(args->getRetType() == QMetaType::type("ito::PythonQObjectMarshal"))
        {
            //add m_object to weakObject-List and pass its ID to python. TODO: right now, we do not check if the object is a child of obj
            ito::PythonQObjectMarshal* m = (ito::PythonQObjectMarshal*)args->args()[0];
            if(m->m_object)
            {
                m->m_objectID = addObjectToList((QObject*)(m->m_object));
                m->m_object = NULL;
            }
        }

        //if return value is set in qt_metacall, this is available in args->args()[0].
        if(success == false)
        {
            retValue += RetVal(retError,errorSlotDoesNotExist, tr("slot could not be found").toAscii().data());
        }

    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getMethodDescriptions(unsigned int objectID, QSharedPointer<MethodDescriptionList> methodList, ItomSharedSemaphore *semaphore)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);
    methodList->clear();

    if(obj)
    {
        const QMetaObject *mo = obj->metaObject();
        QMetaMethod metaMethod;
        QList<QByteArray> paramTypes;
        bool ok = false;
        for(int i=0;i<mo->methodCount();i++)
        {
            metaMethod = mo->method(i);
            //qDebug() << metaMethod.signature();
            ok = true;
            if(metaMethod.access() == QMetaMethod::Public && (metaMethod.methodType() == QMetaMethod::Slot || metaMethod.methodType() == QMetaMethod::Method))
            {
                //check if args can be interpreted by QMetaType:
                if( strcmp(metaMethod.typeName(), "") != 0)
                {
                    if(QMetaType::type( metaMethod.typeName() ) == 0) ok = false;
                }
                if(ok)
                {
                    paramTypes = metaMethod.parameterTypes();
                    for(int j=0;j<paramTypes.size();j++)
                    {
                        if(QMetaType::type( paramTypes[j].data() ) == 0)
                        {
                            ok = false;
                            break;
                        }
                    }
                }
                if(ok)
                {
                    methodList->append( MethodDescription( metaMethod ) );
                }
            }
        }

        methodList->append( m_widgetWrapper->getMethodList(obj) );
    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("object name is not available").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
void UiOrganizer::pythonKeyboardInterrupt(bool /*checked*/)
{
    PyErr_SetInterrupt();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getObjectInfo(unsigned int objectID, int type, QSharedPointer<QVariantMap> infoMap, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retValue(retOk);
    QObject *obj = getWeakObjectReference(objectID);

    if(obj)
    {
		QStringList classInfo;
		QStringList properties;
		QStringList signal;
		QStringList slot;
		QString className;

        const QMetaObject *mo = obj->metaObject();
		className = mo->className();

		while (mo != NULL)
		{
			if ( QString(mo->className()).startsWith("Q") && (type & infoShowAllInheritance) != infoShowAllInheritance )
			{
				break;
			}

			for(int i = mo->classInfoCount() - 1; i >= 0; i--)
            {
                QMetaClassInfo ci = mo->classInfo(i);
                if(i >= mo->classInfoOffset())
                {
                    classInfo.append( QString("%1 : %2").arg( ci.name() ).arg( ci.value() ) );
                }
            }

			for(int i = mo->propertyCount() - 1; i >= 0; i--)
            {
                QMetaProperty prop = mo->property(i);
                if(i >= mo->propertyOffset())
				{
					properties.append( QString("%1 : %2").arg( prop.name() ).arg( prop.typeName() ) );
                }
            }

			for(int i = mo->methodCount() - 1; i >= 0; i--)
            {
                QMetaMethod meth = mo->method(i);

                if(meth.methodType() == QMetaMethod::Signal)
                {
                    if(i >= mo->methodOffset())
                    {
                        signal.append( meth.signature() );
					}
                }
				else if(meth.methodType() == QMetaMethod::Slot && meth.access() == QMetaMethod::Public)
				{
					if(i >= mo->methodOffset())
                    {
                        slot.append( meth.signature() );
					}
				}
            }
			

			if (type & infoShowItomInheritance)
			{
				mo = mo->superClass();
			}
			else
			{
				mo = NULL;
			}
		}

        std::cout << "WIDGET '" << className.toAscii().data() << "'\n--------------------------\n\n" << std::endl;

        if(classInfo.size() > 0)
        {
            std::cout << "Class Info\n---------------\n";

			foreach(const QString &i, classInfo)
			{
				std::cout << " " << i.toAscii().data() << "\n";
			}

            std::cout << "\n" << std::endl;
        }

        if(properties.size() > 0)
        {
            std::cout << "Properties\n---------------\n";

			foreach(const QString &i, properties)
			{
				std::cout << " " << i.toAscii().data() << "\n";
			}

            std::cout << "\n" << std::endl;
        }

        if(signal.size() > 0)
        {
            std::cout << "Signals\n---------------\n";

			foreach(const QString &i, signal)
			{
				std::cout << " " << i.toAscii().data() << "\n";
			}

            std::cout << "\n" << std::endl;
        }
		if(slot.size() > 0)
        {
            std::cout << "Slots\n---------------\n";

			foreach(const QString &i, slot)
			{
				std::cout << " " << i.toAscii().data() << "\n";
			}

            std::cout << "\n" << std::endl;
        }

    }
    else
    {
        retValue += RetVal(retError, errorObjDoesNotExist, tr("There exists no object with the given id.").toAscii().data());
    }

    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
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
//    if(dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, "DesignerWidgetOrganizer is not available");
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
//        if(!retval.containsError())
//        {
//            QObject *window;
//            retval += getNewPluginWindow(plotClassName, *plotHandle, &window);
//
//            if(!retval.containsError() && window)
//            {
//                ito::AbstractDObjFigure *mainWin;
//
//                if (window->inherits("ito::AbstractDObjFigure"))
//                    mainWin = (ito::AbstractDObjFigure*)(window);
//                else
//                    mainWin = NULL;
//
//                if(mainWin)
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
//                    retval += ito::RetVal(ito::retError,0, tr("Plot window could not be loaded").toAscii().data());
//                }
//            }
//        }
//    }
//
//    if(semaphore)
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
//    if(!retVal.containsError() && (*window))
//    {
//        if ((*window)->inherits("ito::AbstractDObjFigure"))
//            mainWin = (ito::AbstractDObjFigure*)(*window);
//        else
//            mainWin = NULL;
//
//        if(mainWin)
//        {
//            mainWin->setWindowMode(ito::AbstractFigure::ModeWindow);
//            //mainWin->setWindowFlags(Qt::Window);
//            //mainWin->setAttribute(Qt::WA_DeleteOnClose, true);
//            mainWin->show();
//        }
//        else
//        {
//            retVal += ito::RetVal(ito::retError,0, tr("Plot window could not be loaded").toAscii().data());
//        }
//    }
//
//    mainWin->setCamera(QPointer<ito::AddInDataIO>(dataIO));
//    /*mainWin->setProperty("liveSource", v);
//    
//    if(dataIO)
//    {
//        AddInManager *aim = AddInManager::getInstance();
//        retVal += aim->incRef((ito::AddInBase*)dataIO);
//        mainWin->getInputParam("liveData")->setVal<void*>(dataIO);
//    }
//
//    QMetaObject::invokeMethod(dataIO, "startDeviceAndRegisterListener", Q_ARG(QObject*, *window), Q_ARG(ItomSharedSemaphore*, NULL));*/
//
//    if(semaphore)
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
//    if(dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, "DesignerWidgetOrganizer is not available");
//    }
//    else
//    {
//        plotClassName = dwo->getFigureClass("DObjLiveImage", plotClassName, retval);
//
//        if(!retval.containsError())
//        {
//            QObject *window = NULL;
//            retval = liveData(dataIO, plotClassName, &window, NULL);
//        //    ito::RetVal retval = liveData(dataIO, "itom2DGVFigure", &window, NULL);
//
//            if (dataIO && !retval.containsError() )
//            {
//                long maxInt = 1;
//                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
//                ito::Param param = dataIO->getParamRec("bpp");
//                if ( param.getName() != NULL)   // Parameter is defined
//                {
//                    QSharedPointer<ito::Param> qsParam(new ito::Param(param));
//                    QMetaObject::invokeMethod(dataIO, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore *, locker.getSemaphore()));
//                    while (!locker.getSemaphore()->wait(PLUGINWAIT))
//                    {
//                        if (!dataIO->isAlive())
//                        {
//                            break;
//                        }
//                    }
//                    if(!locker.getSemaphore()->returnValue.containsError())
//                    {
//                        int bpp = (*qsParam).getVal<int>();
//
//                        if(bpp < 17)    // scale only for int8, uint8, int16, uint16
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
//    if(semaphore)
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
//    if(dwo == NULL)
//    {
//        retval += ito::RetVal(ito::retError, 0, "DesignerWidgetOrganizer is not available");
//    }
//    else
//    {
//        plotClassName = dwo->getFigureClass("DObjLiveLine", plotClassName, retval);
//
//        if(!retval.containsError())
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
//                if ( param.getName() != NULL)   // Parameter is defined
//                {
//                    QSharedPointer<ito::Param> qsParam(new ito::Param(param));
//                    QMetaObject::invokeMethod(dataIO, "getParam", Q_ARG(QSharedPointer<ito::Param>, qsParam), Q_ARG(ItomSharedSemaphore *, locker.getSemaphore()));
//                    while (!locker.getSemaphore()->wait(PLUGINWAIT))
//                    {
//                        if (!dataIO->isAlive())
//                        {
//                            break;
//                        }
//                    }
//                    if(!locker.getSemaphore()->returnValue.containsError())
//                    {
//                        int bpp = (*qsParam).getVal<int>();
//
//                        if(bpp < 16)    // scale only for int8, uint8, int16, uint16
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
//    if(semaphore)
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
RetVal UiOrganizer::figurePlot(QSharedPointer<ito::DataObject> dataObj, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if(*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>() );
        QSharedPointer<unsigned int> initSlotCount(new unsigned int);
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure( guardedFigHandle, initSlotCount, figObjectID, row, col, NULL);
        if(!retval.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
        {
            *figHandle = *(*guardedFigHandle);
            fig = qobject_cast<FigureWidget*>( m_dialogList[*figHandle].container->getUiWidget() );
            fig->setFigHandle( *guardedFigHandle );
        }
    }

    if(!retval.containsError())
    {

        if(m_dialogList.contains(*figHandle))
        {
            fig = qobject_cast<FigureWidget*>( m_dialogList[*figHandle].container->getUiWidget() );
            if(fig)
            {
                QWidget *destWidget;
                retval += fig->plot(dataObj, areaRow, areaCol, className, &destWidget);
                
                *objectID = addObjectToList(destWidget);
            }
            else
            {
                retval += RetVal::format(retError,0,"figHandle %i is not handle for a figure window.",*figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError,0,"figHandle %i not available.",*figHandle);
        }
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureLiveImage(AddInDataIO* dataIO, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if(*figHandle == 0)
    {
        //create new figure and gives it its own reference, since no instance is keeping track of it
        QSharedPointer< QSharedPointer<unsigned int> > guardedFigHandle(new QSharedPointer<unsigned int>() );
        QSharedPointer<unsigned int> initSlotCount(new unsigned int);
        QSharedPointer<unsigned int> figObjectID(new unsigned int);
        QSharedPointer<int> row(new int);
        *row = areaRow + 1;
        QSharedPointer<int> col(new int);
        *col = areaCol + 1;
        retval += createFigure( guardedFigHandle, initSlotCount, figObjectID, row, col, NULL);
        if(!retval.containsError()) //if the figure window is created by this method, it is assumed, that no figure-instance keeps track of this figure, therefore its guardedFigHandle is given to the figure itsself
        {
            *figHandle = *(*guardedFigHandle);
            fig = qobject_cast<FigureWidget*>( m_dialogList[*figHandle].container->getUiWidget() );
            fig->setFigHandle( *guardedFigHandle );
        }
    }

    if(!retval.containsError())
    {

        if(m_dialogList.contains(*figHandle))
        {
            fig = qobject_cast<FigureWidget*>( m_dialogList[*figHandle].container->getUiWidget() );
            if(fig)
            {
                QWidget *destWidget;
                retval += fig->liveImage(dataIO, areaRow, areaCol, className, &destWidget);

                *objectID = addObjectToList(destWidget);
            }
            else
            {
                retval += RetVal::format(retError,0,"figHandle %i is not handle for a figure window.",*figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError,0,"figHandle %i not available.",*figHandle);
        }
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::createFigure(QSharedPointer< QSharedPointer<unsigned int> > guardedFigureHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<int> rows, QSharedPointer<int> cols, ItomSharedSemaphore *semaphore)
{
    RetVal retValue = retOk;
    const FigureWidget *fig = NULL;
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
    if(!guardedFigureHandle.isNull() && !(*guardedFigureHandle).isNull() )
    {
        h = *(*guardedFigureHandle);

        if(m_dialogList.contains(h))
        {
            containerItem = m_dialogList[h];
            if(containerItem.container->getType() == UiContainer::uiTypeFigure)
            {
                fig = qobject_cast<const FigureWidget*>(containerItem.container->getUiWidget());
                if(fig)
                {
                    *rows = fig->rows();
                    *cols = fig->cols();
                    *guardedFigureHandle = (containerItem.guardedHandle).toStrongRef();
                    *initSlotCount = fig->metaObject()->methodOffset();
                    *objectID = addObjectToList(const_cast<FigureWidget*>(fig));
                    found = true;
                }
                else
                {
                    retValue += ito::RetVal(ito::retError,0,"figure window is not available any more");
                }
            }
            else
            {
                retValue += ito::RetVal::format(ito::retError,0,"handle '%i' is no figure.",h);
            }
        }
        else
        {
            if(h == 0)
            {
                retValue += ito::RetVal(ito::retError,0,"handle '0' cannot be assigned.");
            }
            else
            {
                forcedHandle = h;
            }
        }
    }

    if(!found)
    {
        if(m_garbageCollectorTimer == 0)
        {
            m_garbageCollectorTimer = startTimer(5000);
        }

        FigureWidget *fig2 = new FigureWidget("Figure", false, true, *rows, *cols, NULL);
        //fig2->setAttribute( Qt::WA_DeleteOnClose ); //always delete figure window, if user closes it
        //QObject::connect(fig2,SIGNAL(destroyed(QObject*)),this,SLOT(figureDestroyed(QObject*)));

        mainWin = qobject_cast<MainWindow*>(AppManagement::getMainWindow());
        if(mainWin)
        {
            mainWin->addAbstractDock(fig2, Qt::TopDockWidgetArea);
        }

        set = new UiContainer(fig2);
        unsigned int *handle = new unsigned int;
        if(forcedHandle == 0)
        {
            *handle = ++UiOrganizer::autoIncUiDialogCounter;
        }
        else
        {
            *handle = forcedHandle; //does not exist any more!
            if(UiOrganizer::autoIncObjectCounter < forcedHandle)
            {
                UiOrganizer::autoIncObjectCounter = forcedHandle; //the next figure must always get a really new and unique handle number
            }
        }
        *guardedFigureHandle = QSharedPointer<unsigned int>(handle, threadSafeDeleteUi);
        *initSlotCount = fig2->metaObject()->methodOffset();
        *objectID = addObjectToList(fig2);
        containerItem.container = set;
        containerItem.guardedHandle = (*guardedFigureHandle).toWeakRef();
        m_dialogList[*handle] = containerItem;
    }
    
    if(semaphore)
    {
        semaphore->returnValue = retValue;
        semaphore->release();
                semaphore->deleteSemaphore();
        semaphore = NULL;
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::getSubplot(QSharedPointer<unsigned int> figHandle, unsigned int subplotIndex, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;
    QSharedPointer<unsigned int> empty;

    if(m_dialogList.contains(*figHandle))
    {
        fig = qobject_cast<FigureWidget*>( m_dialogList[*figHandle].container->getUiWidget() );
        if(fig)
        {
            QObject* obj = fig->getSubplot(subplotIndex);
            if(obj)
            {
                *objectID = addObjectToList(obj);
                *widgetClassName = obj->metaObject()->className();
                *objectName = obj->objectName().toAscii();
            }
            else
            {
                retval += RetVal::format(retError, errorObjDoesNotExist, tr("subplot at indexed position %i is not available").toAscii().data(), subplotIndex);
            }
        }
        else
        {
            retval += RetVal::format(retError,0,"figHandle %i is not a handle for a figure window.",*figHandle);
        }
    }
    else
    {
        retval += RetVal::format(retError,0,"figHandle %i not available.",*figHandle);
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureRemoveGuardedHandle(unsigned int figHandle, ItomSharedSemaphore *semaphore /*= NULL*/)
{
    RetVal retval;
    ItomSharedSemaphoreLocker locker(semaphore);
    FigureWidget *fig = NULL;

    if(m_dialogList.contains(figHandle))
    {
        fig = qobject_cast<FigureWidget*>( m_dialogList[figHandle].container->getUiWidget() );
        if(fig)
        {
            QSharedPointer<unsigned int> empty;
            fig->setFigHandle( empty );
        }
        else
        {
            retval += RetVal::format(retError,0,"figHandle %i is not handle for a figure window.",figHandle);
        }
    }
    else
    {
        retval += RetVal::format(retError,0,"figHandle %i not available.",figHandle);
    }

    if(semaphore)
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

    if(figHandle > 0)
    {
        if(m_dialogList.contains(figHandle))
        {
            fig = qobject_cast<FigureWidget*>( m_dialogList[figHandle].container->getUiWidget() );
            if(fig)
            {
                fig->setFigHandle( empty );
            }
            else
            {
                retval += RetVal::format(retError,0,"figHandle %i is not a handle for a figure window.",figHandle);
            }
        }
        else
        {
            retval += RetVal::format(retError,0,"figHandle %i not available.",figHandle);
        }
    }
    else
    {
        QMutableHashIterator<unsigned int, ito::UiContainerItem> i(m_dialogList);
        FigureWidget *fig;
        while (i.hasNext() ) 
        {
			i.next();
            fig = qobject_cast<FigureWidget*>( i.value().container->getUiWidget());
            if(fig)
            {
                fig->setFigHandle(empty);
            }
        }
    }

    if(semaphore)
    {
        semaphore->returnValue = retval;
        semaphore->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figurePickPoints(unsigned int objectID, QSharedPointer<ito::DataObject> coords, int maxNrPoints, ItomSharedSemaphore *semaphore)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if(widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1 ||metaObject->indexOfSignal("userInteractionDone(int,bool,QPolygonF)") == -1)
        {
            retval += RetVal(retError,0,tr("The desired widget has no signals/slots defined that enable the pick points interaction").toAscii().data());
        }
        else
        {
            UserInteractionWatcher *watcher = new UserInteractionWatcher(widget, ito::PrimitiveContainer::tMultiPointPick, maxNrPoints, coords, semaphore, this);
            connect(watcher, SIGNAL(finished()), this, SLOT(watcherThreadFinished()));
            QThread *watcherThread = new QThread();
            watcher->moveToThread(watcherThread);
            watcherThread->start();

            m_watcherThreads[watcher] = watcherThread;
        }
    }
   else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore && retval.containsError()) //else the semaphore is released by the userInteractionWatcher-instance, executed in a separate thread and monitoring the widget
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }
    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figureDrawGeometricElements(unsigned int objectID, QSharedPointer<ito::DataObject> coords, int elementType, int maxNrPoints, ItomSharedSemaphore *semaphore)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if(widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1 ||metaObject->indexOfSignal("userInteractionDone(int,bool,QPolygonF)") == -1)
        {
            retval += RetVal(retError,0,tr("The desired widget has no signals/slots defined that enable the pick points interaction").toAscii().data());
        }
        else
        {
            UserInteractionWatcher *watcher = new UserInteractionWatcher(widget, elementType, maxNrPoints, coords, semaphore, this);
            connect(watcher, SIGNAL(finished()), this, SLOT(watcherThreadFinished()));
            QThread *watcherThread = new QThread();
            watcher->moveToThread(watcherThread);
            watcherThread->start();

            m_watcherThreads[watcher] = watcherThread;
        }
    }
   else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    if(semaphore && retval.containsError()) //else the semaphore is released by the userInteractionWatcher-instance, executed in a separate thread and monitoring the widget
    {
        semaphore->returnValue = retval;
        semaphore->release();
        semaphore->deleteSemaphore();
        semaphore = NULL;
    }
    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
RetVal UiOrganizer::figurePickPointsInterrupt(unsigned int objectID)
{
    QObject *obj = getWeakObjectReference(objectID);
    QWidget *widget = qobject_cast<QWidget*>(obj);
    RetVal retval;
    if(widget)
    {
        const QMetaObject* metaObject = widget->metaObject();
        if (metaObject->indexOfSlot("userInteractionStart(int,bool,int)") == -1)
        {
            retval += RetVal(retError,0,tr("The desired widget has no signals/slots defined that enable the pick points interaction").toAscii().data());
        }
        else
        {
            QMetaObject::invokeMethod(obj, "userInteractionStart", Q_ARG(int,1), Q_ARG(bool,false), Q_ARG(int,0));
            /*int type = 1;
            bool aborted = true;
            QPolygonF points;
            void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&type)), const_cast<void*>(reinterpret_cast<const void*>(&aborted)), const_cast<void*>(reinterpret_cast<const void*>(&points)) };
            QMetaObject::activate(obj, obj->metaObject(), metaObject->indexOfSignal("userInteractionDone(int,bool,QPolygonF)") - metaObject->methodOffset(), _a);*/
        }
    }
   else
    {
        retval += RetVal(retError,0,tr("the objectID cannot be cast to a widget").toAscii().data() );
    }

    return retval;
}


//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ void UiOrganizer::threadSafeDeleteUi(unsigned int *handle)
{
    UiOrganizer *orga = qobject_cast<UiOrganizer*>( AppManagement::getUiOrganizer() );
    if(orga)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QMetaObject::invokeMethod(orga, "deleteDialog", Q_ARG(unsigned int, *handle), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        //question: do we need locker here?
        locker.getSemaphore()->wait(-1);
    }
    delete handle;
}

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

} //end namespace ito
