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

#ifndef UIORGANIZER_H
#define UIORGANIZER_H

//#include "../python/pythonQtConversion.h"
#include "../python/pythonItomMetaObject.h"

#include <qwidget.h>

#include "../common/sharedStructuresQt.h"
#include "../DataObject/dataobj.h"

#include "../widgets/userUiDialog.h"
#include "../widgets/figureWidget.h"

#include "../global.h"
#include "../common/sharedStructuresGraphics.h"
#include "../common/addInInterface.h"

#include "../../plot/AbstractFigure.h"

#include <qmap.h>
#include <qsharedpointer.h>
#include <qstring.h>
#include <qvariant.h>
#include <qhash.h>
#include <qtimer.h>
#include <qfiledialog.h>
#include <qmainwindow.h>

namespace ito
{

class WidgetWrapper; //forward declaration

/*!
    \class UiContainer
    \brief Every instance of this class contains information about one user interface (main window, dialog, dock widget...)
		which is organized by UiOrganizer.
*/
struct UiContainer
{
public:
    enum tUiType	 /*!< type of the user interface */
    {
        uiTypeUiDialog    = 0x0001,
        uiTypeQDialog     = 0x0002,
        uiTypeQMainWindow = 0x0003,
        uiTypeQDockWidget = 0x0004,
        uiTypeFigure      = 0x0005
    };

	//! creates new UiContainer from instance of dialog-widget UserUiDialog
	/*!
		The weak reference to uiDialog together with the type uiTypeUiDialog is saved as member variable in UiDialogSet.

		\param uiDialog is the dialog-instance which should be guarded by the instance of UiDialogSet
	*/
    UiContainer(UserUiDialog *uiDialog) : 
        m_type(uiTypeUiDialog) 
    {
        m_weakDialog = QWeakPointer<QWidget>(qobject_cast<QWidget*>(uiDialog));
    }

	//! creates new UiContainer from instance of QDialog
	/*!
		The weak reference to dialog together with the type uiTypeQDialog is saved as member variable in UiDialogSet.

		\param dialog is an instance of QDialog or inherited from it which should be guarded by the instance of UiDialogSet
	*/
    UiContainer(QDialog *dialog) : 
        m_type(uiTypeQDialog) 
    {
        m_weakDialog = QWeakPointer<QWidget>(qobject_cast<QWidget*>(dialog));
    }
	
	//! creates new UiContainer from instance of QMainWindow
	/*!
		The weak reference to mainWindow together with the type uiTypeMainWindow is saved as member variable in UiDialogSet.

		\param mainWindow is the window-instance which should be guarded by the instance of UiDialogSet
	*/
    UiContainer(QMainWindow *mainWindow) : 
        m_type(uiTypeQMainWindow) 
    {
        m_weakDialog = QWeakPointer<QWidget>(qobject_cast<QWidget*>(mainWindow));
    }

    //! creates new UiContainer from instance of QMainWindow
	/*!
		The weak reference to mainWindow together with the type uiTypeMainWindow is saved as member variable in UiDialogSet.

		\param mainWindow is the window-instance which should be guarded by the instance of UiDialogSet
	*/
    UiContainer(FigureWidget *figureWidget) : 
        m_type(uiTypeFigure) 
    {
        m_weakDialog = QWeakPointer<QWidget>(qobject_cast<QWidget*>(figureWidget));
    }
	
	//! general constructor to create an instance of UiContainer from given QWidget*-pointer and type
	/*!
		The weak reference to widget together with the type-parameter is saved as member variable in this instance of UiDialogSet

		\param widget is the pointer to QWidget
		\param type is the corresponding type of widget \sa tUiType
	*/
    UiContainer(QWidget *widget, tUiType type) : 
        m_type(type)
    {
        m_weakDialog = QWeakPointer<QWidget>(widget);
    }
    
	//! copy constructor
    UiContainer(const UiContainer &cpy)
    {
        m_weakDialog = QWeakPointer<QWidget>(cpy.getUiWidget());
        m_type = cpy.m_type;
    }
    

    ~UiContainer(); //comment in source file


	//! returns instance of UiDialog or NULL, if the widget is not longer available of the type is not uiTypeUiDialog
    inline UserUiDialog *getUiDialog() const 
    { 
        if(m_type == uiTypeUiDialog)
        {
            if(m_weakDialog.isNull()) return NULL;
            return qobject_cast<UserUiDialog*>(m_weakDialog.data()); 
        }
        return NULL;
    }

	//! returns instance of Widget or NULL, if the widget is not longer available.
	/*!
		Internally, even a dialog or main windows are casted to QWidget. Therefore, this getter method always
		returns this casted QWidget and NULL, if the QWidget has been deleted before.
	*/
    inline QWidget *getUiWidget() const 
    { 
        if(m_weakDialog.isNull()) return NULL;
        return m_weakDialog.data();
    }
	
	//! returns type of the guarded user interface
    inline tUiType getType() const { return m_type; }

private:
    QWeakPointer<QWidget> m_weakDialog;		/*!< weak pointer to the user interface which is covered by this instance. A weak reference is used, since an external deletion of the user interface is then savely considered. */
    tUiType m_type;							/*!< type of the user interface which is covered by this instance. \sa tUiType */
};

struct UiContainerItem
{
public:
    UiContainerItem() : container(NULL) {}
    
    UiContainerItem(const UiContainerItem &cpy)
    {
        guardedHandle = cpy.guardedHandle;
        container = cpy.container;
    }

    QWeakPointer< unsigned int > guardedHandle;
    UiContainer *container;
};






class UiOrganizer : public QObject
{
    Q_OBJECT
public:
    enum tPropertyFlags  /*!< enumeration describing possible attributes of a property of type QMetaProperty */
    {
        propValid      =   0x0001,
        propConstant   =   0x0002,
        propFinal      =   0x0004,
        propReadable   =   0x0008,
        propWritable   =   0x0010,
        propResettable =   0x0020
    };

    enum tErrorCode  /*!< enumeration with error numbers for different errors which may occure within the class UiOrganizer */
    {
        errorUiHandleInvalid = 0x1001,
        errorObjDoesNotExist = 0x1002,
        errorObjPropWrite = 0x1003,
        errorObjPropRead = 0x1004,
        errorObjPropDoesNotExist = 0x1005,
        errorUnregisteredType = 0x1006,
        errorSlotDoesNotExist = 0x1007,
        errorSignalDoesNotExist = 0x1008,
        errorConnectionError = 0x1009
    };

    enum tWinType /*!< enumeration describing the desired type of the user interface */
    {
        typeDialog     = 0x0000,
        typeMainWindow = 0x0001,
        typeDockWidget = 0x0002
    };

    UiOrganizer();
    ~UiOrganizer();

    void showDialog(QWidget *parent);
    //inline QHash<QString, PluginInfo> & getLoadedPluginList(void) { return m_pluginInfoList; }
    inline QObject *getPluginReference(unsigned int objectID) { return getWeakObjectReference(objectID); }
//    ito::RetVal getNewPluginWindow(const QString pluginName, ito::uint32 &objectID, QObject **newWidget);

    static inline void parseUiDescription(int uiDescription, int* uiType = NULL, int* buttonBarType = NULL, bool* childOfMainWindow = NULL, bool* deleteOnClose = NULL)
    {
        if(uiType) *uiType = (uiDescription & 0x000000FF); //bits 1-8
        if(buttonBarType) *buttonBarType = ((uiDescription & 0x0000FF00) >> 8); //bits 9-16
        if(childOfMainWindow) *childOfMainWindow = ((uiDescription & 0x00FF0000) > 0); //bits 17-24
        if(deleteOnClose) *deleteOnClose = ((uiDescription & 0xFF000000) > 0); //bits 25-32
    }

    static inline int createUiDescription(int uiType, int buttonBarType, bool childOfMainWindow, bool deleteOnClose) 
    { 
        int v = uiType; //bits 1-8
        if(childOfMainWindow) v += (1 << 16); //bits 17-24
        if(deleteOnClose) v+= (1 << 24); //bits 25-32
        v += (buttonBarType << 8); //bits 9-16
        return v;
    }

    RetVal getNewPluginWindow(QString pluginName, unsigned int &objectID, QWidget** newWidget, QWidget *parent = NULL);

    QWidget* loadDesignerPluginWidget(const QString &name, RetVal &retValue, AbstractFigure::WindowMode winMode, QWidget *parent = NULL);

protected:

    static void threadSafeDeleteUi(unsigned int *handle);

private:
    UiContainer* getUiDialogByHandle(unsigned int uiHandle);

    void execGarbageCollection();

    unsigned int addObjectToList(QObject* objPtr);
    QObject *getWeakObjectReference(unsigned int objectID);

    void timerEvent(QTimerEvent *event);

	WidgetWrapper *m_widgetWrapper;					/*!< singleton instance to WidgetWrapper in order to access some public methods of several widgets by python */
    //QHash<QString, PluginInfo> m_pluginInfoList;

    QHash<unsigned int, UiContainerItem> m_dialogList; /*!< Hash-Table mapping a handle to an user interface to its corresponding instance of UiDialogSet */
    QHash<unsigned int, QWeakPointer<QObject> > m_objectList;  /*!< Hash-Table containing weak references to child-objects of any user interface which have recently been accessed. This hash-table helps for faster access to these objects */
    int m_garbageCollectorTimer;					/*!< ID of the garbage collection timer. This timer regularly calls timerEvent in order to check m_dialogList and m_objectList for objects, which already have been destroyed. */
    /*QHash<QString, ito::plotFigureType> m_designerPluginTypeList;
    QHash<QString, ito::plotFeatureType> m_designerPluginFeatureList;*/

    static unsigned int autoIncUiDialogCounter;		/*!< auto incrementing counter for elements in m_dialogList */
    static unsigned int autoIncObjectCounter;		/*!< auto incrementing counter for elements in m_objectList */
    ito::RetVal scanPlugins(QString path, QHash<QString, PluginInfo> &pluginInfoList);

    void setApiPointersToWidgetAndChildren(QWidget *widget);

signals:

public slots:
    void pythonKeyboardInterrupt(bool checked);

    RetVal loadPluginWidget(void* algoWidgetFunc, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore = NULL);
    RetVal addWidgetToOrganizer(QWidget *widget, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int>initSlotCount, QWidget *parent = NULL, ItomSharedSemaphore *semaphore = NULL);
    RetVal createNewDialog(QString filename, int uiDescription, StringMap dialogButtons, QSharedPointer<unsigned int> dialogHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore = NULL);
    RetVal deleteDialog(unsigned int handle, ItomSharedSemaphore *semaphore = NULL);
    RetVal showDialog(unsigned int handle, int modalLevel, QSharedPointer<int> retCodeIfModal, ItomSharedSemaphore *semaphore = NULL);
    RetVal hideDialog(unsigned int handle, ItomSharedSemaphore *semaphore = NULL);
    RetVal setAttribute(unsigned int handle, Qt::WidgetAttribute attribute, bool on = true, ItomSharedSemaphore *semaphore = NULL);
    RetVal isVisible(unsigned int handle, QSharedPointer<bool> visible, ItomSharedSemaphore *semaphore = NULL);

    RetVal getDockedStatus(unsigned int uiHandle, QSharedPointer<bool> docked, ItomSharedSemaphore *semaphore = NULL);
    RetVal setDockedStatus(unsigned int uiHandle, bool docked, ItomSharedSemaphore *semaphore = NULL);

    RetVal showInputDialogGetDouble(QString title, QString label, double defaultValue, QSharedPointer<bool> ok, QSharedPointer<double> value, double min = -2147483647, double max = 2147483647, int decimals = 1, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetInt(QString title, QString label, int defaultValue, QSharedPointer<bool> ok, QSharedPointer<int> value, int min = -2147483647, int max = 2147483647, int step = 1, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetItem(QString title, QString label, QStringList stringList, QSharedPointer<bool> ok, QSharedPointer<QString> value, int currentIndex = 0, bool editable = false, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetText(QString title, QString label, QString defaultString, QSharedPointer<bool> ok, QSharedPointer<QString> value, ItomSharedSemaphore *semaphore = NULL );
    RetVal showMessageBox(unsigned int uiHandle, int type, QString title, QString text, int buttons, int defaultButton, QSharedPointer<int> retButton, QSharedPointer<QString> retButtonText, ItomSharedSemaphore *semaphore = NULL );

    RetVal showFileDialogExistingDir(unsigned int uiHandle, QString caption, QSharedPointer<QString> directory, int options = QFileDialog::ShowDirsOnly, ItomSharedSemaphore *semaphore = NULL); //options are of type QFileDialog::Options
    RetVal showFileOpenDialog(unsigned int uiHandle, QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex = 0, int options = 0, ItomSharedSemaphore *semaphore = NULL);
    RetVal showFileSaveDialog(unsigned int uiHandle, QString caption, QString directory, QString filter, QSharedPointer<QString> file, int selectedFilterIndex = 0, int options = 0, ItomSharedSemaphore *semaphore = NULL);

    RetVal getPropertyInfos(unsigned int objectID, QSharedPointer<QVariantMap> retPropertyMap, ItomSharedSemaphore *semaphore = NULL);
    RetVal readProperties(unsigned int objectID, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal writeProperties(unsigned int objectID, QVariantMap properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal readProperties(unsigned int handle, QString widgetName, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal writeProperties(unsigned int handle, QString widgetName, QVariantMap properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal getAttribute(unsigned int objectID, int attributeNumber, QSharedPointer<bool> value, ItomSharedSemaphore *semaphore = NULL);
    RetVal setAttribute(unsigned int objectID, int attributeNumber, bool value, ItomSharedSemaphore *semaphore = NULL);
    RetVal widgetMetaObjectCounts(unsigned int objectID, QSharedPointer<int> classInfoCount, QSharedPointer<int> enumeratorCount, QSharedPointer<int> methodCount, QSharedPointer<int> propertyCount, ItomSharedSemaphore *semaphore = NULL );

    RetVal getChildObject(unsigned int uiHandle, QString objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore = NULL);
    RetVal getChildObject2(unsigned int parentObjectID, QString objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore = NULL);
    RetVal getChildObject3(unsigned int parentObjectID, QString objectName, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);
    RetVal getSignalIndex(unsigned int objectID, QString signalSignature, QSharedPointer<int> signalIndex, QSharedPointer<QObject*> objPtr, QSharedPointer<IntList> argTypes, ItomSharedSemaphore *semaphore = NULL);
    RetVal callSlotOrMethod(bool slotNotMethod, unsigned int objectID, int slotOrMethodIndex, QSharedPointer<FctCallParamContainer> args, ItomSharedSemaphore *semaphore = NULL);

    RetVal connectWithKeyboardInterrupt(unsigned int objectID, QString signalSignature, ItomSharedSemaphore *semaphore = NULL);

    RetVal getMethodDescriptions(unsigned int objectID, QSharedPointer<MethodDescriptionList> methodList, ItomSharedSemaphore *semaphore = NULL);

    RetVal getObjectInfo(unsigned int objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);

    /*RetVal plotImage(QSharedPointer<ito::DataObject> dataObj, QSharedPointer<unsigned int> plotHandle, QString plotClassName = "", ItomSharedSemaphore *semaphore = NULL);    
    RetVal liveData(AddInDataIO* dataIO, QString widget, QObject **window, ItomSharedSemaphore *semaphore = NULL);
    RetVal liveImage(AddInDataIO* dataIO, QString plotClassName = "", ItomSharedSemaphore *semaphore = NULL);
    RetVal liveLine(AddInDataIO* dataIO, QString plotClassName = "", ItomSharedSemaphore *semaphore = NULL);*/

    RetVal createFigure(QSharedPointer< QSharedPointer<unsigned int> > guardedFigureHandle, QSharedPointer<unsigned int> initSlotCount, QSharedPointer<unsigned int> objectID, QSharedPointer<int> rows, QSharedPointer<int> cols, ItomSharedSemaphore *semaphore = NULL);
    RetVal getSubplot(QSharedPointer<unsigned int> figHandle, unsigned int subplotIndex, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);

    RetVal figurePlot(QSharedPointer<ito::DataObject> dataObj, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, ItomSharedSemaphore *semaphore = NULL);
    RetVal figureLiveImage(AddInDataIO* dataIO, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, ItomSharedSemaphore *semaphore = NULL);
    
    RetVal figureRemoveGuardedHandle(unsigned int figHandle, ItomSharedSemaphore *semaphore = NULL);
    RetVal figureClose(unsigned int figHandle, ItomSharedSemaphore *semaphore = NULL);

    void figureDestroyed(QObject *obj)
    {
        qDebug() << obj;
    }

private slots:


};

} //namespace ito

#endif
