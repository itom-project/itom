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

#ifndef UIORGANIZER_H
#define UIORGANIZER_H

//#include "../python/pythonQtConversion.h"
#include "../python/pythonItomMetaObject.h"

#include <qwidget.h>

#include "../common/sharedStructuresQt.h"
#include "../common/functionCancellationAndObserver.h"

#include "../DataObject/dataobj.h"
#if ITOM_POINTCLOUDLIBRARY > 0
#include "../../PointCloud/pclStructures.h"
#endif

#include "../widgets/userUiDialog.h"
#include "../widgets/figureWidget.h"

#include "../global.h"
#include "../common/sharedStructuresGraphics.h"
#include "../common/addInInterface.h"
#include "../common/shape.h"

#include "../../plot/AbstractFigure.h"

#include <qmap.h>
#include <qsharedpointer.h>
#include <qstring.h>
#include <qvariant.h>
#include <qhash.h>
#include <qtimer.h>
#include <qfiledialog.h>
#include <qmainwindow.h>
#include <qthread.h>
#include <qtranslator.h>
#include <qpoint.h>
#include <qsize.h>

class QUiLoader; //forward declaration

namespace ito
{
class WidgetWrapper; //forward declaration
class TimerModel; //forward declaration

/*!
    \class UiContainer
    \brief Every instance of this class contains information about one user interface (main window, dialog, dock widget...)
        which is organized by UiOrganizer.
*/
struct UiContainer
{
public:
    enum tUiType     /*!< type of the user interface */
    {
        uiTypeUiDialog    = 0x0001,
        uiTypeQDialog     = 0x0002,
        uiTypeQMainWindow = 0x0003,
        uiTypeQDockWidget = 0x0004,
        uiTypeFigure      = 0x0005,
		uiTypeWidget      = 0x0006
    };

    //! creates new UiContainer from instance of dialog-widget UserUiDialog
    /*!
        The weak reference to uiDialog together with the type uiTypeUiDialog is saved as member variable in UiDialogSet.

        \param uiDialog is the dialog-instance which should be guarded by the instance of UiDialogSet
    */
    UiContainer(UserUiDialog *uiDialog) :
        m_type(uiTypeUiDialog)
    {
        m_weakDialog = QPointer<QWidget>(uiDialog);
    }

    //! creates new UiContainer from instance of QDialog
    /*!
        The weak reference to dialog together with the type uiTypeQDialog is saved as member variable in UiDialogSet.

        \param dialog is an instance of QDialog or inherited from it which should be guarded by the instance of UiDialogSet
    */
    UiContainer(QDialog *dialog) :
        m_type(uiTypeQDialog)
    {
        m_weakDialog = QPointer<QDialog>(dialog);
    }

    //! creates new UiContainer from instance of QMainWindow
    /*!
        The weak reference to mainWindow together with the type uiTypeMainWindow is saved as member variable in UiDialogSet.

        \param mainWindow is the window-instance which should be guarded by the instance of UiDialogSet
    */
    UiContainer(QMainWindow *mainWindow) :
        m_type(uiTypeQMainWindow)
    {
        m_weakDialog = QPointer<QWidget>(mainWindow);
    }

    //! creates new UiContainer from instance of QMainWindow
    /*!
        The weak reference to mainWindow together with the type uiTypeMainWindow is saved as member variable in UiDialogSet.

        \param mainWindow is the window-instance which should be guarded by the instance of UiDialogSet
    */
    UiContainer(FigureWidget *figureWidget) :
        m_type(uiTypeFigure)
    {
        m_weakDialog = QPointer<QWidget>(figureWidget);
    }

    //! creates new UiContainer from instance of QDockWidget
    /*!
        The weak reference to dockWidget together with the type uiTypeQDockWidget is saved as member variable in UiDialogSet.

        \param dockWidget is the dockWidget-instance which should be guarded by the instance of UiDialogSet
    */
    UiContainer(QDockWidget *dockWidget) :
        m_type(uiTypeQDockWidget)
    {
        m_weakDialog = QPointer<QWidget>(dockWidget);
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
        m_weakDialog = QPointer<QWidget>(widget);
    }

    //! copy constructor
    UiContainer(const UiContainer &cpy)
    {
        m_weakDialog = QPointer<QWidget>(cpy.getUiWidget());
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
    QPointer<QWidget> m_weakDialog;        /*!< weak pointer to the user interface which is covered by this instance. A weak reference is used, since an external deletion of the user interface is then savely considered. */
    tUiType m_type;                            /*!< type of the user interface which is covered by this instance. \sa tUiType */
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

class UiDataContainer
{
private:
    ito::ParamBase::Type m_dataType;
    QSharedPointer<ito::DataObject> m_dObjPtr;
#if ITOM_POINTCLOUDLIBRARY > 0
    QSharedPointer<ito::PCLPointCloud> m_dPCPtr;
    QSharedPointer<ito::PCLPolygonMesh> m_dPMPtr;
#endif

public:
    UiDataContainer() : m_dataType(ito::ParamBase::DObjPtr) {};
    ~UiDataContainer() {};
    UiDataContainer(const QSharedPointer<ito::DataObject> &sharedDataObject) : m_dataType(ito::ParamBase::DObjPtr), m_dObjPtr(sharedDataObject) {}
#if ITOM_POINTCLOUDLIBRARY > 0
    UiDataContainer(const QSharedPointer<ito::PCLPointCloud> &sharedPointCloud) : m_dataType(ito::ParamBase::PointCloudPtr), m_dPCPtr(sharedPointCloud) {}
    UiDataContainer(const QSharedPointer<ito::PCLPolygonMesh> &sharedPolygonMesh) : m_dataType(ito::ParamBase::PolygonMeshPtr), m_dPMPtr(sharedPolygonMesh) {}

    inline UiDataContainer & operator = (QSharedPointer<ito::PCLPointCloud> sharedPointCloud)
    {
        m_dataType = ito::ParamBase::PointCloudPtr;
        m_dPCPtr = sharedPointCloud;
        m_dObjPtr.clear();
        m_dPMPtr.clear();
        return *this;
    }

    inline UiDataContainer & operator = (QSharedPointer<ito::PCLPolygonMesh> sharedPolygonMesh)
    {
        m_dataType = ito::ParamBase::PolygonMeshPtr;
        m_dPMPtr = sharedPolygonMesh;
        m_dObjPtr.clear();
        m_dPCPtr.clear();
        return *this;
    }
#endif

    inline UiDataContainer & operator = (QSharedPointer<ito::DataObject> sharedDataObject)
    {
        m_dataType = ito::ParamBase::DObjPtr;
        m_dObjPtr = sharedDataObject;
#if ITOM_POINTCLOUDLIBRARY > 0
        m_dPCPtr.clear();
        m_dPMPtr.clear();
#endif
        return *this;
    }

    inline ito::ParamBase::Type getType() const { return m_dataType; }
    inline QSharedPointer<ito::DataObject> getDataObject() const { return m_dObjPtr; }
#if ITOM_POINTCLOUDLIBRARY > 0
    inline QSharedPointer<ito::PCLPointCloud> getPointCloud() const { return m_dPCPtr; }
    inline QSharedPointer<ito::PCLPolygonMesh> getPolygonMesh() const { return m_dPMPtr; }
#endif
};

struct ClassInfoContainer
{
    enum Type {TypeClassInfo, TypeSlot, TypeSignal, TypeProperty, TypeEnum, TypeFlag, TypeInheritance};
    ClassInfoContainer(Type type, const QString &name, const QString &shortDescription = "", const QString &description = "") :
        m_type(type), m_name(name), m_shortDescription(shortDescription), m_description(description)
    {
        if (m_description == "")
        {
            m_description = m_shortDescription;
        }
    }

    Type m_type;
    QString m_name;
    QString m_shortDescription;
    QString m_description;
};

} // namespace ito

namespace ito {

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
        typeDockWidget = 0x0002,
		typeCentralWidget = 0x0003
    };

    enum tObjectInfo
    {
        infoShowNoInheritance = 0x0001,
        infoShowItomInheritance = 0x0002,
        infoShowInheritanceUpToWidget = 0x0004,
        infoShowAllInheritance =0x0008
    };

    typedef QList<ClassInfoContainer> ClassInfoContainerList;

    UiOrganizer(ito::RetVal &retval);
    ~UiOrganizer();

    void showDialog(QWidget *parent);
    inline QObject *getPluginReference(unsigned int objectID) { return getWeakObjectReference(objectID); }

    static inline void parseUiDescription(int uiDescription, int* uiType = NULL, int* buttonBarType = NULL, bool* childOfMainWindow = NULL, bool* deleteOnClose = NULL, int* dockWidgetArea = NULL)
    {
        if(uiType) *uiType =                        (uiDescription & 0x000000FF);        //bits 1-8
        if(buttonBarType) *buttonBarType =         ((uiDescription & 0x0000FF00) >> 8);  //bits 9-16
        if(childOfMainWindow) *childOfMainWindow = ((uiDescription & 0x000F0000) > 0);   //bits 17-20
        if(deleteOnClose) *deleteOnClose =         ((uiDescription & 0x00F00000) > 0);   //bits 21-24
        if(dockWidgetArea) *dockWidgetArea =       ((uiDescription & 0xFF000000) >> 24); //bits 25-32
    }

    static inline int createUiDescription(int uiType, int buttonBarType, bool childOfMainWindow, bool deleteOnClose, int dockWidgetArea)
    {
        int v = uiType & 0x000000FF; //bits 1-8
        v += (buttonBarType << 8); //bits 9-16
        if(childOfMainWindow) v += (1 << 16); //bits 17-24
        if(deleteOnClose) v+= (1 << 20); //bits 21-24
        v += (dockWidgetArea << 24); //bits 25-32

        return v;
    }

    RetVal getNewPluginWindow(const QString &pluginName, unsigned int &objectID, QWidget** newWidget, QWidget *parent = NULL);

    QWidget* loadDesignerPluginWidget(const QString &name, RetVal &retValue, AbstractFigure::WindowMode winMode, QWidget *parent = NULL);

    QWidget* createWidget(const QString &className, RetVal &retValue, QWidget *parent = NULL, const QString &objectName = QString());

    //!< loads a widget from an ui file (including its optional translation) and returns it.
    QWidget* loadUiFile(const QString &filename, RetVal &retValue, QWidget *parent = NULL, const QString &objectNameSuffix = QString());

    TimerModel* getTimerModel() const;

protected:

    //static void threadSafeDeleteUi(unsigned int *handle);

    void startGarbageCollectorTimer();

    RetVal addWidgetToOrganizer(QWidget *widget, int uiDescription, const StringMap &dialogButtons,
                                QSharedPointer<unsigned int>dialogHandle,
                                QSharedPointer<unsigned int> objectID,
                                QSharedPointer<QByteArray> className);

private:
    void execGarbageCollection();

    unsigned int addObjectToList(QObject* objPtr);
    QObject *getWeakObjectReference(unsigned int objectID);

    QByteArray getReadableMethodSignature(const QMetaMethod &method, bool pythonNotCStyle, QByteArray *methodName = NULL, bool *valid = NULL);
    QByteArray getReadableParameter(const QByteArray &parameter, bool pythonNotCStyle, bool *valid = NULL);
    ito::UiOrganizer::ClassInfoContainerList::Iterator parseMetaPropertyForEnumerationTypes(const QMetaProperty &meth, ClassInfoContainerList &currentPropList);

    void timerEvent(QTimerEvent *event);

    WidgetWrapper *m_widgetWrapper;                    /*!< singleton instance to WidgetWrapper in order to access some public methods of several widgets by python */

    QHash<unsigned int, UiContainerItem> m_dialogList; /*!< Hash-Table mapping a handle to an user interface to its corresponding instance of UiDialogSet */
    QHash<unsigned int, QPointer<QObject> > m_objectList;  /*!< Hash-Table containing weak references to child-objects of any user interface which have recently been accessed. This hash-table helps for faster access to these objects */
    int m_garbageCollectorTimer;                    /*!< ID of the garbage collection timer. This timer regularly calls timerEvent in order to check m_dialogList and m_objectList for objects, which already have been destroyed. */
    QMap<QObject*, QThread*> m_watcherThreads;   /*!< map with opened watcher threads and their containing objects (e.g. UserInteractionWatcher) */

    static unsigned int autoIncUiDialogCounter;        /*!< auto incrementing counter for elements in m_dialogList */
    static unsigned int autoIncObjectCounter;        /*!< auto incrementing counter for elements in m_objectList */

    void setApiPointersToWidgetAndChildren(QWidget *widget);
    //moved the uiLoader object to here from loadDesignerPluginWidget and createNewDialog methods as according
    //to valgrind it causes memory leaks. So better have only one instance created and maintain mem leaks low ;-)
    QUiLoader *m_pUiLoader;
    QHash<QString, QTranslator*> m_transFiles;
    TimerModel *m_pTimerModel;

signals:

public slots:
    void pythonKeyboardInterrupt(bool checked);

    RetVal loadPluginWidget(void* algoWidgetFunc, int uiDescription, const StringMap &dialogButtons, QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QSharedPointer<unsigned int>dialogHandle, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore = NULL);
    RetVal createNewDialog(const QString &filename, int uiDescription, const StringMap &dialogButtons, QSharedPointer<unsigned int> dialogHandle, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> className, ItomSharedSemaphore *semaphore = NULL);
    RetVal deleteDialog(unsigned int handle, ItomSharedSemaphore *semaphore = NULL);
    RetVal showDialog(unsigned int handle, int modalLevel, QSharedPointer<int> retCodeIfModal, ItomSharedSemaphore *semaphore = NULL);
    RetVal hideDialog(unsigned int handle, ItomSharedSemaphore *semaphore = NULL);
    RetVal setAttribute(unsigned int handle, Qt::WidgetAttribute attribute, bool on = true, ItomSharedSemaphore *semaphore = NULL);
    RetVal isVisible(unsigned int handle, QSharedPointer<bool> visible, ItomSharedSemaphore *semaphore = NULL);
    RetVal handleExist(unsigned int handle, QSharedPointer<bool> exist, ItomSharedSemaphore *semaphore = NULL);

    UiContainer* getUiDialogByHandle(unsigned int uiHandle);

    RetVal getDockedStatus(unsigned int uiHandle, QSharedPointer<bool> docked, ItomSharedSemaphore *semaphore = NULL);
    RetVal setDockedStatus(unsigned int uiHandle, bool docked, ItomSharedSemaphore *semaphore = NULL);

    RetVal showInputDialogGetDouble(unsigned int objectID, const QString &title, const QString &label, double defaultValue, QSharedPointer<bool> ok, QSharedPointer<double> value, double min = -2147483647, double max = 2147483647, int decimals = 1, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetInt(unsigned int objectID, const QString &title, const QString &label, int defaultValue, QSharedPointer<bool> ok, QSharedPointer<int> value, int min = -2147483647, int max = 2147483647, int step = 1, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetItem(unsigned int objectID, const QString &title, const QString &label, const QStringList &stringList, QSharedPointer<bool> ok, QSharedPointer<QString> value, int currentIndex = 0, bool editable = false, ItomSharedSemaphore *semaphore = NULL );
    RetVal showInputDialogGetText(unsigned int objectID, const QString &title, const QString &label, const QString &defaultString, QSharedPointer<bool> ok, QSharedPointer<QString> value, ItomSharedSemaphore *semaphore = NULL );
    RetVal showMessageBox(unsigned int objectID, int type, const QString &title, const QString &text, int buttons, int defaultButton, QSharedPointer<int> retButton, QSharedPointer<QString> retButtonText, ItomSharedSemaphore *semaphore = NULL );

    RetVal showFileDialogExistingDir(unsigned int objectID, const QString &caption, QSharedPointer<QString> directory, int options = QFileDialog::ShowDirsOnly, ItomSharedSemaphore *semaphore = NULL); //options are of type QFileDialog::Options
    RetVal showFileOpenDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QString> file, int selectedFilterIndex = 0, int options = 0, ItomSharedSemaphore *semaphore = NULL);
    RetVal showFilesOpenDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QStringList> files, int selectedFilterIndex = 0, int options = 0, ItomSharedSemaphore *semaphore = NULL);
    RetVal showFileSaveDialog(unsigned int objectID, const QString &caption, const QString &directory, const QString &filter, QSharedPointer<QString> file, int selectedFilterIndex = 0, int options = 0, ItomSharedSemaphore *semaphore = NULL);

    RetVal exists(unsigned int objectID, QSharedPointer<bool> exists, ItomSharedSemaphore *semaphore = NULL);
    RetVal getPropertyInfos(unsigned int objectID, QSharedPointer<QVariantMap> retPropertyMap, ItomSharedSemaphore *semaphore = NULL);
    RetVal readProperties(unsigned int objectID, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal writeProperties(unsigned int objectID, const QVariantMap &properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal readProperties(unsigned int handle, const QString &widgetName, QSharedPointer<QVariantMap> properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal writeProperties(unsigned int handle, const QString &widgetName, const QVariantMap &properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal getAttribute(unsigned int objectID, int attributeNumber, QSharedPointer<bool> value, ItomSharedSemaphore *semaphore = NULL);
    RetVal setAttribute(unsigned int objectID, int attributeNumber, bool value, ItomSharedSemaphore *semaphore = NULL);
    RetVal getWindowFlags(unsigned int objectID, QSharedPointer<int> flags, ItomSharedSemaphore *semaphore = NULL);
    RetVal setWindowFlags(unsigned int objectID, int flags, ItomSharedSemaphore *semaphore = NULL);
    RetVal widgetMetaObjectCounts(unsigned int objectID, QSharedPointer<int> classInfoCount, QSharedPointer<int> enumeratorCount, QSharedPointer<int> methodCount, QSharedPointer<int> propertyCount, ItomSharedSemaphore *semaphore = NULL );

    RetVal getChildObject(unsigned int uiHandle, const QString &objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore = NULL);
    RetVal getChildObject2(unsigned int parentObjectID, const QString &objectName, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore = NULL);
    RetVal getChildObject3(unsigned int parentObjectID, const QString &objectName, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);
    RetVal getLayout(unsigned int objectID, QSharedPointer<unsigned int> layoutObjectID, QSharedPointer<QByteArray> layoutClassName, QSharedPointer<QString> layoutObjectName, ItomSharedSemaphore *semaphore = NULL);
    RetVal getSignalIndex(unsigned int objectID, const QByteArray &signalSignature, QSharedPointer<int> signalIndex, QSharedPointer<QObject*> objPtr, QSharedPointer<IntList> argTypes, ItomSharedSemaphore *semaphore = NULL);
    RetVal callSlotOrMethod(bool slotNotMethod, unsigned int objectID, int slotOrMethodIndex, QSharedPointer<FctCallParamContainer> args, ItomSharedSemaphore *semaphore = NULL);

    RetVal getObjectInfo(const QString &classname, bool pythonNotCStyle, ito::UiOrganizer::ClassInfoContainerList *objectInfo, ItomSharedSemaphore *semaphore = NULL);
    RetVal getObjectInfo(const QObject *obj, int type, bool pythonNotCStyle, ito::UiOrganizer::ClassInfoContainerList* objectInfo, ItomSharedSemaphore *semaphore = NULL);

    inline RetVal getObjectInfo(unsigned int objectID, int type, bool pythonNotCStyle, ito::UiOrganizer::ClassInfoContainerList *objectInfo, ItomSharedSemaphore *semaphore = NULL)
    {
        return getObjectInfo(getWeakObjectReference(objectID), type, pythonNotCStyle, objectInfo, semaphore);
    }

    RetVal getObjectAndWidgetName(unsigned int objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);
    RetVal getObjectChildrenInfo(unsigned int objectID, bool recursive, QSharedPointer<QStringList> objectNames, QSharedPointer<QStringList> classNames, ItomSharedSemaphore *semaphore = NULL);

    RetVal getObjectID(const QObject *obj, QSharedPointer<unsigned int> objectID, ItomSharedSemaphore *semaphore = NULL);

    RetVal connectWithKeyboardInterrupt(unsigned int objectID, const QByteArray &signalSignature, ItomSharedSemaphore *semaphore = NULL);
    RetVal connectProgressObserverInterrupt(unsigned int objectID, const QByteArray &signalSignature, QPointer<QObject> progressObserver, ItomSharedSemaphore *semaphore = NULL);
    RetVal getMethodDescriptions(unsigned int objectID, QSharedPointer<MethodDescriptionList> methodList, ItomSharedSemaphore *semaphore = NULL);

    RetVal createFigure(QSharedPointer< QSharedPointer<unsigned int> > guardedFigureHandle, QSharedPointer<unsigned int> objectID, QSharedPointer<int> rows, QSharedPointer<int> cols, QPoint offset = QPoint(), QSize size = QSize(), ItomSharedSemaphore *semaphore = NULL);
    RetVal getSubplot(QSharedPointer<unsigned int> figHandle, unsigned int subplotIndex, QSharedPointer<unsigned int> objectID, QSharedPointer<QByteArray> objectName, QSharedPointer<QByteArray> widgetClassName, ItomSharedSemaphore *semaphore = NULL);

    RetVal figurePlot(ito::UiDataContainer &dataCont, ito::UiDataContainer &xAxisCont,QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal figureLiveImage(AddInDataIO* dataIO, QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore = NULL);
    RetVal figureDesignerWidget(QSharedPointer<unsigned int> figHandle, QSharedPointer<unsigned int> objectID, int areaRow, int areaCol, QString className, QVariantMap properties, ItomSharedSemaphore *semaphore = NULL);
    void  figureAssureMinimalSize(ito::FigureWidget* fig);

    RetVal figureClose(unsigned int figHandle, ItomSharedSemaphore *semaphore = NULL);
    RetVal closeAllFloatableFigures(ItomSharedSemaphore *semaphore = NULL);
    RetVal figureShow(const unsigned int& handle = 0, ItomSharedSemaphore *semaphore = NULL);
    RetVal figureMinimizeAll(ItomSharedSemaphore *semaphore = NULL);
    RetVal figurePickPoints(unsigned int objectID, QSharedPointer<QVector<ito::Shape> > shapes, int maxNrPoints, ItomSharedSemaphore *semaphore);
    RetVal figureDrawGeometricShapes(unsigned int objectID, QSharedPointer<QVector<ito::Shape> > shapes, int shapeType, int maxNrElements, ItomSharedSemaphore *semaphore);
    RetVal figurePickPointsInterrupt(unsigned int objectID);
    RetVal isFigureItem(unsigned int objectID,  QSharedPointer<unsigned int> isFigureItem, ItomSharedSemaphore *semaphore);
    RetVal getAllAvailableHandles(QSharedPointer<QList<unsigned int> > list, ItomSharedSemaphore *semaphore = NULL);
    RetVal getPlotWindowTitlebyHandle(const unsigned int& objectID, QSharedPointer<QString> title, ItomSharedSemaphore * semaphore = NULL);

    RetVal connectWidgetsToProgressObserver(bool hasProgressBar, unsigned int progressBarObjectID, bool hasLabel, unsigned int labelObjectID, QSharedPointer<ito::FunctionCancellationAndObserver> progressObserver, ItomSharedSemaphore *semaphore);

    RetVal getAvailableWidgetNames(QSharedPointer<QStringList> widgetNames, ItomSharedSemaphore *semaphore);

	RetVal registerActiveTimer(const QWeakPointer<QTimer> &timer, const QString &name);

    RetVal copyStringToClipboard(const QString &text, ItomSharedSemaphore *semaphore);

    void figureDestroyed(QObject *obj);

private slots:
    void watcherThreadFinished();

};

} //namespace ito

Q_DECLARE_METATYPE(ito::UiOrganizer::ClassInfoContainerList*)

#endif
