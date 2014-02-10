#ifndef CAMERAPLUGIN_H
#define CAMERAPLUGIN_H

#include "common/addInGrabber.h"
//#include "dialogCameraPlugin.h"

#include <qsharedpointer.h>
#include <QTimerEvent>

//----------------------------------------------------------------------------------------------------------------------------------
class CameraPluginInterface : public ito::AddInInterfaceBase
{
    Q_OBJECT
    Q_INTERFACES(ito::AddInInterfaceBase)  /*!< this CameraPluginInterface implements the ito::AddInInterfaceBase-interface, which makes it available as plugin in itom */
    PLUGIN_ITOM_API

    public:
        CameraPluginInterface();                    /*!< Constructor */
        ~CameraPluginInterface();                   /*!< Destructor */
        ito::RetVal getAddInInst(ito::AddInBase **addInInst);   /*!< creates new instance of DummyGrabber and returns this instance */

    private:
        ito::RetVal closeThisInst(ito::AddInBase **addInInst);  /*!< closes any specific instance of DummyGrabber, given by *addInInst */

};


//----------------------------------------------------------------------------------------------------------------------------------
class CameraPlugin : public ito::AddInGrabber
{
    Q_OBJECT

    protected:
        ~CameraPlugin();
        CameraPlugin();

        ito::RetVal retrieveData(ito::DataObject *externalDataObject = NULL);

    public:
        friend class CameraPluginInterface;
        
    private:

    signals:

    public slots:

        ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal close(ItomSharedSemaphore *waitCond);

        ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond = NULL);

        ito::RetVal startDevice(ItomSharedSemaphore *waitCond);
        ito::RetVal stopDevice(ItomSharedSemaphore *waitCond);
        ito::RetVal acquire(const int trigger, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal getVal(void *vpdObj, ItomSharedSemaphore *waitCond);
        ito::RetVal copyVal(void *vpdObj, ItomSharedSemaphore *waitCond);


    private slots:



};



//----------------------------------------------------------------------------------------------------------------------------------

#endif // CAMERAPLUGIN_H
