/* ********************************************************************
    Template for an actuator plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#ifndef MYACTUATOR_H
#define MYACTUATOR_H

#include "common/addInInterface.h"
#include "opencv/cv.h"
#include <qsharedpointer.h>
#include "dialogMyActuator.h"

//----------------------------------------------------------------------------------------------------------------------------------
 /**
  *\class    MyActuatorInterface 
  *
  *\brief    Interface-Class for MyActuator-Class
  *
  *    \sa    AddInActuator, MyActuator
  *
  */
class MyActuatorInterface : public ito::AddInInterfaceBase
{
    Q_OBJECT
#if QT_VERSION >=  QT_VERSION_CHECK(5, 0, 0)
    Q_PLUGIN_METADATA(IID "ito.AddInInterfaceBase" )
#endif
    Q_INTERFACES(ito::AddInInterfaceBase)
    PLUGIN_ITOM_API

    public:
        MyActuatorInterface();
        ~MyActuatorInterface();
        ito::RetVal getAddInInst(ito::AddInBase **addInInst);

    private:
        ito::RetVal closeThisInst(ito::AddInBase **addInInst);
};


//----------------------------------------------------------------------------------------------------------------------------------
 /**
  *\class    MyActuator

  */
class MyActuator : public ito::AddInActuator
{
    Q_OBJECT

    protected:
        //! Destructor
        ~MyActuator();
        //! Constructor
        MyActuator();
        
    public:
        friend class MyActuatorInterface;
        const ito::RetVal showConfDialog(void);
        int hasConfDialog(void) { return 1; }; //!< indicates that this plugin has got a configuration dialog

    private:

        
    public slots:
        //!< Get Camera-Parameter
        ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond);
        //!< Set Camera-Parameter
        ito::RetVal setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond);
        //!< Initialise board, load dll, allocate buffer
        ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
        //!< Free buffer, delete board, unload dll
        ito::RetVal close(ItomSharedSemaphore *waitCond);



    private slots:
        void dockWidgetVisibilityChanged(bool visible);
};

#endif // MYACTUATOR_H
