/* ********************************************************************
    Template for an algorithm plugin for the software itom

    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#ifndef ALGOPLUGIN_H
#define ALGOPLUGIN_H

#include "common/addInInterface.h"
#include "common/sharedStructures.h"

#include "DataObject/dataobj.h"

#include <qsharedpointer.h>

//----------------------------------------------------------------------------------------------------------------------------------
/** @class AlgoPluginInterface
*   @brief short description of this class
*
*   AddIn Interface for the AlgoPlugin class \sa AlgoPlugin
*/
class AlgoPluginInterface : public ito::AddInInterfaceBase
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ito.AddInInterfaceBase" )
    Q_INTERFACES(ito::AddInInterfaceBase)
    PLUGIN_ITOM_API

    public:
        AlgoPluginInterface();       /*! <Class constructor */
        ~AlgoPluginInterface();      /*! <Class destructor */
        ito::RetVal getAddInInst(ito::AddInBase **addInInst);   /*! <Create a new instance of AlgoPlugin-Class */

    private:
        ito::RetVal closeThisInst(ito::AddInBase **addInInst);  /*! <Destroy the loaded instance of AlgoPlugin-Class */
};

//----------------------------------------------------------------------------------------------------------------------------------
/** @class AlgoPlugin
*   @brief short description of this filter class
*
*   long description of this filter class
*
*/
class AlgoPlugin : public ito::AddInAlgo
{
    Q_OBJECT

    protected:
        AlgoPlugin();    /*! <Class constructor */
        ~AlgoPlugin();   /*! <Class destructor */

    public:
        friend class AlgoPluginInterface;

        //for each algorithm copy and adjust the following doc-string and the following two functions
        static const char* algo1doc;
        static ito::RetVal algo1(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal algo1Params(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

    public slots:
        ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal close(ItomSharedSemaphore *waitCond);
};

//----------------------------------------------------------------------------------------------------------------------------------

#endif // ALGOPLUGIN_H
