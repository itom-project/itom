/* ********************************************************************
    Template for an actuator plugin for the software itom

    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#include "dialogMyActuator.h"

#include "common/addInInterface.h"

#include <qdialogbuttonbox.h>
#include <qvector.h>
#include <qsharedpointer.h>

//----------------------------------------------------------------------------------------------------------------------------------
DialogMyActuator::DialogMyActuator(ito::AddInBase *grabber) :
    AbstractAddInConfigDialog(grabber),
    m_firstRun(true)
{
    ui.setupUi(this);

    //disable dialog, since no parameters are known yet. Parameters will immediately be sent by the slot parametersChanged.
    enableDialog(false);
};


//----------------------------------------------------------------------------------------------------------------------------------
void DialogMyActuator::parametersChanged(QMap<QString, ito::Param> params)
{
    //save the currently set parameters to m_currentParameters
    m_currentParameters = params;

    if (m_firstRun)
    {
        setWindowTitle(QString((params)["name"].getVal<char*>()) + " - " + tr("Configuration Dialog"));

        //this is the first time that parameters are sent to this dialog,
        //therefore you can add some initialization work here
        m_firstRun = false;

        //now activate group boxes, since information is available now (at startup, information is not available, since parameters are sent by a signal)
        enableDialog(true);
    }

    //set the status of all widgets depending on the values of params

}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal DialogMyActuator::applyParameters()
{
    ito::RetVal retValue(ito::retOk);
    QVector<QSharedPointer<ito::ParamBase> > values;
    bool success = false;

    //foreach widget, do:
    //   check if the current value of the widget is different than the corresponding
    //   parameter in m_currentParameters.
    //   If so, write something like:
    //   values.append(QSharedPointer<ito::ParamBase>(new ito::ParamBase("name",ito::ParamBase::Int, newValue)))

    retValue += setPluginParameters(values, msgLevelWarningAndError);

    return retValue;
}

//---------------------------------------------------------------------------------------------------------------------
void DialogMyActuator::on_buttonBox_clicked(QAbstractButton* btn)
{
    ito::RetVal retValue(ito::retOk);

    QDialogButtonBox::ButtonRole role = ui.buttonBox->buttonRole(btn);

    if (role == QDialogButtonBox::RejectRole)
    {
        reject(); //close dialog with reject
    }
    else if (role == QDialogButtonBox::AcceptRole)
    {
        accept(); //AcceptRole
    }
    else
    {
        applyParameters(); //ApplyRole
    }
}

//---------------------------------------------------------------------------------------------------------------------
void DialogMyActuator::enableDialog(bool enabled)
{
    //e.g.
    ui.group1->setEnabled(enabled);
    ui.group2->setEnabled(enabled);
}
