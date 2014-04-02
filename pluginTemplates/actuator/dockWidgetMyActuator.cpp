/* ********************************************************************
    Template for an actuator plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#include "dockWidgetMyActuator.h"

//----------------------------------------------------------------------------------------------------------------------------------
DockWidgetMyActuator::DockWidgetMyActuator(ito::AddInActuator *actuator) :
    AbstractAddInDockWidget(actuator),
    m_actuator(actuator),
    m_inEditing(false),
    m_firstRun(true)
{
    ui.setupUi(this);
    enableWidget(true);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::parametersChanged(QMap<QString, ito::Param> params)
{
    if (m_firstRun)
    {
        //first time call
        //get all given parameters and adjust all widgets according to them (min, max, stepSize, values...)

        m_firstRun = false;
    }

    if (!m_inEditing)
    {
        m_inEditing = true;
        //check the value of all given parameters and adjust your widgets according to them (value only should be enough)

        m_inEditing = false;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::identifierChanged(const QString &identifier)
{
    ui.lblIdentifier->setText(identifier);
}

//----------------------------------------------------------------------------------------------------------------------------------
//slot invoked if actuatorStatusChanged is emitted in plugin
/*
    status should always have the length equal to the number of axes,
    actPosition has either the same length or is empty, depending if the current position is known and sent or not.
*/
void DockWidgetMyActuator::actuatorStatusChanged(QVector<int> status, QVector<double> actPosition)
{
    //enable or disable the target position spinboxes if the ito::actuatorEnabled flag is set in status[i] or not
    //ui.widget_i->setEnabled(status[i] & ito::actuatorEnabled);

    if (actPosition.size() > 0)
    {
        //set the widget for the current position
        //ui.widget_i->setValue(actPosition[i]);
    }

    bool running = false;
    QString style;
    
    //modifiy the background color of the current position widget depending on the moving state of any axis
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] & ito::actuatorMoving)
        {
            style = "background-color: yellow";
            running = true;
        }
        else if (status[i] & ito::actuatorInterrupted)
        {
            style = "background-color: red";
        }
        else if (status[i] & ito::actuatorTimeout)
        {
            style = "background-color: #FFA3FD";
        }
        else
        {
            style = "background-color: ";
        }
        
        //todo: set the style to the specific widget, like:
        // ui.widget_i->setStyleSheet(style);
     }

     enableWidget(!running);
}

//----------------------------------------------------------------------------------------------------------------------------------
//slot invoked if targetChanged is emitted in plugin
void DockWidgetMyActuator::targetChanged(QVector<double> targetPositions)
{
    int i = targetPositions.size();
    
    //set the target position of every axis to the value given in targetPositions.
    //please check the size of targetPositions in order to avoid crashes if it is not long enough
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::enableWidget(bool enabled)
{
    //todo: disable/enable all widgets that are use to start or stop a moving operation
    // enable means that a movement is possible
    // if enable=false, the motor is currently running and an interrupt button could be shown
}