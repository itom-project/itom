/* ********************************************************************
    Template for an actuator plugin for the software itom

    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#include "dockWidgetMyActuator.h"

//----------------------------------------------------------------------------------------------------------------------------------
DockWidgetMyActuator::DockWidgetMyActuator(ito::AddInActuator *actuator) :
    AbstractAddInDockWidget(actuator),
    m_inEditing(false),
    m_firstRun(true)
{
    ui.setupUi(this);

    //in order to simplify the communication with the axis specific
    //widgets without the need of programming the same thing multiple
    //times, all relevant widget pointers are now saved in few vectors.

    m_btnRelInc.append(ui.btnXp);
    m_btnRelInc.append(ui.btnYp);
    m_btnRelInc.append(ui.btnZp);
    foreach(QPushButton* btn, m_btnRelInc)
    {
        connect(btn, SIGNAL(clicked()), this, SLOT(btnRelIncClicked()));
    }

    m_btnRelDec.append(ui.btnXm);
    m_btnRelDec.append(ui.btnYm);
    m_btnRelDec.append(ui.btnZm);
    foreach(QPushButton* btn, m_btnRelDec)
    {
        connect(btn, SIGNAL(clicked()), this, SLOT(btnRelDecClicked()));
    }

    m_spinCurrentPos.append(ui.spinCurrentPosX);
    m_spinCurrentPos.append(ui.spinCurrentPosY);
    m_spinCurrentPos.append(ui.spinCurrentPosZ);

    m_spinTargetPos.append(ui.spinTargetPosX);
    m_spinTargetPos.append(ui.spinTargetPosY);
    m_spinTargetPos.append(ui.spinTargetPosZ);

    m_labels.append(ui.lblAxisX);
    m_labels.append(ui.lblAxisY);
    m_labels.append(ui.lblAxisZ);

    enableWidgets(true);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::parametersChanged(QMap<QString, ito::Param> params)
{

}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::targetChanged(QVector<double> targetPos)
{
    for (int i = 0; i < targetPos.size(); i++)
    {
        m_spinTargetPos[i]->setValue(targetPos[i]);
    }
 }

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::actuatorStatusChanged(QVector<int> status, QVector<double> positions)
{
    bool running = false;
    QString style;

    for (int i = 0; i < std::min(status.size(), m_spinCurrentPos.size()); i++)
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
        /*else if (status[i] & ito::actuatorTimeout) //timeout is bad for dummyMotor, since the waitForDone-method always drops into a timeout
        {
            style = "background-color: green";
        }*/
        else
        {
            style = "background-color: ";
        }

        m_spinCurrentPos[i]->setStyleSheet(style);
    }

    enableWidgets(!running);

    for (int i = 0; i < std::min(positions.size(), m_spinCurrentPos.size()); i++)
    {
        m_spinCurrentPos[i]->setValue(positions[i]);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::btnRelDecClicked()                //slot if any button for a relative, negative movement is clicked
{
    double dpos = ui.spinStepSize->value() / -1e3;

    if (qobject_cast<QPushButton*>(sender()))
    {
        int idx = m_btnRelDec.indexOf(qobject_cast<QPushButton*>(sender()));

        if (idx >= 0)
        {
            setActuatorPosition(idx, dpos, true);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::btnRelIncClicked()                //slot if any button for a relative, positive movement is clicked
{
    double dpos = ui.spinStepSize->value() / 1e3;

    if (qobject_cast<QPushButton*>(sender()))
    {
        int idx = m_btnRelInc.indexOf(qobject_cast<QPushButton*>(sender()));

        if (idx >= 0)
        {
            setActuatorPosition(idx, dpos, true);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::on_btnStop_clicked()
{
    setActuatorInterrupt();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::on_btnStart_clicked()
{
    QVector<int> axis;
    QVector<double> dpos;

    for (int i = 0; i < m_btnRelDec.size(); ++i)
    {
        axis << i;
        dpos << m_spinTargetPos[i]->value();
    }

    setActuatorPosition(axis, dpos, false);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::on_btnRefresh_clicked()
{
    requestActuatorStatusAndPositions(true, true);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::enableWidgets(bool enabled)
{
    for (int i = 0; i < m_btnRelDec.size(); i++)
    {
        m_btnRelDec[i]->setEnabled(enabled);
        m_btnRelInc[i]->setEnabled(enabled);
    }

    ui.btnStart->setVisible(enabled);
    ui.btnStop->setVisible(!enabled);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyActuator::identifierChanged(const QString &identifier)
{
    ui.lblIdentifier->setText(identifier);
}
