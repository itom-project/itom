/* ********************************************************************
    Template for a camera / grabber plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#include "dockWidgetMyGrabber.h"

//----------------------------------------------------------------------------------------------------------------------------------
DockWidgetMyGrabber::DockWidgetMyGrabber(ito::AddInDataIO *grabber) :
    AbstractAddInDockWidget(grabber),
    m_inEditing(false),
    m_firstRun(true)
{
    ui.setupUi(this);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyGrabber::parametersChanged(QMap<QString, ito::Param> params)
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
// void DockWidgetMyGrabber::on_contrast_valueChanged(int i)
// {
    // if (!m_inEditing)
    // {
        // m_inEditing = true;
        // QSharedPointer<ito::ParamBase> p(new ito::ParamBase("contrast",ito::ParamBase::Int,d));
        // setPluginParameter(p, msgLevelWarningAndError);
        // m_inEditing = false;
    // }
// }

//----------------------------------------------------------------------------------------------------------------------------------
void DockWidgetMyGrabber::identifierChanged(const QString &identifier)
{
    ui.lblIdentifier->setText(identifier);
}