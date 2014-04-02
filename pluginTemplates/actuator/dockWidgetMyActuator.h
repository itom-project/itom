/* ********************************************************************
    Template for an actuator plugin for the software itom
    
    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#ifndef DOCKWIDGETMYACTUATOR_H
#define DOCKWIDGETMYACTUATOR_H

#include "common/addInInterface.h"
#include "common/abstractAddInDockWidget.h"

#include <qwidget.h>
#include <qmap.h>
#include <qstring.h>

#include "ui_dockWidgetMyActuator.h"

class DockWidgetMyActuator : public ito::AbstractAddInDockWidget
{
    Q_OBJECT

    public:
        DockWidgetMyActuator(ito::AddInActuator *actuator);
        ~DockWidgetMyActuator() {};

    private:
        Ui::DockWidgetMyActuator ui;
        bool m_inEditing;
        bool m_firstRun;
        ito::AddInActuator *m_actuator;
        
        void enableWidget(bool enabled);

    public slots:
        void parametersChanged(QMap<QString, ito::Param> params);
        void identifierChanged(const QString &identifier);
        
        void actuatorStatusChanged(QVector<int> status, QVector<double> actPosition);
        void targetChanged(QVector<double> targetPositions);

    private slots:

};

#endif
