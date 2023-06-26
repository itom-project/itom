/* ********************************************************************
    Template for an actuator plugin for the software itom

    You can use this template, use it in your plugins, modify it,
    copy it and distribute it without any license restrictions.
*********************************************************************** */

#ifndef DIALOGMYACTUATOR_H
#define DIALOGMYACTUATOR_H

#include "common/sharedStructures.h"
#include "common/sharedStructuresQt.h"
#include "common/abstractAddInConfigDialog.h"

#include "ui_dialogMyActuator.h"

#include <qstring.h>
#include <qmap.h>
#include <qabstractbutton.h>

namespace ito
{
    class AddInBase; //forward declaration
}

class DialogMyActuator : public ito::AbstractAddInConfigDialog
{
    Q_OBJECT

    public:
        DialogMyActuator(ito::AddInBase *grabber);
        ~DialogMyActuator() {};

        ito::RetVal applyParameters();

    private:
        void enableDialog(bool enabled);
        bool m_firstRun;

        Ui::DialogMyActuator ui;

    public slots:
        void parametersChanged(QMap<QString, ito::Param> params);

    private slots:
        void on_buttonBox_clicked(QAbstractButton* btn);
};

#endif
