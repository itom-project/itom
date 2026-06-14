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

#include "dialogNewPluginInstance.h"
#include "dialogPluginPicker.h"

#include <qabstractitemmodel.h>
#include <qicon.h>
#include <qgridlayout.h>
#include <qsignalmapper.h>

#include "../../common/helperCommon.h"
#include "../helper/guiHelper.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogNewPluginInstance::DialogNewPluginInstance(QModelIndex &modelIndex, ito::AddInInterfaceBase* aib, bool allowSendToPython) :
    QDialog(),
    m_pMandParser(nullptr),
    m_pOptParser(nullptr)
{
    ui.setupUi(this);

    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

    const QAbstractItemModel *model = modelIndex.model();
    QModelIndex tempIndex;
    QIcon tempIcon;

    //get icon
    QModelIndex parentIdx = model->parent(modelIndex);
    tempIndex = model->index(modelIndex.row(), 0, parentIdx);
    tempIcon = model->data(tempIndex, Qt::DecorationRole).value<QIcon>();
    if (tempIcon.isNull())
    {
        tempIcon = QIcon(":/plugins/icons/plugin.png");
    }

    ui.lblIcon->setPixmap(tempIcon.pixmap(48 * dpiFactor, 48 * dpiFactor));
	ui.lblIcon->setMaximumSize(48 * dpiFactor, 48 * dpiFactor);

    tempIcon = QIcon(":/plugins/icons/sendToPython.png");
    ui.lblImage->setPixmap(tempIcon.pixmap(16 * dpiFactor, 16 * dpiFactor));
	ui.lblImage->setMaximumSize(32 * dpiFactor, 32 * dpiFactor);

    //get name
    ui.lblPluginName->setText(model->data(tempIndex, Qt::DisplayRole).toString());

    //get type
    tempIndex = model->index(modelIndex.row(), 1, parentIdx);
    ui.lblPluginType->setText(model->data(tempIndex, Qt::DisplayRole).toString());

    QWidget *canvas = new QWidget();
    ui.scrollParamsMand->setWidget(canvas);
    m_pMandParser = new ParamInputParser(canvas);

    canvas = new QWidget();
    ui.scrollParamsOpt->setWidget(canvas);
    m_pOptParser = new ParamInputParser(canvas);

    QVector<ito::Param> *params = aib->getInitParamsMand();
    m_pMandParser->createInputMask(*params);
    if (params->size() == 0)
    {
        ui.tabWidget->setTabEnabled(0, false);
    }

    params = aib->getInitParamsOpt();
    m_pOptParser->createInputMask(*params);
    if (params->size() == 0)
    {
        ui.tabWidget->setTabEnabled(1, false);
    }

    ui.groupPython->setVisible(allowSendToPython);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogNewPluginInstance::~DialogNewPluginInstance()
{
    DELETE_AND_SET_NULL(m_pMandParser);
    DELETE_AND_SET_NULL(m_pOptParser);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal DialogNewPluginInstance::getFilledMandParams(QVector<ito::ParamBase> &params)
{
    return m_pMandParser->getParameters(params);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal DialogNewPluginInstance::getFilledOptParams(QVector<ito::ParamBase> &params)
{
    return m_pOptParser->getParameters(params);
}

//----------------------------------------------------------------------------------------------------------------------------------
QString DialogNewPluginInstance::getPythonVariable()
{
    if (ui.groupPython->isChecked() == true)
    {
        return ui.txtPythonVariable->text();
    }

    return QString();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogNewPluginInstance::on_buttonBox_accepted()
{
    ito::RetVal retValue;
    if (!m_pMandParser->validateInput(true, retValue, true))
    {
        return;
    }

    if (!m_pOptParser->validateInput(false, retValue, true))
    {
        return;
    }

    emit accept();
}

} //end namespace ito
