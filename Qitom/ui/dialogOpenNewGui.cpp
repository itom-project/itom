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

#include "dialogOpenNewGui.h"

#include <qgridlayout.h>
#include <qsignalmapper.h>

#include "../../common/helperCommon.h"
#include "../helper/guiHelper.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
DialogOpenNewGui::DialogOpenNewGui(const QString &widgetName, const QVector<ito::Param> &initParamsMand, const QVector<ito::Param> &initParamsOpt) :
    QDialog(),
    m_pMandParser(NULL),
    m_pOptParser(NULL)
{
    ui.setupUi(this);

    //get name
    ui.lblWidgetName->setText(widgetName);

    QWidget *canvas = new QWidget();
    ui.scrollParamsMand->setWidget(canvas);
    m_pMandParser = new ParamInputParser(canvas);

    canvas = new QWidget();
    ui.scrollParamsOpt->setWidget(canvas);
    m_pOptParser = new ParamInputParser(canvas);

    m_pMandParser->createInputMask(initParamsMand);
    if (initParamsMand.size() == 0)
    {
        ui.tabWidget->setTabEnabled(0, false);
    }

    m_pOptParser->createInputMask(initParamsOpt);
    if (initParamsOpt.size() == 0)
    {
        ui.tabWidget->setTabEnabled(1, false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogOpenNewGui::~DialogOpenNewGui()
{
    DELETE_AND_SET_NULL(m_pMandParser);
    DELETE_AND_SET_NULL(m_pOptParser);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal DialogOpenNewGui::getFilledMandParams(QVector<ito::ParamBase> &params)
{
    return m_pMandParser->getParameters(params);
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal DialogOpenNewGui::getFilledOptParams(QVector<ito::ParamBase> &params)
{
    return m_pOptParser->getParameters(params);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogOpenNewGui::on_buttonBox_accepted()
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
