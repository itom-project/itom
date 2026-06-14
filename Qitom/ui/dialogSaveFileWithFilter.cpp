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

#include "dialogSaveFileWithFilter.h"

#include "../helper/guiHelper.h"

#include <QtConcurrent/qtconcurrentrun.h>
#include <qfileinfo.h>
#include <qfileiconprovider.h>

namespace ito {

DialogSaveFileWithFilter::DialogSaveFileWithFilter(const QString &filename, const ito::AddInAlgo::FilterDef *filter, QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QVector<ito::Param> &userMand, QVector<ito::Param> &userOpt, const bool showFilename, QWidget *parent) :
    AbstractFilterDialog(autoMand,autoOut,parent),
    m_pMandParser(NULL),
    m_pOptParser(NULL),
    m_filter(NULL)
{
    ui.setupUi(this);
    QFileInfo info(filename);

    if (showFilename)
    {
        ui.lblFilename->setText(info.fileName());
    }
    else
    {
        ui.lblFilename->setVisible(false);
        ui.lblFilenameLabel->setVisible(false);
    }

    m_filter = filter;

	float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

    QFileIconProvider *provider = new QFileIconProvider();
    QIcon tempIcon = provider->icon(info);
	ui.lblIcon->setPixmap(tempIcon.pixmap(dpiFactor * 48, dpiFactor * 48));
	ui.lblIcon->setMaximumSize(dpiFactor * 48, dpiFactor * 48);
    delete provider;

    ui.lblFilter->setText( filter->m_name );

    QWidget *canvas = new QWidget();
    ui.scrollParamsMand->setWidget( canvas );
    m_pMandParser = new ParamInputParser( canvas );

    canvas = new QWidget();
    ui.scrollParamsOpt->setWidget( canvas );
    m_pOptParser = new ParamInputParser( canvas );

    m_pMandParser->createInputMask( userMand );
    m_pOptParser->createInputMask( userOpt );
}

void DialogSaveFileWithFilter::on_buttonBox_accepted()
{
    RetVal retVal;
    if( m_pMandParser->validateInput(true, retVal, true) == false || m_pOptParser->validateInput(false, retVal, true) == false)
    {
       //
    }
    else
    {
        emit accept();
    }
}



} //end namespace ito
