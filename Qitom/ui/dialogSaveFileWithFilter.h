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

#ifndef DIALOGLOADFILEWITHFILTER_H

#include "../global.h"

#include "../../common/addInInterface.h"
#include "abstractFilterDialog.h"

#include "paramInputParser.h"

#include "ui_dialogSaveFileWithFilter.h"

namespace ito {

class DialogSaveFileWithFilter : public AbstractFilterDialog
{
    Q_OBJECT

public:
    DialogSaveFileWithFilter(const QString &filename, const ito::AddInAlgo::FilterDef *filter, QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QVector<ito::Param> &userMand, QVector<ito::Param> &userOpt, const bool showFilename, QWidget *parent = NULL);
    ~DialogSaveFileWithFilter() {  };

    void getParameters(QVector<ito::ParamBase> &paramsMand, QVector<ito::ParamBase> &paramsOpt)
    {
        m_pMandParser->getParameters( paramsMand );
        m_pOptParser->getParameters( paramsOpt );
    }


protected:

    ParamInputParser *m_pMandParser;
    ParamInputParser *m_pOptParser;
    const ito::AddInAlgo::FilterDef *m_filter;


private:

    Ui::DialogSaveFileWithFilter ui;

private slots:
    void on_buttonBox_accepted();

};

} //end namespace ito

#endif
