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

#ifndef DIALOGOPENNEWGUI_H
#define DIALOGOPENNEWGUI_H

#include "../../common/sharedStructures.h"

#include "../global.h"

#include <QtGui>
#include <qdialog.h>

#include "paramInputParser.h"


#include "ui_dialogOpenNewGui.h"

namespace ito {

class DialogOpenNewGui : public QDialog
{
    Q_OBJECT

public:
    DialogOpenNewGui(const QString &widgetName, const QVector<ito::Param> &initParamsMand, const QVector<ito::Param> &initParamsOpt);
    ~DialogOpenNewGui();

    ito::RetVal getFilledMandParams(QVector<ito::ParamBase> &params);
    ito::RetVal getFilledOptParams(QVector<ito::ParamBase> &params);

protected:
    ParamInputParser *m_pMandParser;
    ParamInputParser *m_pOptParser;

private:

    Ui::DialogOpenNewGui ui;

private slots:
    void on_buttonBox_accepted();

};

} //end namespace ito

#endif
