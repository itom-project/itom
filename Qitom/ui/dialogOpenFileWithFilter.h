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

#ifndef DIALOGOPENFILEWITHFILTER_H
#define DIALOGOPENFILEWITHFILTER_H

#include "../global.h"

#include "../../common/addInInterface.h"
#include "abstractFilterDialog.h"
#include <qfuture.h>
#include <qevent.h>

#include "paramInputParser.h"

#include "ui_dialogOpenFileWithFilter.h"

namespace ito {

class DialogOpenFileWithFilter : public AbstractFilterDialog
{
    Q_OBJECT

public:
    enum CheckVarname { CheckNo, CheckGlobalWorkspace, CheckLocalWorkspace }; /*!< defines options to check the python variable name: no check, check for duplicates in global or local workspace. */

    DialogOpenFileWithFilter(const QString &filename, const ito::AddInAlgo::FilterDef *filter, QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QVector<ito::Param> &userMand, QVector<ito::Param> &userOpt, ito::RetVal &retValue, CheckVarname varnameCheck = CheckNo, QWidget *parent = NULL);
    ~DialogOpenFileWithFilter() { delete m_previewMovie; }

    QString getPythonVariable() const { return ui.txtPythonVariable->text(); }

protected:

    void closeEvent(QCloseEvent *e);

    ParamInputParser *m_pMandParser;
    ParamInputParser *m_pOptParser;
    QString m_filename;
    const ito::AddInAlgo::FilterDef *m_filter;
    bool m_filterExecuted;

    ito::RetVal executeFilter();

    QFuture<ito::RetVal> filterCall;
    QFutureWatcher<ito::RetVal> filterCallWatcher;

private:

    Ui::DialogOpenFileWithFilter ui;
    QMovie *m_previewMovie;
    bool m_acceptedClicked;
    CheckVarname m_checkVarname;

    QVector<ito::ParamBase> m_paramsMand;
    QVector<ito::ParamBase> m_paramsOpt;

private slots:
    void on_buttonBox_accepted();
    void on_tabWidget_currentChanged(int index);
    void on_cmdReload_clicked();

    void filterCallFinished();

};

} //end namespace ito

#endif
