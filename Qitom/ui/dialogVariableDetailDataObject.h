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

#pragma once

#include "ui_dialogVariableDetailDataObject.h"

#include "DataObject/dataobj.h"

#include <qdialog.h>
#include <qspinbox.h>
#include <qlist.h>

namespace ito
{

class DialogVariableDetailDataObject : public QDialog
{
    Q_OBJECT

public:
    DialogVariableDetailDataObject(
        const QString& name,
        const QString& type,
        const char* dtype,
        QSharedPointer<ito::DataObject> data,
        QWidget* parent);
    ~DialogVariableDetailDataObject();

private:
    Ui::DialogVariableDetailDataObject ui;
    QSharedPointer<ito::DataObject> m_dObj;
    ito::Range* m_pAxesRanges;
    QMap<int, QSpinBox*> m_spinBoxToIdxMap;
    bool m_selectedAll;

    void changeDisplayedAxes(int isColNotRow);
    void deleteSlicingWidgets();
    void addSlicingWidgets();

private slots:
    void on_btnCopyClipboard_clicked();
    void spinBoxValueChanged(int idx);
    void on_comboBoxDisplayedCol_currentIndexChanged(int idx);
    void on_comboBoxDisplayedRow_currentIndexChanged(int idx);
    void tableCornerButtonClicked();
};

} //end namespace ito
