/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2021, Institut fuer Technische Optik (ITO),
   Universitaet Stuttgart, Germany

   This file is part of itom.

   itom is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   itom is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#pragma once

#include <QtGui>
#include <qdialog.h>
#include <qlocale.h>

#include "common/interval.h"

#include "ui_dialogHeatmapConfiguration.h"

namespace ito {

class DialogHeatmapConfiguration : public QDialog
{
    Q_OBJECT

public:
    DialogHeatmapConfiguration(const AutoInterval& interval, QWidget* parent = nullptr);
    ~DialogHeatmapConfiguration(){};

    ito::AutoInterval getInterval() const;

private:
    bool checkValue(QLineEdit* lineEdit, const double& min, const double& max, const QString& name);

    double m_min;
    double m_max;
    QLocale m_locale;
    AutoInterval m_interval;

    Ui::DialogHeatmapConfiguration ui;

private slots:
    void on_buttonBox_accepted();
    };

    } // end namespace ito
