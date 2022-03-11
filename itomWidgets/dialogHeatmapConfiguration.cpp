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

#include "dialogHeatmapConfiguration.h"

#include <qmessagebox.h>
#include <qregularexpression.h>
#include <qvalidator.h>

namespace ito {

//-----------------------------------------------------------------------------------------------
DialogHeatmapConfiguration::DialogHeatmapConfiguration(
    const AutoInterval& interval, QWidget* parent) :
    QDialog(parent),
    m_min(-std::numeric_limits<double>::max()), m_max(std::numeric_limits<double>::max()),
    m_locale(QLocale()), m_interval(interval)
{
    ui.setupUi(this);

    m_locale.setNumberOptions(QLocale::OmitGroupSeparator);

    QString numberRegExp;
    if (m_locale.decimalPoint() == '.')
    {
        numberRegExp = "^[\\+-]?(?:0|[1-9]\\d*)(?:\\.\\d*)?(?:[eE][\\+-]?\\d+)?$";
    }
    else
    {
        numberRegExp = "^[\\+-]?(?:0|[1-9]\\d*)(?:,\\d*)?(?:[eE][\\+-]?\\d+)?$";
    }
    auto numberValidator = new QRegularExpressionValidator(QRegularExpression(numberRegExp), this);

    // x
    if (interval.isAuto())
    {
        ui.radioIntervalAuto->setChecked(true);
    }
    else
    {
        ui.radioIntervalManual->setChecked(true);
    }

    ui.txtMax->setValidator(numberValidator);
    ui.txtMin->setValidator(numberValidator);

    ui.txtMin->setText(m_locale.toString(interval.minimum(), 'g'));
    ui.txtMax->setText(m_locale.toString(interval.maximum(), 'g'));
}

//-----------------------------------------------------------------------------------------------
ito::AutoInterval DialogHeatmapConfiguration::getInterval() const
{
    bool ok;
    double number;
    double min = 0.0;
    double max = 0.0;

    number = m_locale.toDouble(ui.txtMin->text(), &ok);

    if (ok)
    {
        min = number;
    }

    number = m_locale.toDouble(ui.txtMax->text(), &ok);

    if (ok)
    {
        max = number;
    }

    AutoInterval interval(min, max, ui.radioIntervalAuto->isChecked());

    return interval;
}

//-----------------------------------------------------------------------------------------------
bool DialogHeatmapConfiguration::checkValue(
    QLineEdit* lineEdit, const double& min, const double& max, const QString& name)
{
    bool ok;
    double val = m_locale.toDouble(lineEdit->text(), &ok);
    if (!ok)
    {
        QMessageBox::critical(
            this,
            tr("Invalid number"),
            tr("The '%1' number is no valid decimal number.").arg(name));
    }
    else if ((val < min) || (val > max))
    {
        ok = false;
        QMessageBox::critical(
            this,
            tr("Out of range"),
            tr("The '%1' number is out of range [%2,%3]")
                .arg(name)
                .arg(m_locale.toString(min, 'g'))
                .arg(m_locale.toString(max, 'g')));
    }

    if (!ok)
    {
        lineEdit->selectAll();
    }

    return ok;
}


//-----------------------------------------------------------------------------------------------
void DialogHeatmapConfiguration::on_buttonBox_accepted()
{
    bool ok = true;

    if (!checkValue(ui.txtMax, m_min, m_max, tr("minimum")))
    {
        return;
    }

    if (!checkValue(ui.txtMax, m_min, m_max, tr("maximum")))
    {
        return;
    }

    emit accept();
}

} // end namespace ito
