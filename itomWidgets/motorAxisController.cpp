/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "motorAxisController.h"

#include "common/addInInterface.h"
#include "common/sharedStructuresQt.h"
#include <qaction.h>
#include <qheaderview.h>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#include <qsignalmapper.h>
#include <qspinbox.h>
#include <qtablewidget.h>
#include <qtoolbutton.h>
#include <qvector.h>

#include "ui_motorAxisController.h"

class MotorAxisControllerPrivate
{
public:
    MotorAxisControllerPrivate() :
        movementType(MotorAxisController::MovementBoth), numAxis(0), isChanging(false),
        defaultAxisType(MotorAxisController::TypeLinear),
        defaultAxisUnit(MotorAxisController::UnitMm), defaultRelativeStepSize(5),
        defaultDecimals(2), cancelAvailable(true), startAllAvailable(true), arbitraryUnit(" a.u."),
        bgColorMoving("yellow"), bgColorInterrupted("red"), bgColorTimeout("#FFA3FD")
    {
    }

    QVector<QDoubleSpinBox*> spinCurrentPos;
    QVector<QDoubleSpinBox*> spinTargetPos;
    QVector<QDoubleSpinBox*> spinStepSize;
    QVector<QWidget*> buttonsRelative;
    QVector<QToolButton*> buttonAbsolute;
    QVector<MotorAxisController::AxisUnit> axisUnit;
    QVector<MotorAxisController::AxisType> axisType;
    QVector<int> axisDecimals;
    QVector<bool> axisEnabled;

    int numAxis;
    bool isChanging;

    MotorAxisController::AxisType defaultAxisType;
    MotorAxisController::AxisUnit defaultAxisUnit;
    int defaultDecimals;

    double defaultRelativeStepSize;
    QStringList verticalHeaderLabels;

    QPointer<ito::AddInActuator> actuator;
    Ui::MotorAxisController ui;
    bool cancelAvailable;
    bool startAllAvailable;

    MotorAxisController::MovementType movementType;
    QString arbitraryUnit;

    QColor bgColorMoving;
    QColor bgColorInterrupted;
    QColor bgColorTimeout;
};

//------------------------------------------------------------------------------------------------------------------------------------------------
const int ColCurrent = 0;
const int ColTarget = 1;
const int ColCommandAbsolute = 2;
const int ColStepSize = 3;
const int ColCommandRelative = 4;

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::MotorAxisController(QWidget* parent) : QWidget(parent)
{
    // register the enumerations that should be callable by slots
    qRegisterMetaType<AxisUnit>("AxisUnit");
    qRegisterMetaType<AxisType>("AxisType");

    d = new MotorAxisControllerPrivate();

    d->ui.setupUi(this);
    d->ui.tableMovement->setColumnCount(5);
    QStringList labels;
    labels << tr("Current Pos.") << tr("Target Pos.") << tr("Abs.") << tr("Step Size")
        << tr("Rel.");
    d->ui.tableMovement->setHorizontalHeaderLabels(labels);

    QHeaderView* header = d->ui.tableMovement->horizontalHeader();
    header->setDefaultSectionSize(120);

    header->setSectionResizeMode(ColCurrent, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(ColTarget, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(ColStepSize, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(ColCommandAbsolute, QHeaderView::Fixed);
    header->setSectionResizeMode(ColCommandRelative, QHeaderView::Fixed);
    header->resizeSection(ColCommandAbsolute, 30);
    header->resizeSection(ColCommandRelative, 64);

    setMovementType(MovementAbsolute);

    connect(
        d->ui.tableMovement,
        SIGNAL(customContextMenuRequested(QPoint)),
        this,
        SLOT(customContextMenuRequested(QPoint)));

    d->ui.tableMovement->setSizeAdjustPolicy(QAbstractScrollArea::AdjustToContents);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::~MotorAxisController()
{
    delete d;
    d = NULL;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setActuator(const QPointer<ito::AddInActuator>& actuator)
{
    if (d->actuator.data() != actuator.data())
    {
        if (d->actuator.data() != NULL)
        {
            disconnect(
                actuator.data(),
                SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)),
                this,
                SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            disconnect(
                actuator.data(),
                SIGNAL(targetChanged(QVector<double>)),
                this,
                SLOT(targetChanged(QVector<double>)));
        }

        d->actuator = actuator;

        if (actuator.data() != NULL)
        {
            connect(
                actuator.data(),
                SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)),
                this,
                SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            connect(
                actuator.data(),
                SIGNAL(targetChanged(QVector<double>)),
                this,
                SLOT(targetChanged(QVector<double>)));
            on_btnRefresh_clicked();
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QPointer<ito::AddInActuator> MotorAxisController::actuator() const
{
    return d->actuator;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QString MotorAxisController::suffixFromAxisUnit(const AxisUnit& unit) const
{
    switch (unit)
    {
    case UnitM:
        return " m";
    case UnitCm:
        return " cm";
    case UnitMm:
        return " mm";
    case UnitMum:
    {
        return QString(" %1m").arg(QChar(0xb5, 0x00)); // QLatin1String(" \u00B5m"); // \mu
    }
    case UnitNm:
        return " nm";
    case UnitDeg:
        return QString(" %1").arg(QChar(0xb0, 0x00)); // \degree
    case UnitAU:
        return d->arbitraryUnit;
    }

    return "";
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::unitToBaseUnit(const double& value, const AxisUnit& unit) const
{
    switch (unit)
    {
    case UnitM:
        return value * 1.e3;
    case UnitCm:
        return value * 1.e2;
    case UnitMm:
        return value;
    case UnitMum:
        return value * 1.e-3;
    case UnitNm:
        return value * 1.e-6;
    case UnitDeg:
    case UnitAU:
        return value;
    }

    return value;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::baseUnitToUnit(const double& value, const AxisUnit& unit) const
{
    switch (unit)
    {
    case UnitM:
        return value * 1.e-3;
    case UnitCm:
        return value * 1.e-2;
    case UnitMm:
        return value;
    case UnitMum:
        return value * 1.e3;
    case UnitNm:
        return value * 1.e6;
    case UnitDeg:
    case UnitAU:
        return value;
    }

    return value;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setNumAxis(int numAxis)
{
    int numExistingRows = d->spinCurrentPos.size();
    d->ui.tableMovement->setRowCount(numAxis);
    d->spinCurrentPos.resize(numAxis);
    d->spinTargetPos.resize(numAxis);
    d->spinStepSize.resize(numAxis);
    d->buttonsRelative.resize(numAxis);
    d->buttonAbsolute.resize(numAxis);

    if (d->axisUnit.size() > numAxis)
    {
        d->axisType.resize(numAxis);
        d->axisUnit.resize(numAxis);
        d->axisDecimals.resize(numAxis);
        d->axisEnabled.resize(numAxis);
    }
    else
    {
        while (d->axisType.size() < numAxis)
        {
            d->axisType.append(d->defaultAxisType);
            d->axisUnit.append(d->defaultAxisUnit);
            d->axisDecimals.append(d->defaultDecimals);
            d->axisEnabled.append(true);
        }
    }

    QDoubleSpinBox* currentPos;
    QDoubleSpinBox* targetPos;
    QDoubleSpinBox* stepSize;
    QToolButton* stepUp;
    QToolButton* stepDown;
    QToolButton* runSingle;
    QWidget* buttonsRelative;
    QHBoxLayout* layout;

    // add missing axes
    for (int i = numExistingRows; i < numAxis; ++i)
    {
        // Spinbox for target position
        targetPos = new QDoubleSpinBox(this);
        targetPos->setMinimum(-std::numeric_limits<double>::max());
        targetPos->setMaximum(std::numeric_limits<double>::max());
        targetPos->setDecimals(d->axisDecimals[i]);
        targetPos->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        targetPos->setEnabled(d->axisEnabled[i]);

        // Spinbox for current position
        currentPos = new QDoubleSpinBox(this);
        currentPos->setReadOnly(true);
        currentPos->setMinimum(-std::numeric_limits<double>::max());
        currentPos->setMaximum(std::numeric_limits<double>::max());
        currentPos->setDecimals(d->axisDecimals[i]);
        currentPos->setButtonSymbols(QAbstractSpinBox::NoButtons);
        currentPos->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        currentPos->setEnabled(d->axisEnabled[i]);

        // Spinbox for single step size of relative movements
        stepSize = new QDoubleSpinBox(this);
        stepSize->setMinimum(-std::numeric_limits<double>::max());
        stepSize->setMaximum(std::numeric_limits<double>::max());
        stepSize->setDecimals(d->axisDecimals[i]);
        stepSize->setValue(baseUnitToUnit(d->defaultRelativeStepSize, d->axisUnit[i]));
        stepSize->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        stepSize->setEnabled(d->axisEnabled[i]);

        // button for step up
        stepUp = new QToolButton(this);
        stepUp->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        stepUp->setText(tr("up"));
        stepUp->setIcon(QIcon(":/icons/up.png"));
        connect(stepUp, &QToolButton::clicked, [=]() { stepUpClicked(i); });

        // button for step down
        stepDown = new QToolButton(this);
        stepDown->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        stepDown->setText(tr("down"));
        stepDown->setIcon(QIcon(":/icons/down.png"));
        connect(stepDown, &QToolButton::clicked, [=]() { stepDownClicked(i); });

        // group of step down and step up buttons
        buttonsRelative = new QWidget(this);
        layout = new QHBoxLayout();
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(5);
        layout->addWidget(stepUp);
        layout->addWidget(stepDown);
        buttonsRelative->setLayout(layout);
        buttonsRelative->setEnabled(d->axisEnabled[i]);

        // button for single axis absolute movement
        runSingle = new QToolButton(this);
        runSingle->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        runSingle->setText(tr("go"));
        runSingle->setIcon(QIcon(":/icons/run.png"));
        connect(runSingle, &QToolButton::clicked, [=]() { runSingleClicked(i); });
        runSingle->setEnabled(d->axisEnabled[i]);

        // add all widgets to table view
        d->spinStepSize[i] = stepSize;
        d->ui.tableMovement->setCellWidget(i, ColStepSize, stepSize);

        d->spinTargetPos[i] = targetPos;
        d->ui.tableMovement->setCellWidget(i, ColTarget, targetPos);

        d->buttonsRelative[i] = buttonsRelative;
        d->ui.tableMovement->setCellWidget(i, ColCommandRelative, buttonsRelative);

        d->buttonAbsolute[i] = runSingle;
        d->ui.tableMovement->setCellWidget(i, ColCommandAbsolute, runSingle);

        d->spinCurrentPos[i] = currentPos;
        d->ui.tableMovement->setCellWidget(i, ColCurrent, currentPos);
    }

    QStringList labels = d->verticalHeaderLabels;
    for (int i = labels.size(); i < numAxis; ++i)
    {
        if (d->verticalHeaderLabels.size() > i)
        {
            labels << d->verticalHeaderLabels[i];
        }
        else
        {
            labels << QString::number(i);
        }
    }
    d->verticalHeaderLabels = labels;
    d->ui.tableMovement->setVerticalHeaderLabels(labels);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
int MotorAxisController::numAxis() const
{
    return d->ui.tableMovement->rowCount();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setAxisUnit(int axisIndex, AxisUnit unit)
{
    ito::RetVal retval;
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (d->axisType[axisIndex] == TypeLinear && unit == UnitDeg)
        {
            unit = UnitMm;
            retval +=
                ito::RetVal(ito::retWarning, 0, "type of axis is linear, the unit is set to 'mm'.");
        }
        else if (d->axisType[axisIndex] == TypeRotational && unit != UnitDeg && unit != UnitAU)
        {
            unit = UnitDeg;
            retval += ito::RetVal::format(
                ito::retWarning,
                0,
                "type of axis is rotational, the unit is set to '%s'.",
                QString("%1").arg(QChar(0xb0, 0x00)).data());
        }

        d->spinTargetPos[axisIndex]->setSuffix(suffixFromAxisUnit(unit));
        d->spinCurrentPos[axisIndex]->setSuffix(suffixFromAxisUnit(unit));
        d->spinStepSize[axisIndex]->setSuffix(suffixFromAxisUnit(unit));

        double baseUnitValue =
            unitToBaseUnit(d->spinTargetPos[axisIndex]->value(), d->axisUnit[axisIndex]);
        double maximum = d->spinTargetPos[axisIndex]->maximum();
        double minimum = d->spinTargetPos[axisIndex]->minimum();
        d->spinTargetPos[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));
        if (maximum < std::numeric_limits<double>::max())
        {
            d->spinTargetPos[axisIndex]->setMaximum(
                baseUnitToUnit(unitToBaseUnit(maximum, d->axisUnit[axisIndex]), unit));
        }
        if (minimum > -std::numeric_limits<double>::max())
        {
            d->spinTargetPos[axisIndex]->setMinimum(
                baseUnitToUnit(unitToBaseUnit(minimum, d->axisUnit[axisIndex]), unit));
        }

        baseUnitValue = unitToBaseUnit(d->spinStepSize[axisIndex]->value(), d->axisUnit[axisIndex]);
        maximum = d->spinStepSize[axisIndex]->maximum();
        minimum = d->spinStepSize[axisIndex]->minimum();
        d->spinStepSize[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));
        if (maximum < std::numeric_limits<double>::max())
        {
            d->spinStepSize[axisIndex]->setMaximum(
                baseUnitToUnit(unitToBaseUnit(maximum, d->axisUnit[axisIndex]), unit));
        }
        if (minimum > -std::numeric_limits<double>::max())
        {
            d->spinStepSize[axisIndex]->setMinimum(
                baseUnitToUnit(unitToBaseUnit(minimum, d->axisUnit[axisIndex]), unit));
        }

        baseUnitValue =
            unitToBaseUnit(d->spinCurrentPos[axisIndex]->value(), d->axisUnit[axisIndex]);
        d->spinCurrentPos[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));


        d->axisUnit[axisIndex] = unit;
    }
    else
    {
        retval +=
            ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::AxisUnit MotorAxisController::axisUnit(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        return d->axisUnit[axisIndex];
    }

    return UnitMm;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setAxisEnabled(int axisIndex, bool enabled)
{
    if (axisIndex >= 0 && axisIndex < d->axisEnabled.size())
    {
        if (d->axisEnabled[axisIndex] != enabled)
        {
            d->spinStepSize[axisIndex]->setEnabled(enabled);
            d->spinTargetPos[axisIndex]->setEnabled(enabled);
            d->buttonsRelative[axisIndex]->setEnabled(enabled);
            d->buttonAbsolute[axisIndex]->setEnabled(enabled);
            d->spinCurrentPos[axisIndex]->setEnabled(enabled);
            d->axisEnabled[axisIndex] = enabled;
        }

        return ito::retOk;
    }
    else
    {
        return ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::axisEnabled(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisEnabled.size())
    {
        return d->axisEnabled[axisIndex];
    }

    return false;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setDefaultAxisUnit(AxisUnit unit)
{
    if (d->defaultAxisUnit != unit)
    {
        if (d->defaultAxisType == TypeRotational)
        {
            d->defaultAxisUnit = UnitDeg;
        }
        else if (unit == UnitDeg)
        {
            d->defaultAxisUnit = UnitMm;
        }
        else
        {
            d->defaultAxisUnit = unit;
        }

        for (int i = 0; i < d->spinCurrentPos.size(); ++i)
        {
            setAxisUnit(i, d->defaultAxisUnit);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::AxisUnit MotorAxisController::defaultAxisUnit() const
{
    return d->defaultAxisUnit;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setAxisType(int axisIndex, AxisType type)
{
    ito::RetVal retval;

    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (d->axisType[axisIndex] != type)
        {
            d->axisType[axisIndex] = type;

            if (type == TypeLinear && axisUnit(axisIndex) == UnitDeg)
            {
                setAxisUnit(axisIndex, UnitMm);
            }
            else if (
                type == TypeRotational &&
                (axisUnit(axisIndex) != UnitDeg && axisUnit(axisIndex) != UnitAU))
            {
                setAxisUnit(axisIndex, UnitDeg);
            }
        }
    }
    else
    {
        retval +=
            ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
    }
    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::AxisType MotorAxisController::axisType(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        return d->axisType[axisIndex];
    }

    return TypeLinear;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setDefaultAxisType(AxisType type)
{
    if (d->defaultAxisType != type)
    {
        d->defaultAxisType = type;

        if (d->defaultAxisType == TypeRotational)
        {
            d->defaultAxisUnit = UnitDeg;
        }
        else if (d->defaultAxisUnit == UnitDeg)
        {
            d->defaultAxisUnit = UnitMm;
        }

        for (int i = 0; i < d->spinCurrentPos.size(); ++i)
        {
            setAxisType(i, type);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::AxisType MotorAxisController::defaultAxisType() const
{
    return d->defaultAxisType;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setDefaultRelativeStepSize(
    double defaultRelativeStepSize) /*in mm or degree*/
{
    if (d->defaultRelativeStepSize != defaultRelativeStepSize)
    {
        d->defaultRelativeStepSize = defaultRelativeStepSize;

        for (int i = 0; i < d->spinStepSize.size(); ++i)
        {
            d->spinStepSize[i]->setValue(
                this->baseUnitToUnit(defaultRelativeStepSize, d->axisUnit[i]));
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::defaultRelativeStepSize() const
{
    return d->defaultRelativeStepSize;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setAxisNames(const QStringList& names)
{
    d->verticalHeaderLabels = names;
    d->ui.tableMovement->setVerticalHeaderLabels(names);
}


//-------------------------------------------------------------------------------------------------------------------------------------------------
QStringList MotorAxisController::axisNames() const
{
    QStringList l = d->verticalHeaderLabels;
    for (int i = l.size(); i < d->ui.tableMovement->rowCount(); ++i)
    {
        l << QString::number(i);
    }
    return l;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setAxisName(int axisIndex, const QString& name)
{
    if (axisIndex >= 0 && axisIndex < d->axisEnabled.size())
    {
        if (d->verticalHeaderLabels[axisIndex] != name)
        {
            d->verticalHeaderLabels[axisIndex] = name;
            d->ui.tableMovement->setVerticalHeaderLabels(d->verticalHeaderLabels);
        }

        return ito::retOk;
    }
    else
    {
        return ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QString MotorAxisController::arbitraryUnit() const
{
    return d->arbitraryUnit.mid(1); // the first char is a space
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setArbitraryUnit(const QString& unit)
{
    if (d->arbitraryUnit.mid(1) != unit)
    {
        d->arbitraryUnit = QString(" %1").arg(unit);
        for (int i = 0; i < d->spinStepSize.size(); ++i)
        {
            setAxisUnit(i, axisUnit(i));
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QString MotorAxisController::axisName(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        return d->verticalHeaderLabels[axisIndex];
    }

    return "";
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setDefaultDecimals(int decimals)
{
    if (decimals != d->defaultDecimals)
    {
        d->defaultDecimals = decimals;

        for (int i = 0; i < d->spinStepSize.size(); ++i)
        {
            d->spinTargetPos[i]->setDecimals(decimals);
            d->spinCurrentPos[i]->setDecimals(decimals);
            d->spinStepSize[i]->setDecimals(decimals);
            d->axisDecimals[i] = decimals;
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
int MotorAxisController::defaultDecimals() const
{
    return d->defaultDecimals;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QColor MotorAxisController::backgroundColorMoving() const
{
    return d->bgColorMoving;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setBackgroundColorMoving(const QColor& color)
{
    if (color != d->bgColorMoving)
    {
        d->bgColorMoving = color;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QColor MotorAxisController::backgroundColorInterrupted() const
{
    return d->bgColorInterrupted;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setBackgroundColorInterrupted(const QColor& color)
{
    if (color != d->bgColorInterrupted)
    {
        d->bgColorInterrupted = color;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
QColor MotorAxisController::backgroundColorTimeout() const
{
    return d->bgColorTimeout;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setBackgroundColorTimeout(const QColor& color)
{
    if (color != d->bgColorTimeout)
    {
        d->bgColorTimeout = color;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setAxisDecimals(int axisIndex, int decimals)
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (decimals != d->axisDecimals[axisIndex])
        {
            d->axisDecimals[axisIndex] = decimals;
            d->spinCurrentPos[axisIndex]->setDecimals(decimals);
            d->spinTargetPos[axisIndex]->setDecimals(decimals);
            d->spinStepSize[axisIndex]->setDecimals(decimals);
        }

        return ito::retOk;
    }
    else
    {
        return ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
int MotorAxisController::axisDecimals(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        return d->axisDecimals[axisIndex];
    }

    return 2;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval MotorAxisController::stepSizeInterval(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if ((d->spinStepSize[axisIndex]->minimum() == -std::numeric_limits<double>::max()) &&
            (d->spinStepSize[axisIndex]->maximum() == std::numeric_limits<double>::max()))
        {
            return ito::AutoInterval(); // auto
        }
        else
        {
            return ito::AutoInterval(
                unitToBaseUnit(d->spinStepSize[axisIndex]->minimum(), axisUnit(axisIndex)),
                unitToBaseUnit(d->spinStepSize[axisIndex]->maximum(), axisUnit(axisIndex)));
        }
    }

    return ito::AutoInterval();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::AutoInterval MotorAxisController::targetInterval(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if ((d->spinTargetPos[axisIndex]->minimum() == -std::numeric_limits<double>::max()) &&
            (d->spinTargetPos[axisIndex]->maximum() == std::numeric_limits<double>::max()))
        {
            return ito::AutoInterval(); // auto
        }
        else
        {
            return ito::AutoInterval(
                unitToBaseUnit(d->spinTargetPos[axisIndex]->minimum(), axisUnit(axisIndex)),
                unitToBaseUnit(d->spinTargetPos[axisIndex]->maximum(), axisUnit(axisIndex)));
        }
    }

    return ito::AutoInterval();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setStepSizeInterval(
    int axisIndex, const ito::AutoInterval& interval)
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (interval.isAuto())
        {
            d->spinStepSize[axisIndex]->setMinimum(-std::numeric_limits<double>::max());
            d->spinStepSize[axisIndex]->setMaximum(std::numeric_limits<double>::max());
        }
        else
        {
            d->spinStepSize[axisIndex]->setMinimum(
                baseUnitToUnit(interval.minimum(), axisUnit(axisIndex)));
            d->spinStepSize[axisIndex]->setMaximum(
                baseUnitToUnit(interval.maximum(), axisUnit(axisIndex)));
        }
    }

    return ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::setTargetInterval(int axisIndex, const ito::AutoInterval& interval)
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (interval.isAuto())
        {
            d->spinTargetPos[axisIndex]->setMinimum(-std::numeric_limits<double>::max());
            d->spinTargetPos[axisIndex]->setMaximum(std::numeric_limits<double>::max());
        }
        else
        {
            d->spinTargetPos[axisIndex]->setMinimum(
                baseUnitToUnit(interval.minimum(), axisUnit(axisIndex)));
            d->spinTargetPos[axisIndex]->setMaximum(
                baseUnitToUnit(interval.maximum(), axisUnit(axisIndex)));
        }
    }

    return ito::RetVal(ito::retError, 0, tr("axisIndex is out of bounds.").toLatin1().data());
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setMovementType(MovementType type)
{
    if (type != d->movementType)
    {
        d->ui.comboType->setCurrentIndex(type);

        switch (type)
        {
        case MovementAbsolute:
            d->ui.tableMovement->showColumn(ColTarget);
            d->ui.tableMovement->hideColumn(ColStepSize);
            d->ui.tableMovement->hideColumn(ColCommandRelative);
            d->ui.tableMovement->showColumn(ColCommandAbsolute);
            d->ui.btnStart->setVisible(d->startAllAvailable);
            break;
        case MovementRelative:
            d->ui.tableMovement->hideColumn(ColTarget);
            d->ui.tableMovement->showColumn(ColStepSize);
            d->ui.tableMovement->showColumn(ColCommandRelative);
            d->ui.tableMovement->hideColumn(ColCommandAbsolute);
            d->ui.btnStart->setVisible(false);
            break;
        case MovementBoth:
            d->ui.tableMovement->showColumn(ColTarget);
            d->ui.tableMovement->showColumn(ColStepSize);
            d->ui.tableMovement->showColumn(ColCommandRelative);
            d->ui.tableMovement->showColumn(ColCommandAbsolute);
            d->ui.btnStart->setVisible(d->startAllAvailable);
            break;
        case MovementNo:
            d->ui.tableMovement->showColumn(ColTarget);
            d->ui.tableMovement->hideColumn(ColStepSize);
            d->ui.tableMovement->hideColumn(ColCommandRelative);
            d->ui.tableMovement->hideColumn(ColCommandAbsolute);
            d->ui.btnStart->setVisible(false);
            break;
        }

        d->movementType = type;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::MovementType MotorAxisController::movementType() const
{
    return d->movementType;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setRefreshAvailable(bool available)
{
    d->ui.btnRefresh->setVisible(available);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::refreshAvailable() const
{
    return d->ui.btnRefresh->isVisible();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setCancelAvailable(bool available)
{
    if (d->cancelAvailable != available)
    {
        d->cancelAvailable = available;
        d->ui.btnCancel->setVisible(d->cancelAvailable);
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::cancelAvailable() const
{
    return d->cancelAvailable;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setStartAllAvailable(bool available)
{
    if (d->startAllAvailable != available)
    {
        d->startAllAvailable = available;

        switch (d->movementType)
        {
        case MovementAbsolute:
        case MovementBoth:
            d->ui.btnStart->setVisible(d->startAllAvailable);
            break;
        case MovementNo:
        case MovementRelative:
            d->ui.btnStart->setVisible(false);
            break;
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::startAllAvailable() const
{
    return d->startAllAvailable;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setMovementTypeVisible(bool visible)
{
    d->ui.comboType->setVisible(visible);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::movementTypeVisible() const
{
    return d->ui.comboType->isVisible();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::actuatorStatusChanged(
    QVector<int> status,
    QVector<double> actPosition) //!< slot to receive information about status and position changes.
{
    bool globalRunning = false;
    bool running = false;
    QString style;

    for (int i = 0; i < std::min(actPosition.size(), d->spinCurrentPos.size()); ++i)
    {
        d->spinCurrentPos[i]->setValue(baseUnitToUnit(actPosition[i], d->axisUnit[i]));
    }

    for (int i = 0; i < std::min(status.size(), d->spinCurrentPos.size()); ++i)
    {
        running = false;
        d->spinTargetPos[i]->setEnabled((status[i] & ito::actuatorEnabled) && d->axisEnabled[i]);

        if (status[i] & ito::actuatorMoving)
        {
            style = "background-color: " + d->bgColorMoving.name();
            running = true;
            globalRunning = true;
        }
        else if (status[i] & ito::actuatorInterrupted)
        {
            style = "background-color: " + d->bgColorInterrupted.name();
        }
        else if (status[i] & ito::actuatorTimeout)
        {
            style = "background-color: " + d->bgColorTimeout.name();
        }
        else
        {
            style = "background-color: ";
        }
        d->spinTargetPos[i]->setStyleSheet(style);
        d->spinCurrentPos[i]->setStyleSheet(style);
        d->spinStepSize[i]->setStyleSheet(style);
        d->buttonsRelative[i]->setEnabled(
            (status[i] & ito::actuatorEnabled) && !running && d->axisEnabled[i]);
        d->buttonAbsolute[i]->setEnabled(
            (status[i] & ito::actuatorEnabled) && !running && d->axisEnabled[i]);
    }

    d->ui.btnStart->setEnabled(!globalRunning && d->startAllAvailable);
    d->ui.btnCancel->setVisible(globalRunning && d->cancelAvailable);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::targetChanged(QVector<double> targetPositions)
{
    for (int i = 0; i < std::min(targetPositions.size(), d->spinTargetPos.size()); ++i)
    {
        d->spinTargetPos[i]->setValue(baseUnitToUnit(targetPositions[i], d->axisUnit[i]));
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::on_btnRefresh_clicked()
{
    ito::RetVal retval;

    if (d->actuator)
    {
        if (!QMetaObject::invokeMethod(
            d->actuator,
            "requestStatusAndPosition",
            Qt::QueuedConnection,
            Q_ARG(bool, true),
            Q_ARG(bool, true)))
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("slot 'requestStatusAndPosition' could not be invoked since it does not exist.")
                .toLatin1()
                .data());
        }
    }
    else
    {
        retval +=
            ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    retValToMessageBox(retval, tr("refresh"));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::on_btnCancel_clicked()
{
    if (d->actuator)
    {
        d->actuator->setInterrupt();
    }
    else
    {
        retValToMessageBox(
            ito::RetVal(ito::retError, 0, tr("Actuator not available").toLatin1().data()),
            tr("interrupt movement"));
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::on_btnStart_clicked()
{
    ito::RetVal retval;

    if (d->actuator)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        QVector<int> axes;
        QVector<double> positions;

        for (int i = 0; i < d->spinCurrentPos.size(); ++i)
        {
            axes << i;
            positions << unitToBaseUnit(d->spinTargetPos[i]->value(), d->axisUnit[i]);
        }

        d->actuator->resetInterrupt();

        if (QMetaObject::invokeMethod(
            d->actuator,
            "setPosAbs",
            Q_ARG(const QVector<int>, axes),
            Q_ARG(QVector<double>, positions),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore());
        }
        else
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("slot 'setPosAbs' could not be invoked since it does not exist.")
                .toLatin1()
                .data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("Actuator not available").toLatin1().data());
    }

    retValToMessageBox(retval, tr("start movement"));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::moveRelOrAbs(int axis, double value, bool relNotAbs)
{
    ito::RetVal retval;
    QByteArray func = relNotAbs ? "setPosRel" : "setPosAbs";

    if (d->actuator)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        double valueBase = unitToBaseUnit(value, d->axisUnit[axis]);

        d->actuator->resetInterrupt();

        if (QMetaObject::invokeMethod(
            d->actuator,
            func,
            Q_ARG(const int, axis),
            Q_ARG(double, valueBase),
            Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore());
        }
        else
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("slot '%s' could not be invoked since it does not exist.")
                .arg(QLatin1String(func))
                .toLatin1()
                .data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("Actuator not available").toLatin1().data());
    }

    retValToMessageBox(retval, tr("start movement"));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::observeInvocation(ItomSharedSemaphore* waitCond) const
{
    ito::RetVal retval;

    if (d->actuator)
    {
        bool timeout = false;

        while (!timeout && waitCond->waitAndProcessEvents(PLUGINWAIT) == false)
        {
            if (d->actuator->isAlive() == false)
            {
                retval += ito::RetVal(
                    ito::retError,
                    0,
                    tr("Timeout while waiting for answer from plugin instance.").toLatin1().data());
                timeout = true;
            }
        }

        if (!timeout)
        {
            retval += waitCond->returnValue;
        }
    }

    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::retValToMessageBox(
    const ito::RetVal& retval, const QString& methodName) const
{
    if ((retval.containsWarningOrError()) && d->actuator.isNull())
    {
        // it seems that the covered actuator has been deleted in the meantime.
        // Do not show any messages any more!
        return;
    }

    if (retval.containsError())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while calling '%1'").arg(methodName));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning())
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while calling '%1'").arg(methodName));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::on_comboType_currentIndexChanged(int index)
{
    if (!d->isChanging)
    {
        d->isChanging = true;
        setMovementType((MovementType)index);
        d->isChanging = false;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::stepUpClicked(int index)
{
    moveRelOrAbs(index, d->spinStepSize[index]->value(), true);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::stepDownClicked(int index)
{
    moveRelOrAbs(index, -d->spinStepSize[index]->value(), true);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::runSingleClicked(int index)
{
    moveRelOrAbs(index, d->spinTargetPos[index]->value(), false);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::customContextMenuRequested(const QPoint& pos)
{
    QModelIndex index = d->ui.tableMovement->indexAt(pos);
    if (index.isValid() && index.row() >= 0 && index.row() < d->ui.tableMovement->rowCount())
    {
        QMenu contextMenu;
        QAction* a;

        if (axisUnit(index.row()) != UnitAU)
        {
            QMenu* unitMenu = contextMenu.addMenu(tr("Unit"));
            if (axisType(index.row()) == TypeLinear)
            {
                a = new QAction("m", this);
                a->setCheckable(true);
                a->setData(UnitM);
                if (axisUnit(index.row()) == UnitM)
                {
                    a->setChecked(true);
                }
                unitMenu->addAction(a);

                a = new QAction("cm", this);
                a->setCheckable(true);
                a->setData(UnitCm);
                if (axisUnit(index.row()) == UnitCm)
                {
                    a->setChecked(true);
                }
                unitMenu->addAction(a);

                a = new QAction("mm", this);
                a->setCheckable(true);
                a->setData(UnitMm);
                if (axisUnit(index.row()) == UnitMm)
                {
                    a->setChecked(true);
                }
                unitMenu->addAction(a);

                a = new QAction(QString("%1m").arg(QChar(0xb5, 0x00)), this); // \mu m
                a->setCheckable(true);
                a->setData(UnitMum);
                if (axisUnit(index.row()) == UnitMum)
                {
                    a->setChecked(true);
                }
                unitMenu->addAction(a);

                a = new QAction("nm", this);
                a->setCheckable(true);
                a->setData(UnitNm);
                if (axisUnit(index.row()) == UnitNm)
                {
                    a->setChecked(true);
                }
                unitMenu->addAction(a);
            }
            else
            {
                a = new QAction(QString(QChar(0xb0, 0x00)), this); // \degree
                a->setCheckable(true);
                a->setChecked(true);
                a->setData(UnitDeg);
                unitMenu->addAction(a);
            }
        }

        QMenu* decimalsMenu = contextMenu.addMenu(tr("Decimals"));

        int dec = axisDecimals(index.row());
        for (int i = 0; i < 6; ++i)
        {
            a = new QAction(QString::number(i), this);
            a->setData(1000 + i);
            a->setCheckable(true);
            if (i == dec)
            {
                a->setChecked(true);
            }
            decimalsMenu->addAction(a);
        }

        a = contextMenu.exec(mapToGlobal(pos));

        if (a)
        {
            int data = a->data().toInt();
            if (data >= 1000)
            {
                setAxisDecimals(index.row(), data - 1000);
            }
            else
            {
                setAxisUnit(index.row(), (AxisUnit)data);
            }
        }
    }
}
