/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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
#include <qmessagebox.h>
#include <qvector.h>
#include <qspinbox.h>
#include <qpushbutton.h>
#include <qtablewidget.h>
#include <qvector.h>
#include <qheaderview.h>
#include <qsignalmapper.h>
#include <qmenu.h>
#include <qaction.h>
#include <qtoolbutton.h>

#include "ui_motorAxisController.h"

class MotorAxisControllerPrivate
{
public:
    MotorAxisControllerPrivate() :
        movementType(MotorAxisController::MovementBoth),
        numAxis(0),
        isChanging(false),
        defaultAxisType(MotorAxisController::TypeLinear),
        defaultAxisUnit(MotorAxisController::UnitMm),
        defaultRelativeStepSize(5),
        defaultDecimals(2),
        cancelAvailable(true)
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
    int numAxis;
    bool isChanging;

    QSignalMapper *stepUpMapper;
    QSignalMapper *stepDownMapper;
    QSignalMapper *runSingleMapper;

    MotorAxisController::AxisType defaultAxisType;
    MotorAxisController::AxisUnit defaultAxisUnit;
    int defaultDecimals;

    double defaultRelativeStepSize;
    QStringList verticalHeaderLabels;

    QPointer<ito::AddInActuator> actuator;
    Ui::MotorAxisController ui;
    bool cancelAvailable;

    MotorAxisController::MovementType movementType;
};

//------------------------------------------------------------------------------------------------------------------------------------------------
const int ColCurrent = 0;
const int ColTarget = 1;
const int ColCommandAbsolute = 2;
const int ColStepSize = 3;
const int ColCommandRelative = 4;



//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::MotorAxisController(QWidget *parent) :
    QWidget(parent)
{
    d = new MotorAxisControllerPrivate();

    d->stepUpMapper = new QSignalMapper(this);
    d->stepDownMapper = new QSignalMapper(this);
    d->runSingleMapper = new QSignalMapper(this);
    connect(d->stepUpMapper, SIGNAL(mapped(int)), this, SLOT(stepUpClicked(int)));
    connect(d->stepDownMapper, SIGNAL(mapped(int)), this, SLOT(stepDownClicked(int)));
    connect(d->runSingleMapper, SIGNAL(mapped(int)), this, SLOT(runSingleClicked(int)));

    d->ui.setupUi(this);
    d->ui.tableMovement->setColumnCount(5);
    QStringList labels;
    labels << tr("Current Pos.") << tr("Target Pos.") << tr("Absolute")  << tr("Step Size") << tr("Relative");
    d->ui.tableMovement->setHorizontalHeaderLabels(labels);

#if QT_VERSION >= 0x050000
    QHeaderView *header = d->ui.tableMovement->horizontalHeader();
    header->setSectionResizeMode(ColCurrent, QHeaderView::Stretch);
    header->setSectionResizeMode(ColTarget, QHeaderView::Stretch);
    header->setSectionResizeMode(ColStepSize, QHeaderView::Stretch);
    header->setSectionResizeMode(ColCommandAbsolute, QHeaderView::Fixed);
    header->resizeSection(ColCommandAbsolute, 40);
    header->setSectionResizeMode(ColCommandRelative, QHeaderView::Fixed);
    header->resizeSection(ColCommandRelative, 85);
#endif

    setMovementType(MovementAbsolute);

    connect(d->ui.tableMovement, SIGNAL(customContextMenuRequested(QPoint)), this, SLOT(customContextMenuRequested(QPoint)));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
MotorAxisController::~MotorAxisController()
{
    delete d;
    d = NULL;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setActuator(const QPointer<ito::AddInActuator> &actuator)
{
    if (d->actuator.data() != actuator.data())
    {
        if (d->actuator.data() != NULL)
        {
            disconnect(actuator.data(), SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)), this, SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            disconnect(actuator.data(), SIGNAL(targetChanged(QVector<double>)), this, SLOT(targetChanged(QVector<double>)));
        }

        d->actuator = actuator;

        if (actuator.data() != NULL)
        {
            connect(actuator.data(), SIGNAL(actuatorStatusChanged(QVector<int>, QVector<double>)), this, SLOT(actuatorStatusChanged(QVector<int>, QVector<double>)));
            connect(actuator.data(), SIGNAL(targetChanged(QVector<double>)), this, SLOT(targetChanged(QVector<double>)));
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
QString MotorAxisController::suffixFromAxisUnit(const AxisUnit &unit)
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
        return QLatin1String(" µm");
    case UnitNm:
        return " nm";
    case UnitDeg:
        return QLatin1String(" °");
    }

    return "";
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::unitToBaseUnit(const double &value, const AxisUnit &unit)
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
        return value;
    }

    return value;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::baseUnitToUnit(const double &value, const AxisUnit &unit)
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
    }
    else
    {
        while (d->axisType.size() < numAxis)
        {
            d->axisType.append(d->defaultAxisType);
            d->axisUnit.append(d->defaultAxisUnit);
            d->axisDecimals.append(d->defaultDecimals);
        }
    }

    QDoubleSpinBox *currentPos;
    QDoubleSpinBox *targetPos;
    QDoubleSpinBox *stepSize;
    QToolButton *stepUp;
    QToolButton *stepDown;
    QToolButton *runSingle;
    QWidget *buttonsRelative;
    QHBoxLayout *layout;

    //add missing axes
    for (int i = numExistingRows; i < numAxis; ++i)
    {
        targetPos = new QDoubleSpinBox(this);
        targetPos->setMinimum(-std::numeric_limits<double>::max());
        targetPos->setMaximum(std::numeric_limits<double>::max());
        targetPos->setDecimals(d->axisDecimals[i]);
        targetPos->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        d->spinTargetPos[i] = targetPos;
        d->ui.tableMovement->setCellWidget(i, ColTarget, targetPos);

        currentPos = new QDoubleSpinBox(this);
        currentPos->setReadOnly(true);
        currentPos->setMinimum(-std::numeric_limits<double>::max());
        currentPos->setMaximum(std::numeric_limits<double>::max());
        currentPos->setDecimals(d->axisDecimals[i]);
        currentPos->setButtonSymbols(QAbstractSpinBox::NoButtons);
        currentPos->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        d->spinCurrentPos[i] = currentPos;
        d->ui.tableMovement->setCellWidget(i, ColCurrent, currentPos);

        stepSize = new QDoubleSpinBox(this);
        stepSize->setMinimum(-std::numeric_limits<double>::max());
        stepSize->setMaximum(std::numeric_limits<double>::max());
        stepSize->setDecimals(d->axisDecimals[i]);
        stepSize->setValue(baseUnitToUnit(d->defaultRelativeStepSize, d->axisUnit[i]));
        stepSize->setSuffix(suffixFromAxisUnit(d->axisUnit[i]));
        d->spinStepSize[i] = stepSize;
        d->ui.tableMovement->setCellWidget(i, ColStepSize, stepSize);

        stepUp = new QToolButton(this);
        stepUp->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        stepUp->setText(tr("up"));
        stepUp->setIcon(QIcon(":/icons/up.png"));
        connect(stepUp, SIGNAL(clicked()), d->stepUpMapper, SLOT(map()));
        d->stepUpMapper->setMapping(stepUp, i);

        stepDown = new QToolButton(this);
        stepDown->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        stepDown->setText(tr("down"));
        stepDown->setIcon(QIcon(":/icons/down.png"));
        connect(stepDown, SIGNAL(clicked()), d->stepDownMapper, SLOT(map()));
        d->stepDownMapper->setMapping(stepDown, i);

        runSingle = new QToolButton(this);
        runSingle->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        runSingle->setText(tr("go"));
        runSingle->setIcon(QIcon(":/icons/run.png"));
        connect(runSingle, SIGNAL(clicked()), d->runSingleMapper, SLOT(map()));
        d->runSingleMapper->setMapping(runSingle, i);

        d->buttonAbsolute[i] = runSingle;
        d->ui.tableMovement->setCellWidget(i, ColCommandAbsolute, runSingle);

        buttonsRelative = new QWidget(this);
        layout = new QHBoxLayout();
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(5);
        layout->addWidget(stepUp);
        layout->addWidget(stepDown);
        buttonsRelative->setLayout(layout);
        d->buttonsRelative[i] = buttonsRelative;
        QWidget *w = d->buttonsRelative[i]->layout()->itemAt(0)->widget();
        d->ui.tableMovement->setCellWidget(i, ColCommandRelative, buttonsRelative);
    }

    QStringList labels = d->verticalHeaderLabels;
    for (int i = labels.size(); i < numAxis; ++i)
    {
        labels << QString::number(i);
    }
    d->ui.tableMovement->setVerticalHeaderLabels(labels);

#if QT_VERSION >= 0x050000
    QHeaderView *header = d->ui.tableMovement->horizontalHeader();
    header->setSectionResizeMode(ColCurrent, QHeaderView::Stretch);
    header->setSectionResizeMode(ColTarget, QHeaderView::Stretch);
    header->setSectionResizeMode(ColStepSize, QHeaderView::Stretch);
    header->setSectionResizeMode(ColCommandAbsolute, QHeaderView::Fixed);
    header->resizeSection(ColCommandAbsolute, 40);
    header->setSectionResizeMode(ColCommandRelative, QHeaderView::Fixed);
    header->resizeSection(ColCommandRelative, 85);
#endif
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
int MotorAxisController::numAxis() const
{
    return d->ui.tableMovement->rowCount();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setAxisUnit(int axisIndex, AxisUnit unit)
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (d->axisType[axisIndex] == TypeLinear && unit == UnitDeg)
        {
            unit = UnitMm;
        }
        else if (d->axisType[axisIndex] == TypeRotational && unit != UnitDeg)
        {
            unit = UnitDeg;
        }

        d->spinTargetPos[axisIndex]->setSuffix(suffixFromAxisUnit(unit));
        d->spinCurrentPos[axisIndex]->setSuffix(suffixFromAxisUnit(unit));
        d->spinStepSize[axisIndex]->setSuffix(suffixFromAxisUnit(unit));

        double baseUnitValue = unitToBaseUnit(d->spinTargetPos[axisIndex]->value(), d->axisUnit[axisIndex]);
        d->spinTargetPos[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));
        
        baseUnitValue = unitToBaseUnit(d->spinStepSize[axisIndex]->value(), d->axisUnit[axisIndex]);
        d->spinStepSize[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));

        baseUnitValue = unitToBaseUnit(d->spinCurrentPos[axisIndex]->value(), d->axisUnit[axisIndex]);
        d->spinCurrentPos[axisIndex]->setValue(baseUnitToUnit(baseUnitValue, unit));

        d->axisUnit[axisIndex] = unit;
    }
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
void MotorAxisController::setAxisType(int axisIndex, AxisType type)
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        if (d->axisType[axisIndex] != type)
        {
            d->axisType[axisIndex] = type;

            if (type == TypeLinear && axisUnit(axisIndex) == UnitDeg)
            {
                setAxisUnit(axisIndex, UnitMm);
            }
            else if (type == TypeRotational && axisUnit(axisIndex) != UnitDeg)
            {
                setAxisUnit(axisIndex, UnitDeg);
            }
        }
    }
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
void MotorAxisController::setDefaultRelativeStepSize(double defaultRelativeStepSize) /*in mm or degree*/
{
    if (d->defaultRelativeStepSize != defaultRelativeStepSize)
    {
        d->defaultRelativeStepSize = defaultRelativeStepSize;

        for (int i = 0; i < d->spinStepSize.size(); ++i)
        {
            d->spinStepSize[i]->setValue(this->baseUnitToUnit(defaultRelativeStepSize, d->axisUnit[i]));
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
double MotorAxisController::defaultRelativeStepSize() const
{
    return d->defaultRelativeStepSize;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::setAxisNames(const QStringList &names)
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
void MotorAxisController::setDecimals(int axisIndex, int decimals)
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
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
int MotorAxisController::decimals(int axisIndex) const
{
    if (axisIndex >= 0 && axisIndex < d->axisType.size())
    {
        return d->axisDecimals[axisIndex];
    }

    return 2;
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
            d->ui.btnStart->setVisible(true);
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
            d->ui.btnStart->setVisible(true);
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
    d->cancelAvailable = available;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
bool MotorAxisController::cancelAvailable() const
{
    return d->cancelAvailable;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void MotorAxisController::actuatorStatusChanged(QVector<int> status, QVector<double> actPosition) //!< slot to receive information about status and position changes.
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
        d->spinTargetPos[i]->setEnabled(status[i] & ito::actuatorEnabled);

        if (status[i] & ito::actuatorMoving)
        {
            style = "background-color: yellow";
            running = true;
            globalRunning = true;
        }
        else if (status[i] & ito::actuatorInterrupted)
        {
            style = "background-color: red";
        }
        else if (status[i] & ito::actuatorTimeout)
        {
            style = "background-color: #FFA3FD";
        }
        else
        {
            style = "background-color: ";
        }
        d->spinTargetPos[i]->setStyleSheet(style);
        d->spinCurrentPos[i]->setStyleSheet(style);
        d->spinStepSize[i]->setStyleSheet(style);
        d->buttonsRelative[i]->setEnabled(!running);
        d->buttonAbsolute[i]->setEnabled(!running);
    }

    d->ui.btnStart->setEnabled(!globalRunning);
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
        if (!QMetaObject::invokeMethod(d->actuator, "requestStatusAndPosition", Qt::QueuedConnection, Q_ARG(bool, true), Q_ARG(bool, true)))
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'requestStatusAndPosition' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }

    retValToMessageBox(retval, "refresh");
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
        retValToMessageBox(ito::RetVal(ito::retError, 0, "Actuator not available"), "interrupt movement");
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

        if (QMetaObject::invokeMethod(d->actuator, "setPosAbs", Q_ARG(const QVector<int>, axes), Q_ARG(QVector<double>, positions), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore());
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'setPosAbs' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, "Actuator not available");
    }

    retValToMessageBox(retval, "start movement");
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

        if (QMetaObject::invokeMethod(d->actuator, func, Q_ARG(const int, axis), Q_ARG(double, valueBase), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore());
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot '%s' could not be invoked since it does not exist.").arg(QLatin1String(func)).toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, "Actuator not available");
    }

    retValToMessageBox(retval, "start movement");
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MotorAxisController::observeInvocation(ItomSharedSemaphore *waitCond) const
{
    ito::RetVal retval;

    if (d->actuator)
    {
        bool timeout = false;

        while (!timeout && waitCond->waitAndProcessEvents(PLUGINWAIT) == false)
        {
            if (d->actuator->isAlive() == false)
            {
                retval += ito::RetVal(ito::retError, 0, tr("Timeout while waiting for answer from plugin instance.").toLatin1().data());
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
void MotorAxisController::retValToMessageBox(const ito::RetVal &retval, const QString &methodName) const
{
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
void MotorAxisController::customContextMenuRequested(const QPoint &pos)
{
    QModelIndex index = d->ui.tableMovement->indexAt(pos);
    if (index.isValid() && index.row() >= 0 && index.row() < d->ui.tableMovement->rowCount())
    {
        QMenu contextMenu;
        QAction *a;

        QMenu *unitMenu = contextMenu.addMenu(tr("Unit"));
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

            a = new QAction(QLatin1String("µm"), this);
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
            a = new QAction(QLatin1String("°"), this);
            a->setCheckable(true);
            a->setChecked(true);
            a->setData(UnitDeg);
            unitMenu->addAction(a);
        }


        QMenu *decimalsMenu = contextMenu.addMenu(tr("Decimals"));

        int dec = decimals(index.row());
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
                setDecimals(index.row(), data - 1000);
            }
            else
            {
                setAxisUnit(index.row(), (AxisUnit)data);
            }
        }
    }
}
