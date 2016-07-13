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

#ifndef MOTORAXISCONTROLLER_H
#define MOTORAXISCONTROLLER_H


#include "commonWidgets.h"
#include <qwidget.h>
#include <qstring.h>
#include <qpoint.h>
#include <qpointer.h>
#include <qstringlist.h>
#include "common/retVal.h"

class ItomSharedSemaphore; //forward declaration

class MotorAxisControllerPrivate; //forward declaration

namespace ito {
    class AddInActuator; //forward declaration
};

class ITOMWIDGETS_EXPORT MotorAxisController : public QWidget
{
    Q_OBJECT

    Q_ENUMS(AxisUnit)
    Q_ENUMS(AxisType)
    Q_ENUMS(MovementType)

    Q_PROPERTY(QPointer<ito::AddInActuator> actuator READ actuator WRITE setActuator)
    Q_PROPERTY(int numAxis READ numAxis WRITE setNumAxis)
    Q_PROPERTY(AxisUnit defaultAxisUnit READ defaultAxisUnit WRITE setDefaultAxisUnit)
    Q_PROPERTY(AxisType defaultAxisType READ defaultAxisType WRITE setDefaultAxisType)
    Q_PROPERTY(bool refreshAvailable READ refreshAvailable WRITE setRefreshAvailable)
    Q_PROPERTY(bool cancelAvailable READ cancelAvailable WRITE setCancelAvailable)
    Q_PROPERTY(double defaultRelativeStepSize READ defaultRelativeStepSize WRITE setDefaultRelativeStepSize)
    Q_PROPERTY(QStringList axisNames READ axisNames WRITE setAxisNames)
    Q_PROPERTY(int defaultDecimals READ defaultDecimals WRITE setDefaultDecimals)
    Q_PROPERTY(MovementType movementType READ movementType WRITE setMovementType)

public:
    enum AxisUnit {
        UnitNm = 0,
        UnitMum,
        UnitMm,
        UnitCm,
        UnitM,
        UnitDeg
    };

    enum AxisType {
        TypeRotational = 0,
        TypeLinear = 1
    };

    enum MovementType {
        MovementAbsolute = 0,
        MovementRelative = 1,
        MovementBoth = 2
    };

    MotorAxisController(QWidget *parent = NULL);
    ~MotorAxisController();

    void setActuator(const QPointer<ito::AddInActuator> &actuator);
    QPointer<ito::AddInActuator> actuator() const;

    void setNumAxis(int numAxis);
    int numAxis() const;

    AxisUnit axisUnit(int axisIndex) const;

    void setDefaultAxisUnit(AxisUnit unit);
    AxisUnit defaultAxisUnit() const;

    AxisType axisType(int axisIndex) const;

    void setDefaultAxisType(AxisType type);
    AxisType defaultAxisType() const;

    void setRefreshAvailable(bool available);
    bool refreshAvailable() const;

    void setCancelAvailable(bool available);
    bool cancelAvailable() const;

    void setDefaultRelativeStepSize(double defaultRelativeStepSize); /*in mm or degree*/
    double defaultRelativeStepSize() const;

    void setAxisNames(const QStringList &names);
    QStringList axisNames() const;

    QString axisName(int axisIndex) const;

    void setDefaultDecimals(int decimals);
    int defaultDecimals() const;

    int axisDecimals(int axisIndex) const;

    void setMovementType(MovementType type);
    MovementType movementType() const;

    bool axisEnabled(int axisIndex) const;

private:
    void retValToMessageBox(const ito::RetVal &retval, const QString &methodName) const;
    QString suffixFromAxisUnit(const AxisUnit &unit);
    double baseUnitToUnit(const double &value, const AxisUnit &unit);
    double unitToBaseUnit(const double &value, const AxisUnit &unit);
    ito::RetVal observeInvocation(ItomSharedSemaphore *waitCond) const;
    void moveRelOrAbs(int axis, double value, bool relNotAbs);

    
    MotorAxisControllerPrivate *d;

public slots:
    virtual void actuatorStatusChanged(QVector<int> status, QVector<double> actPosition);
    virtual void targetChanged(QVector<double> targetPositions);

    ito::RetVal setAxisUnit(int axisIndex, AxisUnit unit);
    ito::RetVal setAxisEnabled(int axisIndex, bool enabled);
    ito::RetVal setAxisDecimals(int axisIndex, int decimals);
    ito::RetVal setAxisType(int axisIndex, AxisType type);
    ito::RetVal setAxisName(int axisIndex, const QString &name);

private slots:
    void on_btnCancel_clicked();
    void on_btnStart_clicked();
    void on_btnRefresh_clicked();
    void on_comboType_currentIndexChanged(int index);
    void stepUpClicked(int index);
    void stepDownClicked(int index);
    void runSingleClicked(int index);
    void customContextMenuRequested(const QPoint &pos);
};

#endif
