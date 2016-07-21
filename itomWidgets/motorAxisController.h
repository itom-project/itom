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

#if QT_VERSION < 0x050500
    //for >= Qt 5.5.0 see Q_ENUM definition below
    Q_ENUMS(AxisUnit)
    Q_ENUMS(AxisType)
    Q_ENUMS(MovementType)
#endif

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
    Q_PROPERTY(bool movementTypeVisible READ movementTypeVisible WRITE setMovementTypeVisible)
    Q_PROPERTY(QString arbitraryUnit READ arbitraryUnit WRITE setArbitraryUnit)

public:
    enum AxisUnit {
        UnitNm = 0,
        UnitMum,
        UnitMm,
        UnitCm,
        UnitM,
        UnitDeg,
        UnitAU /*Arbitrary unit*/
    };

    enum AxisType {
        TypeRotational = 0,
        TypeLinear = 1
    };

    enum MovementType {
        MovementAbsolute = 0,
        MovementRelative = 1,
        MovementBoth = 2,
        MovementNo = 3
    };

#if QT_VERSION >= 0x050500
    //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
    //values are always accessible.
    Q_ENUM(AxisUnit);
    Q_ENUM(AxisType);
    Q_ENUM(MovementType);
#endif

    MotorAxisController(QWidget *parent = NULL);
    ~MotorAxisController();

    void setActuator(const QPointer<ito::AddInActuator> &actuator);
    QPointer<ito::AddInActuator> actuator() const;

    int numAxis() const;
    AxisUnit axisUnit(int axisIndex) const;
    AxisUnit defaultAxisUnit() const;
    AxisType axisType(int axisIndex) const;
    AxisType defaultAxisType() const;
    bool refreshAvailable() const;
    bool cancelAvailable() const;
    double defaultRelativeStepSize() const;
    QStringList axisNames() const;
    QString axisName(int axisIndex) const;
    int defaultDecimals() const;
    int axisDecimals(int axisIndex) const;
    MovementType movementType() const;
    bool movementTypeVisible() const;
    bool axisEnabled(int axisIndex) const;
    QString arbitraryUnit() const;

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

    void setDefaultAxisUnit(AxisUnit unit);
    void setMovementTypeVisible(bool visible);
    void setMovementType(MovementType type);
    void setDefaultDecimals(int decimals);
    void setAxisNames(const QStringList &names);
    void setDefaultRelativeStepSize(double defaultRelativeStepSize); /*in mm or degree*/
    void setCancelAvailable(bool available);
    void setRefreshAvailable(bool available);
    void setDefaultAxisType(AxisType type);
    void setNumAxis(int numAxis);
    void setArbitraryUnit(const QString &unit);

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
