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

#ifndef MOTORAXISCONTROLLER_H
#define MOTORAXISCONTROLLER_H


#include "commonWidgets.h"
#include <qwidget.h>
#include <qstring.h>
#include <qpoint.h>
#include <qpointer.h>
#include <qcolor.h>
#include <qstringlist.h>
#include "common/retVal.h"
#include "common/interval.h"

#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
// https://stackoverflow.com/questions/66581395/q-property-must-be-fully-defined-error-in-qt-6
Q_MOC_INCLUDE("common/addInInterface.h")
#endif

class ItomSharedSemaphore; //forward declaration

class MotorAxisControllerPrivate; //forward declaration

namespace ito {
    class AddInActuator; //forward declaration

};

class ITOMWIDGETS_EXPORT MotorAxisController : public QWidget
{
    Q_OBJECT

    Q_PROPERTY(QPointer<ito::AddInActuator> actuator READ actuator WRITE setActuator)
    Q_PROPERTY(int numAxis READ numAxis WRITE setNumAxis)
    Q_PROPERTY(AxisUnit defaultAxisUnit READ defaultAxisUnit WRITE setDefaultAxisUnit)
    Q_PROPERTY(AxisType defaultAxisType READ defaultAxisType WRITE setDefaultAxisType)
    Q_PROPERTY(bool refreshAvailable READ refreshAvailable WRITE setRefreshAvailable)
    Q_PROPERTY(bool cancelAvailable READ cancelAvailable WRITE setCancelAvailable)
    Q_PROPERTY(bool startAllAvailable READ startAllAvailable WRITE setStartAllAvailable)
    Q_PROPERTY(double defaultRelativeStepSize READ defaultRelativeStepSize WRITE setDefaultRelativeStepSize)
    Q_PROPERTY(QStringList axisNames READ axisNames WRITE setAxisNames)
    Q_PROPERTY(int defaultDecimals READ defaultDecimals WRITE setDefaultDecimals)
    Q_PROPERTY(MovementType movementType READ movementType WRITE setMovementType)
    Q_PROPERTY(bool movementTypeVisible READ movementTypeVisible WRITE setMovementTypeVisible)
    Q_PROPERTY(QString arbitraryUnit READ arbitraryUnit WRITE setArbitraryUnit)
    Q_PROPERTY(QColor backgroundColorMoving READ backgroundColorMoving WRITE setBackgroundColorMoving)
    Q_PROPERTY(QColor backgroundColorInterrupted READ backgroundColorInterrupted WRITE setBackgroundColorInterrupted)
    Q_PROPERTY(QColor backgroundColorTimeout READ backgroundColorTimeout WRITE setBackgroundColorTimeout)

    Q_CLASSINFO("prop://actuator", "Actuator instance that is monitored and controlled by this widget (or None in order to remove a previous actuator).")
    Q_CLASSINFO("prop://numAxis", "Number of axes that are monitored.")
    Q_CLASSINFO("prop://defaultAxisUnit", "Default unit for all axes. A different unit can be set for distinct axes using the slot 'setAxisUnit'.")
    Q_CLASSINFO("prop://defaultAxisType", "Default type for all axes. A different type can be set for any axis using the slot 'setAxisType'.")
    Q_CLASSINFO("prop://refreshAvailable", "Hide or show a button to manually refresh the positions of all covered axes.")
    Q_CLASSINFO("prop://cancelAvailable", "Hide or show a button to cancel a running movement of any axis (should only be used, if the specific actuator is able to handle interrupts).")
    Q_CLASSINFO("prop://startAllAvailable", "Hide or show a button to start a simultaneous movement of all covered axes to their current target positions.")
    Q_CLASSINFO("prop://defaultRelativeStepSize", "Default relative step size for all axes (in mm or degree, depending on their types).")
    Q_CLASSINFO("prop://axisNames", "Names of all axes as string list.")
    Q_CLASSINFO("prop://defaultDecimals", "Default number of decimals of all axes. The number of decimals can also be set individually for each axis using the slot 'setAxisDecimals'.")
    Q_CLASSINFO("prop://movementType", "Style of the widget depending if it should be optimized for an absolute movement, relative movement, both or no movement.")
    Q_CLASSINFO("prop://movementTypeVisible", "Hide or show a combobox above the axes values that can be used to select an appropriate movement type.")
    Q_CLASSINFO("prop://arbitraryUnit", "Unit name that is used for axes, whose unit is set to UnitAU (Arbitrary unit).")
    Q_CLASSINFO("prop://backgroundColorMoving", "Background color for spinboxes of axes that are currently moving.")
    Q_CLASSINFO("prop://backgroundColorInterrupted", "Background color for spinboxes of axes that were interrupted.")
    Q_CLASSINFO("prop://backgroundColorTimeout", "Background color for spinboxes of axes that run into a timeout.")

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

    //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
    //values are always accessible.
    Q_ENUM(AxisUnit);
    Q_ENUM(AxisType);
    Q_ENUM(MovementType);

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
    bool startAllAvailable() const;
    double defaultRelativeStepSize() const;
    QStringList axisNames() const;
    QString axisName(int axisIndex) const;
    int defaultDecimals() const;
    int axisDecimals(int axisIndex) const;
    MovementType movementType() const;
    bool movementTypeVisible() const;
    bool axisEnabled(int axisIndex) const;
    QString arbitraryUnit() const;

    QColor backgroundColorMoving() const;
    void setBackgroundColorMoving(const QColor &color);

    QColor backgroundColorInterrupted() const;
    void setBackgroundColorInterrupted(const QColor &color);

    QColor backgroundColorTimeout() const;
    void setBackgroundColorTimeout(const QColor &color);

private:
    void retValToMessageBox(const ito::RetVal &retval, const QString &methodName) const;
    QString suffixFromAxisUnit(const AxisUnit &unit) const;
    double baseUnitToUnit(const double &value, const AxisUnit &unit) const;
    double unitToBaseUnit(const double &value, const AxisUnit &unit) const;
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
    void setStartAllAvailable(bool available);
    void setRefreshAvailable(bool available);
    void setDefaultAxisType(AxisType type);
    void setNumAxis(int numAxis);
    void setArbitraryUnit(const QString &unit);

    ito::AutoInterval stepSizeInterval(int axisIndex) const;
    ito::AutoInterval targetInterval(int axisIndex) const;

    ito::RetVal setStepSizeInterval(int axisIndex, const ito::AutoInterval &interval);
    ito::RetVal setTargetInterval(int axisIndex, const ito::AutoInterval &interval);


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
