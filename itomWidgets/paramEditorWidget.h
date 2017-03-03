/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2017, Institut fuer Technische Optik (ITO),
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

#ifndef PARAMEDITORWIDGET_H
#define PARAMEDITORWIDGET_H


#include "commonWidgets.h"

#include "common/sharedStructuresQt.h"
#include "common/param.h"

#include <qwidget.h>
#include <qscopedpointer.h>
#include <qpointer.h>
#include <qvector.h>

class ParamEditorModel;
class ParamEditorWidgetPrivate;
class QtProperty;
class QTimerEvent;

namespace ito {
	class AddInBase; //forward declaration
};

class ITOMWIDGETS_EXPORT ParamEditorWidget : public QWidget
{
    Q_OBJECT

	Q_PROPERTY(QPointer<ito::AddInBase> plugin READ plugin WRITE setPlugin)

public:
    /**
     * \brief Constructor 
     *
     * Creates a new editor widget based on QTreeView
     * @param parent optional parent widget
     */
	ParamEditorWidget(QWidget* parent = 0);
    
    /// Destructor
	virtual ~ParamEditorWidget();

	QPointer<ito::AddInBase> plugin() const;
	void setPlugin(QPointer<ito::AddInBase> plugin);

protected:
    /**
    * MessageLevel enumeration
    * defines whether warnings and/or errors that might occur during some executions should be displayed with a message box.
    */
    enum MessageLevel 
    {
        msgLevelNo = 0,          /*!< no messagebox should information about warnings or errors */
        msgLevelErrorOnly = 1,   /*!< a message box should only inform about errors */
        msgLevelWarningOnly = 2, /*!< a message box should only inform about warnings */
        msgLevelWarningAndError = msgLevelErrorOnly | msgLevelWarningOnly /*!< a message box should inform about warnings and errors */
    };

    ito::RetVal setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel = msgLevelWarningAndError) const;
    ito::RetVal setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel = msgLevelWarningAndError) const;
    ito::RetVal observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const;

    ito::RetVal addParam(const ito::Param &param);
    ito::RetVal addParamInt(const ito::Param &param, QtProperty *groupProperty);

protected:
    void timerEvent(QTimerEvent *event);

private:
	QScopedPointer<ParamEditorWidgetPrivate> d_ptr;

	Q_DECLARE_PRIVATE(ParamEditorWidget);
	Q_DISABLE_COPY(ParamEditorWidget);

private slots:
    void valueChanged(QtProperty* prop, int value);
    void valueChanged(QtProperty* prop, bool value);

    void parametersChanged(QMap<QString, ito::Param> parameters);


};

#endif
