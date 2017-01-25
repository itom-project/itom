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

#include <qtreeview.h>
#include <qscopedpointer.h>
#include <qpointer.h>

class ParamEditorModel;
class ParamEditorWidgetPrivate;

namespace ito {
	class AddInBase; //forward declaration
};

class ITOMWIDGETS_EXPORT ParamEditorWidget : public QTreeView
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

    void setSorted(bool value);

    bool sorted() const;

	QPointer<ito::AddInBase> plugin() const;
	void setPlugin(QPointer<ito::AddInBase> plugin);

protected:
    void mousePressEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);

private:
	QScopedPointer<ParamEditorWidgetPrivate> d_ptr;

	Q_DECLARE_PRIVATE(ParamEditorWidget);
	Q_DISABLE_COPY(ParamEditorWidget);

signals:

private slots :
    void sortedAction(bool checked);


};

#endif
