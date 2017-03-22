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
class QtBrowserItem;
class QTimerEvent;

namespace ito {
	class AddInBase; //forward declaration
};

class ITOMWIDGETS_EXPORT ParamEditorWidget : public QWidget
{
    Q_OBJECT

#if QT_VERSION < 0x050500
    //for >= Qt 5.5.0 see Q_ENUM definition below
    Q_ENUMS(ResizeMode)
#endif
	Q_PROPERTY(QPointer<ito::AddInBase> plugin READ plugin WRITE setPlugin)
    Q_PROPERTY(int indentation READ indentation WRITE setIndentation)
    Q_PROPERTY(bool rootIsDecorated READ rootIsDecorated WRITE setRootIsDecorated)
    Q_PROPERTY(bool alternatingRowColors READ alternatingRowColors WRITE setAlternatingRowColors)
    Q_PROPERTY(bool headerVisible READ isHeaderVisible WRITE setHeaderVisible)
    Q_PROPERTY(ResizeMode resizeMode READ resizeMode WRITE setResizeMode)
    Q_PROPERTY(int splitterPosition READ splitterPosition WRITE setSplitterPosition)
    Q_PROPERTY(bool propertiesWithoutValueMarked READ propertiesWithoutValueMarked WRITE setPropertiesWithoutValueMarked)
    Q_PROPERTY(bool readonly READ readonly WRITE setReadonly)
    Q_PROPERTY(bool showDescriptions READ showDescriptions WRITE setShowDescriptions)
    Q_PROPERTY(QStringList filteredCategories READ filteredCategories WRITE setFilteredCategories)

public:
    enum ResizeMode
    {
        Interactive,
        Stretch,
        Fixed,
        ResizeToContents
    };

#if QT_VERSION >= 0x050500
    //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
    //values are always accessible.
    Q_ENUM(ResizeMode)
#endif

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

    void refresh();

    int indentation() const;
    void setIndentation(int i);

    bool rootIsDecorated() const;
    void setRootIsDecorated(bool show);

    bool alternatingRowColors() const;
    void setAlternatingRowColors(bool enable);

    bool readonly() const;
    void setReadonly(bool enable);

    bool isHeaderVisible() const;
    void setHeaderVisible(bool visible);

    ResizeMode resizeMode() const;
    void setResizeMode(ResizeMode mode);

    int splitterPosition() const;
    void setSplitterPosition(int position);

    void setPropertiesWithoutValueMarked(bool mark);
    bool propertiesWithoutValueMarked() const;

    void setShowDescriptions(bool show);
    bool showDescriptions() const;

    void setFilteredCategories(const QStringList &filteredCategories);
    QStringList filteredCategories() const;

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
    ito::RetVal addParamChar(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamDouble(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamString(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamOthers(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamInterval(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamRect(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamIntArray(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamCharArray(const ito::Param &param, QtProperty *groupProperty);
    ito::RetVal addParamDoubleArray(const ito::Param &param, QtProperty *groupProperty);

    ito::RetVal loadPlugin(QPointer<ito::AddInBase> plugin);

protected:
    void timerEvent(QTimerEvent *event);

private:
	QScopedPointer<ParamEditorWidgetPrivate> d_ptr;

	Q_DECLARE_PRIVATE(ParamEditorWidget);
	Q_DISABLE_COPY(ParamEditorWidget);

private slots:
    void valueChanged(QtProperty* prop, int value);
    void valueChanged(QtProperty* prop, char value);
    void valueChanged(QtProperty* prop, double value);
    void valueChanged(QtProperty* prop, int num, const char* values);
    void valueChanged(QtProperty* prop, int num, const int* values);
    void valueChanged(QtProperty* prop, int num, const double* values);
    void valueChanged(QtProperty* prop, const QByteArray &value);
    void valueChanged(QtProperty* prop, int min, int max);
    void valueChanged(QtProperty* prop, int left, int top, int width, int height);

    void currentItemChanged(QtBrowserItem *);

    void parametersChanged(QMap<QString, ito::Param> parameters);


};

#endif