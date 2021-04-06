/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2018, Institut fuer Technische Optik (ITO), 
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

#ifndef DATAOBJECTTABLE_H
#define DATAOBJECTTABLE_H

#include "DataObject/dataobj.h"

#include <qtableview.h>
#include <qsharedpointer.h>
#include <qheaderview.h>

#include "commonWidgets.h"

class QKeyEvent;
class QContextMenuEvent;
class DataObjectModel;
class DataObjectDelegate;

class ITOMWIDGETS_EXPORT DataObjectTable : public QTableView
{
    Q_OBJECT

    Q_PROPERTY(QSharedPointer<ito::DataObject> data READ getData WRITE setData DESIGNABLE false);
    Q_PROPERTY(bool readOnly READ getReadOnly WRITE setReadOnly DESIGNABLE true);
    Q_PROPERTY(double min READ getMin WRITE setMin DESIGNABLE true);
    Q_PROPERTY(double max READ getMax WRITE setMax DESIGNABLE true);
    Q_PROPERTY(int decimals READ getDecimals WRITE setDecimals DESIGNABLE true);
    Q_PROPERTY(int editorDecimals READ getEditorDecimals WRITE setEditorDecimals DESIGNABLE true);
    Q_PROPERTY(int defaultCols READ getDefaultCols WRITE setDefaultCols DESIGNABLE true);
    Q_PROPERTY(int defaultRows READ getDefaultRows WRITE setDefaultRows DESIGNABLE true);
    Q_PROPERTY(QStringList horizontalLabels READ getHorizontalLabels WRITE setHorizontalLabels DESIGNABLE true);
    Q_PROPERTY(QStringList verticalLabels READ getVerticalLabels WRITE setVerticalLabels DESIGNABLE true);
    Q_PROPERTY(QStringList suffixes READ getSuffixes WRITE setSuffixes DESIGNABLE true);
    Q_PROPERTY(QHeaderView::ResizeMode horizontalResizeMode READ getHorizontalResizeMode WRITE setHorizontalResizeMode DESIGNABLE true);
    Q_PROPERTY(QHeaderView::ResizeMode verticalResizeMode READ getVerticalResizeMode WRITE setVerticalResizeMode DESIGNABLE true);
    Q_PROPERTY(Qt::Alignment alignment READ getAlignment WRITE setAlignment DESIGNABLE true);
    

    Q_CLASSINFO("prop://data", "dataObject that is displaye in the table view");
    Q_CLASSINFO("prop://readOnly", "enable write protection");
    Q_CLASSINFO("prop://min", "minimum acceptable value (if editing is allowed)");
    Q_CLASSINFO("prop://max", "maximum acceptable value (if editing is allowed)");
    Q_CLASSINFO("prop://decimals", "number of visible decimals for floating point numbers");
    Q_CLASSINFO("prop://editorDecimals", "number of possible decimals during the edit of floating point numbers");
    Q_CLASSINFO("prop://defaultCols", "number of column to be shown");
    Q_CLASSINFO("prop://defaultRows", "number of rows to be shown");
    Q_CLASSINFO("prop://horizontalLabels", "list with labels for each shown column (if more columns are shown than labels, a default numbering is used for additional columns)");
    Q_CLASSINFO("prop://verticalLabels", "list with labels for each shown row (if more rows are shown than labels, a default numbering is used for additional rows)");
    Q_CLASSINFO("prop://suffixes", "list with suffixes for each column. If less suffixes than columns are indicated, the last suffix is repeated.");
    Q_CLASSINFO("prop://horizontalResizeMode", "defines the mode how the rows can be resized or are stretched over the available space (ResizeToContents, Interactive, Stretch, Fixed, Custom -> see QHeaderView::ResizeMode).");
    Q_CLASSINFO("prop://verticalResizeMode", "defines the mode how the columns can be resized or are stretched over the available space (ResizeToContents, Interactive, Stretch, Fixed, Custom -> see QHeaderView::ResizeMode).");
    Q_CLASSINFO("prop://alignment", "alignment of the text cells.");

    Q_CLASSINFO("signal://activated", "signal emitted if a cell is activated. Arguments are (row,column) of the cell.")
    Q_CLASSINFO("signal://clicked", "signal emitted if a cell is clicked by the mouse. Arguments are (row,column) of the cell.")
    Q_CLASSINFO("signal://doubleClicked", "signal emitted if a cell is double clicked by the mouse. Arguments are (row,column) of the cell.")
    Q_CLASSINFO("signal://entered", "signal emitted if a cell is entered by the mouse cursor. Arguments are (row,column) of the cell. Property 'mouseTracking' needs to be enabled for this feature to work.")
    Q_CLASSINFO("signal://pressed", "signal emitted if a cell if the mouse is pressed on a cell. Arguments are (row,column) of the cell.")

public:
    DataObjectTable(QWidget *parent = 0);
    ~DataObjectTable();

    void setData(QSharedPointer<ito::DataObject> dataObj);
    QSharedPointer<ito::DataObject> getData() const;

    bool getReadOnly() const;
    void setReadOnly(bool value);

    double getMin() const;
    void setMin(double value);

    double getMax() const;
    void setMax(double value);

    int getDecimals() const;
    void setDecimals(int value);

    Qt::Alignment getAlignment() const;
    void setAlignment(Qt::Alignment alignment);

    int getEditorDecimals() const;
    void setEditorDecimals(int value);

    QHeaderView::ResizeMode getHorizontalResizeMode() const;
    void setHorizontalResizeMode(QHeaderView::ResizeMode mode);

    QHeaderView::ResizeMode getVerticalResizeMode() const;
    void setVerticalResizeMode(QHeaderView::ResizeMode mode);

    int getDefaultCols() const;
    void setDefaultCols(int value);

    int getDefaultRows() const;
    void setDefaultRows(int value);

    QStringList getVerticalLabels() const;
    void setVerticalLabels(QStringList value);

    QStringList getHorizontalLabels() const;
    void setHorizontalLabels(QStringList value);

    QStringList getSuffixes() const;
    void setSuffixes(QStringList value);

    virtual QSize sizeHint() const;


protected:
    DataObjectModel *m_pModel;
    DataObjectDelegate *m_pDelegate;

    void keyPressEvent(QKeyEvent *e);
    void contextMenuEvent(QContextMenuEvent *event);

private:

private slots:
    inline void _activated (const QModelIndex &index) { emit activated(index.row(), index.column()); }
    inline void _clicked (const QModelIndex &index) { emit clicked(index.row(), index.column()); }
    inline void _doubleClicked (const QModelIndex &index) { emit doubleClicked(index.row(), index.column()); }
    inline void _entered (const QModelIndex &index) { emit entered(index.row(), index.column()); }
    inline void _pressed (const QModelIndex &index) { emit pressed(index.row(), index.column()); }
    void copySelectionToClipboard();
    void copyAllToClipboard();
    void setDecimalsGUI();

signals:
    void activated (int row, int column);
    void clicked (int row, int column);
    void doubleClicked (int row, int column);
    void entered (int row, int column);
    void pressed (int row, int column);
};

#endif