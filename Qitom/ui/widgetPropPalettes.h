/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2016, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public Licence as published by
the Free Software Foundation; either version 2 of the Licence, or (at
your option) any later version.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public Licence for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */
#ifndef WIDGETPROPPALETTES_H
#define WIDGETPROPPALETTES_H

#include "abstractPropertyPageWidget.h"
#include <qwidget.h>
#include <qgraphicsitem.h>

#include "ui_widgetPropPalettes.h"
#include "../organizer/paletteOrganizer.h"

namespace ito
{  

class WidgetPropPalettes; //!> forward declaration

//----------------------------------------------------------------------------------------------------------------------------------
class ColCurve : public QObject, public QGraphicsPathItem
{
    Q_OBJECT

    public:
        ColCurve(WidgetPropPalettes *parentWidget, int colChannel) 
            : QObject(), QGraphicsPathItem(), m_parentWidget(parentWidget), m_colChannel(colChannel) 
        { }
        int getColChannel() { return m_colChannel; }

    private:
        int m_colChannel;
        WidgetPropPalettes *m_parentWidget;
        QPointF m_insertPos;

    protected slots:
        void mousePressEvent(QGraphicsSceneMouseEvent*);
        void mouseMoveEvent(QGraphicsSceneMouseEvent*);
        void mouseReleaseEvent(QGraphicsSceneMouseEvent*);
        void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);
        void removeDataPoint();
        void addDataPoint();
};

//----------------------------------------------------------------------------------------------------------------------------------
class WidgetPropPalettes : public AbstractPropertyPageWidget
{
    Q_OBJECT

    public:
        WidgetPropPalettes(QWidget *parent = NULL);
        void readSettings();
        void writeSettings();
        //void setCurPaletteCols(QVector<uint> palCols) { m_curCols = palCols; }
        //QVector<QGradientStop>* getCurPalData() { return &m_curPalData; }
        ItomPaletteBase* getCurPalette() { return &m_curPalette; }
        void drawPalCurves(int selPt = -1, int x0 = 5, int y0 = 5, int dx = 10, int dy = 10);
        void updatePalette();

        friend class ColCurve;

    private:
        Ui::WidgetPropPalettes ui;
        QImage m_imgGVCurPalette;
        //QVector<uint> m_curCols;
        //QVector<QGradientStop> m_curPalData;
        ItomPaletteBase m_curPalette;
        int m_selPt;
        int m_isUpdating;
        int m_isDirty;

        void updatePaletteList();

    public slots:
        void lwCurrentRowChanged(int row);
        void mousePressEvent(QMouseEvent* event);
        void sbValueChanged(int value);
        void pbColToggled(bool);
        void pbAddClicked();
        void pbRemoveClicked();
        void pbSaveClicked();
        void palSpecialColorChanged(QColor color);
        //void gvPalCurvesMouseMove(QEvent *event);
        //void widgetResize(QResizeEvent * event);

    private slots :
        void on_defaultBtn_clicked();
        void resizeEvent(QResizeEvent *event);
};

}//end namespace
#endif