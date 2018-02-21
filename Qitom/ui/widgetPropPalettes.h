/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
        ColCurve(WidgetPropPalettes *parentWidget, int colChannel, QObject *parent = NULL) 
            : QObject(parent), QGraphicsPathItem(), m_parentWidget(parentWidget), m_colChannel(colChannel) 
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
        const ItomPaletteBase* getCurPalette() const { return &m_currentPalette; }
        void drawPalCurves(int selPt = -1, int x0 = 5, int y0 = 5, int dx = 10, int dy = 10);
        void updateOptionPalette();

        void removeColorStop(int index);
        void addColorStop(int index_before, float percent_to_next = 0.5);
        void changeSelectedColorStop(int new_index);
        int getSelectedColorStop() const { return m_selectedColorStop; }


        ito::RetVal saveCurrentPalette();

        friend class ColCurve;

    private:
        Ui::WidgetPropPalettes ui;
        QImage m_imgGVCurPalette;
        int m_selectedColorStop;
        int m_isUpdating;
        int m_isDirty;

        void updatePaletteList();

        QList<ito::ItomPaletteBase> m_palettes; //all existing palettes
        int m_curPaletteIndex; //index of current palette in m_palettes or -1 if new and not saved, yet
        ito::ItomPaletteBase m_currentPalette;

    public slots:
        void lwCurrentRowChanged(int row);
        void colorComponentChanged(int value);
        void colorComponentVisibilityChanged(bool);
        
        void palSpecialColorChanged(QColor color);

    private slots :
        void on_sbIndex_valueChanged(double value);
        void on_pbAdd_clicked();
        void on_pbDuplicate_clicked();
        void on_pbRemove_clicked();
        void on_pbPalSave_clicked();
        void resizeEvent(QResizeEvent *event);
        void on_pbEquidistantColorStop_clicked();
        void on_pbRemoveColorStop_clicked();
        void on_pbAddColorStop_clicked();
        void on_lePalName_textChanged(const QString & text);
        void on_pbImportPalette_clicked();
        void on_pbExportPalette_clicked();
};

}//end namespace
#endif