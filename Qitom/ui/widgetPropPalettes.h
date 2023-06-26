/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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
            : QObject(parent), QGraphicsPathItem(),
            m_parentWidget(parentWidget), m_colChannel(colChannel),
            m_editable(true)
        {
        }

        int getColChannel() const { return m_colChannel; }

        void setEditable(bool editable) { m_editable = editable; }
        bool editable() const { return m_editable; }

        void setActiveSceneSize(const QSizeF size) { m_activeSceneSize = size; }

    private:
        int m_colChannel;
        WidgetPropPalettes *m_parentWidget;
        QPointF m_insertPos;
        bool m_editable;

        QSizeF m_activeSceneSize;

    protected:
        void mousePressEvent(QGraphicsSceneMouseEvent*);
        void mouseMoveEvent(QGraphicsSceneMouseEvent*);
        void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);

    protected slots:
        void removeDataPoint();
        void addDataPoint();
};

//----------------------------------------------------------------------------------------------------------------------------------
class WidgetPropPalettes : public AbstractPropertyPageWidget
{
    Q_OBJECT

    public:
        WidgetPropPalettes(QWidget *parent = NULL);
        ~WidgetPropPalettes();

        void readSettings();
        void writeSettings();
        const ItomPaletteBase* getCurPalette() const { return &m_currentPalette; }
        void drawPalCurves(int selPt = -1);
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

        float m_gvSceneMarginLeftRight;
        float m_gvPaletteSceneMarginTopBottom;
        float m_gvCurveSceneMarginTopBottom;

        void updatePaletteList();

        void updateViewOnResize();

        QList<ito::ItomPaletteBase> m_palettes; //all existing palettes
        int m_curPaletteIndex; //index of current palette in m_palettes or -1 if new and not saved, yet
        ito::ItomPaletteBase m_currentPalette;

        QGraphicsScene *m_pSceneCurPalette;
        QGraphicsScene *m_pScenePalCurves;

    public slots:
        void lwCurrentRowChanged(int row);
        void colorComponentChanged(int value);
        void colorComponentVisibilityChanged(bool);

        void palSpecialColorChanged(QColor color);

    private slots:
        void on_sbIndex_valueChanged(double value);
        void on_pbAdd_clicked();
        void on_pbDuplicate_clicked();
        void on_pbRemove_clicked();
        void on_pbPalSave_clicked();
        void on_pbEquidistantColorStop_clicked();
        void on_pbRemoveColorStop_clicked();
        void on_pbAddColorStop_clicked();
        void on_lePalName_textChanged(const QString & text);
        void on_pbImportPalette_clicked();
        void on_pbExportPalette_clicked();
        void on_btnColor_colorChanged(QColor color);

    protected:
        bool eventFilter(QObject *obj, QEvent *event);
};

}//end namespace
#endif
