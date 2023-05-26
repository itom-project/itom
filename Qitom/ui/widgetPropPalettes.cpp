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
#include "widgetPropPalettes.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../organizer/paletteOrganizer.h"
#include "../helper/guiHelper.h"

#include <qsettings.h>
#include <qgraphicsitem.h>
#include <qgraphicswidget.h>
#include <qgraphicssceneevent.h>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qinputdialog.h>
#include <qfiledialog.h>
#include <qtimer.h>

namespace ito
{

// --------------------------------------------------------------------------------------------------------------------
void setColorButton(const QColor &color, ColorPickerButton *btn, QLabel *lbl, QLabel *ico)
{
    btn->setColor(color);

    QString text = QString("#%1%2%3").arg(color.red(), 2, 16, QLatin1Char('0')) \
        .arg(color.green(), 2, 16, QLatin1Char('0')).arg(color.blue(), 2, 16, QLatin1Char('0'));
    lbl->setText(text);

    int _iconSize = ico->style()->pixelMetric(QStyle::PM_SmallIconSize);
    QPixmap pix(_iconSize, _iconSize);
    pix.fill(color.isValid() ?
        ico->palette().button().color() : Qt::transparent);
    QPainter p(&pix);
    p.setPen(QPen(Qt::gray));
    p.setBrush(color.isValid() ?
        color : QBrush(Qt::NoBrush));
    p.drawRect(2, 2, pix.width() - 5, pix.height() - 5);

    ico->setPixmap(pix);
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (!m_editable)
    {
        event->ignore();
        return;
    }

    event->accept();

    if (m_parentWidget)
    {
        int selectedColorStop = m_parentWidget->getSelectedColorStop();
        QVector<QGradientStop> pts = m_parentWidget->m_currentPalette.getColorStops();

        if (selectedColorStop >= 0 && selectedColorStop < pts.length() && event->buttons() == Qt::LeftButton)
        {
            m_parentWidget->m_isDirty = 1;

            QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());
            float posXRel = qBound(0.0, event->lastScenePos().x() / m_activeSceneSize.width(), 1.0);

            //define x-coordinate of gradient
            if (selectedColorStop == (pts.length() - 1))
            {
                //last point stays at pos 1
                pts[selectedColorStop].first = 1.0;
            }
            else if (selectedColorStop == 0)
            {
                //first point stays at pos 0
                pts[selectedColorStop].first = 0.0;
            }
            else if (selectedColorStop > 0 && posXRel <= pts[selectedColorStop - 1].first)
            {
                pts[selectedColorStop].first = (pts[selectedColorStop - 1].first);
            }
            else if (selectedColorStop < (pts.length() - 1) && posXRel >= pts[selectedColorStop + 1].first)
            {
                pts[selectedColorStop].first = pts[selectedColorStop + 1].first;
            }
            else
            {
                pts[selectedColorStop].first = posXRel;
            }

            int colVal = 255 - qBound(0.0, 255.0 * event->lastScenePos().y() / m_activeSceneSize.height(), 255.0);

            QColor currentColor(pts[selectedColorStop].second);

            switch (m_colChannel)
            {
            case 0: //red
                currentColor.setRed(colVal);
                break;
            case 1: //green
                currentColor.setGreen(colVal);
                break;
            case 2: //blue
                currentColor.setBlue(colVal);
                break;
            }

            pts[selectedColorStop].second = currentColor;

            m_parentWidget->m_isUpdating = 1;
            m_parentWidget->m_currentPalette.setColorStops(pts);
            m_parentWidget->drawPalCurves(selectedColorStop);
            m_parentWidget->updateOptionPalette();
            m_parentWidget->m_isUpdating = 0;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

    double minDist = 1e208;
    int thisPt = -1;

    QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());

    if (m_parentWidget && par)
    {
        const QGradientStops &pts = m_parentWidget->m_currentPalette.getColorStops();
        float scaleX = m_activeSceneSize.width();
        float scaleY = m_activeSceneSize.height();
        QPointF cursorPos = event->scenePos();

        for (int ne = 0; ne < pts.length(); ne++)
        {
            QLineF l;

            switch (m_colChannel)
            {
            case 0:
                l = QLineF(QPointF(pts[ne].first * scaleX, m_activeSceneSize.height() - pts[ne].second.redF() * scaleY), cursorPos);
                break;
            case 1:
                l = QLineF(QPointF(pts[ne].first * scaleX, m_activeSceneSize.height() - pts[ne].second.greenF() * scaleY), cursorPos);
                break;
            case 2:
                l = QLineF(QPointF(pts[ne].first * scaleX, m_activeSceneSize.height() - pts[ne].second.blueF() * scaleY), cursorPos);
                break;
            default:
                continue;
            }

            if (l.length() < minDist)
            {
                minDist = l.length();
                thisPt = ne;
            }
        }

        if (minDist < (20 * dpiFactor))
        {
            m_parentWidget->changeSelectedColorStop(thisPt);
            event->accept();
        }
        else
        {
            m_parentWidget->changeSelectedColorStop(-1);
            event->ignore();
        }
    }
    else if (m_parentWidget)
    {
        m_parentWidget->changeSelectedColorStop(-1);
        event->ignore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    if (!m_editable)
    {
        event->ignore();
        return;
    }

    event->accept();

    QMenu contextMenu(tr("Context menu"), m_parentWidget);
    QAction *action;

    int numColorStops = m_parentWidget->m_currentPalette.getColorStops().size();
    int selectedColorStop = m_parentWidget->getSelectedColorStop();

    action = new QAction(tr("Delete Color Stop"), this);
    connect(action, SIGNAL(triggered()), this, SLOT(removeDataPoint()));
    contextMenu.addAction(action);
    action->setEnabled(selectedColorStop> 0 && selectedColorStop < (numColorStops - 1));

    action = new QAction(tr("Add color stop"), this);
    connect(action, SIGNAL(triggered()), this, SLOT(addDataPoint()));
    contextMenu.addAction(action);

    m_insertPos = event->scenePos();
    contextMenu.exec(event->screenPos());
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::removeDataPoint()
{
    if (m_parentWidget)
    {
        m_parentWidget->removeColorStop(m_parentWidget->getSelectedColorStop());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::addDataPoint()
{
    QGraphicsView *view = qobject_cast<QGraphicsView*>(parent());

    if (m_parentWidget && view)
    {
        float scaleX = view->size().width();
        QGradientStops pts = m_parentWidget->getCurPalette()->getColorStops();
        float xpos = m_insertPos.x() / scaleX;

        int first = 0;

        while (pts[first].first <= xpos && first < pts.length())
        {
            first++;
        }

        first--;

        if (first >= 0)
        {
            m_parentWidget->addColorStop(first, (xpos - pts[first].first) / (pts[first + 1].first - pts[first].first));
        }
    }
}





//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPalettes::WidgetPropPalettes(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_selectedColorStop(-1),
    m_isUpdating(0),
    m_isDirty(0),
    m_curPaletteIndex(-1),
    m_pSceneCurPalette(NULL),
    m_pScenePalCurves(NULL),
    m_gvSceneMarginLeftRight(5),
    m_gvPaletteSceneMarginTopBottom(1),
    m_gvCurveSceneMarginTopBottom(5)
{
    ui.setupUi(this);

    connect(ui.lwPalettes, &QListWidget::currentRowChanged, this, &WidgetPropPalettes::lwCurrentRowChanged);
    connect(ui.sbR, SIGNAL(valueChanged(int)), this, SLOT(colorComponentChanged(int)));
    connect(ui.sbG, SIGNAL(valueChanged(int)), this, SLOT(colorComponentChanged(int)));
    connect(ui.sbB, SIGNAL(valueChanged(int)), this, SLOT(colorComponentChanged(int)));
    connect(ui.pbR, SIGNAL(toggled(bool)), this, SLOT(colorComponentVisibilityChanged(bool)));
    connect(ui.pbG, SIGNAL(toggled(bool)), this, SLOT(colorComponentVisibilityChanged(bool)));
    connect(ui.pbB, SIGNAL(toggled(bool)), this, SLOT(colorComponentVisibilityChanged(bool)));
    connect(ui.btnInvColor1, SIGNAL(colorChanged(QColor)), this, SLOT(palSpecialColorChanged(QColor)));
    connect(ui.btnInvColor2, SIGNAL(colorChanged(QColor)), this, SLOT(palSpecialColorChanged(QColor)));
    connect(ui.btnInvColor, SIGNAL(colorChanged(QColor)), this, SLOT(palSpecialColorChanged(QColor)));

    ui.gvPalCurves->installEventFilter(this);
    ui.gvPalCurves->setRenderHint(QPainter::Antialiasing);
    m_pScenePalCurves = new QGraphicsScene();
    ui.gvPalCurves->setScene(m_pScenePalCurves);

    ui.gvCurPalette->installEventFilter(this);
    m_pSceneCurPalette = new QGraphicsScene();
    ui.gvCurPalette->setScene(m_pSceneCurPalette);

}

//--------------------------------------------------------------------------------------------------------------------------------
WidgetPropPalettes::~WidgetPropPalettes()
{
    m_pSceneCurPalette->deleteLater();
    m_pScenePalCurves->deleteLater();
}


//--------------------------------------------------------------------------------------------------------------------------------
bool WidgetPropPalettes::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == ui.gvCurPalette || obj == ui.gvPalCurves)
    {
        if (event->type() == QEvent::Resize)
        {
            QTimer::singleShot(1, this, &WidgetPropPalettes::updateViewOnResize);
        }
    }

    // standard event processing
    return QObject::eventFilter(obj, event);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::updateViewOnResize()
{
    updateOptionPalette();
    drawPalCurves();
    ui.gvPalCurves->fitInView(-m_gvSceneMarginLeftRight, -m_gvCurveSceneMarginTopBottom, ui.gvPalCurves->width(), ui.gvPalCurves->height());
}

//----------------------------------------------------------------------------------------------------------------------------------
/* updates the list of all color palettes including the special icons
*/
void WidgetPropPalettes::updatePaletteList()
{
    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)
    int width = 256;
    int height = qRound(dpiFactor * 16);
    int borderVertical = qRound(dpiFactor * 3);
    int iconSize = height;

    ui.lwPalettes->clear();
    ui.lwPalettes->setIconSize(QSize(width, height));

    for (int nc = 0; nc < m_palettes.size(); nc++)
    {
        const ito::ItomPaletteBase &pal = m_palettes[nc];

        QVector<uint> cols = pal.get256Colors();
        QImage img(width, height - 2 * borderVertical, QImage::Format_RGB32);

        for (int np = 0; np < width; np++)
        {
            for (int nh = 0; nh < height - 2 * borderVertical; nh++)
            {
#if QTVERSION < 0x050600
                img.setPixel(np, nh, QColor((cols[np] >> 16) & 0xFF, (cols[np] >> 8) & 0xFF, cols[np] & 0xFF).rgb());
#else
                img.setPixelColor(np, nh, QColor((cols[np] >> 16) & 0xFF, (cols[np] >> 8) & 0xFF, cols[np] & 0xFF));
#endif
            }
        }

        QPixmap pixmap_with_icon(img.width() + iconSize, height);
        pixmap_with_icon.fill(Qt::transparent);
        QPainter painter(&pixmap_with_icon);
        painter.drawPixmap(0, borderVertical, QPixmap::fromImage(img));

        if (pal.getType() & ito::tPaletteReadOnly)
        {
            QIcon lock(":/misc/icons/lock.png");
            painter.drawPixmap(width, 0, iconSize, iconSize, lock.pixmap(iconSize, iconSize));
        }

        QIcon icon(pixmap_with_icon);
        QListWidgetItem *item = new QListWidgetItem(icon, pal.getName());
        ui.lwPalettes->insertItem(nc, item);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/* updates the horizontal color bar in the palette options bar
*/
void WidgetPropPalettes::updateOptionPalette()
{
    QImage img(256, 1, QImage::Format_RGB32);
    QVector<uint> curCols = m_currentPalette.get256Colors();

    for (int np = 0; np < 256; np++)
    {
#if QT_VERSION < 0x050600
        img.setPixel(np, 0, QColor((curCols[np] >> 16) & 0xFF, (curCols[np] >> 8) & 0xFF, curCols[np] & 0xFF).rgb());
#else
        img.setPixelColor(np, 0, QColor((curCols[np] >> 16) & 0xFF, (curCols[np] >> 8) & 0xFF, curCols[np] & 0xFF));
#endif
    }

    m_pSceneCurPalette->clear();

    int width = ui.gvCurPalette->width() - 2 * m_gvSceneMarginLeftRight;
    int height = ui.gvCurPalette->height() - 2 * m_gvPaletteSceneMarginTopBottom;

    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(
        QPixmap::fromImage(img.scaled(
            width,
            height)));
    item->setOffset(0, 0);

    m_pSceneCurPalette->addItem(item);

    ui.gvCurPalette->setSceneRect(-m_gvSceneMarginLeftRight, -m_gvPaletteSceneMarginTopBottom, ui.gvCurPalette->width(), ui.gvCurPalette->height());
    ui.gvCurPalette->fitInView(-m_gvSceneMarginLeftRight, -m_gvPaletteSceneMarginTopBottom, ui.gvCurPalette->width(), ui.gvCurPalette->height());

    ui.gvCurPalette->show();

    ui.pbPalSave->setEnabled(m_isDirty);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::drawPalCurves(int selPt)
{
    QGraphicsScene* sceneCurves = m_pScenePalCurves;
    double sceneHeight = ui.gvPalCurves->height() - 2 * m_gvCurveSceneMarginTopBottom;
    double sceneWidth = ui.gvPalCurves->width() - 2 * m_gvSceneMarginLeftRight;

    QList<QGraphicsItem*> items = sceneCurves->items();
    ColCurve *curveR, *curveG, *curveB;
    bool editable = (m_curPaletteIndex >= 0) && !m_currentPalette.isWriteProtected();

    if (items.size() < 3)
    {
        sceneCurves->clear();
        curveR = new ColCurve(this, 0, ui.gvPalCurves);
        curveR->setActiveSceneSize(QSizeF(sceneWidth, sceneHeight));
        curveG = new ColCurve(this, 1, ui.gvPalCurves);
        curveG->setActiveSceneSize(QSizeF(sceneWidth, sceneHeight));
        curveB = new ColCurve(this, 2, ui.gvPalCurves);
        curveB->setActiveSceneSize(QSizeF(sceneWidth, sceneHeight));

        QPen rPen(Qt::red, 2);
        QPen gPen(Qt::green, 2);
        QPen bPen(Qt::blue, 2);
        curveR->setPen(rPen);
        curveG->setPen(gPen);
        curveB->setPen(bPen);

        sceneCurves->addItem(curveR);
        sceneCurves->addItem(curveG);
        sceneCurves->addItem(curveB);
    }
    else
    {
        QMutableListIterator<QGraphicsItem *> iter(items);
        while(iter.hasNext())
        {
            ColCurve *curve = dynamic_cast<ColCurve*>(iter.next());

            if (curve)
            {
                if (curve->getColChannel() == 0)
                    curveR = curve;
                else if (curve->getColChannel() == 1)
                    curveG = curve;
                else
                    curveB = curve;

                curve->setActiveSceneSize(QSizeF(sceneWidth, sceneHeight));
            }
            else
            {
                // removing remaining markers
                sceneCurves->removeItem(iter.value());
            }
        }
    }

    QPainterPath pathR, pathG, pathB;
    // draw line

    QPolygonF polyR, polyG, polyB;
    int gval, rval, bval, x0, y0;
    QVector<QGradientStop> curPalData = m_currentPalette.getColorStops();

    for (int cs = 0; cs < curPalData.size(); cs++)
    {
        gval = curPalData[cs].second.green();
        rval = curPalData[cs].second.red();
        bval = curPalData[cs].second.blue();
        x0 = curPalData[cs].first * sceneWidth;

        if (ui.pbR->isChecked())
        {
            y0 = sceneHeight * (255.0 - rval) / 255.0;
            polyR.append(QPointF(x0, y0));
            pathR.addRect(QRectF(x0 - 2, y0 - 2, 5, 5));
        }

        if (ui.pbG->isChecked())
        {
            y0 = sceneHeight * (255.0 - gval) / 255.0;
            polyG.append(QPointF(x0, y0));
            pathG.addRect(QRectF(x0 - 2, y0 - 2, 5, 5));
        }

        if (ui.pbB->isChecked())
        {
            y0 = sceneHeight * (255.0 - bval) / 255.0;
            polyB.append(QPointF(x0, y0));
            pathB.addRect(QRectF(x0 - 2, y0 - 2, 5, 5));
        }
    }

    pathR.addPolygon(polyR);
    pathG.addPolygon(polyG);
    pathB.addPolygon(polyB);
    curveR->setPath(pathR);
    curveG->setPath(pathG);
    curveB->setPath(pathB);
    curveR->setEditable(editable);
    curveB->setEditable(editable);
    curveG->setEditable(editable);

    if (selPt >= 0 && selPt < curPalData.size())
    {
        QPainterPath markerPath;
        if (ui.pbR->isChecked())
        {
            markerPath.addRect(
                (int)(curPalData[selPt].first * sceneWidth - 4),
                sceneHeight * (255.0 - curPalData[selPt].second.red()) / 255.0 - 5,
                9,
                9);
        }
        if (ui.pbG->isChecked())
        {
            markerPath.addRect(
                (int)(curPalData[selPt].first * sceneWidth - 4),
                sceneHeight * (255.0 - curPalData[selPt].second.green()) / 255.0 - 5,
                9,
                9);
        }
        if (ui.pbB->isChecked())
        {
            markerPath.addRect(
                (int)(curPalData[selPt].first * sceneWidth - 4),
                sceneHeight * (255.0 - curPalData[selPt].second.blue()) / 255.0 - 5,
                9,
                9);
        }

        //highlight color with respect to current stylesheet
        QColor c = this->palette().color(QPalette::Normal, QPalette::HighlightedText);
        sceneCurves->addPath(markerPath, QPen(c, 2));

        ui.sbR->setValue(curPalData[m_selectedColorStop].second.red());
        ui.sbG->setValue(curPalData[m_selectedColorStop].second.green());
        ui.sbB->setValue(curPalData[m_selectedColorStop].second.blue());
        setColorButton(curPalData[m_selectedColorStop].second, ui.btnColor, ui.lblColor, ui.icoColor);

        double minimum = 0.0;
        double maximum = 1.0;

        if (m_selectedColorStop > 0 && m_selectedColorStop < curPalData.size() - 1)
        {
            minimum = curPalData[m_selectedColorStop - 1].first;
            maximum = curPalData[m_selectedColorStop + 1].first;
        }

        ui.sbIndex->setMinimum(minimum);
        ui.sbIndex->setMaximum(maximum);

        ui.sbIndex->setValue(curPalData[m_selectedColorStop].first);
    }
    else
    {
        setColorButton(Qt::black, ui.btnColor, ui.lblColor, ui.icoColor);
    }

    ui.gvPalCurves->setSceneRect(-m_gvSceneMarginLeftRight, -m_gvCurveSceneMarginTopBottom, ui.gvPalCurves->width(), ui.gvPalCurves->height());

    ui.gvPalCurves->show();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::colorComponentVisibilityChanged(bool)
{
    drawPalCurves(m_selectedColorStop);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::colorComponentChanged(int value)
{
    if (m_curPaletteIndex < 0 || m_curPaletteIndex >= m_palettes.size())
    {
        return;
    }

    if (m_selectedColorStop >= 0 && !m_isUpdating)
    {
        QVector<QGradientStop> colorStops = m_currentPalette.getColorStops();
        int r = colorStops[m_selectedColorStop].second.red();
        int g = colorStops[m_selectedColorStop].second.green();
        int b = colorStops[m_selectedColorStop].second.blue();

        QObject *sender = QObject::sender();
        if (sender == ui.sbR)
        {
            colorStops[m_selectedColorStop].second = QColor(value > 255 ? 255 : value < 0 ? 0 : value, g, b);
        }
        else if (sender == ui.sbG)
        {
            colorStops[m_selectedColorStop].second = QColor(r, value > 255 ? 255 : value < 0 ? 0 : value, b);
        }
        else
        {
            colorStops[m_selectedColorStop].second = QColor(r, g, value > 255 ? 255 : value < 0 ? 0 : value);
        }

        ui.btnColor->blockSignals(true);
        ui.btnColor->setColor(colorStops[m_selectedColorStop].second);
        ui.btnColor->blockSignals(false);

        m_currentPalette.setColorStops(colorStops);
        drawPalCurves(m_selectedColorStop);
        updateOptionPalette();

        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_btnColor_colorChanged(QColor color)
{
    if (m_curPaletteIndex < 0 || m_curPaletteIndex >= m_palettes.size())
    {
        return;
    }

    if (m_selectedColorStop >= 0 && !m_isUpdating)
    {
        QVector<QGradientStop> colorStops = m_currentPalette.getColorStops();
        int r = colorStops[m_selectedColorStop].second.red();
        int g = colorStops[m_selectedColorStop].second.green();
        int b = colorStops[m_selectedColorStop].second.blue();

        colorStops[m_selectedColorStop].second = color;

        ui.sbR->blockSignals(true);
        ui.sbG->blockSignals(true);
        ui.sbB->blockSignals(true);
        ui.sbR->setValue(color.red());
        ui.sbB->setValue(color.blue());
        ui.sbG->setValue(color.green());
        ui.sbR->blockSignals(false);
        ui.sbG->blockSignals(false);
        ui.sbB->blockSignals(false);

        m_currentPalette.setColorStops(colorStops);
        drawPalCurves(m_selectedColorStop);
        updateOptionPalette();

        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::palSpecialColorChanged(QColor color)
{
    if (!m_isUpdating)
    {
        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_sbIndex_valueChanged(double value)
{
    if (!m_isUpdating)
    {
        if (m_curPaletteIndex < 0 || m_curPaletteIndex >= m_palettes.size())
        {
            return;
        }

        if (m_selectedColorStop > 0 && m_selectedColorStop < (m_currentPalette.getColorStops().size() - 1) && !m_isUpdating)
        {
            QVector<QGradientStop> colorStops = m_currentPalette.getColorStops();
            colorStops[m_selectedColorStop].first = value;

            m_currentPalette.setColorStops(colorStops);
            drawPalCurves(m_selectedColorStop);
            updateOptionPalette();

            m_isDirty = 1;
            ui.pbPalSave->setEnabled(true);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_lePalName_textChanged(const QString & text)
{
    if (!m_isUpdating)
    {
        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::lwCurrentRowChanged(int row)
{
    if (row == -1 || row >= m_palettes.size())
    {
        m_curPaletteIndex = qBound(-1, m_curPaletteIndex, m_palettes.size() - 1);
        return;
    }

    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();

    // handling of altered palette
    if (m_isDirty)
    {
        if (QMessageBox::question(this, tr("Color palette altered"), \
            tr("The current color palette was altered and is currently unsaved. Save changes or discard?"), QMessageBox::Save, QMessageBox::Discard) == QMessageBox::Save)
        {
            if (saveCurrentPalette().containsError())
            {
                ui.lwPalettes->setCurrentRow(m_curPaletteIndex);
                return;
            }
        }

        m_isDirty = 0;
    }

    m_isUpdating = 1;

    m_curPaletteIndex = row;

    m_selectedColorStop = -1;

    m_currentPalette = m_palettes[m_curPaletteIndex];

    setColorButton(m_currentPalette.getInverseColorOne(), ui.btnInvColor1, ui.lblInvColor1, ui.icoInvColor1);
    setColorButton(m_currentPalette.getInverseColorTwo(), ui.btnInvColor2, ui.lblInvColor2, ui.icoInvColor2);
    setColorButton(m_currentPalette.getInvalidColor(), ui.btnInvColor, ui.lblInvColor, ui.icoInvColor);

    ui.lePalName->setText(m_currentPalette.getName());

    ui.pbDuplicate->setEnabled(row >= 0);
    bool editable = (row >= 0) && !m_currentPalette.isWriteProtected();

    ui.pbRemove->setEnabled(editable);
    ui.pbExportPalette->setEnabled(row >= 0);
    ui.groupColorStops->setEnabled(editable);
    ui.sbR->setEnabled(editable);
    ui.sbG->setEnabled(editable);
    ui.sbB->setEnabled(editable);
    ui.btnInvColor->setVisible(editable);
    ui.btnInvColor1->setVisible(editable);
    ui.btnInvColor2->setVisible(editable);
    ui.btnColor->setVisible(editable);
    ui.icoInvColor1->setVisible(!editable);
    ui.lblInvColor1->setVisible(!editable);
    ui.icoInvColor2->setVisible(!editable);
    ui.lblInvColor2->setVisible(!editable);
    ui.icoInvColor->setVisible(!editable);
    ui.lblInvColor->setVisible(!editable);
    ui.icoColor->setVisible(!editable);
    ui.lblColor->setVisible(!editable);
    ui.sbIndex->setEnabled(editable);
    ui.pbPalSave->setEnabled(editable && m_isDirty);
    ui.lePalName->setReadOnly(!editable);

    ui.pbAddColorStop->setEnabled(false);
    ui.pbRemoveColorStop->setEnabled(false);

    m_isUpdating = 0;

    drawPalCurves(m_selectedColorStop);
    updateOptionPalette();

}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbAdd_clicked()
{
    // handling of altered palette
    if (m_isDirty)
    {
        if (QMessageBox::question(this, tr("Color palette altered"), \
            tr("The current color palette was altered and is currently unsaved. Save changes or discard?"), QMessageBox::Save, QMessageBox::Discard) == QMessageBox::Save)
        {
            if (saveCurrentPalette().containsError())
            {
                ui.lwPalettes->setCurrentRow(m_curPaletteIndex);
                return;
            }
        }
    }

    QString newPalName = tr("User Palette");
    int newCnt = 1;

    QString tmpPalName = newPalName;

    QStringList currentPalettesNames;
    foreach (const ito::ItomPaletteBase &palette, m_palettes)
    {
        currentPalettesNames.append(palette.getName());
    }

    while (currentPalettesNames.contains(tmpPalName))
    {
        tmpPalName = newPalName + " " + QString::number(newCnt++);
    }
    newPalName = tmpPalName;

    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    ito::ItomPaletteBase pal = palOrganizer->getColorPalette(palOrganizer->getColorBarIndex("gray"));

    m_currentPalette = ito::ItomPaletteBase(newPalName, pal.getType() & (~ito::tPaletteReadOnly), pal.getInverseColorOne(), \
        pal.getInverseColorTwo(), pal.getInvalidColor(), pal.getColorStops());

    m_palettes.append(m_currentPalette);
    m_curPaletteIndex = m_palettes.size() - 1;
    m_isDirty = 0;
    ui.pbPalSave->setEnabled(false);

    updatePaletteList();

    ui.lwPalettes->setCurrentRow(m_curPaletteIndex);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbDuplicate_clicked()
{
    // handling of altered palette
    if (m_isDirty)
    {
        if (QMessageBox::question(this, tr("Color palette altered"), \
            tr("The current color palette was altered and is currently unsaved. Save changes or discard?"), QMessageBox::Save, QMessageBox::Discard) == QMessageBox::Save)
        {
            if (saveCurrentPalette().containsError())
            {
                ui.lwPalettes->setCurrentRow(m_curPaletteIndex);
                return;
            }
        }
    }

    ito::ItomPaletteBase pal = m_palettes[ui.lwPalettes->currentRow()]; // current palette

    QString tmpPalName = pal.getName() + tr(" - Copy");
    QString newPalName = tmpPalName;

    int newCnt = 1;
    QStringList currentPalettesNames;
    foreach (const ito::ItomPaletteBase &palette, m_palettes)
    {
        currentPalettesNames.append(palette.getName());
    }

    while (currentPalettesNames.contains(tmpPalName))
    {
        tmpPalName = newPalName + QString(" (%i)").arg(newCnt++);
    }
    newPalName = tmpPalName;


    m_currentPalette = ito::ItomPaletteBase(newPalName, pal.getType() & (~ito::tPaletteReadOnly), pal.getInverseColorOne(), \
        pal.getInverseColorTwo(), pal.getInvalidColor(), pal.getColorStops());

    m_palettes.append(m_currentPalette);
    m_curPaletteIndex = m_palettes.size() - 1;
    m_isDirty = 0;
    ui.pbPalSave->setEnabled(false);

    updatePaletteList();

    ui.lwPalettes->setCurrentRow(m_curPaletteIndex);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbRemove_clicked()
{
    int idx = ui.lwPalettes->currentRow();

    if (idx >= 0 && idx < m_palettes.size())
    {
        const ito::ItomPaletteBase &currentPalette = m_palettes[idx];
        if (currentPalette.isWriteProtected())
        {
            QMessageBox::information(this, tr("Palette is read only"), tr("Palette is read only and cannot be removed!"));
            return;
        }

		m_isDirty = 0;

        ui.lwPalettes->takeItem(idx);
        m_palettes.takeAt(idx);

        ui.pbPalSave->setEnabled(false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbPalSave_clicked()
{
    saveCurrentPalette();
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal WidgetPropPalettes::saveCurrentPalette()
{
    QString newName = ui.lePalName->text().trimmed();

    bool nameExists = false;
    for (int idx = 0; idx < m_palettes.size(); ++idx)
    {
        if ((idx != m_curPaletteIndex) && (newName == m_palettes[idx].getName()))
        {
            nameExists = true;
            break;
        }
    }

    if (nameExists)
    {
        QMessageBox::critical(this, tr("name conflict"), tr("Another color palette with the same name already exists."));
        return ito::retError;
    }
    else if (newName == "")
    {
        QMessageBox::critical(this, tr("invalid name"), tr("An empty palette name is not valid."));
        return ito::retError;
    }
    else if (m_isDirty)
    {
        m_isUpdating = true;

        m_currentPalette = ItomPaletteBase(ui.lePalName->text(), m_currentPalette.getType()&~tPaletteReadOnly, ui.btnInvColor1->color(),
            ui.btnInvColor2->color(), ui.btnInvColor->color(), m_currentPalette.getColorStops());

        if (m_curPaletteIndex == -1)
        {
            m_palettes.append(m_currentPalette);
            m_curPaletteIndex = m_palettes.size() - 1;
        }
        else
        {
            m_palettes[m_curPaletteIndex] = m_currentPalette;
        }

        m_isDirty = 0;
        ui.pbPalSave->setEnabled(false);
        updatePaletteList();

        m_isUpdating = false;
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::readSettings()
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    ito::ItomPaletteBase palette;
    bool found;

    foreach (const QString &colorBarName, palOrganizer->getColorPaletteList())
    {
        palette = palOrganizer->getColorPalette(colorBarName, &found);
        if (found)
        {
            m_palettes.append(palette);
        }
    }

    updatePaletteList();
    ui.lwPalettes->setCurrentRow(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::writeSettings()
{
    // handling of altered palette
    if (m_isDirty)
    {
        if (QMessageBox::question(this, tr("Color palette altered"), \
            tr("The current color palette was altered and is currently unsaved. Save changes or ignore?"), QMessageBox::Save, QMessageBox::Ignore) == QMessageBox::Save)
        {
            if (saveCurrentPalette().containsError())
            {
                QMessageBox::information(this, tr("Error saving current color palette."), tr("The current color palette could not be saved."));
            }

            m_isDirty = 0;
            ui.pbPalSave->setEnabled(false);
            updatePaletteList();
        }
    }

    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    QList<QString> remainingPalettes;

    foreach (const ito::ItomPaletteBase &palette, m_palettes)
    {
        if (!palette.isWriteProtected())
        {
            palOrganizer->setColorBarThreaded(palette.getName(), palette);
        }

        remainingPalettes << palette.getName();
    }

    //remove deleted palettes...
    QList<QString> builtInPalettes = palOrganizer->getBuiltInPaletteNames();
    QList<QString> colorBarList = palOrganizer->getColorPaletteList();

    for (int i = colorBarList.size() - 1; i >= 0; --i)
    {
        if (!remainingPalettes.contains(colorBarList[i]))
        {
            palOrganizer->removeColorPalette(i);
        }
    }

    //update list
    colorBarList = palOrganizer->getColorPaletteList();

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    settings.beginGroup("ColorPalettes");

    foreach(QString child, settings.childGroups())
    {
        // check for left over user palettes and removed them
        if (!builtInPalettes.contains(child) && !colorBarList.contains(child))
        {
            settings.beginGroup(child);
            settings.remove("");
            settings.endGroup();
        }
    }

    for (int np = 0; np < colorBarList.length(); np++)
    {
        if (!builtInPalettes.contains(colorBarList[np]))
        {
            ItomPaletteBase pal = palOrganizer->getColorPalette(colorBarList[np]);
            palOrganizer->saveColorPaletteToSettings(pal, settings);
        }
    }
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::changeSelectedColorStop(int new_index)
{
    int numColorStops = m_currentPalette.getColorStops().size();
    new_index = qBound(-1, new_index, numColorStops - 1);

    ui.pbAddColorStop->setEnabled(new_index != -1);
    ui.pbRemoveColorStop->setEnabled(new_index != -1 && numColorStops > 2);

    if (new_index != m_selectedColorStop)
    {
        int isUpdating = m_isUpdating;
        m_isUpdating = 1;
        m_selectedColorStop = new_index;
        drawPalCurves(m_selectedColorStop);

        if (new_index == 0 ||
            (new_index == m_currentPalette.getColorStops().size() - 1))
        {
            ui.sbIndex->setEnabled(false);
        }
        else
        {
            bool editable = !m_currentPalette.isWriteProtected();
            ui.sbIndex->setEnabled(editable);
        }

        m_isUpdating = isUpdating;
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::removeColorStop(int index)
{
    QGradientStops pts = m_currentPalette.getColorStops();

    if (index >= 0 && index < (pts.size()))
    {
        if (pts.size() > 2)
        {
            pts.remove(index);

            //make sure that the first point is at the 0-position and that the last point at the 1-position
            pts[0].first = 0.0;
            pts[pts.size() - 1].first = 1.0;

            ito::ItomPaletteBase palette(m_currentPalette.getName(), m_currentPalette.getType(),
                ui.btnInvColor1->color(), ui.btnInvColor2->color(), ui.btnInvColor->color(), pts);
            m_currentPalette = palette;

            changeSelectedColorStop(-1);

            updateOptionPalette();

            m_isDirty = 1;
            ui.pbPalSave->setEnabled(true);
        }
        else
        {
            QMessageBox::information(this, tr("Remove color stop"), tr("A color palette must have at least two color stops."));
        }
    }
    else
    {
        QMessageBox::information(this, tr("Remove color stop"), tr("No color stop has been selected."));
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::addColorStop(int index_before, float percent_to_next /*=0.5*/)
{
    QGradientStops pts = m_currentPalette.getColorStops();

    if (index_before >= 0 && index_before < (pts.size() - 1))
    {
        float xpos = pts[index_before].first + percent_to_next * (pts[index_before + 1].first - pts[index_before].first);

        int first = index_before;

        QColor col1 = pts[first].second;
        QColor col2 = pts[first < pts.length() - 1 ? first + 1 : first].second;
        pts.insert(first + 1, QGradientStop(xpos, QColor(col1.red() + (col2.red() - col1.red()) * percent_to_next,
            col1.green() + (col2.green() - col1.green()) * percent_to_next,
            col1.blue() + (col2.blue() - col1.blue()) * percent_to_next)));

        ito::ItomPaletteBase palette(m_currentPalette.getName(), m_currentPalette.getType(),
            ui.btnInvColor1->color(), ui.btnInvColor2->color(), ui.btnInvColor->color(), pts);
        m_currentPalette = palette;

        m_selectedColorStop = -1; //to force an update with the next command!
        changeSelectedColorStop(first + 1);
        updateOptionPalette();

        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
    else
    {
        QMessageBox::information(this, tr("Add color stop"), tr("No color stop has been selected."));
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbEquidistantColorStop_clicked()
{
    QGradientStops pts = m_currentPalette.getColorStops();

    if (pts.size() > 2)
    {
        float first_index = pts.first().first;
        float last_index = pts.last().first;
        float step = (last_index - first_index) / (pts.size() - 1);

        for (int idx = 1; idx < (pts.size() - 1); ++idx)
        {
            pts[idx].first = first_index + idx * step;
        }

        ito::ItomPaletteBase palette(m_currentPalette.getName(), m_currentPalette.getType(),
            ui.btnInvColor1->color(), ui.btnInvColor2->color(), ui.btnInvColor->color(), pts);

        m_currentPalette = palette;
        drawPalCurves(m_selectedColorStop);
        updateOptionPalette();

        m_isDirty = 1;
        ui.pbPalSave->setEnabled(true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbRemoveColorStop_clicked()
{
    removeColorStop(m_selectedColorStop);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbAddColorStop_clicked()
{
    if (m_selectedColorStop >= 0 && m_selectedColorStop < (m_currentPalette.getColorStops().size() - 1))
    {
        addColorStop(m_selectedColorStop);
    }
    else if (m_selectedColorStop > 0 && m_selectedColorStop == m_currentPalette.getColorStops().size() - 1)
    {
        addColorStop(m_selectedColorStop - 1);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbImportPalette_clicked()
{
    const ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    QString filename = QFileDialog::getOpenFileName(this, tr("Color palette import"), QString(), tr("Itom color palette (*.icp)"));
    bool save = false;

    if (filename != "" && palOrganizer)
    {
        QSettings settings(filename, QSettings::IniFormat);
        settings.beginGroup("ColorPalettes");
        foreach(QString child, settings.childGroups())
        {
            ItomPaletteBase pal;
            ito::RetVal retval = palOrganizer->loadColorPaletteFromSettings(child, pal, settings);
            if (retval.containsError())
            {
                QMessageBox::critical(this, tr("Wrong file format"), tr("The color palette '%1' in the itom color palette file is no valid color palette").arg(child));
                continue;
            }

            bool ok = false;
            QString name = pal.getName();
            if (name != child)
            {
                QMessageBox::critical(this, tr("Wrong file format"), tr("The color palette '%1' in the itom color palette file is no valid color palette").arg(child));
                continue;
            }

            while (!ok)
            {
                ok = true;
                save = true;

                for (int i = 0; i < m_palettes.size(); ++i)
                {
                    if (m_palettes[i].getName() == name)
                    {
                        name = QInputDialog::getText(this, tr("Name already exists"), tr("The name '%1' of the color palette already exists. Please indicate a new name to load the color palette:").arg(name), QLineEdit::Normal, pal.getName(), &ok);
                        if (!ok)
                        {
                            ok = true;
                            save = false;
                            break;
                        }
                        ok = false;
                        save = false;
                        break;
                    }
                }
            }

            if (save)
            {
                pal.setName(name);
                m_palettes.append(pal);
            }
        }
        settings.endGroup();

        updatePaletteList();

    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_pbExportPalette_clicked()
{
    const ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    ItomPaletteBase pal = m_palettes[ui.lwPalettes->currentRow()];
    QString filename = QFileDialog::getSaveFileName(this, tr("Color palette export"), pal.getName(), tr("Itom color palette (*.icp)"));
    if (filename != "" && palOrganizer)
    {
        QSettings settings(filename, QSettings::IniFormat);
        settings.clear();
        pal.removeWriteProtection();
        settings.beginGroup("ColorPalettes");
        palOrganizer->saveColorPaletteToSettings(pal, settings);
        settings.endGroup();
    }
}


//----------------------------------------------------------------------------------------------------------------------------------

}//endNamespace ito
