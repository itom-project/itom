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
#include "widgetPropPalettes.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../organizer/paletteOrganizer.h"
#include <qsettings.h>
#include <qgraphicsitem.h>
#include <qgraphicswidget.h>
#include <qgraphicssceneevent.h>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qinputdialog.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    if (m_parentWidget)
    {
        QVector<QGradientStop> pts = m_parentWidget->m_curPalette.getColorStops();
        if (m_parentWidget->m_selPt >= 0 && m_parentWidget->m_selPt < pts.length() && event->buttons() == Qt::LeftButton)
        {
            m_parentWidget->m_isDirty = 1;

            QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());
            float posXRel = event->lastScenePos().x() / par->size().width();
            if (m_parentWidget->m_selPt > 0 && posXRel <= pts[m_parentWidget->m_selPt - 1].first)
                pts[m_parentWidget->m_selPt].first = (pts[m_parentWidget->m_selPt - 1].first);
            else if (m_parentWidget->m_selPt < (pts.length() - 1) && posXRel >= pts[m_parentWidget->m_selPt + 1].first)
                pts[m_parentWidget->m_selPt].first = pts[m_parentWidget->m_selPt + 1].first;
            else if (m_parentWidget->m_selPt == 0 && posXRel != 0)
                pts[m_parentWidget->m_selPt].first = 0;
            else if (m_parentWidget->m_selPt == (pts.length() - 1) && par && posXRel != 1)
                pts[m_parentWidget->m_selPt].first = 1;
            else
                pts[m_parentWidget->m_selPt].first = posXRel;

            int colVal = (par->size().height() - event->lastScenePos().y()) / par->size().height() * 255;
            int r = (pts[m_parentWidget->m_selPt].second.rgb() & 0xFF0000) >> 16;
            int g = (pts[m_parentWidget->m_selPt].second.rgb() & 0xFF00) >> 8;
            int b = (pts[m_parentWidget->m_selPt].second.rgb() & 0xFF);
            if (m_colChannel == 0)
                pts[m_parentWidget->m_selPt].second = QColor(colVal, g, b);
            else if (m_colChannel == 1)
                pts[m_parentWidget->m_selPt].second = QColor(r, colVal, b);
            else
                pts[m_parentWidget->m_selPt].second = QColor(r, g, colVal);

            ito::ItomPaletteBase palette(m_parentWidget->getCurPalette()->getName(), m_parentWidget->getCurPalette()->getType(),
                m_parentWidget->getCurPalette()->getInverseColorOne(), m_parentWidget->getCurPalette()->getInverseColorTwo(),
                m_parentWidget->getCurPalette()->getInvalidColor(), pts);

            m_parentWidget->m_isUpdating = 1;
            m_parentWidget->m_curPalette = palette;
            m_parentWidget->drawPalCurves(m_parentWidget->m_selPt);
            m_parentWidget->updatePalette();
            m_parentWidget->m_isUpdating = 0;
            //qDebug() << r << " " << g << " " << b;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    double minDist = 1e208;
    int thisPt;

    QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());
    if (m_parentWidget && par)
    {
        QVector<QGradientStop> pts = m_parentWidget->m_curPalette.getColorStops();
        float scaleX = par->size().width();
        float scaleY = par->size().height();

        for (int ne = 0; ne < pts.length(); ne++)
        {
            QLineF l;
            if (m_colChannel == 0)
                l = QLineF(QPointF(pts[ne].first * scaleX, par->size().height() - pts[ne].second.redF() * scaleY),
                    QPointF(event->buttonDownScenePos(Qt::LeftButton).x(), event->buttonDownScenePos(Qt::LeftButton).y()));
            else if (m_colChannel == 1)
                l = QLineF(QPointF(pts[ne].first * scaleX, par->size().height() - pts[ne].second.greenF() * scaleY),
                    QPointF(event->buttonDownScenePos(Qt::LeftButton).x(), event->buttonDownScenePos(Qt::LeftButton).y()));
            else
                l = QLineF(QPointF(pts[ne].first * scaleX, par->size().height() - pts[ne].second.blueF() * scaleY),
                    QPointF(event->buttonDownScenePos(Qt::LeftButton).x(), event->buttonDownScenePos(Qt::LeftButton).y()));

            if (l.length() < minDist)
            {
                minDist = l.length();
                thisPt = ne;
            }
        }

        //if (minDist < 20)
        {
            m_parentWidget->m_isUpdating = 1;
            m_parentWidget->m_selPt = thisPt;
            m_parentWidget->drawPalCurves(m_parentWidget->m_selPt);
            m_parentWidget->m_isUpdating = 0;
        }
    }
    else
    {
        if (m_parentWidget)
            m_parentWidget->m_selPt = -1;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
    QMenu contextMenu(tr("Context menu"), m_parentWidget);
    QAction *action;

    if (m_parentWidget->m_selPt >= 0)
    {
        action = new QAction(tr("Delete Color Stop"), this);
        connect(action, SIGNAL(triggered()), this, SLOT(removeDataPoint()));
        contextMenu.addAction(action);
    }
    m_insertPos = event->scenePos();
    action = new QAction(tr("Add Color Stop"), this);
    connect(action, SIGNAL(triggered()), this, SLOT(addDataPoint()));
    contextMenu.addAction(action);
    contextMenu.exec(event->screenPos());
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::removeDataPoint()
{
    QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());
    if (m_parentWidget && par && m_parentWidget->m_selPt >= 0)
    {
        m_parentWidget->m_isDirty = 1;

        QVector<QGradientStop> pts = m_parentWidget->m_curPalette.getColorStops();

        if (pts.size() > 2)
        {
            pts.remove(m_parentWidget->m_selPt);
            m_parentWidget->m_selPt = -1;

            ito::ItomPaletteBase palette(m_parentWidget->getCurPalette()->getName(), m_parentWidget->getCurPalette()->getType(),
                m_parentWidget->getCurPalette()->getInverseColorOne(), m_parentWidget->getCurPalette()->getInverseColorTwo(),
                m_parentWidget->getCurPalette()->getInvalidColor(), pts);
            m_parentWidget->m_curPalette = palette;
            m_parentWidget->drawPalCurves(-1);
            m_parentWidget->updatePalette();
        }
        else
        {
            QMessageBox::information(par, tr("Too few color stops"), tr("A color palette must have at least two color stops."));
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void ColCurve::addDataPoint()
{
    QGraphicsView *par = qobject_cast<QGraphicsView*>(parent());
    if (m_parentWidget && par)
    {
        m_parentWidget->m_isDirty = 1;

        float scaleX = par->size().width();
        float scaleY = par->size().height();
        QVector<QGradientStop> pts = m_parentWidget->m_curPalette.getColorStops();
        float xpos = m_insertPos.x() / scaleX;

        int first = 0;
        while (pts[first].first <= xpos && first < pts.length())
            first++;
        first--;

        QColor col1 = pts[first].second;
        QColor col2 = pts[first < pts.length() - 1 ? first + 1 : first].second;
        float cfact = xpos - pts[first].first;
        pts.insert(first + 1, QGradientStop(xpos, QColor((col2.red() - col1.red()) * cfact, 
            (col2.green() - col1.green()) * cfact,
            (col2.blue() - col1.blue()) * cfact)));

        ito::ItomPaletteBase palette(m_parentWidget->getCurPalette()->getName(), m_parentWidget->getCurPalette()->getType(),
            m_parentWidget->getCurPalette()->getInverseColorOne(), m_parentWidget->getCurPalette()->getInverseColorTwo(),
            m_parentWidget->getCurPalette()->getInvalidColor(), pts);
        m_parentWidget->m_curPalette = palette;
        m_parentWidget->drawPalCurves(first + 1);
        m_parentWidget->updatePalette();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropPalettes::WidgetPropPalettes(QWidget *parent) :
    AbstractPropertyPageWidget(parent),
    m_selPt(-1),
    m_isUpdating(0),
    m_isDirty(0)
{
    ui.setupUi(this);

    updatePaletteList();

    connect(ui.lwPalettes, SIGNAL(currentRowChanged(int)), this, SLOT(lwCurrentRowChanged(int)));
    connect(ui.gvPalCurves, SIGNAL(mousePressEvent(QMouseEvent*)), this, SLOT(mousePressEvent(QMouseEvent*)));
    connect(ui.sbR, SIGNAL(valueChanged(int)), this, SLOT(sbValueChanged(int)));
    connect(ui.sbG, SIGNAL(valueChanged(int)), this, SLOT(sbValueChanged(int)));
    connect(ui.sbB, SIGNAL(valueChanged(int)), this, SLOT(sbValueChanged(int)));
    connect(ui.pbR, SIGNAL(toggled(bool)), this, SLOT(pbColToggled(bool)));
    connect(ui.pbG, SIGNAL(toggled(bool)), this, SLOT(pbColToggled(bool)));
    connect(ui.pbB, SIGNAL(toggled(bool)), this, SLOT(pbColToggled(bool)));
    connect(ui.pbAdd, SIGNAL(clicked()), this, SLOT(pbAddClicked()));
    connect(ui.pbDuplicate, SIGNAL(clicked()), this, SLOT(pbDuplicateClicked()));
    connect(ui.pbRemove, SIGNAL(clicked()), this, SLOT(pbRemoveClicked()));
    connect(ui.pbPalSave, SIGNAL(clicked()), this, SLOT(pbSaveClicked()));
    connect(ui.btnInvColor, SIGNAL(colorChanged(QColor)), this, SLOT(palSpecialColorChanged(QColor)));

    //connect(this, SIGNAL(resizeEvent(QResizeEvent*)), this, SLOT(widgetResize(QResizeEvent*)));
    ui.lwPalettes->setCurrentRow(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::updatePaletteList()
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    QList<QString> paletteNames = palOrganizer->getColorBarList();
    ui.lwPalettes->clear();
    ui.lwPalettes->setIconSize(QSize(256, 10));
    for (int nc = 0; nc < paletteNames.size(); nc++)
    {
        ito::ItomPaletteBase pal = palOrganizer->getColorBar(nc);
        QVector<uint> cols = pal.get256Colors();
        QImage img(256, 10, QImage::Format_RGB32);

        for (int np = 0; np < 256; np++)
        {
            for (int nh = 0; nh < 10; nh++)
            {
#if QTVERSION < 0x050600
                img.setPixel(np, nh, QColor((cols[np] >> 16) & 0xFF, (cols[np] >> 8) & 0xFF, cols[np] & 0xFF).rgb());
#else
                img.setPixelColor(np, nh, QColor((cols[np] >> 16) & 0xFF, (cols[np] >> 8) & 0xFF, cols[np] & 0xFF));
#endif
            }
        }

        QPixmap pixmap = QPixmap::fromImage(img);
        QIcon icon(pixmap);
        QListWidgetItem *item = new QListWidgetItem(icon, paletteNames[nc]);
        ui.lwPalettes->insertItem(nc, item);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::mousePressEvent(QMouseEvent *event)
{
    if (event->buttons() == Qt::LeftButton)
    {
        m_selPt = -1;
        drawPalCurves(-1);
        ui.sbR->setValue(0);
        ui.sbG->setValue(0);
        ui.sbB->setValue(0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::pbColToggled(bool)
{
    drawPalCurves(m_selPt);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::sbValueChanged(int value)
{
    m_isDirty = 1;

    if (m_selPt >= 0 && !m_isUpdating)
    {
        QVector<QGradientStop> curPalData = m_curPalette.getColorStops();
        int r = (curPalData[m_selPt].second.rgb() & 0xFF0000) >> 16;
        int g = (curPalData[m_selPt].second.rgb() & 0xFF00) >> 8;
        int b = (curPalData[m_selPt].second.rgb() & 0xFF);

        QObject *sender = QObject::sender();
        if (sender->objectName() == "sbR")
        {
            curPalData[m_selPt].second = QColor(value > 255 ? 255 : value < 0 ? 0 : value, g, b);
        }
        else if (sender->objectName() == "sbG")
        {
            curPalData[m_selPt].second = QColor(r, value > 255 ? 255 : value < 0 ? 0 : value, b);
        }
        else
        {
            curPalData[m_selPt].second = QColor(r, g, value > 255 ? 255 : value < 0 ? 0 : value);
        }

        ito::ItomPaletteBase palette(m_curPalette.getName(), m_curPalette.getType(),
            m_curPalette.getInverseColorOne(), m_curPalette.getInverseColorTwo(),
            m_curPalette.getInvalidColor(), curPalData);

        m_curPalette = palette;
        QVector<uint> pal = m_curPalette.get256Colors();
        drawPalCurves(m_selPt);
        updatePalette();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::drawPalCurves(int selPt, int sx0, int sy0, int sdx, int sdy)
{
    QGraphicsScene* sceneCurves;
    if (!ui.gvPalCurves->scene())
    {
        sceneCurves = new QGraphicsScene;
        ui.gvPalCurves->setScene(sceneCurves);
    }
    else
    {
        sceneCurves = ui.gvPalCurves->scene();
    }
    ui.gvPalCurves->setSceneRect(0, 0, ui.gvPalCurves->frameSize().width(), ui.gvPalCurves->frameSize().height());
    
    QList<QGraphicsItem*> items = sceneCurves->items();
    ColCurve *curveR, *curveG, *curveB;
    if (items.size() < 3)
    {
        sceneCurves->clear();
        curveR = new ColCurve(this, 0);
        curveR->setParent(ui.gvPalCurves);
        curveG = new ColCurve(this, 1);
        curveG->setParent(ui.gvPalCurves);
        curveB = new ColCurve(this, 2);
        curveB->setParent(ui.gvPalCurves);

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
            //ColCurve *curve = (ColCurve*)(items[nc]);
            ColCurve *curve = dynamic_cast<ColCurve*>(iter.next());
            if (curve)
            {
                if (curve->getColChannel() == 0)
                    curveR = curve;
                else if (curve->getColChannel() == 1)
                    curveG = curve;
                else
                    curveB = curve;
            }
            else
            {
                // removing remaining markers
                //items.removeOne(item.next());
                sceneCurves->removeItem(iter.value());
            }
        }
    }

    QPainterPath pathR, pathG, pathB;
    // draw line

    float scalex = ui.gvPalCurves->frameSize().width() - sdx;
    float scaley = (ui.gvPalCurves->frameSize().height() - sdy) / 255;
    QPolygonF polyR, polyG, polyB;
    QVector<QGradientStop> curPalData = m_curPalette.getColorStops();
    for (int cs = 0; cs < curPalData.size(); cs++)
    {
        int gval = curPalData[cs].second.green();
        int rval = curPalData[cs].second.red();
        int bval = curPalData[cs].second.blue();
        int x0 = curPalData[cs].first * scalex + sx0;
        //int x1 = curPalData[cs + 1].first * scalex;
        int y0 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs].second.red() * scaley - sy0;
        //int y1 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs + 1].second.red() * scaley;
        if (ui.pbR->isChecked())
        {
            polyR.append(QPointF(x0, y0));
            pathR.addRect(QRectF(x0 - 2, y0 - 2, 5, 5));
        }

        if (ui.pbG->isChecked())
        {
            y0 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs].second.green() * scaley - sy0;
            //y1 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs + 1].second.green() * scaley;
            polyG.append(QPointF(x0, y0));
            pathG.addRect(QRectF(x0 - 2, y0 - 2, 5, 5));
        }

        if (ui.pbB->isChecked())
        {
            y0 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs].second.blue() * scaley - sy0;
            //y1 = ui.gvPalCurves->frameSize().height() - 2 - curPalData[cs + 1].second.blue() * scaley;
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

    if (selPt >= 0 && selPt < curPalData.size())
    {
        QPainterPath markerPath;
        if (ui.pbR->isChecked())
            markerPath.addRect((int)(curPalData[selPt].first * scalex + sx0 - 4),
                ui.gvPalCurves->frameSize().height() - (int)(curPalData[selPt].second.red() * scaley + sy0 + 5), 9, 9);
        if (ui.pbG->isChecked())
            markerPath.addRect((int)(curPalData[selPt].first * scalex + sx0 - 4),
                ui.gvPalCurves->frameSize().height() - (int)(curPalData[selPt].second.green() * scaley + sy0 + 5), 9, 9);
        if (ui.pbB->isChecked())
            markerPath.addRect((int)(curPalData[selPt].first * scalex + sx0 - 4),
                ui.gvPalCurves->frameSize().height() - (int)(curPalData[selPt].second.blue() * scaley + sy0 + 5), 9, 9);

        sceneCurves->addPath(markerPath, QPen(Qt::black, 2));

        ui.sbR->setValue(curPalData[m_selPt].second.red());
        ui.sbG->setValue(curPalData[m_selPt].second.green());
        ui.sbB->setValue(curPalData[m_selPt].second.blue());
    }

    ui.gvPalCurves->show();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::updatePalette()
{
    QImage img(256, 1, QImage::Format_RGB32);
    QVector<uint> curCols = m_curPalette.get256Colors();
    for (int np = 0; np < 256; np++)
    {
#if QTVERSION < 0x050600
        img.setPixel(np, 0, QColor((curCols[np] >> 16) & 0xFF, (curCols[np] >> 8) & 0xFF, curCols[np] & 0xFF).rgb());
#else
        img.setPixelColor(np, 0, QColor((curCols[np] >> 16) & 0xFF, (curCols[np] >> 8) & 0xFF, curCols[np] & 0xFF));
#endif
    }

    m_imgGVCurPalette = img;
    ui.gvCurPalette->setSceneRect(0, 0, ui.gvCurPalette->width(), ui.gvCurPalette->height());
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(QPixmap::fromImage(img.scaled(ui.gvCurPalette->sceneRect().width() - 6,
        ui.gvCurPalette->sceneRect().height() - 6)));
    QGraphicsScene* scene = new QGraphicsScene;
    scene->addItem(item);
    ui.gvCurPalette->setScene(scene);
    ui.gvCurPalette->show();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::lwCurrentRowChanged(int row)
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();

    // handling of altered palette
    if (m_isDirty)
    {
        if (QMessageBox::question(this, tr("Palette altered"), tr("Current palette has been altered and is unsaved. Save changes or discard?")) == QMessageBox::Yes)
        {
            QList<QString> palettes = palOrganizer->getColorBarList();
            if (palettes.contains(ui.lePalName->text()))
            {

            }
            else
            {
                m_curPalette.setInvalidColor(ui.btnInvColor->color());
                m_curPalette.setInverseColorOne(ui.btnInvColor1->color());
                m_curPalette.setInverseColorTwo(ui.btnInvColor2->color());
                palOrganizer->setColorBarThreaded(ui.lePalName->text(), m_curPalette, NULL);
            }
        }
    }

    m_curPalette = palOrganizer->getColorBar(row);
    ui.btnInvColor1->setColor(m_curPalette.getInverseColorOne());
    ui.btnInvColor2->setColor(m_curPalette.getInverseColorTwo());
    ui.btnInvColor->setColor(m_curPalette.getInvalidColor());
    ui.lePalName->setText(m_curPalette.getName());

    ui.pbDuplicate->setEnabled(row >= 0);
    ui.pbRemove->setEnabled((row >= 0) && !(m_curPalette.getType() & ito::tPaletteReadOnly));
    ui.groupOptions->setEnabled((row >= 0) && !(m_curPalette.getType() & ito::tPaletteReadOnly));

    //m_curCols = m_curPalette.get256Colors();
    //m_curPalData = pal.getColorStops();

    drawPalCurves();
    updatePalette();
    m_isDirty = 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::resizeEvent(QResizeEvent * event)
{
    ui.gvCurPalette->setSceneRect(0, 0, ui.gvCurPalette->width(), ui.gvCurPalette->height());
    QGraphicsPixmapItem *item = new QGraphicsPixmapItem(QPixmap::fromImage(m_imgGVCurPalette.scaled(ui.gvCurPalette->sceneRect().width() - 6, 
        ui.gvCurPalette->sceneRect().height() - 6)));
    ui.gvCurPalette->scene()->clear();
    ui.gvCurPalette->scene()->addItem(item);

    //ui.gvPalCurves->setSceneRect(0, 0, ui.gvPalCurves->frameSize().width(), ui.gvPalCurves->frameSize().height());
    drawPalCurves();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::pbAddClicked()
{
    QString newPalName = tr("User Palette");
    int newCnt = 1;

    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    QList<QString> paletteNames = palOrganizer->getColorBarList();
    QString tmpPalName = newPalName;

    while (paletteNames.contains(tmpPalName))
    {
        tmpPalName = newPalName + " " + QString::number(newCnt++);
    }
    newPalName = tmpPalName;

    ito::ItomPaletteBase pal = palOrganizer->getColorBar(palOrganizer->getColorBarIndex("gray"));
    m_curPalette = ito::ItomPaletteBase(newPalName, pal.getType(), pal.getInverseColorOne(), \
        pal.getInverseColorTwo(), pal.getInvalidColor(), pal.getColorStops());

    ui.btnInvColor1->setColor(m_curPalette.getInverseColorOne());
    ui.btnInvColor2->setColor(m_curPalette.getInverseColorTwo());
    ui.btnInvColor->setColor(m_curPalette.getInvalidColor());
    ui.lePalName->setText(m_curPalette.getName());

    ui.groupOptions->setEnabled(true);

    drawPalCurves();
    updatePalette();

    m_isDirty = 1;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::pbDuplicateClicked()
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    ito::ItomPaletteBase pal = palOrganizer->getColorBar(ui.lwPalettes->currentRow()); // current palette


    QString tmpPalName = pal.getName() + tr(" - Copy");
    QString newPalName = tmpPalName;

    int newCnt = 1;
    QList<QString> paletteNames = palOrganizer->getColorBarList();

    while (paletteNames.contains(tmpPalName))
    {
        tmpPalName = newPalName + QString(" (%i)").arg(newCnt++);
    }
    newPalName = tmpPalName;

    
    m_curPalette = ito::ItomPaletteBase(newPalName, pal.getType() & (~ito::tPalette::tPaletteReadOnly), pal.getInverseColorOne(), \
        pal.getInverseColorTwo(), pal.getInvalidColor(), pal.getColorStops());

    ui.btnInvColor1->setColor(m_curPalette.getInverseColorOne());
    ui.btnInvColor2->setColor(m_curPalette.getInverseColorTwo());
    ui.btnInvColor->setColor(m_curPalette.getInvalidColor());
    ui.lePalName->setText(m_curPalette.getName());

    ui.groupOptions->setEnabled(true);

    drawPalCurves();
    updatePalette();

    m_isDirty = 1;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::palSpecialColorChanged(QColor color)
{
    m_isDirty = 1;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::pbRemoveClicked()
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (ui.lwPalettes->currentRow())
    {
        ItomPaletteBase pal;
        pal = palOrganizer->getColorBar(ui.lwPalettes->currentRow());
        if (pal.getType() & ito::tPaletteReadOnly)
        {
            QMessageBox::information(this, tr("Palette is read only"), tr("Palette is read only, cannot remove!"));
            return;
        }
        palOrganizer->removeColorbar(ui.lwPalettes->currentRow());
        ui.lwPalettes->takeItem(ui.lwPalettes->currentRow());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::pbSaveClicked()
{
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    if (palOrganizer->getColorBarList().contains(ui.lePalName->text()) && m_curPalette.getType() & ito::tPaletteReadOnly)
    {
        if (QMessageBox::question(this, tr("Palette is readonly"), tr("Trying to overwrite read-only (maybe itom internal) palette. Create copy or discard changes")) == QMessageBox::Yes)
        {
            QString newPalName("User Palette");
            QString tmpPalName = newPalName;
            int newCnt = 1;

            QList<QString> paletteNames = palOrganizer->getColorBarList();
            while (paletteNames.contains(tmpPalName))
                tmpPalName = newPalName + " " + newCnt++;
            newPalName = tmpPalName;
            bool ok;
            QString text = QInputDialog::getText(this, tr("Palette Name"),
                tr("Palette Name:"), QLineEdit::Normal,
                newPalName, &ok);
            if (ok && !text.isEmpty())
                ui.lePalName->setText(text);
            else
                return;
        }
        else
            return;
    }

    m_curPalette = ItomPaletteBase(ui.lePalName->text(), m_curPalette.getType()&~tPaletteReadOnly, m_curPalette.getInverseColorOne(),
        m_curPalette.getInverseColorTwo(), m_curPalette.getInvalidColor(), m_curPalette.getColorStops());
    palOrganizer->setColorBarThreaded(ui.lePalName->text(), m_curPalette, NULL);
    updatePaletteList();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    settings.beginGroup("ito::Palettes");
    foreach(QString child, settings.childGroups())
    {
        settings.beginGroup(child);
        const QString name = settings.value("name").toString();
        int type = settings.value("type").toInt();
        QRgb uinvalidCol = settings.value("invalidColor").toUInt();
        QRgb uinvCol1 = settings.value("inverseColor1").toUInt();
        QRgb uinvCol2 = settings.value("inverseColor2").toUInt();
        QColor invalidCol(uinvalidCol);
        QColor invCol1(uinvCol1);
        QColor invCol2(uinvCol2);
        QVariant numColStops = settings.value("numColorStops");
        QVector<QGradientStop> colorStops;
        for (int ns = 0; ns < numColStops.toInt(); ns++)
        {
            QVariant val = settings.value(QString("cs%1_1").arg(ns));
            QVariant col = settings.value(QString("cs%1_2").arg(ns));
            QColor color(val.toUInt());
            colorStops.append(QGradientStop(val.toFloat(), color));
        }
        ItomPaletteBase newPal(name, type, invCol1, invCol2, invalidCol, colorStops);
        palOrganizer->setColorBarThreaded(name, newPal);

        settings.endGroup();
    }

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);

    settings.beginGroup("ito::Palettes");
    ito::PaletteOrganizer *palOrganizer = (PaletteOrganizer*)AppManagement::getPaletteOrganizer();
    QList<QString> builtInPalettes = palOrganizer->getBuiltInPaletteNames();
    QList<QString> colorBarList = palOrganizer->getColorBarList();
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
            settings.beginGroup(colorBarList[np]);
            ItomPaletteBase pal = palOrganizer->getColorBar(colorBarList[np]);
            settings.setValue("name", pal.getName());
            settings.setValue("type", pal.getType());
            settings.setValue("invalidColor", pal.getInvalidColor().rgb());
            settings.setValue("inverseColor1", pal.getInverseColorOne().rgb());
            settings.setValue("inverseColor2", pal.getInverseColorTwo().rgb());
            QVector<QGradientStop> colorStops = pal.getColorStops();
            settings.setValue("numColorStops", colorStops.length());
            for (int ns = 0; ns < colorStops.length(); ns++)
            {
                settings.setValue(QString("cs%1_1").arg(ns), colorStops[ns].first);
                settings.setValue(QString("cs%1_2").arg(ns), colorStops[ns].second.rgb());
            }
            settings.endGroup();
        }
    }
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropPalettes::on_defaultBtn_clicked()
{
    this->update();
}

//----------------------------------------------------------------------------------------------------------------------------------

}//endNamespace ito