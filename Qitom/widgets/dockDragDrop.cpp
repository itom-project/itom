/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "dockDragDrop.h"

#include "../AppManagement.h"
#include "../organizer/scriptEditorOrganizer.h"
#include "abstractDockWidget.h"
#include "scriptDockWidget.h"
#include "scriptEditorWidget.h"

#include <qapplication.h>
#include <qcursor.h>
#include <qdrag.h>
#include <qfontmetrics.h>
#include <qmainwindow.h>
#include <qmimedata.h>
#include <qpainter.h>
#include <qpalette.h>

namespace ito {

/*static*/ const char* DockDragDropController::ScriptTabMimeType = "application/x-itom-scripttab";

//-------------------------------------------------------------------------------------
static QMainWindow* itomMainWindow()
{
    return qobject_cast<QMainWindow*>(AppManagement::getMainWindow());
}

//=====================================================================================
ScriptTabBar::ScriptTabBar(QWidget* parent /*= nullptr*/) :
    QTabBar(parent), m_pressedIndex(-1)
{
}

//-------------------------------------------------------------------------------------
void ScriptTabBar::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton)
    {
        m_pressPos = event->pos();
        m_pressedIndex = tabAt(m_pressPos);
    }

    QTabBar::mousePressEvent(event);
}

//-------------------------------------------------------------------------------------
void ScriptTabBar::mouseMoveEvent(QMouseEvent* event)
{
    if (m_pressedIndex >= 0 && (event->buttons() & Qt::LeftButton))
    {
        // moving inside of the tab bar is handled by QTabBar. If the cursor leaves the
        // bar by some distance, the tab is dragged out of the bar.
        const int margin = qMax(12, QApplication::startDragDistance() * 2);
        const QRect keepRect = rect().adjusted(-margin, -margin, margin, margin);

        if (!keepRect.contains(event->pos()) &&
            (event->pos() - m_pressPos).manhattanLength() >= QApplication::startDragDistance())
        {
            const int index = m_pressedIndex;
            m_pressedIndex = -1;

            // finish the internal move of QTabBar
            QMouseEvent release(
                QEvent::MouseButtonRelease,
                QPointF(event->pos()),
                Qt::LeftButton,
                Qt::NoButton,
                event->modifiers());
            QTabBar::mouseReleaseEvent(&release);

            emit tabDragOutRequested(index);
            return;
        }
    }

    QTabBar::mouseMoveEvent(event);
}

//-------------------------------------------------------------------------------------
void ScriptTabBar::mouseReleaseEvent(QMouseEvent* event)
{
    m_pressedIndex = -1;
    QTabBar::mouseReleaseEvent(event);
}

//=====================================================================================
/*static*/ DockDragDropController* DockDragDropController::instance()
{
    static DockDragDropController* controller = new DockDragDropController();
    return controller;
}

//-------------------------------------------------------------------------------------
DockDragDropController::DockDragDropController() :
    QObject(nullptr), m_indicator(nullptr), m_tabDragActive(false)
{
    m_windowDragTimer.setInterval(40);
    connect(&m_windowDragTimer, &QTimer::timeout, this, &DockDragDropController::windowDragTick);
}

//-------------------------------------------------------------------------------------
DockDragDropController::~DockDragDropController()
{
    delete m_indicator;
}

//-------------------------------------------------------------------------------------
/*static*/ ScriptDockWidget* DockDragDropController::scriptDockOfWidget(QWidget* widget)
{
    for (QWidget* w = widget; w; w = w->parentWidget())
    {
        ScriptDockWidget* sdw = qobject_cast<ScriptDockWidget*>(w);

        if (sdw)
        {
            return sdw;
        }
    }

    return nullptr;
}

//-------------------------------------------------------------------------------------
void DockDragDropController::showIndicator(const QRect& globalRect)
{
    if (!m_indicator)
    {
        m_indicator = new QRubberBand(QRubberBand::Rectangle);
        m_indicator->setWindowFlags(m_indicator->windowFlags() | Qt::WindowTransparentForInput);
        m_indicator->setAttribute(Qt::WA_TransparentForMouseEvents, true);
        m_indicator->setAttribute(Qt::WA_ShowWithoutActivating, true);
    }

    if (m_indicator->geometry() != globalRect)
    {
        m_indicator->setGeometry(globalRect);
    }

    if (!m_indicator->isVisible())
    {
        m_indicator->show();
    }
}

//-------------------------------------------------------------------------------------
void DockDragDropController::hideIndicator()
{
    if (m_indicator)
    {
        m_indicator->hide();
    }
}

//-------------------------------------------------------------------------------------
//! finds the drop target at the given global position.
/*!
    If exclude is nullptr, a script tab is dragged. Then all script windows (docked or
    not) are possible targets and the docking areas are selected using large edge zones
    of the main window.

    If exclude is given, this undocked script window is dragged by its title bar.
    Since moving such a window above the main window is a common action, which must not
    dock the window by accident, only the edge zones of the main window and the tab
    bars of docked script windows are possible targets.
*/
DockDropTarget DockDragDropController::findTarget(
    const QPoint& globalPos, ScriptDockWidget* exclude) const
{
    DockDropTarget target;
    QMainWindow* mw = itomMainWindow();

    if (!mw)
    {
        return target;
    }

    const bool windowDrag = (exclude != nullptr);
    ScriptDockWidget* hit = nullptr;
    bool overMainWindow = false;
    const QRect mainRect(mw->mapToGlobal(QPoint(0, 0)), mw->size());

    if (!windowDrag)
    {
        QWidget* w = QApplication::widgetAt(globalPos);

        if (!w)
        {
            return target;
        }

        hit = scriptDockOfWidget(w);
        overMainWindow = (w->window() == mw);
    }
    else
    {
        // the dragged window is below the cursor. Therefore, use the geometry instead of widgetAt
        if (!mw->isVisible() || !mainRect.contains(globalPos))
        {
            return target;
        }

        overMainWindow = true;

        foreach (ScriptDockWidget* d, mw->findChildren<ScriptDockWidget*>())
        {
            if (d != exclude && d->docked() && static_cast<QWidget*>(d)->isVisible() &&
                d->contentGlobalRect().contains(globalPos))
            {
                hit = d;
                break;
            }
        }
    }

    if (hit)
    {
        const QRect content = hit->contentGlobalRect();
        const QRect tabBarRect = hit->tabBarGlobalRect();

        if (tabBarRect.contains(globalPos))
        {
            target.kind = DockDropTarget::Tab;
            target.dock = hit;
            target.tabIndex = windowDrag ? -1 : hit->tabInsertIndexAt(globalPos);
            target.indicator = tabBarRect;
            return target;
        }

        if (!windowDrag)
        {
            if (hit->docked())
            {
                // zones at the edges of the docked script window create a new split window
                const int mx = content.width() / 4;
                const int my = content.height() / 4;
                QRect half = content;

                if (globalPos.x() < content.left() + mx)
                {
                    target.area = Qt::LeftDockWidgetArea;
                    half.setRight(content.center().x());
                }
                else if (globalPos.x() > content.right() - mx)
                {
                    target.area = Qt::RightDockWidgetArea;
                    half.setLeft(content.center().x());
                }
                else if (globalPos.y() < content.top() + my)
                {
                    target.area = Qt::TopDockWidgetArea;
                    half.setBottom(content.center().y());
                }
                else if (globalPos.y() > content.bottom() - my)
                {
                    target.area = Qt::BottomDockWidgetArea;
                    half.setTop(content.center().y());
                }

                if (target.area != Qt::NoDockWidgetArea)
                {
                    target.kind = DockDropTarget::Split;
                    target.dock = hit;
                    target.indicator = half;
                    return target;
                }
            }

            // center of the window: append to the tabs
            target.kind = DockDropTarget::Tab;
            target.dock = hit;
            target.tabIndex = -1;
            target.indicator = content;
            return target;
        }
    }

    if (!overMainWindow)
    {
        return target;
    }

    // docking areas at the edges of the main window
    const double bandFactor = windowDrag ? 0.06 : 0.18;
    const int bandX = qMax(windowDrag ? 40 : 0, (int)(mainRect.width() * bandFactor));
    const int bandY = qMax(windowDrag ? 40 : 0, (int)(mainRect.height() * bandFactor));
    QRect indicator = mainRect;

    if (globalPos.x() < mainRect.left() + bandX)
    {
        target.area = Qt::LeftDockWidgetArea;
        indicator.setWidth(mainRect.width() / 3);
    }
    else if (globalPos.x() > mainRect.right() - bandX)
    {
        target.area = Qt::RightDockWidgetArea;
        indicator.setLeft(mainRect.right() - mainRect.width() / 3);
    }
    else if (globalPos.y() < mainRect.top() + bandY)
    {
        target.area = Qt::TopDockWidgetArea;
        indicator.setHeight(mainRect.height() / 3);
    }
    else if (globalPos.y() > mainRect.bottom() - bandY)
    {
        target.area = Qt::BottomDockWidgetArea;
        indicator.setTop(mainRect.bottom() - mainRect.height() / 3);
    }
    else if (!windowDrag)
    {
        // center of the main window: default area for script windows
        target.area = Qt::TopDockWidgetArea;
        indicator.setHeight(mainRect.height() / 2);
    }

    if (target.area != Qt::NoDockWidgetArea)
    {
        target.kind = DockDropTarget::Area;
        target.indicator = indicator;
    }

    return target;
}

//-------------------------------------------------------------------------------------
QPixmap DockDragDropController::createDragPixmap(const QString& text) const
{
    QFont font = QApplication::font();
    QFontMetrics fm(font);
    const QString elided = fm.elidedText(text, Qt::ElideMiddle, 300);
    const int w = fm.horizontalAdvance(elided) + 24;
    const int h = fm.height() + 12;

    QPixmap pm(w, h);
    pm.fill(Qt::transparent);

    QPainter painter(&pm);
    painter.setRenderHint(QPainter::Antialiasing);
    QColor bg = QApplication::palette().color(QPalette::Highlight);
    bg.setAlpha(200);
    painter.setBrush(bg);
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(pm.rect(), 4, 4);
    painter.setPen(QApplication::palette().color(QPalette::HighlightedText));
    painter.setFont(font);
    painter.drawText(pm.rect(), Qt::AlignCenter, elided);
    painter.end();

    return pm;
}

//-------------------------------------------------------------------------------------
void DockDragDropController::startTabDrag(ScriptDockWidget* source, int index)
{
    if (m_tabDragActive || !source || !source->isTabIndexValid(index))
    {
        return;
    }

    QPointer<ScriptEditorWidget> editor = source->editorAtIndex(index);

    if (!editor)
    {
        return;
    }

    QDrag* drag = new QDrag(source);
    QMimeData* mime = new QMimeData();
    mime->setData(ScriptTabMimeType, QByteArray::number(editor->getUID()));
    drag->setMimeData(mime);

    QPixmap pm = createDragPixmap(source->tabTitle(index));
    drag->setPixmap(pm);
    drag->setHotSpot(QPoint(pm.width() / 2, pm.height() / 2));

    m_tabDragActive = true;
    m_pendingTarget = DockDropTarget();
    qApp->installEventFilter(this);

    // the drag object is deleted by Qt after exec returns
    const Qt::DropAction result = drag->exec(Qt::MoveAction);

    qApp->removeEventFilter(this);
    m_tabDragActive = false;
    hideIndicator();

    const DockDropTarget target = m_pendingTarget;
    m_pendingTarget = DockDropTarget();
    const QPoint pos = QCursor::pos();

    if (!editor)
    {
        return;
    }

    if (target.kind != DockDropTarget::None)
    {
        applyTabDrop(editor, target, pos);
    }
    else if (result == Qt::IgnoreAction && QApplication::widgetAt(pos) == nullptr)
    {
        // dropped outside of all windows of itom: open the tab in a new window
        ScriptEditorOrganizer* seo =
            qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

        if (seo)
        {
            seo->moveScriptEditorToNewWindow(editor, pos);
        }
    }
}

//-------------------------------------------------------------------------------------
void DockDragDropController::applyTabDrop(
    ScriptEditorWidget* editor, const DockDropTarget& target, const QPoint& globalPos)
{
    Q_UNUSED(globalPos);
    ScriptEditorOrganizer* seo =
        qobject_cast<ScriptEditorOrganizer*>(AppManagement::getScriptEditorOrganizer());

    if (!seo || !editor)
    {
        return;
    }

    switch (target.kind)
    {
    case DockDropTarget::Tab:
        if (target.dock)
        {
            seo->moveScriptEditor(editor, target.dock, target.tabIndex);
        }
        break;
    case DockDropTarget::Split:
        if (target.dock)
        {
            seo->moveScriptEditorToNewDock(editor, target.area, target.dock);
        }
        break;
    case DockDropTarget::Area:
        seo->moveScriptEditorToNewDock(editor, target.area, nullptr);
        break;
    default:
        break;
    }
}

//-------------------------------------------------------------------------------------
bool DockDragDropController::eventFilter(QObject* watched, QEvent* event)
{
    if (!m_tabDragActive)
    {
        return QObject::eventFilter(watched, event);
    }

    switch (event->type())
    {
    case QEvent::DragEnter:
    case QEvent::DragMove:
    case QEvent::Drop: {
        QDropEvent* de = static_cast<QDropEvent*>(event);

        if (!de->mimeData()->hasFormat(ScriptTabMimeType))
        {
            break;
        }

        const DockDropTarget target = findTarget(QCursor::pos(), nullptr);
        m_pendingTarget = target;

        if (target.kind == DockDropTarget::None)
        {
            hideIndicator();
            de->ignore();
        }
        else
        {
            if (event->type() != QEvent::Drop)
            {
                showIndicator(target.indicator);
            }
            else
            {
                hideIndicator();
            }

            de->setDropAction(Qt::MoveAction);
            de->accept();
        }

        return true; // the event is completely handled here
    }
    case QEvent::DragLeave:
        hideIndicator();
        m_pendingTarget = DockDropTarget();
        break;
    default:
        break;
    }

    return QObject::eventFilter(watched, event);
}

//-------------------------------------------------------------------------------------
void DockDragDropController::beginWindowDrag(ScriptDockWidget* window)
{
    if (!window || window->docked() || m_tabDragActive)
    {
        return;
    }

    if (!m_windowDragTimer.isActive())
    {
        m_draggedWindow = window;
        m_windowTarget = DockDropTarget();
        m_windowDragTimer.start();
    }
}

//-------------------------------------------------------------------------------------
void DockDragDropController::windowDragTick()
{
    if (!m_draggedWindow || m_draggedWindow->docked())
    {
        m_windowDragTimer.stop();
        hideIndicator();
        m_draggedWindow.clear();
        return;
    }

    if (QApplication::mouseButtons() & Qt::LeftButton)
    {
        m_windowTarget = findTarget(QCursor::pos(), m_draggedWindow);

        if (m_windowTarget.kind == DockDropTarget::None)
        {
            hideIndicator();
        }
        else
        {
            showIndicator(m_windowTarget.indicator);
        }
    }
    else
    {
        // the window has been released
        m_windowDragTimer.stop();
        hideIndicator();

        QPointer<ScriptDockWidget> window = m_draggedWindow;
        const DockDropTarget target = m_windowTarget;
        m_draggedWindow.clear();
        m_windowTarget = DockDropTarget();

        if (window && target.kind != DockDropTarget::None)
        {
            applyWindowDrop(window, target);
        }
    }
}

//-------------------------------------------------------------------------------------
void DockDragDropController::applyWindowDrop(
    ScriptDockWidget* window, const DockDropTarget& target)
{
    QMainWindow* mw = itomMainWindow();

    if (!mw || !window || window->docked())
    {
        return;
    }

    // dockWidget() transforms the window into a dock widget and docks it at its last
    // position in the main window. Afterwards, it is moved to the selected position.
    window->dockWidget();

    switch (target.kind)
    {
    case DockDropTarget::Tab:
        if (target.dock && target.dock != window)
        {
            mw->tabifyDockWidget(target.dock, window);
        }
        break;
    case DockDropTarget::Area:
        mw->addDockWidget(target.area, window);
        break;
    default:
        break;
    }

    window->setFloating(false);
    window->show();
    window->raiseAndActivate();
}

} // end namespace ito
