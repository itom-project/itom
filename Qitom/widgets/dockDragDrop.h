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

#ifndef DOCKDRAGDROP_H
#define DOCKDRAGDROP_H

#include <qdatetime.h>
#include <qevent.h>
#include <qobject.h>
#include <qpoint.h>
#include <qpointer.h>
#include <qrect.h>
#include <qrubberband.h>
#include <qtabbar.h>
#include <qtimer.h>

namespace ito {

class AbstractDockWidget;
class ScriptDockWidget;
class ScriptEditorWidget;

//-------------------------------------------------------------------------------------
//! tab bar of the script editors, that can start a drag operation if a tab is dragged out of the bar.
/*!
    Moving a tab inside of the tab bar is still handled by the default implementation
    of QTabBar. Only if the tab is moved away from the bar, a drag and drop operation is requested.
*/
class ScriptTabBar : public QTabBar
{
    Q_OBJECT

public:
    explicit ScriptTabBar(QWidget* parent = nullptr);

signals:
    //! emitted if the tab with the given index has been dragged out of the tab bar
    void tabDragOutRequested(int index);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    int m_pressedIndex;
    QPoint m_pressPos;
};

//-------------------------------------------------------------------------------------
//! describes where a dragged script tab or a dragged script window will be docked
struct DockDropTarget
{
    enum Kind
    {
        None, //!< no valid target
        Tab, //!< into the tabs of the given dock widget (for windows: tabified with it)
        Split, //!< a new dock widget next to the given dock widget (side is given by area)
        Area //!< a new dock widget in the given docking area of the main window
    };

    DockDropTarget() : kind(None), tabIndex(-1), area(Qt::NoDockWidgetArea)
    {
    }

    Kind kind;
    QPointer<ScriptDockWidget> dock; //!< target dock widget for kind Tab or Split
    int tabIndex; //!< insertion index for kind Tab (-1: append)
    Qt::DockWidgetArea area; //!< docking area (kind Area) or side of 'dock' (kind Split)
    QRect indicator; //!< global rectangle, that is shown as drop indicator
};

//-------------------------------------------------------------------------------------
//! Controller for all drag and drop operations that dock script editors.
/*!
    The controller handles two different operations:

    1. Tabs of a script editor window can be dragged to another script editor window,
       to an edge of the main window (new docked script window) or outside of all
       windows of itom (new floating script window).
    2. An undocked script window, that is moved by its title bar, can be docked by releasing
       it above the main window.

    While a drag is active, the controller is an application wide event filter in order to
    accept the drag events for all widgets of itom. Dropping is not executed during the
    drop event itself but after the drag has been finished (the source of the drag
    might be deleted by the move).
*/
class DockDragDropController : public QObject
{
    Q_OBJECT

public:
    static DockDragDropController* instance();

    static const char* ScriptTabMimeType;

    //! starts a drag operation for the tab with the given index of the given script window
    void startTabDrag(ScriptDockWidget* source, int index);

    //! starts to observe the movement of the given undocked window. If it is released above
    //! the main window, it is docked.
    void beginWindowDrag(ScriptDockWidget* window);

    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    DockDragDropController();
    ~DockDragDropController();

    DockDropTarget findTarget(const QPoint& globalPos, ScriptDockWidget* exclude) const;
    void showIndicator(const QRect& globalRect);
    void hideIndicator();

    void applyTabDrop(
        ScriptEditorWidget* editor, const DockDropTarget& target, const QPoint& globalPos);
    void applyWindowDrop(ScriptDockWidget* window, const DockDropTarget& target);

    QPixmap createDragPixmap(const QString& text) const;

    static ScriptDockWidget* scriptDockOfWidget(QWidget* widget);

    QRubberBand* m_indicator;
    bool m_tabDragActive;
    DockDropTarget m_pendingTarget; //!< target of the active tab drag

    QTimer m_windowDragTimer;
    QPointer<ScriptDockWidget> m_draggedWindow;
    DockDropTarget m_windowTarget;

private slots:
    void windowDragTick();
};

} // end namespace ito

#endif
