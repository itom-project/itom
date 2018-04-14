#include "panel.h"

#include <qapplication.h>
#include <qpainter.h>
#include "codeEditor.h"
#include "managers/panelsManager.h"

//-------------------------------------------------------
Panel::Panel(const QString &name, bool dynamic, const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Mode(name, description),
    QWidget(parent),
    m_dynamic(dynamic),
    m_orderInZone(-1),
    m_scrollable(false),
    m_position(Left)
{
}

//-------------------------------------------------------
Panel::~Panel()
{
}

//-------------------------------------------------------
/*
A scrollable panel will follow the editor's scroll-bars. Left and right
panels follow the vertical scrollbar. Top and bottom panels follow the
horizontal scrollbar.

:type: bool
*/
bool Panel::scrollable() const
{
    return m_scrollable;
}

//-------------------------------------------------------
void Panel::setScrollable(bool value)
{
    m_scrollable = value;
}

//-------------------------------------------------------
int Panel::orderInZone() const
{
    return m_orderInZone;
}

//-------------------------------------------------------
void Panel::setOrderInZone(int orderInZone)
{
    m_orderInZone = orderInZone;
}

//-------------------------------------------------------
Panel::Position Panel::position() const
{
    return m_position;
}

//-------------------------------------------------------
void Panel::setPosition(Position pos)
{
    m_position = pos;
}

//-------------------------------------------------------
/*
Fills the panel background using QPalette
*/
void Panel::paintEvent(QPaintEvent *e)
{
    if (isVisible())
    {
        //fill background
        m_backgroundBrush = QBrush(QColor(palette().window().color()));
        m_foregroundPen = QPen(QColor(palette().windowText().color()));
        QPainter painter(this);
        painter.fillRect(e->rect(), m_backgroundBrush);
    }
}

//-------------------------------------------------------
/*
Extends :meth:`pyqode.core.api.Mode.on_install` method to set the
editor instance as the parent widget.

.. warning:: Don't forget to call **super** if you override this
    method!

:param editor: editor instance
:type editor: pyqode.core.api.CodeEdit
*/
void Panel::onInstall(CodeEditor *editor)
{
    Mode::onInstall(editor);
    setParent(editor);
    setPalette(qApp->palette());

}

//-------------------------------------------------------
/*
Shows/Hides the panel

Automatically call CodeEdit.refresh_panels.

:param visible: Visible state
*/
void Panel::setVisible(bool visible)
{
    QWidget::setVisible(visible);
    if (m_pEditor)
    {
        m_pEditor->panels()->refresh();
    }
}