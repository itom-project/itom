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

    --------------------------------
    This class is a modified version of the class QToolTip of the
    Qt framework (licensed under LGPL):
    https://code.woboq.org/qt5/qtbase/src/widgets/kernel/qtooltip.cpp.html
*********************************************************************** */

#include <qguiapplication.h>
#include <qapplication.h>

#include <qevent.h>
#include <qpointer.h>
#include <qstyle.h>
#include <qstyleoption.h>
#include <qstylepainter.h>
#include <qtimer.h>
#include <qscreen.h>

#include <qtextdocument.h>
#include <qdebug.h>

#include <qlabel.h>

#include "toolTip.h"

#if QT_VERSION < QT_VERSION_CHECK(5, 15, 0)
#include <qdesktopwidget.h>
#endif

ToolTipLabel *ToolTipLabel::instance = 0;
ToolTipLabel::ToolTipLabel(const QString &text, const QPoint &pos, QWidget *w, int msecDisplayTime)
#ifndef QT_NO_STYLE_STYLESHEET
    : QLabel(w, Qt::ToolTip | Qt::BypassGraphicsProxyWidget), styleSheetParent(nullptr), widget(nullptr)
#else
    : QLabel(w, Qt::ToolTip | Qt::BypassGraphicsProxyWidget), widget(nullptr)
#endif
{
    delete instance;
    instance = this;
    setForegroundRole(QPalette::ToolTipText);
    setBackgroundRole(QPalette::ToolTipBase);
    setPalette(ToolTip::palette());
    ensurePolished();
    setMargin(1 + style()->pixelMetric(QStyle::PM_ToolTipLabelFrameWidth, nullptr, this));
    setFrameStyle(QFrame::NoFrame);
    setAlignment(Qt::AlignLeft);
    setIndent(1);
    //setTextFormat(Qt::RichText);
    qApp->installEventFilter(this);
    setWindowOpacity(style()->styleHint(QStyle::SH_ToolTipLabel_Opacity, nullptr, this) / 255.0);
    setMouseTracking(true);
    fadingOut = false;
    reuseTip(text, msecDisplayTime, pos);
    setWordWrap(false);
}
void ToolTipLabel::restartExpireTimer(int msecDisplayTime)
{
    int time = 10000 + 40 * qMax(0, text().length() - 100);
    if (msecDisplayTime > 0)
        time = msecDisplayTime;
    expireTimer.start(time, this);
    hideTimer.stop();
}
void ToolTipLabel::reuseTip(const QString &text, int msecDisplayTime, const QPoint &pos)
{
#ifndef QT_NO_STYLE_STYLESHEET
    if (styleSheetParent) {
        disconnect(styleSheetParent, SIGNAL(destroyed()),
            ToolTipLabel::instance, SLOT(styleSheetParentDestroyed()));
        styleSheetParent = nullptr;
    }
#endif
    setText(text);
    updateSize(pos);
    restartExpireTimer(msecDisplayTime);
}
#if (QT_VERSION >= QT_VERSION_CHECK(5, 15, 0))
void ToolTipLabel::updateSize(const QPoint& pos)
{
    // d_func()->setScreenForPoint(pos);
    // Ensure that we get correct sizeHints by placing this window on the right screen.
    QFontMetrics fm(font());
    QSize extra(1, 0);
    // Make it look good with the default ToolTip font on Mac, which has a small descent.
    if (fm.descent() == 2 && fm.ascent() >= 11)
        ++extra.rheight();
    setWordWrap(Qt::mightBeRichText(text()));
    QSize sh = sizeHint();
    const QScreen* screen = getTipScreen(pos, this);
    if (!wordWrap() && sh.width() > screen->geometry().width())
    {
        setWordWrap(true);
        sh = sizeHint();
    }
    resize(sh + extra);
}
#else
void ToolTipLabel::updateSize(const QPoint& pos)
{
#ifndef Q_OS_WINRT
    // ### The code below does not always work well on WinRT
    // (e.g COIN fails an auto test - tst_ItomToolTip::qtbug64550_stylesheet - QTBUG-72652)
    // d_func()->setScreenForPoint(pos);
#endif
    // Ensure that we get correct sizeHints by placing this window on the right screen.
    QFontMetrics fm(font());
    QSize extra(1, 0);
    // Make it look good with the default ToolTip font on Mac, which has a small descent.
    if (fm.descent() == 2 && fm.ascent() >= 11)
        ++extra.rheight();
    setWordWrap(Qt::mightBeRichText(text()));
    QSize sh = sizeHint();
    // ### When the above WinRT code is fixed, windowhandle should be used to find the screen.
    int screenWidth = QApplication::desktop()->geometry().width();
    if (!wordWrap() && sh.width() > screenWidth)
    {
        setWordWrap(true);
        sh = sizeHint();
    }
    resize(sh + extra);
}
#endif

void ToolTipLabel::paintEvent(QPaintEvent *ev)
{
    QStylePainter p(this);
    QStyleOptionFrame opt;
    opt.initFrom(this);
    p.drawPrimitive(QStyle::PE_PanelTipLabel, opt);
    p.end();
    QLabel::paintEvent(ev);
}

void ToolTipLabel::resizeEvent(QResizeEvent *e)
{
    QStyleHintReturnMask frameMask;
    QStyleOption option;
    option.initFrom(this);
    if (style()->styleHint(QStyle::SH_ToolTip_Mask, &option, this, &frameMask))
        setMask(frameMask.region);
    QLabel::resizeEvent(e);
}

void ToolTipLabel::mouseMoveEvent(QMouseEvent *e)
{
    if (!rect.isNull()) {
        QPoint pos = e->globalPos();
        if (widget)
            pos = widget->mapFromGlobal(pos);
        if (!rect.contains(pos))
            hideTip();
    }
    QLabel::mouseMoveEvent(e);
}

ToolTipLabel::~ToolTipLabel()
{
    instance = nullptr;
}

void ToolTipLabel::hideTip()
{
    if (!hideTimer.isActive())
        hideTimer.start(300, this);
}

void ToolTipLabel::hideTipImmediately()
{
    close(); // to trigger QEvent::Close which stops the animation
    deleteLater();
}

void ToolTipLabel::setTipRect(QWidget *w, const QRect &r)
{
    if (Q_UNLIKELY(!r.isNull() && !w)) {
        qWarning("ItomToolTip::setTipRect: Cannot pass null widget if rect is set");
        return;
    }
    widget = w;
    rect = r;
}

void ToolTipLabel::timerEvent(QTimerEvent *e)
{
    if (e->timerId() == hideTimer.timerId()
        || e->timerId() == expireTimer.timerId()) {
        hideTimer.stop();
        expireTimer.stop();
        hideTipImmediately();
    }
}

bool ToolTipLabel::eventFilter(QObject *o, QEvent *e)
{
    switch (e->type()) {
#ifdef Q_OS_MACOS
    case QEvent::KeyPress:
    case QEvent::KeyRelease: {
        const int key = static_cast<QKeyEvent *>(e)->key();
        // Anything except key modifiers or caps-lock, etc.
        if (key < Qt::Key_Shift || key > Qt::Key_ScrollLock)
            hideTipImmediately();
        break;
    }
#endif
    case QEvent::Leave:
        //hideTip();
        break;
#if defined (Q_OS_QNX) // On QNX the window activate and focus events are delayed and will appear
        // after the window is shown.
    case QEvent::WindowActivate:
    case QEvent::FocusIn:
        return false;
    case QEvent::WindowDeactivate:
        if (o != this)
            return false;
        hideTipImmediately();
        break;
    case QEvent::FocusOut:
        if (reinterpret_cast<QWindow*>(o) != windowHandle())
            return false;
        hideTipImmediately();
        break;
#else
    case QEvent::WindowActivate:
    case QEvent::WindowDeactivate:
    case QEvent::FocusIn:
    case QEvent::FocusOut:
#endif
    case QEvent::Close:
        // For QTBUG-55523 (QQC) specifically: Hide tooltip when windows are closed,
        // unless the close event is for the this tooltip, in which case we don't
        // recursively hide/close again.
        if (o != (QObject*)windowHandle() && o != this)
            hideTipImmediately();
        break;
    case QEvent::MouseButtonPress:
    case QEvent::MouseButtonRelease:
    case QEvent::MouseButtonDblClick:
    case QEvent::Wheel:
        hideTipImmediately();
        break;
    case QEvent::MouseMove:
        if (o == widget && !rect.isNull() && !rect.contains(static_cast<QMouseEvent*>(e)->pos()))
            hideTip();
    default:
        break;
    }
    return false;
}

#if (QT_VERSION >= QT_VERSION_CHECK(5, 15, 0))
QScreen *ToolTipLabel::getTipScreen(const QPoint& pos, QWidget* w)
{
    QScreen* guess = w ? w->screen() : QGuiApplication::primaryScreen();
    QScreen* exact = guess->virtualSiblingAt(pos);
    return exact ? exact : guess;
}
#else
int ToolTipLabel::getTipScreen(const QPoint& pos, QWidget* w)
{
    if (QApplication::desktop()->isVirtualDesktop())

        return QApplication::desktop()->screenNumber(pos);

    else

        return QApplication::desktop()->screenNumber(w);
}
#endif

void ToolTipLabel::placeTip(
    const QPoint& pos,
    QWidget* w,
    const QPoint& alternativeTopRightPos /*= QPoint()*/,
    bool doNotForceYToBeWithinScreen /*= false*/)
{
#ifndef QT_NO_STYLE_STYLESHEET
    if (testAttribute(Qt::WA_StyleSheet) || (w)) { // && qt_styleSheet(w->style()))) {
        //the stylesheet need to know the real parent
        ToolTipLabel::instance->setProperty("_q_stylesheet_parent", QVariant::fromValue(w));
        //we force the style to be the QStyleSheetStyle, and force to clear the cache as well.
        ToolTipLabel::instance->setStyleSheet(QLatin1String("/* */"));
        // Set up for cleaning up this later...
        ToolTipLabel::instance->styleSheetParent = w;
        if (w) {
            connect(w, SIGNAL(destroyed()),
                ToolTipLabel::instance, SLOT(styleSheetParentDestroyed()));
            // QTBUG-64550: A font inherited by the style sheet might change the size,
            // particular on Windows, where the tip is not parented on a window.
            ToolTipLabel::instance->updateSize(pos);
        }
    }
#endif //QT_NO_STYLE_STYLESHEET

    QPoint p = pos;
    p += QPoint(2, 16);

    const auto screens = QGuiApplication::screens();
    QRect allScreensRect;

    if (screens.size() > 0)
    {
        allScreensRect = screens[0]->geometry();
    }

    for (int i = 1; i < screens.size(); ++i)
    {
        const QScreen* screen = screens[i];
        allScreensRect = allScreensRect.united(screen->geometry());
    }

    if (p.x() + width() > allScreensRect.x() + allScreensRect.width())
    {
        if (alternativeTopRightPos.isNull())
        {
            p.rx() -= 4 + this->width();
        }
        else
        {
            p.rx() = 2 + alternativeTopRightPos.x() - 4 - this->width();
        }
    }

    if (p.y() + this->height() > allScreensRect.y() + allScreensRect.height())
    {
        if (alternativeTopRightPos.isNull())
        {
            p.ry() -= 24 + this->height();
        }
        else
        {
            p.ry() = 16 + alternativeTopRightPos.y() - 24 - this->height();
        }
    }

    if ((p.y() < allScreensRect.y()) && !doNotForceYToBeWithinScreen)
    {
        p.setY(allScreensRect.y());
    }

    if (p.x() + this->width() > allScreensRect.x() + allScreensRect.width())
    {
        p.setX(allScreensRect.x() + allScreensRect.width() - this->width());
    }

    if (p.x() < allScreensRect.x())
    {
        p.setX(allScreensRect.x());
    }

    if ((p.y() + this->height() > allScreensRect.y() + allScreensRect.height()) &&
        !doNotForceYToBeWithinScreen)
    {
        p.setY(allScreensRect.y() + allScreensRect.height() - this->height());
    }

    this->move(p);
}
bool ToolTipLabel::tipChanged(const QPoint &pos, const QString &text, QObject *o)
{
    if (ToolTipLabel::instance->text() != text)
        return true;
    if (o != widget)
        return true;
    if (!rect.isNull())
        return !rect.contains(pos);
    else
        return false;
}
/*!
    Shows \a text as a tool tip, with the global position \a pos as
    the point of interest. The tool tip will be shown with a platform
    specific offset from this point of interest.
    If you specify a non-empty rect the tip will be hidden as soon
    as you move your cursor out of this area.
    The \a rect is in the coordinates of the widget you specify with
    \a w. If the \a rect is not empty you must specify a widget.
    Otherwise this argument can be \nullptr but it is used to
    determine the appropriate screen on multi-head systems.
    If \a text is empty the tool tip is hidden. If the text is the
    same as the currently shown tooltip, the tip will \e not move.
    You can force moving by first hiding the tip with an empty text,
    and then showing the new tip at the new position.

    \param doNotForceYToBeWithinScreen usually the tooltip is either
        below or above the desired position. However, if the tooltip is
        to big in height, it will be vertically moved to not exceed the
        screen limits. For code completions, this might lead to the
        situation, that the tooltip overlaps the current cursor position.
        If this is not desired, set this value to true.
*/
void ToolTip::showText(
    const QPoint& pos,
    const QString& text,
    QWidget* w,
    const QRect& rect,
    bool doNotForceYToBeWithinScreen /*= false*/)
{
    showText(pos, text, w, rect, -1, QPoint(), doNotForceYToBeWithinScreen);
}
/*!
   \since 5.2
   \overload
   This is similar to ItomToolTip::showText(\a pos, \a text, \a w, \a rect) but with an extra parameter \a msecDisplayTime
   that specifies how long the tool tip will be displayed, in milliseconds.

   \param doNotForceYToBeWithinScreen usually the tooltip is either
        below or above the desired position. However, if the tooltip is
        to big in height, it will be vertically moved to not exceed the
        screen limits. For code completions, this might lead to the
        situation, that the tooltip overlaps the current cursor position.
        If this is not desired, set this value to true.
*/
void ToolTip::showText(
    const QPoint& pos,
    const QString& text,
    QWidget* w,
    const QRect& rect,
    int msecDisplayTime,
    const QPoint& alternativeTopRightPos /* = QPoint()*/,
    bool doNotForceYToBeWithinScreen /*= false*/)
{
    if (ToolTipLabel::instance && ToolTipLabel::instance->isVisible()) { // a tip does already exist
        if (text.isEmpty()) { // empty text means hide current tip
            ToolTipLabel::instance->hideTip();
            return;
        }
        else if (!ToolTipLabel::instance->fadingOut) {
            // If the tip has changed, reuse the one
            // that is showing (removes flickering)
            QPoint localPos = pos;
            if (w)
                localPos = w->mapFromGlobal(pos);
            if (ToolTipLabel::instance->tipChanged(localPos, text, w)) {
                ToolTipLabel::instance->reuseTip(text, msecDisplayTime, pos);
                ToolTipLabel::instance->setTipRect(w, rect);
                ToolTipLabel::instance->placeTip(
                    pos, w, alternativeTopRightPos, doNotForceYToBeWithinScreen);
            }
            return;
        }
    }
    if (!text.isEmpty()) { // no tip can be reused, create new tip:
        QWidget* tipLabelParent = [w]() -> QWidget* {
#ifdef Q_OS_WIN32
            // On windows, we can't use the widget as parent otherwise the window will be
            // raised when the tooltip will be shown
            Q_UNUSED(w);
            return nullptr;
#else
            return w;
#endif
        }();
        new ToolTipLabel(text, pos, tipLabelParent, msecDisplayTime);
        ToolTipLabel::instance->setTipRect(w, rect);
        ToolTipLabel::instance->placeTip(
            pos, w, alternativeTopRightPos, doNotForceYToBeWithinScreen);
        ToolTipLabel::instance->setObjectName(QLatin1String("ToolTip_label"));
        ToolTipLabel::instance->showNormal();
    }
}
/*!
    \overload
    This is analogous to calling ItomToolTip::showText(\a pos, \a text, \a w, QRect())
*/
void ToolTip::showText(
    const QPoint& pos,
    const QString& text,
    QWidget* w,
    const QPoint& alternativeTopRightPos /*= QPoint()*/,
    bool doNotForceYToBeWithinScreen /*= false*/)
{
    ToolTip::showText(
        pos, text, w, QRect(), -1, alternativeTopRightPos, doNotForceYToBeWithinScreen);
}
/*!
    \fn void ItomToolTip::hideText()
    \since 4.2
    Hides the tool tip. This is the same as calling showText() with an
    empty string.
    \sa showText()
*/
/*!
  \since 4.4
  Returns \c true if this tooltip is currently shown.
  \sa showText()
 */
bool ToolTip::isVisible()
{
    return (ToolTipLabel::instance != 0 && ToolTipLabel::instance->isVisible());
}
/*!
  \since 4.4
  Returns the tooltip text, if a tooltip is visible, or an
  empty string if a tooltip is not visible.
 */
QString ToolTip::text()
{
    if (ToolTipLabel::instance)
        return ToolTipLabel::instance->text();
    return QString();
}
Q_GLOBAL_STATIC(QPalette, tooltip_palette)
/*!
    Returns the palette used to render tooltips.
    \note Tool tips use the inactive color group of QPalette, because tool
    tips are not active windows.
*/
QPalette ToolTip::palette()
{
    return *tooltip_palette();
}
/*!
    \since 4.2
    Returns the font used to render tooltips.
*/
QFont ToolTip::font()
{
    return QApplication::font("ToolTipLabel");
}
/*!
    \since 4.2
    Sets the \a palette used to render tooltips.
    \note Tool tips use the inactive color group of QPalette, because tool
    tips are not active windows.
*/
void ToolTip::setPalette(const QPalette &palette)
{
    *tooltip_palette() = palette;
    if (ToolTipLabel::instance)
        ToolTipLabel::instance->setPalette(palette);
}
/*!
    \since 4.2
    Sets the \a font used to render tooltips.
*/
void ToolTip::setFont(const QFont &font)
{
    QApplication::setFont(font, "ToolTipLabel");
}
