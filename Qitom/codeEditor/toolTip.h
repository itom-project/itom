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

#ifndef TOOLTIP_H
#define TOOLTIP_H

//#include <QtWidgets/qtwidgetsglobal.h>
#include <QtWidgets/qwidget.h>
#include <qlabel.h>
#include <qbasictimer.h>

class ToolTipLabel : public QLabel
{
    Q_OBJECT
public:
    ToolTipLabel(const QString &text, const QPoint &pos, QWidget *w, int msecDisplayTime);
    ~ToolTipLabel();
    static ToolTipLabel *instance;
    void adjustTooltipScreen(const QPoint &pos);
    void updateSize(const QPoint &pos);
    bool eventFilter(QObject *, QEvent *) override;
    QBasicTimer hideTimer, expireTimer;
    bool fadingOut;
    void reuseTip(const QString &text, int msecDisplayTime, const QPoint &pos);
    void hideTip();
    void hideTipImmediately();
    void setTipRect(QWidget *w, const QRect &r);
    void restartExpireTimer(int msecDisplayTime);
    bool tipChanged(const QPoint &pos, const QString &text, QObject *o);
    void placeTip(const QPoint &pos, QWidget *w, const QPoint &alternativeTopRightPos = QPoint(), bool doNotForceYToBeWithinScreen = false);
#if (QT_VERSION >= QT_VERSION_CHECK(5, 15, 0))
    static QScreen *getTipScreen(const QPoint& pos, QWidget* w);
#else
    static int getTipScreen(const QPoint& pos, QWidget* w);
#endif

protected:
    void timerEvent(QTimerEvent *e) override;
    void paintEvent(QPaintEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void resizeEvent(QResizeEvent *e) override;
#ifndef QT_NO_STYLE_STYLESHEET
public slots:
    /** \internal
      Cleanup the _q_stylesheet_parent propery.
     */
    void styleSheetParentDestroyed() {
        setProperty("_q_stylesheet_parent", QVariant());
        styleSheetParent = 0;
    }
private:
    QWidget *styleSheetParent;
#endif
private:
    QWidget *widget;
    QRect rect;
};

class ToolTip
{
    ToolTip() = delete;
public:
    // ### Qt 6 - merge the three showText functions below
    static void showText(
        const QPoint& pos,
        const QString& text,
        QWidget* w = nullptr,
        const QPoint& alternativeTopRightPos = QPoint(),
        bool doNotForceYToBeWithinScreen = false);
    static void showText(
        const QPoint& pos,
        const QString& text,
        QWidget* w,
        const QRect& rect,
        bool doNotForceYToBeWithinScreen = false);
    static void showText(
        const QPoint& pos,
        const QString& text,
        QWidget* w,
        const QRect& rect,
        int msecShowTime,
        const QPoint& alternativeTopRightPos = QPoint(),
        bool doNotForceYToBeWithinScreen = false);
    static inline void hideText() { showText(QPoint(), QString()); }
    static bool isVisible();
    static QString text();
    static QPalette palette();
    static void setPalette(const QPalette &);
    static QFont font();
    static void setFont(const QFont &);
};


#endif // TOOLTIP_H
