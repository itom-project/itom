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

    Further hints:
    ------------------------

    This file belongs to the code editor of itom. The code editor is
    in major parts a fork / rewritten version of the python-based source
    code editor PyQode from Colin Duquesnoy and others
    (see https://github.com/pyQode). PyQode itself is licensed under
    the MIT License (MIT).

    Some parts of the code editor of itom are also inspired by the
    source code editor of the Spyder IDE (https://github.com/spyder-ide),
    also licensed under the MIT License and developed by the Spyder Project
    Contributors.

*********************************************************************** */

#include "checkerBookmarkPanel.h"

#include "../codeEditor.h"
#include <qpainter.h>
#include <qapplication.h>
#include <qicon.h>
#include <qmenu.h>
#include <qtooltip.h>
#include "../textBlockUserData.h"
#include "../codeEditor.h"
#include "../delayJobRunner.h"

#include "helper/guiHelper.h"

namespace ito {

//----------------------------------------------------------
/*
*/
CheckerBookmarkPanel::CheckerBookmarkPanel(BookmarkModel *bookmarkModel, const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Panel("CheckerBookmarkPanel", false, description, parent),
    m_previousLine(-1),
    m_pJobRunner(NULL),
    m_contextMenuLine(-1),
    m_pBookmarkModel(bookmarkModel)
{
    setScrollable(true);
    setMouseTracking(true);
    setContextMenuPolicy(Qt::DefaultContextMenu);

    m_pJobRunner = new DelayJobRunner<CheckerBookmarkPanel, void(CheckerBookmarkPanel::*)(QList<QVariant>)>(100);

    m_pContextMenu = new QMenu(this);
    m_contextMenuActions["toggleBM"] = m_pContextMenu->addAction(QIcon(":/bookmark/icons/bookmarkToggle.png"), tr("&Toggle Bookmark"), this, SLOT(menuToggleBookmark()));
    m_contextMenuActions["nextBM"] = m_pBookmarkModel->bookmarkNextAction();
    m_contextMenuActions["prevBM"] = m_pBookmarkModel->bookmarkPreviousAction();
    m_contextMenuActions["clearAllBM"] = m_pBookmarkModel->bookmarkClearAllAction();
}

//----------------------------------------------------------
/*
*/
CheckerBookmarkPanel::~CheckerBookmarkPanel()
{
    m_pJobRunner->deleteLater();
    m_pJobRunner = NULL;
}

//----------------------------------------------------------
/*
*/
void CheckerBookmarkPanel::onUninstall()
{
    DELAY_JOB_RUNNER(m_pJobRunner, CheckerBookmarkPanel, void(CheckerBookmarkPanel::*)(QList<QVariant>))->cancelRequests();
    Panel::onUninstall();
}

//------------------------------------------------------------
/*
Returns the panel size hint. (fixed with of 16px)
*/
QSize CheckerBookmarkPanel::sizeHint() const
{
    int dpi = GuiHelper::getScreenLogicalDpi();
    int size = 16 * dpi / 96;

    QFontMetricsF metrics(editor()->font());
    QSize size_hint(metrics.height(), metrics.height());
    if (size_hint.width() > size)
    {
        size_hint.setWidth(size);
    }
    return size_hint;
}

//------------------------------------------------------------
/*
Returns the marker that is displayed at the specified line number if
any.

:param line: The marker line.

:return: Marker of None
:rtype: pyqode.core.Marker
*/
QList<CodeCheckerItem> CheckerBookmarkPanel::markersForLine(int line) const
{
    QTextBlock block = editor()->document()->findBlockByNumber(line);
    TextBlockUserData* tbud = dynamic_cast<TextBlockUserData*>(block.userData());
    if (tbud)
    {
        return tbud->m_checkerMessages;
    }
    else
    {
        return QList<CodeCheckerItem>();
    }
}

//----------------------------------------------------------
/*static*/ QIcon CheckerBookmarkPanel::iconFromMessages(bool hasCheckerMessages, bool hasBookmark, CodeCheckerItem::CheckerType checkerStatus)
{
    if (!hasBookmark && hasCheckerMessages)
    {
        switch (checkerStatus)
        {
        case CodeCheckerItem::Info:
            return QIcon(":/script/icons/checkerInfo.png");
        case CodeCheckerItem::Warning:
            return QIcon(":/script/icons/checkerWarning.png");
        case CodeCheckerItem::Error:
            return QIcon(":/script/icons/syntaxError.png");
        }
    }
    else if (hasBookmark && !hasCheckerMessages)
    {
        return QIcon(":/bookmark/icons/bookmark.png");
    }
    else if (hasBookmark && hasCheckerMessages)
    {
        switch (checkerStatus)
        {
        case CodeCheckerItem::Info:
            return QIcon(":/script/icons/bookmarkAndInfo.png");
        case CodeCheckerItem::Warning:
            return QIcon(":/script/icons/bookmarkAndWarning.png");
        case CodeCheckerItem::Error:
            return QIcon(":/script/icons/bookmarkSyntaxError.png");
        }
    }

    return QIcon();
}

//----------------------------------------------------------
void CheckerBookmarkPanel::paintEvent(QPaintEvent *e)
{
    Panel::paintEvent(e);
    QPainter painter(this);
    TextBlockUserData *tbud;
    QIcon icon;
    CodeCheckerItem::CheckerType worstStatus = CodeCheckerItem::Info;
    bool hasCheckerMessage;
    QRect rect;

    foreach (const VisibleBlock &b, editor()->visibleBlocks())
    {
        worstStatus = CodeCheckerItem::Info;
        hasCheckerMessage = false;

        tbud = dynamic_cast<TextBlockUserData*>(b.textBlock.userData());

        if (tbud)
        {
            foreach (const CodeCheckerItem &cci,tbud->m_checkerMessages)
            {
                hasCheckerMessage = true;

                if (cci.type() > worstStatus)
                {
                    worstStatus = cci.type();
                }
            }

            icon = iconFromMessages(hasCheckerMessage, tbud->m_bookmark, worstStatus);

            if (!icon.isNull())
            {
                rect = QRect();
                rect.setX(0);
                rect.setY(b.topPosition);
                rect.setWidth(sizeHint().width());
                rect.setHeight(sizeHint().height());
                icon.paint(&painter, rect);
            }
        }
    }
}

//----------------------------------------------------------
/*
Hide tooltip when leaving the panel region.
*/
void CheckerBookmarkPanel::leaveEvent(QEvent *e)
{
    QToolTip::hideText();
    m_previousLine = -1;
}

//----------------------------------------------------------
/*
*/
void CheckerBookmarkPanel::mouseMoveEvent(QMouseEvent *e)
{
    // Requests a tooltip if the cursor is currently over a marker.
    int line = editor()->lineNbrFromPosition(e->pos().y());
    if (line != -1)
    {
        QList<CodeCheckerItem> markers = markersForLine(line);
        QStringList texts;
        int msgTypes = 0;

        foreach(const CodeCheckerItem &cci, markers)
        {
            msgTypes |= cci.type();
        }

        bool addShortType = !(msgTypes == CodeCheckerItem::Info ||
            msgTypes == CodeCheckerItem::Warning ||
            msgTypes == CodeCheckerItem::Error);

        //all messages are of the same type
        foreach(const CodeCheckerItem &cm, markers)
        {
            texts.append(cm.checkerItemText(addShortType, 100));
        }

        QString text = texts.join("\n");

        if (text != "")
        {
            if (m_previousLine != line)
            {
                int top = editor()->linePosFromNumber(line);
                if (top > 0)
                {
                    QList<QVariant> args;
                    args << text;
                    args << top;
                    DELAY_JOB_RUNNER(m_pJobRunner, CheckerBookmarkPanel, void(CheckerBookmarkPanel::*)(QList<QVariant>))->requestJob( \
                        this, &CheckerBookmarkPanel::displayTooltip, args);
                }
            }
        }
        else
        {
            DELAY_JOB_RUNNER(m_pJobRunner, CheckerBookmarkPanel, void(CheckerBookmarkPanel::*)(QList<QVariant>))->cancelRequests();
        }
        m_previousLine = line;
    }
}

//----------------------------------------------------------
/*
*/
void CheckerBookmarkPanel::mouseReleaseEvent(QMouseEvent *e)
{
    /*# Handle mouse press:
    # - emit add marker signal if there were no marker under the mouse
    #   cursor
    # - emit remove marker signal if there were one or more markers under
    #   the mouse cursor.*/
    int line = editor()->lineNbrFromPosition(e->pos().y());

    if (line != -1)
    {
        if (e->button() == Qt::LeftButton)
        {
            emit toggleBookmarkRequested(line);
        }
    }
}

//----------------------------------------------------------
/*

*/
/*virtual*/ void CheckerBookmarkPanel::contextMenuEvent (QContextMenuEvent * e)
{
    e->accept();

    int line = editor()->lineNbrFromPosition(e->pos().y());

    if (line != -1)
    {
        m_contextMenuLine = line;
        m_pContextMenu->exec(e->globalPos());
    }

    m_contextMenuLine = -1;
}

//----------------------------------------------------------
/*
Display tooltip at the specified top position.
*/
void CheckerBookmarkPanel::displayTooltip(QList<QVariant> args)
{
    QString tooltip = args[0].toString();
    int top = args[1].toInt();

    QToolTip::showText(mapToGlobal(QPoint(sizeHint().width(), top)), tooltip, this);
}


//----------------------------------------------------------------------------------------------------------------------------------
void CheckerBookmarkPanel::menuToggleBookmark()
{
    if (m_contextMenuLine >= 0)
    {
        emit toggleBookmarkRequested(m_contextMenuLine);
    }
}


} //end namespace ito
