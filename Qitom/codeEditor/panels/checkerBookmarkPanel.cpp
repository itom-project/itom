#include "checkerBookmarkPanel.h"

#include "../codeEditor.h"
#include <qpainter.h>
#include <qapplication.h>
#include <qicon.h>
#include <qtooltip.h>
#include "../syntaxHighlighter/textBlockUserData.h"
#include "../codeEditor.h"
#include "../delayJobRunner.h"


//----------------------------------------------------------
/*
*/
CheckerBookmarkPanel::CheckerBookmarkPanel(const QString &description /*= ""*/, QWidget *parent /*= NULL*/) :
    Panel("CheckerBookmarkPanel", false, description, parent),
    m_previousLine(-1),
    m_pJobRunner(NULL)
{
    setScrollable(true);
    setMouseTracking(true);

    m_pJobRunner = new DelayJobRunner<CheckerBookmarkPanel, void(CheckerBookmarkPanel::*)(QList<QVariant>)>(100);
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
    QFontMetricsF metrics(editor()->font());
    QSize size_hint(metrics.height(), metrics.height());
    if (size_hint.width() > 16)
    {
        size_hint.setWidth(16);
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
QList<CheckerMessage> CheckerBookmarkPanel::markersForLine(int line) const
{
    QTextBlock block = editor()->document()->findBlockByLineNumber(line);
    TextBlockUserData* tbud = dynamic_cast<TextBlockUserData*>(block.userData());
    if (tbud)
    {
        return tbud->m_checkerMessages;
    }
    else
    {
        return QList<CheckerMessage>();
    }
}

//----------------------------------------------------------
/*static*/ QIcon CheckerBookmarkPanel::iconFromMessages(bool hasCheckerMessages, bool hasBookmark, CheckerMessage::CheckerStatus checkerStatus)
{
    if (!hasBookmark && hasCheckerMessages)
    {
        switch (checkerStatus)
        {
        case CheckerMessage::StatusInfo:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\dialog-info.png");
        case CheckerMessage::StatusWarning:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\dialog-warning.png");
        case CheckerMessage::StatusError:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\dialog-error.png");
        }
    }
    else if (hasBookmark && !hasCheckerMessages)
    {
        return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\bookmark.png");
    }
    else if (hasBookmark && hasCheckerMessages)
    {
        switch (checkerStatus)
        {
        case CheckerMessage::StatusInfo:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\bookmarkSyntaxError.png");
        case CheckerMessage::StatusWarning:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\bookmarkSyntaxError.png");
        case CheckerMessage::StatusError:
            return QIcon("D:\\itom-git2\\sources\\itom\\codeEditor\\pysources\\pyqode.core-master\\forms\\rc\\bookmarkSyntaxError.png");
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
    CheckerMessage::CheckerStatus worstStatus = CheckerMessage::StatusInfo;
    bool hasCheckerMessage;
    QRect rect;

    foreach (const VisibleBlock &b, editor()->visibleBlocks())
    {
        worstStatus = CheckerMessage::StatusInfo;
        hasCheckerMessage = false;

        tbud = dynamic_cast<TextBlockUserData*>(b.textBlock.userData());

        if (tbud)
        {
            foreach (const CheckerMessage &cm,tbud->m_checkerMessages)
            {
                hasCheckerMessage = true;
                if (cm.m_status > worstStatus)
                {
                    worstStatus = cm.m_status;
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
    if (line)
    {
        QList<CheckerMessage> markers = markersForLine(line);
        QStringList texts;
        foreach (const CheckerMessage &cm, markers)
        {
            if (cm.m_description != "")
            {
                texts.append(cm.m_description);
            }
        }

        QString text = texts.join("\n");

        if (markers.size() > 0)
        {
            if (m_previousLine != line)
            {
                int top = editor()->linePosFromNumber(markers[0].m_line);
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

    QTextBlock block = editor()->document()->findBlockByLineNumber(line);
    TextBlockUserData* tbud = dynamic_cast<TextBlockUserData*>(block.userData());
    if (tbud)
    {
        if (e->button() == Qt::LeftButton)
        {
            emit removeBookmarkRequested(line);
        }
    }
    else
    {
        emit addBookmarkRequested(line);
    }
}

//----------------------------------------------------------
/*
Display tooltip at the specified top position.
*/
void CheckerBookmarkPanel::displayTooltip(QList<QVariant> args)
{
    QString tooltip = args[0].toString();
    int top = args[1].toInt();

    QToolTip::showText(mapToGlobal(QPoint(
            sizeHint().width(), top)), tooltip, this);
}


