#ifndef CHECKERBOOKMARKPANEL_H
#define CHECKERBOOKMARKPANEL_H

/*
Checker panels:

- CheckerPanel: draw checker messages in front of each line
- GlobalCheckerPanel: draw all checker markers as colored rectangle to
  offer a global view of all errors
*/

#include "../panel.h"
#include "../utils/utils.h"
#include "../textBlockUserData.h"

#include <qevent.h>
#include <qsize.h>
#include <qcolor.h>

class DelayJobRunnerBase;
/*
Shows messages collected by one or more checker modes
*/
class CheckerBookmarkPanel : public Panel
{
    Q_OBJECT
public:
    CheckerBookmarkPanel(const QString &description = "", QWidget *parent = NULL);
    virtual ~CheckerBookmarkPanel();

    virtual QSize sizeHint() const;
    
    virtual void onUninstall();

    QList<CheckerMessage> markersForLine(int line) const;

    static QIcon iconFromMessages(bool hasCheckerMessages, bool hasBookmark, CheckerMessage::CheckerStatus checkerStatus);

protected:
    virtual void paintEvent(QPaintEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void leaveEvent(QEvent *e);

protected:
    void displayTooltip(QList<QVariant> args);

private:
    int m_previousLine;
    DelayJobRunnerBase *m_pJobRunner;

signals:
    void removeBookmarkRequested(int line);
    void addBookmarkRequested(int line);
};

#endif


