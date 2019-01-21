#ifndef OCCURRENCES_H
#define OCCURRENCES_H

/*
This module contains the occurrences highlighter mode.
*/

#include "textDecoration.h"
#include "mode.h"
#include "delayJobRunner.h"

#include <qfuturewatcher.h>


/*
Highlights the caret line
*/
class OccurrencesHighlighterMode : public QObject, public Mode
{
    Q_OBJECT
public:
    OccurrencesHighlighterMode(const QString &description = "", QObject *parent = NULL);
    virtual ~OccurrencesHighlighterMode();

    QColor background() const;
    void setBackground(const QColor &color);

    QColor foreground() const;
    void setForeground(const QColor &color);

    int delay() const;
    void setDelay(int delay);

    bool underlined() const;
    void setUnderlined(bool value);

    bool caseSensitive() const;
    void setCaseSensitive(bool value);

    bool wholeWord() const;
    void setWholeWord(bool value);

    virtual void onStateChanged(bool state);

private slots:
    void requestHighlight();
    void sendRequest();
    void onResultsAvailable(QList<QPair<int,int> > results);
    void asyncFindAllWatcherFinished();

protected:
    typedef QList<QPair<int, int> > MatchesList;
    void clearDecorations();
    MatchesList findAll(const QString &text, const QString &sub, bool wholeWord, bool caseSensitive);

    QColor m_background;
    QColor m_foreground;
    bool m_underlined;
    bool m_caseSensitive;
    QString m_sub;
    bool m_wholeWord;

    QList<TextDecoration::Ptr> m_decorations;

    QFutureWatcher<MatchesList> m_asyncFindAllWatcher; 

    DelayJobRunnerBase *m_pTimer;
};

#endif
