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

#ifndef OCCURRENCES_H
#define OCCURRENCES_H

/*
This module contains the occurrences highlighter mode.
*/

#include "../textDecoration.h"
#include "../mode.h"
#include "../delayJobRunner.h"

#include <qfuturewatcher.h>


namespace ito {
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

    bool selectOnDoubleClick() const;
    void setSelectOnDoubleClick(bool value);

    virtual void onStateChanged(bool state);

private slots:
    void requestHighlightPosChanged();
    void requestHighlightDoubleClick();
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
    bool m_selectOnDoubleClick;

    QList<TextDecoration::Ptr> m_decorations;

    QFutureWatcher<MatchesList> m_asyncFindAllWatcher;

    DelayJobRunnerBase *m_pTimer;
};

} //end namespace ito

#endif
