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

#ifndef CODECOMPLETION_H
#define CODECOMPLETION_H


#include "../../python/pythonJedi.h"

#include "../utils/utils.h"
#include "../toolTip.h"
#include "../mode.h"
#include <qevent.h>
#include <qobject.h>
#include <qpair.h>
#include <qstring.h>
#include <qlist.h>
#include <qsortfilterproxymodel.h>
#include <qregularexpression.h>
#include <qcompleter.h>
#include <qsharedpointer.h>

class QStandardItemModel;

namespace ito {

/*
This module contains the code completion mode and the related classes.
*/

/*
Performs subsequence matching/sorting (see pyQode/pyQode#1)
*/
class SubsequenceSortFilterProxyModel : public QSortFilterProxyModel
{
public:
    SubsequenceSortFilterProxyModel(Qt::CaseSensitivity caseSensitivity, QObject *parent = NULL);

    void setPrefix(const QString &prefix);

protected:
    virtual bool filterAcceptsRow(int source_row, const QModelIndex &source_parent) const;

    Qt::CaseSensitivity m_caseSensitivity;
    QList<QRegularExpression> m_filterPatterns;
    QList<QRegularExpression> m_filterPatternsCaseSensitive;
    QList<QRegularExpression> m_sortPatterns;
    QString m_prefix;
};

/*
QCompleter specialised for subsequence matching
*/
class SubsequenceCompleter : public QCompleter
{
public:
    SubsequenceCompleter(QObject *parent = NULL);
    void setModel(QAbstractItemModel *model);
    void updateModel();
    virtual QStringList splitPath(const QString &path) const;

protected:
    QString m_localCompletionPrefix;
    SubsequenceSortFilterProxyModel *m_pFilterProxyModel;
    QAbstractItemModel *m_pSourceModel;
    bool m_forceNextUpdate;
};


/*
Provides code completions when typing or when pressing Ctrl+Space.

This mode provides a code completion system which is extensible.
It takes care of running the completion request in a background process
using one or more completion provider and display the results in a
QCompleter.

To add code completion for a specific language, you only need to
implement a new
:class:`pyqode.core.backend.workers.CodeCompletionWorker.Provider`

The completion popup is shown when the user press **ctrl+space** or
automatically while the user is typing some code (this can be configured
using a series of properties).
*/
class CodeCompletionMode : public QObject, public Mode
{
    Q_OBJECT
public:
    CodeCompletionMode(const QString &name, const QString &description = "", QObject *parent = NULL);
    virtual ~CodeCompletionMode();

    enum FilterMode
    {
        FilterPrefix = 0,   //Filter completions based on the prefix, FAST
        FilterContains = 1, //Filter completions based on whether the prefix is contained in the
                            //suggestion. Only available with PyQt5, if set with PyQt4, FILTER_PREFIX
                            //will be used instead. FAST
        FilterFuzzy = 2,    //Fuzzy filtering, using the subsequence matcher. This is the most powerful filter mode but also the SLOWEST.
    };

    FilterMode filterMode() const;
    void setFilterMode(FilterMode mode);

    Qt::Key triggerKey() const;
    void setTriggerKey(Qt::Key key);

    bool selectWithReturn() const;
    void setSelectWithReturn(bool select);

    int triggerLength() const;
    void setTriggerLength(int length);

    QStringList triggerSymbols() const;
    void setTriggerSymbols(const QStringList &symbols);

    bool caseSensitive() const;
    void setCaseSensitive(bool cs);

    QString completionPrefix() const;

    bool showTooltips() const;
    void setShowTooltips(bool show);

    int tooltipsMaxLength() const;
    void setTooltipsMaxLength(int length);

    virtual void onStateChanged(bool state);
    virtual void onInstall(CodeEditor *editor);
    virtual void onUninstall();

    void hidePopup();

private slots:
    void onJediCompletionResultAvailable(int line, int col, int requestId, QVector<ito::JediCompletion> completions);

    virtual void onKeyPressed(QKeyEvent *e);
    virtual void onKeyReleased(QKeyEvent *e);
    virtual void onFocusIn(QFocusEvent *e);

    void insertCompletion(const QString &completion);
    void onSelectedCompletionChanged(const QString &completion);
    void displayCompletionTooltip(const QString &completion) const;

protected:
    bool requestCompletion();

    void createCompleter();


    void handleCompleterEvents(QKeyEvent *e);
    bool isPopupVisible() const;
    void resetSyncDataAndHidePopup();
    bool isShortcut(QKeyEvent *e) const;
    QRect getPopupRect() const;
    void showPopup(int index = 0);
    void showCompletions(const QVector<JediCompletion> &completions);
    QStandardItemModel* updateModel(const QVector<JediCompletion> &completions);

    /*
    \returns (stringlist os signatures, docstring)
    */
    QPair<QStringList, QString> parseTooltipDocstring(const QString &docstring) const;

    static bool isNavigationKey(QKeyEvent *e);

private:
    QObject *m_pPythonEngine;
    int m_requestCount;

    /* maps the completion name to a a list of (string list of signatures and a docstring) */
    QMap<QString, QList<QPair<QStringList, QString>>> m_tooltips;
    bool m_showTooltips;
    QCompleter *m_pCompleter;
    QString m_completionPrefix;
    bool m_caseSensitive;
    int m_lastCursorColumn;
    int m_lastCursorLine;
    Qt::Key m_triggerKey;
    int m_requestId; //!< auto-incremented number for the last enqueued completion request.
    int m_lastRequestId;
    QString m_currentCompletion;
    QStringList m_triggerSymbols;
    int m_triggerLen;
    FilterMode m_filterMode;
    int m_tooltipsMaxLength;
    bool m_selectWithReturn;
};

} //end namespace ito

#endif
