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
*********************************************************************** */

#ifndef WIDGETFINDWORD_H
#define WIDGETFINDWORD_H

#include <qwidget.h>
#include <qstring.h>
#include <qevent.h>

#include "ui_widgetFindWord.h"

namespace ito
{

class WidgetFindWord : public QWidget
{
    Q_OBJECT

public:
    WidgetFindWord(QWidget *parent = NULL);
    ~WidgetFindWord();

    void setCursorToTextField();
	void setText(const QString &text);

protected:
    bool eventFilter(QObject *obj, QEvent *event);

private:
    Ui::WidgetFindWord ui;

private slots:
    void on_cmdClose_clicked();
    void on_txtFind_returnPressed();
    void on_cmdFindUp_clicked();
    void on_cmdFindDown_clicked();
    void on_txtFind_textChanged ( const QString & text );

public slots:
    void setFindBarEnabled(bool enabled, bool reduced);
    void setSuccessState(bool successfull);


signals:
    void findNext(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward = true, bool isQuickSeach = true);
    void hideSearchBar();
};

} //end namespace ito

#endif
