/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef WIDGETPROPEDITORSTYLES_H
#define WIDGETPROPEDITORSTYLES_H

#include "abstractPropertyPageWidget.h"

#include <QtGui>
#include <qvector.h>
#include <qstring.h>
#include <qfont.h>
#include <qcolor.h>

// Under Windows, define QSCINTILLA_MAKE_DLL to create a Scintilla DLL, or
// define QSCINTILLA_DLL to link against a Scintilla DLL, or define neither
// to either build or link against a static Scintilla library.
//!< this text is coming from qsciglobal.h
#define QSCINTILLA_DLL  //http://www.riverbankcomputing.com/pipermail/qscintilla/2007-March/000034.html

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>

#include "ui_widgetPropEditorStyles.h"

class WidgetPropEditorStyles : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:

    struct StyleNode
    {
        StyleNode(int index, QString name, QFont font, bool fillToEOL, QColor foregroundColor, QColor backgroundColor) : m_index(index), m_name(name), m_font(font), m_fillToEOL(fillToEOL), m_foregroundColor(foregroundColor), m_backgroundColor(backgroundColor) {};
        StyleNode() {};
        StyleNode(int index, QString name) : m_index(index), m_name(name) {};
        int m_index;
        QString m_name;
        QFont m_font;
        bool m_fillToEOL;
        QColor m_foregroundColor;
        QColor m_backgroundColor;
    };

    WidgetPropEditorStyles(QWidget *parent = NULL);
    ~WidgetPropEditorStyles();

    void readSettings();
    void writeSettings();

protected:

private:
    Ui::WidgetPropEditorStyles ui;

    QVector<StyleNode> m_styles;

    QsciLexerPython* qSciLex;
    

signals:

public slots:

private slots:
    void on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);
    void on_btnBackgroundColor_clicked();
    void on_btnFont_clicked();
    void on_btnForegroundColor_clicked();
    void on_checkFillEOL_stateChanged(int state);

};

#endif