/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#include <qstring.h>
#include <qcolor.h>

#include "ui_widgetPropEditorStyles.h"

namespace ito
{

class WidgetPropEditorStyles : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    struct StyleNode
    {
        StyleNode(int index, QString name, QFont font, bool fillToEOL, QColor foregroundColor, QColor backgroundColor) : m_index(index), m_name(name), m_font(font), m_fillToEOL(fillToEOL), m_foregroundColor(foregroundColor), m_backgroundColor(backgroundColor) {}
        StyleNode() {}
        StyleNode(int index, QString name) : m_index(index), m_name(name), m_fillToEOL(0) {}
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
    bool m_changing;
    
    void setFontSizeGeneral(const int fontSizeAdd);

    void writeSettingsInternal(const QString &filename);
    void readSettingsInternal(const QString &filename);

    QColor m_paperBgcolor;
    QColor m_foldMarginFgcolor;
    QColor m_foldMarginBgcolor;
    QColor m_marginFgcolor;
    QColor m_marginBgcolor;
    QColor m_markerCurrentBgcolor;
    QColor m_markerInputBgcolor;
    QColor m_markerErrorBgcolor;
    QColor m_whitespaceFgcolor;
    QColor m_matchedBraceFgcolor;
    QColor m_matchedBraceBgcolor;
    QColor m_unmatchedBraceFgcolor;
    QColor m_unmatchedBraceBgcolor;
    QColor m_caretBgcolor;
    QColor m_caretFgcolor;
    QColor m_selectionBgcolor;
    QColor m_selectionFgcolor;
    QColor m_markerSameStringBgcolor;

signals:

public slots:

private slots:
    void on_listWidget_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);
    void on_btnBackgroundColor_colorChanged(QColor color);
    void on_btnFont_clicked();
    void on_btnForegroundColor_colorChanged(QColor color);
    void on_checkFillEOL_stateChanged(int state);
    void on_btnFontSizeDec_clicked();
    void on_btnFontSizeInc_clicked();
    void on_btnReset_clicked();
    void on_btnImport_clicked();
    void on_btnExport_clicked();
};

} //end namespace ito

#endif
