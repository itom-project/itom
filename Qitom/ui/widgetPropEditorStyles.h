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

#ifndef WIDGETPROPEDITORSTYLES_H
#define WIDGETPROPEDITORSTYLES_H

#include "abstractPropertyPageWidget.h"

#include <QtGui>
#include <qvector.h>
#include <qstring.h>
#include <qfont.h>
#include <qcolor.h>
#include <qstring.h>
#include <qcolor.h>

#include "../codeEditor/syntaxHighlighter/codeEditorStyle.h"

#include "ui_widgetPropEditorStyles.h"

namespace ito
{

class CodeEditorStyle;

class WidgetPropEditorStyles : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    struct StyleNode
    {
        ito::StyleItem::StyleType m_index;
        QString m_name;
        QFont m_font;
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
    bool m_changing;
    CodeEditorStyle* m_pCodeEditorStyle;

    void setFontSizeGeneral(const int fontSizeAdd);

    void writeSettingsInternal(const QString &filename);
    void readSettingsInternal(const QString &filename);

    QString colorStringMixedWithPaperBgColor(const QColor &color);

    QColor m_paperBgcolor;
    QColor m_markerScriptErrorBgcolor;
    QColor m_markerCurrentBgcolor;
    QColor m_markerInputBgcolor;
    QColor m_markerErrorBgcolor;
    QColor m_whitespaceFgcolor;
    QColor m_whitespaceBgcolor;
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
    void on_btnFontSizeDec_clicked();
    void on_btnFontSizeInc_clicked();
    void on_btnReset_clicked();
    void on_btnImport_clicked();
    void on_btnExport_clicked();
    void on_btnTextBackgroundsTransparent_clicked();
};

} //end namespace ito

#endif
