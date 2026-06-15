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

#ifndef CODEEDITORSTYLE_H
#define CODEEDITORSTYLE_H

#include <qstring.h>
#include <qbrush.h>
#include <QTextCharFormat>
#include <qmetaobject.h>
#include <qlist.h>
#include <qmap.h>

namespace ito {

//------------------------------------------------------------------
/*
*/
class StyleItem : public QObject
{
    Q_OBJECT

public:
    enum StyleGroup
    {
        GroupCommentOrString = QTextFormat::UserObject,
        GroupNumber = QTextFormat::UserObject + 1
    };

    enum StyleType
    {
        //the pre-defined indices are used to provide a backward-compatibility with old Scintilla styles!
        KeyDefault = 0,
        KeyComment = 1,
        KeyNumber = 2,
        KeyString = 3,
        KeyKeyword = 5,
        KeyDocstring = 6,
        KeyClass = 8,
        KeyFunction = 9,
        KeyOperator = 10,
        KeyDecorator = 15,

        KeyBackground = 1000, /*special style: directly set by settings*/
        KeyHighlight, /* not used any more, should be there to keep ids in settings. */
        KeyNamespace,
        KeyType,
        KeyKeywordReserved,
        KeyBuiltin,
        KeyDefinition,
        KeyInstance,
        KeyWhitespace, /*special style: directly set by settings*/
        KeyTag,
        KeySelf,
        KeyPunctuation,
        KeyConstant,
        KeyOperatorWord,
        KeyStreamOutput,
        KeyStreamError
    };

    //Q_ENUM exposes a meta object to the enumeration types, such that the key names for the enumeration
    //values are always accessible.
    Q_ENUM(StyleType)

    StyleItem() : m_valid(false) {}

    StyleItem(StyleType type, const QTextCharFormat &format);

    StyleItem(const StyleItem &item);

    StyleItem& operator= (const StyleItem &rhs);

    bool isValid() const { return m_valid; }
    QString name() const { return m_name; }
    StyleType type() const { return m_type; }
    QTextCharFormat& rformat() { return m_format; }
    QTextCharFormat format() const { return m_format; }

    static QMetaEnum styleTypeEnum();

    static QList<StyleItem::StyleType> availableStyleTypes();

    static QTextCharFormat createFormat(const QBrush &color, const QBrush &bgcolor = QBrush(), \
                bool bold = false, bool italic = false, bool underline = false, \
                QFont::StyleHint styleHint = QFont::SansSerif);

    static QTextCharFormat createFormat(const QString &familyName, int pointSize, const QColor &color, \
        const QColor &bgcolor = QColor(), bool bold = false, QFont::StyleHint styleHint = QFont::SansSerif);

    void setObjectType(int type) { m_format.setObjectType(type); }

private:
    StyleType m_type;
    QString m_name;
    QTextCharFormat m_format;
    bool m_valid;
};

//------------------------------------------------------------------
/*
*/
class CodeEditorStyle
{
public:
    CodeEditorStyle();
    virtual ~CodeEditorStyle();

    int numStyles() const { return m_formats.size(); }
    QList<int> styleKeys() const { return m_formats.keys(); }
    StyleItem operator[](StyleItem::StyleType type) const;
    StyleItem& operator[](StyleItem::StyleType type);
    StyleItem at(StyleItem::StyleType type) const { return (*this)[type]; }
    StyleItem& at(StyleItem::StyleType type) { return (*this)[type]; }
    QTextCharFormat format(StyleItem::StyleType type) const;
    QTextCharFormat& rformat(StyleItem::StyleType type);

    QColor background() const;
    void setBackground(const QColor &color);

private:
    QMap<int, StyleItem> m_formats;
};

} //end namespace ito

#endif
