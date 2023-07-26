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

#include "codeEditorStyle.h"

#include <qdebug.h>

namespace ito {

//------------------------------------------------------------------
StyleItem::StyleItem(StyleType type, const QTextCharFormat &format) :
        m_type(type),
        m_format(format),
        m_valid(true)
{
    switch (type)
    {
    case KeyBackground:
        m_name = ""; //QObject::tr("Paper color");
        break;

    case KeyDefault:
        m_name = QObject::tr("Default");
        break;
    case KeyKeyword:
        m_name = QObject::tr("Keyword");
        break;
    case KeyNamespace:
        m_name = QObject::tr("Namespace");
        break;
    case KeyType:
        m_name = QObject::tr("Type");
        break;
    case KeyKeywordReserved:
        m_name = QObject::tr("Keyword Reserved");
        break;
    case KeyBuiltin:
        m_name = QObject::tr("Builtin");
        break;
    case KeyDefinition:
        m_name = QObject::tr("Definition");
        break;
    case KeyComment:
        m_name = QObject::tr("Comment");
        m_format.setObjectType(GroupCommentOrString);
        break;
    case KeyString:
        m_name = QObject::tr("String");
        m_format.setObjectType(GroupCommentOrString);
        break;
    case KeyDocstring:
        m_name = QObject::tr("Docstring");
        m_format.setObjectType(GroupCommentOrString);
        break;
    case KeyNumber:
        m_name = QObject::tr("Number");
        m_format.setObjectType(GroupNumber);
        break;
    case KeyInstance:
        m_name = QObject::tr("Instance");
        break;
    case KeyWhitespace:
        m_name = ""; //special style
        break;
    case KeyTag:
        m_name = QObject::tr("Tag");
        break;
    case KeySelf:
        m_name = QObject::tr("Self");
        break;
    case KeyDecorator:
        m_name = QObject::tr("Decorator");
        break;
    case KeyPunctuation:
        m_name = QObject::tr("Punctuation");
        break;
    case KeyConstant:
        m_name = QObject::tr("Constant"); //all methods starting and ending with two underlines
        break;
    case KeyFunction:
        m_name = QObject::tr("Function"); //method name
        break;
    case KeyOperator:
        m_name = QObject::tr("Operator");
        break;
    case KeyOperatorWord:
        m_name = QObject::tr("Operator Word");
        break;
    case KeyClass:
        m_name = QObject::tr("Class Name");
        break;
    case KeyStreamOutput:
        m_name = QObject::tr("Stream Output (Command Line)");
        break;
    case KeyStreamError:
        m_name = QObject::tr("Stream Error (Command Line)");
        break;
    default:
        m_name = "";
        m_valid = false;
        break;
    }
}

//------------------------------------------------------------------
StyleItem::StyleItem(const StyleItem &item)
{
    m_type = item.m_type;
    m_name = item.m_name;
    m_format = item.m_format;
    m_valid = item.m_valid;
}

//------------------------------------------------------------------
StyleItem& StyleItem::operator= (const StyleItem &rhs)
{
    m_type = rhs.m_type;
    m_name = rhs.m_name;
    m_format = rhs.m_format;
    m_valid = rhs.m_valid;
    return *this;
}

//------------------------------------------------------------------
QTextCharFormat StyleItem::createFormat(const QBrush &color, const QBrush &bgcolor /*= QBrush()*/, bool bold /*= false*/, \
    bool italic /*= false*/, bool underline /*= false*/, QFont::StyleHint styleHint /* = QFont::SansSerif*/)
{
    QTextCharFormat f;
    f.setForeground(color);
    f.setBackground(bgcolor);
    if (bold)
    {
        f.setFontWeight(QFont::Bold);
    }
    f.setFontItalic(italic);
    if (underline)
    {
        f.setUnderlineStyle(QTextCharFormat::SingleUnderline);
    }
    f.setFontStyleHint(styleHint);
    return f;
}

//------------------------------------------------------------------
QTextCharFormat StyleItem::createFormat(const QString &familyName, int pointSize, \
    const QColor &color, const QColor &bgcolor /*= QColor()*/, \
    bool bold /*= false*/, QFont::StyleHint styleHint /* = QFont::SansSerif*/)
{
    QTextCharFormat f;
    f.setForeground(color);
    f.setBackground(bgcolor);
    f.setFontFamily(familyName);
    f.setFontPointSize(pointSize);

    if (bold)
    {
        f.setFontWeight(QFont::Bold);
    }
    f.setFontItalic(false);
    f.setFontStyleHint(styleHint);
    return f;
}

//------------------------------------------------------------------
/*static*/ QMetaEnum StyleItem::styleTypeEnum()
{
    const QMetaObject &mo = StyleItem::staticMetaObject;
    int idx = mo.indexOfEnumerator("ito::StyleType");
    if (idx == -1)
    {
        idx = mo.indexOfEnumerator("StyleType");
    }
    return mo.enumerator(idx);
}

//------------------------------------------------------------------
/*static*/ QList<StyleItem::StyleType> StyleItem::availableStyleTypes()
{
    QList<StyleItem::StyleType> types;
    QMetaEnum styleTypeEnum = StyleItem::styleTypeEnum();

    //create defaults
    for (int i = 0; i < styleTypeEnum.keyCount(); ++i)
    {
        types << (StyleItem::StyleType)styleTypeEnum.value(i);
    }

    return types;
}

//------------------------------------------------------------------
CodeEditorStyle::CodeEditorStyle()
{
    QBrush bgcolor;
    bgcolor.setColor("white");

    QString defaultFontName = "Verdana";
    int defaultPointSize = 10;

    QTextCharFormat defaultFormat = StyleItem::createFormat(defaultFontName, defaultPointSize, "#000000" /*808080"*/, Qt::white);

    //create defaults
    foreach (StyleItem::StyleType styleType, StyleItem::availableStyleTypes())
    {
        m_formats[styleType] = StyleItem(styleType, defaultFormat);
    }

    m_formats[StyleItem::KeyKeyword] = StyleItem(StyleItem::KeyKeyword, StyleItem::createFormat(defaultFontName, defaultPointSize, "#0000ff", Qt::white, true)); //pygments, vs style
    m_formats[StyleItem::KeyOperator] = StyleItem(StyleItem::KeyOperator, StyleItem::createFormat(defaultFontName, defaultPointSize, Qt::black, Qt::white, true));
    m_formats[StyleItem::KeyConstant] = StyleItem(StyleItem::KeyConstant, StyleItem::createFormat(defaultFontName, defaultPointSize, "#0000ff", Qt::white, true)); //pygments, vs style
    m_formats[StyleItem::KeyNamespace] = StyleItem(StyleItem::KeyNamespace, StyleItem::createFormat(defaultFontName, defaultPointSize, "#0000ff", Qt::white, true)); //pygments, vs style

    m_formats[StyleItem::KeyClass] = StyleItem(StyleItem::KeyClass, StyleItem::createFormat(defaultFontName, defaultPointSize, "#5aaac1", Qt::white, true)); //pygments, vs style
    m_formats[StyleItem::KeyString] = StyleItem(StyleItem::KeyString, StyleItem::createFormat("Courier New", defaultPointSize, "#7f007f", Qt::white, false));
    m_formats[StyleItem::KeyComment] = StyleItem(StyleItem::KeyComment, StyleItem::createFormat(defaultFontName, defaultPointSize, "#007f00", Qt::white, false)); //pygments, vs style
    m_formats[StyleItem::KeySelf] = StyleItem(StyleItem::KeySelf, StyleItem::createFormat(defaultFontName, defaultPointSize, "#007020", Qt::white, false)); //pygments, vs style
    m_formats[StyleItem::KeyNumber] = StyleItem(StyleItem::KeyNumber, StyleItem::createFormat(defaultFontName, defaultPointSize, "#40a070", Qt::white, false)); //pygments, vs style
    m_formats[StyleItem::KeyDocstring] = StyleItem(StyleItem::KeyDocstring, StyleItem::createFormat(defaultFontName, defaultPointSize, "#a31515", Qt::white, false)); //pygments, vs style
    m_formats[StyleItem::KeyDocstring].rformat().setFontItalic(true);
    m_formats[StyleItem::KeyDecorator] = StyleItem(StyleItem::KeyDecorator, StyleItem::createFormat(defaultFontName, defaultPointSize, "#805000", Qt::white, false));
    m_formats[StyleItem::KeyFunction] = StyleItem(StyleItem::KeyFunction, StyleItem::createFormat(defaultFontName, defaultPointSize, "#007f7f", Qt::white, true));

    m_formats[StyleItem::KeyBuiltin] = StyleItem(StyleItem::KeyBuiltin, StyleItem::createFormat(defaultFontName, defaultPointSize, "#06287e", Qt::white, true)); //pygments, vs style
    m_formats[StyleItem::KeyOperatorWord] = StyleItem(StyleItem::KeyOperatorWord, StyleItem::createFormat(defaultFontName, defaultPointSize, "#0000ff", Qt::white, true)); //pygments, vs style
    m_formats[StyleItem::KeyDefinition] = StyleItem(StyleItem::KeyDefinition, StyleItem::createFormat(defaultFontName, defaultPointSize, "#5aaac1", Qt::white, true)); //pygments, vs style

    m_formats[StyleItem::KeyStreamOutput] = StyleItem(StyleItem::KeyStreamOutput, StyleItem::createFormat("Courier New", defaultPointSize, "#000000", Qt::white, false));
    m_formats[StyleItem::KeyStreamError] = StyleItem(StyleItem::KeyStreamError, StyleItem::createFormat("Courier New", defaultPointSize, "#000000", Qt::white, false));

}


//------------------------------------------------------------------
CodeEditorStyle::~CodeEditorStyle()
{
}

//------------------------------------------------------------------
/*
Gets the background color.
:return:
*/
QColor CodeEditorStyle::background() const
{
    //qDebug() << m_formats[StyleItem::KeyBackground].format().background().color();
    return m_formats[StyleItem::KeyBackground].format().background().color();
}

//------------------------------------------------------------------
void CodeEditorStyle::setBackground(const QColor &color)
{
    QBrush bg = m_formats[StyleItem::KeyBackground].format().background();
    if (bg.color() != color)
    {
        bg.setColor(color);
        m_formats[StyleItem::KeyBackground].rformat().setBackground(bg);
        //qDebug() << m_formats[StyleItem::KeyBackground].format().background().color();
    }
}

//------------------------------------------------------------------
StyleItem CodeEditorStyle::operator[](StyleItem::StyleType type) const
{
    if (m_formats.contains(type))
    {
        return m_formats[type];
    }

    return StyleItem();
}

//------------------------------------------------------------------
StyleItem& CodeEditorStyle::operator[](StyleItem::StyleType type)
{
    if (m_formats.contains(type))
    {
        return m_formats[type];
    }

    return m_formats[StyleItem::KeyDefault];
}

//------------------------------------------------------------------
QTextCharFormat CodeEditorStyle::format(StyleItem::StyleType type) const
{
    if (m_formats.contains(type))
    {
        return m_formats[type].format();
    }

    return QTextCharFormat();
}

//------------------------------------------------------------------
QTextCharFormat& CodeEditorStyle::rformat(StyleItem::StyleType type)
{
    if (m_formats.contains(type))
    {
        return m_formats[type].rformat();
    }

    return m_formats[StyleItem::KeyDefault].rformat();
}



} //end namespace ito
