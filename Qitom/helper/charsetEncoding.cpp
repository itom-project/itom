/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

#include "charsetEncoding.h"

#include <qcoreapplication.h>

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
#include <qtextcodec.h>
#else
#include <qstringconverter.h>
#endif

namespace ito {

QList<CharsetEncoding::Item> CharsetEncoding::m_supportedScriptEncodings;

// the translation context is kept from the former location of this code in IOHelper
#define ENC_TR(text) QCoreApplication::translate("IOHelper", text)

//-------------------------------------------------------------------------------------
//! return a list of default encodings, that are officially supported by Qt (as well as Python).
/*
    The aliases are mostly aliases used in Python scripts (e.g. #coding=... line) and
    are compared case insensitive.
*/
/*static*/ QList<CharsetEncoding::Item> CharsetEncoding::supportedScriptEncodings()
{
    if (m_supportedScriptEncodings.size() > 0)
    {
        return m_supportedScriptEncodings;
    }

    // aliases contain the name as well as all aliases
    // from https://docs.python.org/3.11/library/codecs.html#standard-encodings

    auto item = Item();
    item.aliases = QStringList() << "latin-1" << "latin1" << "latin_1" << "iso-8859-15"
                                 << "iso8859-1" << "8859" << "cp819" << "latin" << "L1";
    item.bom = "";
    item.encodingName = "Latin1";
    item.displayName = "Latin 1";
    item.displayNameShort = "Latin 1";
    m_supportedScriptEncodings.append(item);

    item = Item();
    item.aliases = QStringList() << "utf8" << "utf-8" << "utf_8" << "u8" << "utf" << "cp650001";
    item.bom = "";
    item.encodingName = "UTF-8";
    item.displayName = "UTF-8";
    item.displayNameShort = "UTF-8";
    m_supportedScriptEncodings.append(item);

    item = Item();
    item.aliases = QStringList() << "utf_8_sig";
    item.bom = QByteArray::fromHex("EFBBBF");
    item.encodingName = "UTF-8"; // do not change or remove, see defaultScriptEncoding
    item.displayName = "UTF-8 with BOM";
    item.displayNameShort = "UTF-8 BOM";
    m_supportedScriptEncodings.append(item);

    // UTF-16 and UTF-32 files are detected by their BOM. This is common for text files
    // written by Windows tools (e.g. exported data), but not recommended for Python
    // scripts, since Python requires an UTF-8 compatible encoding of the source file.
    // The BOM is always written again when such a file is saved.
    // Python's "utf-16" and "utf-32" write the BOM with the native (little endian) order.
    item = Item();
    item.aliases = QStringList() << "utf-16be" << "utf_16_be";
    item.bom = QByteArray::fromHex("FEFF");
    item.encodingName = "UTF-16BE";
    item.displayName = ENC_TR("UTF-16BE with BOM (not recommended for Python scripts)");
    item.displayNameShort = ENC_TR("UTF-16BE BOM");
    m_supportedScriptEncodings.append(item);

    item = Item();
    item.aliases = QStringList() << "utf-16le" << "utf_16_le" << "utf16" << "utf-16" << "utf_16"
                                 << "u16";
    item.bom = QByteArray::fromHex("FFFE");
    item.encodingName = "UTF-16LE";
    item.displayName = ENC_TR("UTF-16LE with BOM (not recommended for Python scripts)");
    item.displayNameShort = ENC_TR("UTF-16LE BOM");
    m_supportedScriptEncodings.append(item);

    item = Item();
    item.aliases = QStringList() << "utf-32be" << "utf_32_be";
    item.bom = QByteArray::fromHex("0000FEFF");
    item.encodingName = "UTF-32BE";
    item.displayName = ENC_TR("UTF-32BE with BOM (not recommended for Python scripts)");
    item.displayNameShort = ENC_TR("UTF-32BE BOM");
    m_supportedScriptEncodings.append(item);

    item = Item();
    item.aliases = QStringList() << "utf-32le" << "utf_32_le" << "utf32" << "utf-32" << "utf_32"
                                 << "u32";
    item.bom = QByteArray::fromHex("FFFE0000");
    item.encodingName = "UTF-32LE";
    item.displayName = ENC_TR("UTF-32LE with BOM (not recommended for Python scripts)");
    item.displayNameShort = ENC_TR("UTF-32LE BOM");
    m_supportedScriptEncodings.append(item);

    return m_supportedScriptEncodings;
}

//-------------------------------------------------------------------------------------
/*static*/ CharsetEncoding::Item CharsetEncoding::defaultScriptEncoding()
{
    foreach (const Item& enc, supportedScriptEncodings())
    {
        if (enc.encodingName == "UTF-8" && enc.bom == "")
        {
            return enc;
        }
    }

    return Item();
}

//-------------------------------------------------------------------------------------
/*static*/ CharsetEncoding::Item CharsetEncoding::fromAlias(
    const QString& alias, bool* found /*= nullptr*/)
{
    foreach (const Item& item, supportedScriptEncodings())
    {
        foreach (const QString& s, item.aliases)
        {
            if (QString::compare(alias, s, Qt::CaseInsensitive) == 0)
            {
                if (found)
                {
                    *found = true;
                }

                return item;
            }
        }
    }

    // alias not found, create a user defined one, as long as Qt supports it.
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QTextCodec* tc = QTextCodec::codecForName(alias.toLatin1());

    if (tc)
#else
    QStringDecoder decoder(alias.toLatin1());

    if (decoder.isValid())
#endif
    {
        Item item;
        item.encodingName = alias;
        item.displayName = item.displayNameShort = alias;
        item.userDefined = true;

        if (found)
        {
            *found = true;
        }

        return item;
    }

    if (found)
    {
        *found = false;
    }

    return Item();
}

//-------------------------------------------------------------------------------------
/*static*/ CharsetEncoding::Item CharsetEncoding::fromBom(
    const QByteArray& content, bool* found /*= nullptr*/)
{
    // the longest BOM must win (the UTF-32LE BOM starts with the UTF-16LE BOM)
    Item best;
    bool anyFound = false;

    foreach (const Item& item, supportedScriptEncodings())
    {
        if (item.bom.size() > best.bom.size() && content.startsWith(item.bom))
        {
            best = item;
            anyFound = true;
        }
    }

    if (found)
    {
        *found = anyFound;
    }

    return best;
}

//-------------------------------------------------------------------------------------
/*static*/ CharsetEncoding::Item CharsetEncoding::guessUtf16WithoutBom(
    const QByteArray& content, bool* found /*= nullptr*/)
{
    // inspect at most the first 1024 bytes (an even number of bytes)
    const int n = qMin(content.size(), 1024) & ~1;
    int zeroEven = 0; // zero bytes at even positions (high bytes of UTF-16BE)
    int zeroOdd = 0; // zero bytes at odd positions (high bytes of UTF-16LE)
    int nonZeroEven = 0;
    int nonZeroOdd = 0;

    for (int i = 0; i < n; i += 2)
    {
        if (content[i] == '\0')
        {
            zeroEven++;
        }
        else
        {
            nonZeroEven++;
        }

        if (content[i + 1] == '\0')
        {
            zeroOdd++;
        }
        else
        {
            nonZeroOdd++;
        }
    }

    QString name;

    // at least 8 characters, nearly all high bytes zero, no zero low bytes
    // (a text with only null bytes in one half is also no valid 8 bit text)
    if (n >= 16)
    {
        if (zeroOdd >= n / 2 * 0.9 && zeroEven == 0)
        {
            name = "UTF-16LE";
        }
        else if (zeroEven >= n / 2 * 0.9 && zeroOdd == 0)
        {
            name = "UTF-16BE";
        }
    }

    if (found)
    {
        *found = !name.isEmpty();
    }

    if (!name.isEmpty())
    {
        foreach (const Item& item, supportedScriptEncodings())
        {
            if (item.encodingName == name)
            {
                return item;
            }
        }
    }

    return Item();
}

//-------------------------------------------------------------------------------------
/*static*/ QString CharsetEncoding::decode(
    const QByteArray& content, const Item& encoding, bool* ok /*= nullptr*/)
{
    QByteArray data = content;

    // the BOM is removed here, such that the decoder never has to interpret it
    if (!encoding.bom.isEmpty() && data.startsWith(encoding.bom))
    {
        data.remove(0, encoding.bom.size());
    }

    bool valid = true;

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QTextCodec* tc = QTextCodec::codecForName(encoding.encodingName.toLatin1());

    if (!tc)
    {
        valid = false;
        tc = QTextCodec::codecForName(defaultScriptEncoding().encodingName.toLatin1());
    }

    // IgnoreHeader: a possible BOM has been removed already
    QTextCodec::ConverterState state(QTextCodec::IgnoreHeader);
    QString text = tc->toUnicode(data.constData(), data.size(), &state);
#else
    QStringDecoder decoder(encoding.encodingName.toLatin1());

    if (!decoder.isValid())
    {
        valid = false;
        decoder = QStringDecoder(defaultScriptEncoding().encodingName.toLatin1());
    }

    QString text = decoder(data);
#endif

    if (ok)
    {
        *ok = valid;
    }

    return text;
}

//-------------------------------------------------------------------------------------
/*static*/ QByteArray CharsetEncoding::encode(
    const QString& text, const Item& encoding, bool* ok /*= nullptr*/)
{
    bool valid = true;
    QByteArray data;

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QTextCodec* tc = QTextCodec::codecForName(encoding.encodingName.toLatin1());

    if (!tc)
    {
        valid = false;
        tc = QTextCodec::codecForName(defaultScriptEncoding().encodingName.toLatin1());
    }

    // IgnoreHeader: the BOM is written explicitly below (QTextCodec would write it for
    // UTF-16/32 but not for UTF-8).
    QTextCodec::ConverterState state(QTextCodec::IgnoreHeader);
    data = tc->fromUnicode(text.constData(), text.size(), &state);
#else
    // QStringEncoder does not write a BOM by default (flag WriteBom is not set)
    QStringEncoder encoder(encoding.encodingName.toLatin1());

    if (!encoder.isValid())
    {
        valid = false;
        encoder = QStringEncoder(defaultScriptEncoding().encodingName.toLatin1());
    }

    data = encoder(text);
#endif

    if (valid && !encoding.bom.isEmpty())
    {
        data.prepend(encoding.bom);
    }

    if (ok)
    {
        *ok = valid;
    }

    return data;
}

} // end namespace ito
