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

#include "helper/charsetEncoding.h"
#include <qfile.h>
#include <qset.h>
#include <qtemporarydir.h>
#include "gtest/gtest.h"

using namespace ito;

//! Tests for the charset encodings of script and text files (BOM detection, decode, encode).

//-------------------------------------------------------------------------------------
//! text with ASCII, latin-1 and non latin-1 characters
static QString sampleText()
{
    return QString::fromUtf8("Listing of Data\r\n\xC3\xA4\xC3\xB6\xC3\xBC \xE2\x82\xAC\n -2.96E-01\t 1\n");
}

//! builds the UTF-16 or UTF-32 raw bytes of a text without any Qt codec
static QByteArray rawUtf(const QString& text, int bytesPerChar, bool littleEndian)
{
    QByteArray result;

    foreach (uint cp, text.toUcs4())
    {
        if (bytesPerChar == 2)
        {
            EXPECT_LT(cp, 0x10000u); // no surrogates in the sample
        }

        for (int i = 0; i < bytesPerChar; ++i)
        {
            int shift = littleEndian ? i : (bytesPerChar - 1 - i);
            result.append((char)((cp >> (8 * shift)) & 0xFF));
        }
    }

    return result;
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, supportedEncodingsAndDefault)
{
    auto list = CharsetEncoding::supportedScriptEncodings();
    EXPECT_GE(list.size(), 7);

    auto def = CharsetEncoding::defaultScriptEncoding();
    EXPECT_EQ(def.encodingName, "UTF-8");
    EXPECT_TRUE(def.bom.isEmpty());
    EXPECT_FALSE(def.userDefined);

    // all BOMs are unique
    QSet<QByteArray> boms;

    foreach (const auto& item, list)
    {
        if (!item.bom.isEmpty())
        {
            EXPECT_FALSE(boms.contains(item.bom)) << item.encodingName.toStdString();
            boms.insert(item.bom);
        }
    }
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, fromAlias)
{
    bool found = false;

    EXPECT_EQ(CharsetEncoding::fromAlias("utf-8", &found).encodingName, "UTF-8");
    EXPECT_TRUE(found);
    EXPECT_TRUE(CharsetEncoding::fromAlias("utf-8").bom.isEmpty());

    // case insensitive
    EXPECT_EQ(CharsetEncoding::fromAlias("UTF8").encodingName, "UTF-8");
    EXPECT_EQ(CharsetEncoding::fromAlias("Latin-1").encodingName, "Latin1");

    // the python alias for utf-8 with BOM
    EXPECT_EQ(CharsetEncoding::fromAlias("utf_8_sig").bom, QByteArray::fromHex("EFBBBF"));

    // utf-16 / utf-32 (python writes the BOM in little endian order)
    EXPECT_EQ(CharsetEncoding::fromAlias("utf-16", &found).encodingName, "UTF-16LE");
    EXPECT_TRUE(found);
    EXPECT_EQ(CharsetEncoding::fromAlias("utf-16be").encodingName, "UTF-16BE");
    EXPECT_EQ(CharsetEncoding::fromAlias("utf-32").encodingName, "UTF-32LE");

    // an encoding known by Qt but not in the list is user defined
    found = false;
    auto item = CharsetEncoding::fromAlias("ISO-8859-2", &found);
    EXPECT_TRUE(found);
    EXPECT_TRUE(item.userDefined);
    EXPECT_EQ(item.encodingName, "ISO-8859-2");

    // unknown encoding
    found = true;
    item = CharsetEncoding::fromAlias("no-such-encoding-xyz", &found);
    EXPECT_FALSE(found);
    EXPECT_TRUE(item.encodingName.isEmpty());
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, fromBom)
{
    bool found = false;

    EXPECT_EQ(
        CharsetEncoding::fromBom(QByteArray::fromHex("EFBBBF") + "abc", &found).displayNameShort,
        "UTF-8 BOM");
    EXPECT_TRUE(found);

    EXPECT_EQ(
        CharsetEncoding::fromBom(QByteArray::fromHex("FFFE") + "L\0i\0", &found).encodingName,
        "UTF-16LE");
    EXPECT_TRUE(found);

    EXPECT_EQ(
        CharsetEncoding::fromBom(QByteArray::fromHex("FEFF") + "\0L\0i", &found).encodingName,
        "UTF-16BE");

    // the UTF-32LE BOM starts with the UTF-16LE BOM: the longer BOM must win
    EXPECT_EQ(
        CharsetEncoding::fromBom(QByteArray::fromHex("FFFE0000") + "L\0\0\0", &found).encodingName,
        "UTF-32LE");
    EXPECT_EQ(
        CharsetEncoding::fromBom(QByteArray::fromHex("0000FEFF") + "\0\0\0L", &found).encodingName,
        "UTF-32BE");

    // no BOM
    found = true;
    EXPECT_TRUE(CharsetEncoding::fromBom("plain text", &found).encodingName.isEmpty());
    EXPECT_FALSE(found);

    found = true;
    EXPECT_TRUE(CharsetEncoding::fromBom(QByteArray(), &found).encodingName.isEmpty());
    EXPECT_FALSE(found);
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, decodeUtf16LeWithBomLikeZemaxFile)
{
    // files written by many windows tools (e.g. Zemax "Listing of Universal Plot 2D Data")
    const QString text = sampleText();
    QByteArray content = QByteArray::fromHex("FFFE") + rawUtf(text, 2, true);

    bool found = false;
    auto enc = CharsetEncoding::fromBom(content, &found);
    ASSERT_TRUE(found);

    bool ok = false;
    QString decoded = CharsetEncoding::decode(content, enc, &ok);
    EXPECT_TRUE(ok);
    EXPECT_EQ(decoded, text);

    // the BOM must not be part of the text
    EXPECT_FALSE(decoded.startsWith(QChar(0xFEFF)));
    EXPECT_TRUE(decoded.startsWith("Listing"));
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, decodeAndEncodeAllBomEncodings)
{
    const QString text = sampleText();

    struct Case
    {
        const char* alias;
        int bytesPerChar;
        bool littleEndian;
    };

    const Case cases[] = {
        {"utf-16le", 2, true}, {"utf-16be", 2, false}, {"utf-32le", 4, true}, {"utf-32be", 4, false}};

    for (const Case& c : cases)
    {
        auto enc = CharsetEncoding::fromAlias(c.alias);
        ASSERT_FALSE(enc.bom.isEmpty()) << c.alias;

        QByteArray expectedRaw = enc.bom + rawUtf(text, c.bytesPerChar, c.littleEndian);

        // encode: exactly one BOM at the beginning
        bool ok = false;
        QByteArray encoded = CharsetEncoding::encode(text, enc, &ok);
        EXPECT_TRUE(ok) << c.alias;
        EXPECT_EQ(encoded.toHex(), expectedRaw.toHex()) << c.alias;

        // decode with and without BOM
        EXPECT_EQ(CharsetEncoding::decode(expectedRaw, enc, &ok), text) << c.alias;
        EXPECT_TRUE(ok);
        EXPECT_EQ(CharsetEncoding::decode(expectedRaw.mid(enc.bom.size()), enc), text) << c.alias;

        // round trip
        EXPECT_EQ(CharsetEncoding::decode(CharsetEncoding::encode(text, enc), enc), text) << c.alias;
    }
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, utf8WithAndWithoutBom)
{
    const QString text = sampleText();
    auto utf8 = CharsetEncoding::fromAlias("utf-8");
    auto utf8bom = CharsetEncoding::fromAlias("utf_8_sig");

    EXPECT_EQ(CharsetEncoding::encode(text, utf8), text.toUtf8());
    EXPECT_EQ(CharsetEncoding::encode(text, utf8bom), QByteArray::fromHex("EFBBBF") + text.toUtf8());

    EXPECT_EQ(CharsetEncoding::decode(text.toUtf8(), utf8), text);
    EXPECT_EQ(CharsetEncoding::decode(QByteArray::fromHex("EFBBBF") + text.toUtf8(), utf8bom), text);

    // a file with BOM, that is decoded as UTF-8 (without BOM item), keeps the BOM character.
    // This is the behavior of Qt and only happens if the auto detection is disabled.
    QString withBom = CharsetEncoding::decode(QByteArray::fromHex("EFBBBF") + text.toUtf8(), utf8);
    EXPECT_TRUE(withBom.endsWith(text));
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, latin1)
{
    auto latin1 = CharsetEncoding::fromAlias("latin1");
    QString text = QString::fromUtf8("na\xC3\xAFve");
    QByteArray raw = CharsetEncoding::encode(text, latin1);
    EXPECT_EQ(raw.toHex(), QByteArray("6e61ef7665"));
    EXPECT_EQ(CharsetEncoding::decode(raw, latin1), text);
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, unsupportedEncodingFallsBackToUtf8)
{
    CharsetEncoding::Item bogus;
    bogus.encodingName = "no-such-encoding-xyz";
    const QString text = sampleText();

    bool ok = true;
    QByteArray raw = CharsetEncoding::encode(text, bogus, &ok);
    EXPECT_FALSE(ok);
    EXPECT_EQ(raw, text.toUtf8());

    ok = true;
    EXPECT_EQ(CharsetEncoding::decode(text.toUtf8(), bogus, &ok), text);
    EXPECT_FALSE(ok);
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, utf16FileWithCrlfMustBeReadInBinaryMode)
{
    // Files from Windows tools have "\r\n" line endings. In UTF-16 this is the byte
    // sequence 0D 00 0A 00. QIODevice::Text removes '\r' bytes before the decoding and
    // destroys the alignment of all following characters. Therefore the script editor
    // reads files in binary mode and normalizes the line endings after the decoding.
    const QString text = "Listing of Data\r\n\r\nTitle: \r\nDate : 10/30/2026\r\n 1.0E-01\t 2.0E-01\r\n";
    auto enc = CharsetEncoding::fromAlias("utf-16le");

    QTemporaryDir dir;
    QString path = dir.path() + "/crlf.txt";

    {
        QFile f(path);
        ASSERT_TRUE(f.open(QIODevice::WriteOnly));
        f.write(CharsetEncoding::encode(text, enc));
    }

    // binary mode: correct
    QFile binary(path);
    ASSERT_TRUE(binary.open(QIODevice::ReadOnly));
    QByteArray raw = binary.readAll();
    bool found = false;
    EXPECT_EQ(CharsetEncoding::fromBom(raw, &found).encodingName, "UTF-16LE");
    QString decoded = CharsetEncoding::decode(raw, enc);
    EXPECT_EQ(decoded, text);
    decoded.replace("\r\n", "\n");
    EXPECT_EQ(decoded.split("\n").size(), 6);
    EXPECT_EQ(decoded.split("\n")[3], "Date : 10/30/2026");

    // text mode: one byte per line is lost, the rest is garbage (this documents the pitfall)
    QFile textMode(path);
    ASSERT_TRUE(textMode.open(QIODevice::ReadOnly | QIODevice::Text));
    QByteArray rawText = textMode.readAll();
    EXPECT_LT(rawText.size(), raw.size());
    QString garbage = CharsetEncoding::decode(rawText, enc);
    EXPECT_NE(garbage, text);
    EXPECT_FALSE(garbage.contains("Date : 10/30/2026"));
}

//-------------------------------------------------------------------------------------
TEST(CharsetEncodingTest, guessUtf16WithoutBom)
{
    const QString text = sampleText() + "Listing of Universal Plot 2D Data\r\n 1.0E-01\t 2.0E-01\r\n";
    bool found = false;

    // UTF-16LE without BOM
    QByteArray le = rawUtf(text, 2, true);
    auto enc = CharsetEncoding::guessUtf16WithoutBom(le, &found);
    EXPECT_TRUE(found);
    EXPECT_EQ(enc.encodingName, "UTF-16LE");
    EXPECT_EQ(CharsetEncoding::decode(le, enc), text);

    // UTF-16BE without BOM
    QByteArray be = rawUtf(text, 2, false);
    enc = CharsetEncoding::guessUtf16WithoutBom(be, &found);
    EXPECT_TRUE(found);
    EXPECT_EQ(enc.encodingName, "UTF-16BE");
    EXPECT_EQ(CharsetEncoding::decode(be, enc), text);

    // 8 bit texts are never detected as UTF-16
    found = true;
    EXPECT_TRUE(CharsetEncoding::guessUtf16WithoutBom(text.toUtf8(), &found).encodingName.isEmpty());
    EXPECT_FALSE(found);
    found = true;
    EXPECT_TRUE(CharsetEncoding::guessUtf16WithoutBom(text.toLatin1(), &found).encodingName.isEmpty());
    EXPECT_FALSE(found);

    // too short, empty, binary data with some zeros
    found = true;
    EXPECT_TRUE(CharsetEncoding::guessUtf16WithoutBom(rawUtf("abc", 2, true), &found).encodingName.isEmpty());
    EXPECT_FALSE(found);
    EXPECT_TRUE(CharsetEncoding::guessUtf16WithoutBom(QByteArray(), &found).encodingName.isEmpty());
    EXPECT_FALSE(found);

    QByteArray binary;

    for (int i = 0; i < 512; ++i)
    {
        binary.append((char)((i * 7) % 13 == 0 ? 0 : i));
    }

    EXPECT_TRUE(CharsetEncoding::guessUtf16WithoutBom(binary, &found).encodingName.isEmpty());
    EXPECT_FALSE(found);

    // UTF-16 text with non latin characters (high byte not zero) in the first bytes: not
    // detected, which is acceptable (the heuristic is for latin text listings)
}
