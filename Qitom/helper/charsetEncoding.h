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

#ifndef CHARSETENCODING_H
#define CHARSETENCODING_H

#include <qbytearray.h>
#include <qlist.h>
#include <qstring.h>
#include <qstringlist.h>

namespace ito {

//! Charset encodings of script and text files, that are opened in the script editor.
/*!
    This class only depends on QtCore, such that it can be used in unittests. IOHelper
    forwards to it (IOHelper::CharsetEncodingItem, IOHelper::getSupportedScriptEncodings, ...).
*/
class CharsetEncoding
{
public:
    struct Item
    {
        Item() : encodingName(""), displayName(""), displayNameShort(""), bom(""), userDefined(false)
        {
        }

        //!< the name that is accepted by QTextCodec or QStringConverter.
        //!< This is also used as value in the itom settings.
        QString encodingName;

        //!< the name used as display name in a config dialog
        QString displayName;

        //!< the short version, e.g. for a status bar
        QString displayNameShort;

        //!< all possible aliases, e.g. in Python scripts (coding=...)
        QStringList aliases;

        //!< the Byte-Order-Mark used to automatically detect this encoding (or empty, if not used)
        QByteArray bom;

        //!< false if part of the officially supported list of encodings, else true
        bool userDefined;
    };

    //! list of default encodings, that are officially supported by Qt (as well as Python).
    static QList<Item> supportedScriptEncodings();

    //! the default encoding (UTF-8 without BOM)
    static Item defaultScriptEncoding();

    //! the item that belongs to the given alias (case insensitive). If the alias is unknown,
    //! but Qt supports it, a user defined item is returned. Else an empty item.
    static Item fromAlias(const QString& alias, bool* found = nullptr);

    //! the encoding that fits to the BOM at the beginning of the content. If no BOM
    //! is found, 'found' is set to false and an empty item is returned.
    static Item fromBom(const QByteArray& content, bool* found = nullptr);

    //! guesses UTF-16 (LE or BE) files without BOM by their pattern of null bytes.
    /*!
        Windows tools sometimes write UTF-16 files without BOM. In the first bytes of such
        a file every second byte is zero (for latin text). If this pattern is found,
        the UTF-16LE or UTF-16BE item (with BOM, since the file is saved with BOM then)
        is returned. 'found' is set to false and an empty item is returned else.
    */
    static Item guessUtf16WithoutBom(const QByteArray& content, bool* found = nullptr);

    //! decodes the raw content of a file. A leading BOM of the encoding is removed.
    /*!
        \param ok is set to false, if the encoding is not supported by Qt. Then, the
               content is decoded with the default encoding.
    */
    static QString decode(const QByteArray& content, const Item& encoding, bool* ok = nullptr);

    //! encodes a text to the raw content of a file. The BOM of the encoding (if any) is
    //! written at the beginning of the output.
    /*!
        \param ok is set to false, if the encoding is not supported by Qt. Then, the
               text is encoded with the default encoding.
    */
    static QByteArray encode(const QString& text, const Item& encoding, bool* ok = nullptr);

private:
    static QList<Item> m_supportedScriptEncodings;
};

} // end namespace ito

#endif
