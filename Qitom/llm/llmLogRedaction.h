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

#pragma once

#include <QString>

namespace ito
{
namespace llm
{

/*! \class ito::llm::LlmLogRedaction
    \brief Utility helpers for redacting sensitive values from text logs.

    The static API masks common credential patterns such as bearer tokens,
    api keys, basic auth headers and generic secret fields.

    \sa redactAll
*/
class LlmLogRedaction
{
    public:
        //! Redact bearer token occurrences.
        /*! \param [in] text Input text.
            \return Text with bearer tokens masked.
            \sa redactAll
        */
        static QString redactBearerToken(const QString &text);

        //! Redact api key occurrences.
        /*! \param [in] text Input text.
            \return Text with api keys masked.
            \sa redactAll
        */
        static QString redactApiKey(const QString &text);

        //! Redact HTTP basic authorization values.
        /*! \param [in] text Input text.
            \return Text with basic auth values masked.
            \sa redactAll
        */
        static QString redactBasicAuth(const QString &text);

        //! Redact generic secret-like fields.
        /*! \param [in] text Input text.
            \return Text with generic secret fields masked.
            \sa redactAll
        */
        static QString redactGenericSecrets(const QString &text);

        //! Apply all configured redaction rules.
        /*! \param [in] text Input text.
            \return Text with all supported secret patterns masked.
            \sa redactBearerToken
        */
        static QString redactAll(const QString &text);
};

} // namespace llm
} // namespace ito
