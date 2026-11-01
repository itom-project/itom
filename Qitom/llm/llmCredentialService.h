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

#include "iOsKeyStore.h"
#include "llmSettings.h"

#include "../../common/retVal.h"

#include <QScopedPointer>

namespace ito
{
namespace llm
{

/*! \class ito::llm::LlmCredentialService
    \brief Persists and retrieves provider-specific API credentials.

    The service maps active settings to key-store identifiers and delegates
    secure storage operations to the configured OS key-store backend.

    \sa IOsKeyStore, LlmSettings
*/
class LlmCredentialService
{
    public:
        //! Construct service with platform key-store backend.
        /*! \sa IOsKeyStore */
        LlmCredentialService();

        //! Load API key for the given LLM settings.
        /*! \param [in] settings Active LLM settings.
            \param [out] apiKeyOut Loaded API key value.
            \return retOk on success, otherwise retError.
            \sa credentialKeyForSettings
        */
        RetVal loadApiKey(const LlmSettings &settings, QString &apiKeyOut) const;

        //! Store API key for the given LLM settings.
        /*! \param [in] settings Active LLM settings.
            \param [in] apiKey API key value to store.
            \return retOk on success, otherwise retError.
            \sa credentialKeyForSettings
        */
        RetVal storeApiKey(const LlmSettings &settings, const QString &apiKey) const;

        //! Remove stored API key for the given LLM settings.
        /*! \param [in] settings Active LLM settings.
            \return retOk on success, otherwise retError.
            \sa credentialKeyForSettings
        */
        RetVal removeApiKey(const LlmSettings &settings) const;

        //! Build key-store identifier for active settings.
        /*! \param [in] settings Active LLM settings.
            \return Provider-specific credential key identifier.
            \sa loadApiKey
        */
        static QString credentialKeyForSettings(const LlmSettings &settings);

    private:
        QScopedPointer<IOsKeyStore> m_keyStore; //!< Platform key-store backend.
};

} // namespace llm
} // namespace ito
