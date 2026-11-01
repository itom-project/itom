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

#include "llmCredentialService.h"

#include "osKeyStoreWin.h"

namespace ito
{
namespace llm
{

LlmCredentialService::LlmCredentialService() :
    m_keyStore(new OsKeyStoreWin())
{
}

RetVal LlmCredentialService::loadApiKey(const LlmSettings &settings, QString &apiKeyOut) const
{
    apiKeyOut.clear();

    if (!m_keyStore)
    {
        return RetVal(retError, 0, "No OS key store backend available.");
    }

    return m_keyStore->readSecret(credentialKeyForSettings(settings), apiKeyOut);
}

RetVal LlmCredentialService::storeApiKey(const LlmSettings &settings, const QString &apiKey) const
{
    if (!m_keyStore)
    {
        return RetVal(retError, 0, "No OS key store backend available.");
    }

    if (apiKey.trimmed().isEmpty())
    {
        return m_keyStore->removeSecret(credentialKeyForSettings(settings));
    }

    return m_keyStore->writeSecret(credentialKeyForSettings(settings), apiKey);
}

RetVal LlmCredentialService::removeApiKey(const LlmSettings &settings) const
{
    if (!m_keyStore)
    {
        return RetVal(retError, 0, "No OS key store backend available.");
    }

    return m_keyStore->removeSecret(credentialKeyForSettings(settings));
}

QString LlmCredentialService::credentialKeyForSettings(const LlmSettings &settings)
{
    const QString provider = settings.provider.trimmed().isEmpty() ? QString("default") : settings.provider.trimmed().toLower();
    return QString("itom/llm/%1/apikey").arg(provider);
}

} // namespace llm
} // namespace ito
