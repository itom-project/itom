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

#include "llmProviderFactory.h"

#include "gitHubModelsProvider.h"
#include "ollamaProvider.h"

namespace ito
{
namespace llm
{

RetVal LlmProviderFactory::createProvider(
    const LlmRequest &request,
    QSharedPointer<ILlmProvider> &provider,
    LlmError *errorOut)
{
    LlmSettings settings;
    settings.mode = request.mode;
    settings.provider = request.provider;
    settings.model = request.model;
    settings.endpoint = request.endpoint;
    settings.apiKey = request.apiKey;
    settings.timeoutMs = request.timeoutMs;
    settings.enabled = request.mode != LlmMode::Disabled;

    return createProvider(settings, provider, errorOut);
}

RetVal LlmProviderFactory::createProvider(
    const LlmSettings &settings,
    QSharedPointer<ILlmProvider> &provider,
    LlmError *errorOut)
{
    provider.clear();

    if (!settings.enabled || settings.mode == LlmMode::Disabled)
    {
        return RetVal(retOk);
    }

    const QString providerName = settings.provider.trimmed().toLower();

    if (settings.mode == LlmMode::Cloud)
    {
        if (providerName.isEmpty() || providerName == "github-models" || providerName == "github")
        {
            provider.reset(new GitHubModelsProvider());
            return RetVal(retOk);
        }
    }
    else if (settings.mode == LlmMode::Offline)
    {
        if (providerName.isEmpty() || providerName == "ollama")
        {
            provider.reset(new OllamaProvider());
            return RetVal(retOk);
        }
    }

    if (errorOut)
    {
        errorOut->category = LlmErrorCategory::Configuration;
        errorOut->userMessage = QString("Unsupported LLM provider or mode configuration.");
        errorOut->technicalMessage =
            QString("Provider '%1' is invalid for mode '%2'.")
                .arg(settings.provider)
                .arg(static_cast<int>(settings.mode));
    }

    return RetVal(retError, 0, "Unsupported LLM provider or mode configuration.");
}

LlmProviderFactory::LlmProviderFactory()
{
}

} // namespace llm
} // namespace ito
