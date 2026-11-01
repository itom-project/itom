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

#include "llmSettingsManager.h"

#include "../AppManagement.h"

#include <QSettings>

namespace ito
{
namespace llm
{

RetVal LlmSettingsManager::load(LlmSettings &settings)
{
    QSettings qsettings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    qsettings.beginGroup("llm");

    settings.enabled = qsettings.value("enabled", false).toBool();
    settings.mode = parseMode(qsettings.value("mode", "disabled").toString());
    settings.provider = qsettings.value("provider", "github-models").toString();
    settings.model = qsettings.value("model", "").toString();
    settings.endpoint = qsettings.value("endpoint", "").toString();
    settings.timeoutMs = qsettings.value("timeoutMs", 30000).toInt();

    settings.streamingEnabled = qsettings.value("streaming/enabled", true).toBool();
    settings.telemetryEnabled = qsettings.value("telemetry/enabled", false).toBool();
    settings.sessionOnlyHistory = qsettings.value("chat/sessionOnlyHistory", true).toBool();
    settings.stubsContextEnabled = qsettings.value("knowledge/stubs/enabled", false).toBool();
    settings.stubsPath = qsettings.value("knowledge/stubs/path", "").toString();
    settings.docsContextEnabled = qsettings.value("knowledge/docs/enabled", false).toBool();
    settings.docsPath = qsettings.value("knowledge/docs/path", "").toString();
    settings.restrictToKnownModels =
        qsettings.value("policy/restrictToKnownModels", false).toBool();

    qsettings.endGroup();

    return RetVal(retOk);
}

LlmMode LlmSettingsManager::parseMode(const QString &modeText)
{
    const QString mode = modeText.trimmed().toLower();

    if (mode == "cloud")
    {
        return LlmMode::Cloud;
    }
    else if (mode == "offline")
    {
        return LlmMode::Offline;
    }

    return LlmMode::Disabled;
}

} // namespace llm
} // namespace ito
