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

#include "widgetPropLlm.h"

#include "../AppManagement.h"

#include <qsettings.h>

namespace ito
{

WidgetPropLlm::WidgetPropLlm(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);
}

WidgetPropLlm::~WidgetPropLlm()
{
}

llm::LlmSettings WidgetPropLlm::collectSettingsFromUi() const
{
    llm::LlmSettings settings;
    settings.enabled = ui.checkEnabled->isChecked();

    if (ui.comboMode->currentIndex() == 1)
    {
        settings.mode = llm::LlmMode::Cloud;
    }
    else if (ui.comboMode->currentIndex() == 2)
    {
        settings.mode = llm::LlmMode::Offline;
    }
    else
    {
        settings.mode = llm::LlmMode::Disabled;
    }

    settings.provider = ui.comboProvider->currentText().trimmed();
    settings.model = ui.editModel->text().trimmed();
    settings.endpoint = ui.editEndpoint->text().trimmed();
    settings.timeoutMs = ui.spinTimeoutMs->value();
    settings.streamingEnabled = ui.checkStreamingEnabled->isChecked();
    settings.telemetryEnabled = ui.checkTelemetryEnabled->isChecked();
    settings.sessionOnlyHistory = ui.checkSessionOnlyHistory->isChecked();
    settings.stubsContextEnabled = ui.checkStubsContextEnabled->isChecked();
    settings.stubsPath = ui.editStubsPath->text().trimmed();
    settings.docsContextEnabled = ui.checkDocsContextEnabled->isChecked();
    settings.docsPath = ui.editDocsPath->text().trimmed();
    settings.restrictToKnownModels = ui.checkRestrictKnownModels->isChecked();

    return settings;
}

void WidgetPropLlm::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("llm");

    ui.checkEnabled->setChecked(settings.value("enabled", false).toBool());

    const QString mode = settings.value("mode", "disabled").toString().trimmed().toLower();
    if (mode == "cloud")
    {
        ui.comboMode->setCurrentIndex(1);
    }
    else if (mode == "offline")
    {
        ui.comboMode->setCurrentIndex(2);
    }
    else
    {
        ui.comboMode->setCurrentIndex(0);
    }

    ui.comboProvider->setCurrentText(settings.value("provider", "github-models").toString());
    ui.editModel->setText(settings.value("model", "").toString());
    ui.editEndpoint->setText(settings.value("endpoint", "").toString());
    ui.spinTimeoutMs->setValue(settings.value("timeoutMs", 30000).toInt());

    ui.checkStreamingEnabled->setChecked(settings.value("streaming/enabled", true).toBool());
    ui.checkTelemetryEnabled->setChecked(settings.value("telemetry/enabled", false).toBool());
    ui.checkSessionOnlyHistory->setChecked(settings.value("chat/sessionOnlyHistory", true).toBool());
    ui.checkStubsContextEnabled->setChecked(settings.value("knowledge/stubs/enabled", false).toBool());
    ui.editStubsPath->setText(settings.value("knowledge/stubs/path", "").toString());
    ui.checkDocsContextEnabled->setChecked(settings.value("knowledge/docs/enabled", false).toBool());
    ui.editDocsPath->setText(settings.value("knowledge/docs/path", "").toString());
    ui.checkRestrictKnownModels->setChecked(
        settings.value("policy/restrictToKnownModels", false).toBool());

    settings.endGroup();

    const llm::LlmSettings llmSettings = collectSettingsFromUi();
    QString apiKey;
    if (!m_credentialService.loadApiKey(llmSettings, apiKey).containsError() && !apiKey.isEmpty())
    {
        ui.editApiKey->setText(QString("********"));
    }
    else
    {
        ui.editApiKey->clear();
    }
}

void WidgetPropLlm::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("llm");

    const llm::LlmSettings llmSettings = collectSettingsFromUi();

    settings.setValue("enabled", llmSettings.enabled);

    QString mode = "disabled";
    if (llmSettings.mode == llm::LlmMode::Cloud)
    {
        mode = "cloud";
    }
    else if (llmSettings.mode == llm::LlmMode::Offline)
    {
        mode = "offline";
    }

    settings.setValue("mode", mode);
    settings.setValue("provider", llmSettings.provider);
    settings.setValue("model", llmSettings.model);
    settings.setValue("endpoint", llmSettings.endpoint);
    settings.setValue("timeoutMs", llmSettings.timeoutMs);

    settings.setValue("streaming/enabled", llmSettings.streamingEnabled);
    settings.setValue("telemetry/enabled", llmSettings.telemetryEnabled);
    settings.setValue("chat/sessionOnlyHistory", llmSettings.sessionOnlyHistory);
    settings.setValue("knowledge/stubs/enabled", llmSettings.stubsContextEnabled);
    settings.setValue("knowledge/stubs/path", llmSettings.stubsPath);
    settings.setValue("knowledge/docs/enabled", llmSettings.docsContextEnabled);
    settings.setValue("knowledge/docs/path", llmSettings.docsPath);
    settings.setValue("policy/restrictToKnownModels", llmSettings.restrictToKnownModels);

    settings.endGroup();

    const QString apiKeyText = ui.editApiKey->text();
    if (apiKeyText != QString("********"))
    {
        m_credentialService.storeApiKey(llmSettings, apiKeyText);
    }
}

} //end namespace ito
