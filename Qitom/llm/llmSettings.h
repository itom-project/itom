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

#include "llmTypes.h"

#include <QString>

namespace ito
{
namespace llm
{

struct LlmSettings
{
    bool enabled = false;
    LlmMode mode = LlmMode::Disabled;
    QString provider;
    QString model;
    QString endpoint;
    QString apiKey; // sensitive runtime-only value loaded from OS key store
    int timeoutMs = 30000;

    bool streamingEnabled = true;
    bool telemetryEnabled = false;
    bool sessionOnlyHistory = true;

    bool stubsContextEnabled = false;
    QString stubsPath;
    bool docsContextEnabled = false;
    QString docsPath;

    // no hard allowlist; optional restricted startup with known models
    bool restrictToKnownModels = false;
};

} // namespace llm
} // namespace ito
