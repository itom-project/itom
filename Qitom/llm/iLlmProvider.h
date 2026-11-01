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

namespace ito
{
namespace llm
{

class IStreamObserver
{
    public:
        virtual ~IStreamObserver() {}

        virtual void onChunk(const LlmChunk &chunk) = 0;
        virtual void onCompleted(const LlmResponse &finalResponse) = 0;
        virtual void onError(const LlmError &error) = 0;
};

class ILlmProvider
{
    public:
        virtual ~ILlmProvider() {}

        virtual ProviderCapabilities capabilities() const = 0;
        virtual LlmResponse chat(const LlmRequest &request, LlmError *errorOut = NULL) = 0;
        virtual StreamHandle streamChat(
            const LlmRequest &request,
            IStreamObserver *observer,
            LlmError *errorOut = NULL) = 0;
        virtual void cancel(const StreamHandle &handle) = 0;
        virtual ProviderHealth checkHealth() = 0;
};

} // namespace llm
} // namespace ito
