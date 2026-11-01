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

#include "iLlmProvider.h"
#include "llmSettings.h"

#include "../../common/retVal.h"

#include <QSharedPointer>

namespace ito
{
namespace llm
{

/*! \class ito::llm::LlmProviderFactory
    \brief Creates concrete provider instances for active LLM mode.

    The factory resolves cloud/offline provider implementations from request
    or settings data and returns an initialized provider instance.

    \sa ILlmProvider, LlmSettings
*/
class LlmProviderFactory
{
    public:
        //! Create provider from a prepared request.
        /*! \param [in] request Request containing provider selection data.
            \param [out] provider Created provider instance.
            \param [out] errorOut Optional detailed error object.
            \return retOk on success, otherwise retError.
            \sa createProvider
        */
        static RetVal createProvider(
            const LlmRequest &request,
            QSharedPointer<ILlmProvider> &provider,
            LlmError *errorOut = NULL);

        //! Create provider from settings.
        /*! \param [in] settings Active LLM settings.
            \param [out] provider Created provider instance.
            \param [out] errorOut Optional detailed error object.
            \return retOk on success, otherwise retError.
            \sa createProvider
        */
        static RetVal createProvider(
            const LlmSettings &settings,
            QSharedPointer<ILlmProvider> &provider,
            LlmError *errorOut = NULL);

    private:
        //! Disable instantiation of static factory class.
        /*! \sa createProvider */
        LlmProviderFactory();
};

} // namespace llm
} // namespace ito
