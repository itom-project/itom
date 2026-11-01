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

#include <QSet>

namespace ito
{
namespace llm
{

/*! \class ito::llm::GitHubModelsProvider
    \brief Cloud provider implementation for GitHub Models chat API.

    This provider performs synchronous chat requests and a stream fallback,
    including cancellation tracking and health probing.

    \sa ILlmProvider
*/
class GitHubModelsProvider : public ILlmProvider
{
    public:
        //! Construct provider instance.
        /*! \sa capabilities */
        GitHubModelsProvider();

        //! Return provider capabilities.
        /*! \return Capability flags for this provider.
            \sa ILlmProvider
        */
        virtual ProviderCapabilities capabilities() const;

        //! Execute non-streaming chat request.
        /*! \param [in] request Prepared LLM request.
            \param [out] errorOut Optional detailed error object.
            \return Chat response payload.
            \sa streamChat
        */
        virtual LlmResponse chat(const LlmRequest &request, LlmError *errorOut = NULL);

        //! Execute streaming chat request.
        /*! \param [in] request Prepared LLM request.
            \param [in,out] observer Stream observer for callbacks.
            \param [out] errorOut Optional detailed error object.
            \return Stream handle for cancellation and tracking.
            \sa cancel
        */
        virtual StreamHandle streamChat(
            const LlmRequest &request,
            IStreamObserver *observer,
            LlmError *errorOut = NULL);

        //! Cancel active stream by handle.
        /*! \param [in] handle Stream handle to cancel.
            \sa streamChat
        */
        virtual void cancel(const StreamHandle &handle);

        //! Perform provider health check.
        /*! \return Provider health state.
            \sa capabilities
        */
        virtual ProviderHealth checkHealth();

    private:
        quint64 m_nextStreamHandle; //!< Monotonic stream handle counter.
        QSet<quint64> m_cancelledHandles; //!< Set of cancelled stream handles.
};

} // namespace llm
} // namespace ito
