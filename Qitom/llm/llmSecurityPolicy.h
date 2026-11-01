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

#include "llmSettings.h"
#include "llmTypes.h"
#include "workspaceChangeSet.h"

#include "../../common/retVal.h"

#include <QStringList>

namespace ito
{
namespace llm
{

struct WorkspacePolicyDecision
{
    bool allowed = true; //!< True if the change set may proceed in current mode.
    bool requiresConfirmation = false; //!< True if explicit confirmation is required.
    QString reason; //!< Human-readable decision explanation.
}; //!< Policy decision for workspace apply requests.

/*! \class ito::llm::LlmSecurityPolicy
    \brief Validates request configuration and workspace apply risk.

    This policy layer enforces mode-specific configuration requirements,
    model allow-list checks, and confirmation gates for risky workspace changes.

    \sa WorkspacePolicyDecision, LlmSettings
*/
class LlmSecurityPolicy
{
    public:
        //! Validate a request against active LLM settings.
        /*! \param [in] settings Active LLM settings.
            \param [in] request Prepared request to validate.
            \param [out] errorOut Optional detailed validation error.
            \return retOk if request is allowed; otherwise retError.
            \sa fail
        */
        RetVal validateRequest(
            const LlmSettings &settings,
            const LlmRequest &request,
            LlmError *errorOut = NULL) const;

        //! Evaluate a workspace change set for policy gating.
        /*! \param [in] changeSet Parsed change set from assistant output.
            \param [in] autoApplyRequested True if auto-apply mode is active.
            \return Decision with allow/confirm flags and reason text.
            \sa WorkspacePolicyDecision
        */
        WorkspacePolicyDecision evaluateWorkspaceChangeSet(
            const WorkspaceChangeSet &changeSet,
            bool autoApplyRequested) const;

        //! Mask sensitive values for safer UI or logging output.
        /*! \param [in] value Sensitive string value.
            \return Masked representation of the input value.
            \sa validateRequest
        */
        QString maskSensitive(const QString &value) const;

        //! Set allowed model names for cloud mode restrictions.
        /*! \param [in] models Case-sensitive list of allowed model names.
            \sa validateRequest
        */
        void setKnownModels(const QStringList &models);

    private:
        //! Fill an error object and return a policy/configuration error.
        /*! \param [in] category Error category.
            \param [in] userMessage User-facing error text.
            \param [in] technicalMessage Technical diagnostic details.
            \param [out] errorOut Optional detailed error object.
            \return retError containing userMessage.
            \sa validateRequest
        */
        RetVal fail(
            LlmErrorCategory category,
            const QString &userMessage,
            const QString &technicalMessage,
            LlmError *errorOut) const;

        QStringList m_knownModels; //!< Allow-list of cloud model names.
};

} // namespace llm
} // namespace ito
