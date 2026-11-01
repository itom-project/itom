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

#include <QElapsedTimer>
#include <QHash>
#include <QMutex>
#include <QString>

namespace ito
{
namespace llm
{

struct LlmMetricsSnapshot
{
    quint64 requestCount = 0; //!< Number of chat requests started.
    quint64 streamRequestCount = 0; //!< Number of streaming requests started.
    quint64 streamCancelCount = 0; //!< Number of streams finished as cancelled.
    quint64 streamChunkCount = 0; //!< Total number of stream chunks received.
    quint64 applySuccessCount = 0; //!< Number of successful apply operations.
    quint64 applyConflictCount = 0; //!< Number of apply operations with conflicts.
    quint64 workspaceCreateCount = 0; //!< Total count of created files.
    quint64 workspaceUpdateCount = 0; //!< Total count of updated files.
    quint64 workspaceDeleteCount = 0; //!< Total count of deleted files.
    quint64 policyRejectCount = 0; //!< Number of policy rejections.
    qint64 totalRequestDurationMs = 0; //!< Accumulated non-stream request duration.
    qint64 totalStreamDurationMs = 0; //!< Accumulated stream request duration.
    QHash<QString, quint64> errorsByCategory; //!< Error counters grouped by category.
};

/*! \class ito::llm::LlmMetricsCollector
    \brief Thread-safe collector for LLM runtime usage metrics.

    The collector tracks request, stream, apply, workspace and policy counters
    and provides snapshot access for diagnostics and operations.

    \sa LlmMetricsSnapshot
*/
class LlmMetricsCollector
{
    public:
        //! Construct an empty metrics collector.
        /*! \sa snapshot */
        LlmMetricsCollector();

        //! Record start of a request.
        /*! \sa onRequestFinished */
        void onRequestStarted();

        //! Record completion of a request.
        /*! \param [in] success Indicates if request completed successfully.
            \sa onRequestStarted
        */
        void onRequestFinished(bool success);

        //! Record start of a streaming request.
        /*! \param [in] handleValue Stream handle identifier.
            \sa onStreamFinished
        */
        void onStreamStarted(quint64 handleValue);

        //! Record receipt of one stream chunk.
        /*! \sa onStreamStarted */
        void onStreamChunk();

        //! Record completion of a streaming request.
        /*! \param [in] handleValue Stream handle identifier.
            \param [in] cancelled True if stream ended due to cancellation.
            \sa onStreamStarted
        */
        void onStreamFinished(quint64 handleValue, bool cancelled);

        //! Record an error occurrence by category.
        /*! \param [in] category Error category key.
            \sa snapshot
        */
        void onError(const QString &category);

        //! Record result of apply operation.
        /*! \param [in] success True if apply operation succeeded.
            \param [in] conflict True if conflict state was detected.
            \sa onWorkspaceApply
        */
        void onApplyResult(bool success, bool conflict);

        //! Record aggregated workspace apply counters.
        /*! \param [in] created Number of created files.
            \param [in] updated Number of updated files.
            \param [in] deleted Number of deleted files.
            \sa onApplyResult
        */
        void onWorkspaceApply(int created, int updated, int deleted);

        //! Record a policy rejection event.
        /*! \sa onError */
        void onPolicyRejected();

        //! Return a copy of current metrics snapshot.
        /*! \return Aggregated LLM metrics snapshot.
            \sa LlmMetricsSnapshot
        */
        LlmMetricsSnapshot snapshot() const;

    private:
        mutable QMutex m_mutex; //!< Mutex protecting metrics state.
        LlmMetricsSnapshot m_metrics; //!< Aggregated metrics values.
        QElapsedTimer m_requestTimer; //!< Timer for non-stream request duration.
        QHash<quint64, QElapsedTimer> m_streamTimers; //!< Timers for active streams.
};

} // namespace llm
} // namespace ito
