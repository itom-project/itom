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

#include "llmMetricsCollector.h"

namespace ito
{
namespace llm
{

LlmMetricsCollector::LlmMetricsCollector()
{
}

void LlmMetricsCollector::onRequestStarted()
{
    QMutexLocker locker(&m_mutex);
    m_metrics.requestCount += 1;
    m_requestTimer.restart();
}

void LlmMetricsCollector::onRequestFinished(bool)
{
    QMutexLocker locker(&m_mutex);
    if (m_requestTimer.isValid())
    {
        m_metrics.totalRequestDurationMs += m_requestTimer.elapsed();
    }
}

void LlmMetricsCollector::onStreamStarted(quint64 handleValue)
{
    QMutexLocker locker(&m_mutex);
    m_metrics.streamRequestCount += 1;
    QElapsedTimer timer;
    timer.start();
    m_streamTimers.insert(handleValue, timer);
}

void LlmMetricsCollector::onStreamChunk()
{
    QMutexLocker locker(&m_mutex);
    m_metrics.streamChunkCount += 1;
}

void LlmMetricsCollector::onStreamFinished(quint64 handleValue, bool cancelled)
{
    QMutexLocker locker(&m_mutex);

    if (cancelled)
    {
        m_metrics.streamCancelCount += 1;
    }

    if (m_streamTimers.contains(handleValue))
    {
        m_metrics.totalStreamDurationMs += m_streamTimers.value(handleValue).elapsed();
        m_streamTimers.remove(handleValue);
    }
}

void LlmMetricsCollector::onError(const QString &category)
{
    QMutexLocker locker(&m_mutex);
    m_metrics.errorsByCategory[category] += 1;
}

void LlmMetricsCollector::onApplyResult(bool success, bool conflict)
{
    QMutexLocker locker(&m_mutex);

    if (success)
    {
        m_metrics.applySuccessCount += 1;
    }

    if (conflict)
    {
        m_metrics.applyConflictCount += 1;
    }
}

void LlmMetricsCollector::onWorkspaceApply(int created, int updated, int deleted)
{
    QMutexLocker locker(&m_mutex);
    m_metrics.workspaceCreateCount += qMax(0, created);
    m_metrics.workspaceUpdateCount += qMax(0, updated);
    m_metrics.workspaceDeleteCount += qMax(0, deleted);
}

void LlmMetricsCollector::onPolicyRejected()
{
    QMutexLocker locker(&m_mutex);
    m_metrics.policyRejectCount += 1;
}

LlmMetricsSnapshot LlmMetricsCollector::snapshot() const
{
    QMutexLocker locker(&m_mutex);
    return m_metrics;
}

} // namespace llm
} // namespace ito
