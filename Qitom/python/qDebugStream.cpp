/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#include "qDebugStream.h"

#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
#include <QRandomGenerator>
#endif

#include <qthread.h>

namespace ito {

//-----------------------------------------------------------------------------------
//! constructor
/*!
    initializes this instance and stores actual content of stream in m_old_buf

    \param stream Stream of type std::ostream which should be observed
    \param type message type of enumeration tMsgType which corresponds to the stream
    \param lineBreak string representation of line break, default: \n
    \return description
    \sa tMsgType
*/
QDebugStream::QDebugStream(std::ostream& stream, ito::tStreamMessageType type) :
    m_stream(stream), msg_type(type)
{
    // Performance issue: if any cout or cerr stream is called very often in a short time,
    // the flushStream signal is emitted very often and the receiving slots can probably not
    // handle all these emitted signals. This leads to an increasing input buffer of the receiver
    // thread This can cause performance issues or crashes due to buffer overflows or deadlocks. In
    // order to avoid this, a short delay is inserted after each flushStream emission. However the
    // shortest delay of 1us will take a lot longer in reality. Therefore, the delay is only
    // inserted in rare cases, defined by a random value and a threshold, given below (here: delay in
    // 4% of all cases)

#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
    // QRandomGenerator produces a value random value between 0 and 2**32-1
    m_randWaitThreshold = std::numeric_limits<uint32>::max() * 0.96;
#else
    // qRand() produces a value between 0 and RAND_MAX
    m_randWaitThreshold = RAND_MAX * 0.96;
#endif


    m_old_buf = stream.rdbuf();
    stream.rdbuf(this);
}

//-----------------------------------------------------------------------------------
//! destructor
/*!
    destroys this instance and the stream observation and emits remaining string in the buffer.
    Restores m_old_buf to the stream.
*/
QDebugStream::~QDebugStream()
{
    // output anything that is left
    if (!m_string.empty())
    {
        // Python stdout and stderr streams as well as std::cout streams in itom and plugins are
        // encoded with latin1.
        QString str = QString::fromUtf8(m_string.c_str());

        // the c_str will be converted into QString using the codec set by
        // QTextCodec::setCodecForCStrings(textCodec) in MainApplication
        emit flushStream(str, msg_type);
    }
    m_stream.rdbuf(m_old_buf);
}

//-----------------------------------------------------------------------------------
//! method invoked if new content has been added to stream
std::streamsize QDebugStream::xsputn(const char* p, std::streamsize n)
{
    m_string.append(p, p + n);

    // Python stdout and stderr streams as well as std::cout streams in itom and plugins are encoded
    // with latin1.
    QString str = QString::fromUtf8(m_string.c_str());

    // the c_str will be converted into QString using the codec set by
    // QTextCodec::setCodecForCStrings(textCodec) in MainApplication
    emit flushStream(str, msg_type);
    m_string.clear();

#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
    if (QRandomGenerator::global()->generate() > m_randWaitThreshold)
    {
        QThread::usleep(1);
    }
#else
    if (qrand() > m_randWaitThreshold)
    {
        QThread::usleep(1);
    }
#endif

    return n;
}

//-----------------------------------------------------------------------------------
//! this method overwrites a corresponding method in basic_streambuf class and is invoked, if buffer
//! risks to overflow
std::basic_streambuf<char>::int_type QDebugStream::overflow(int_type v)
{
    if (v == '\n')
    {
        // Python stdout and stderr streams as well as std::cout streams in itom and plugins are
        // encoded with latin1.
        QString str = QString::fromUtf8(m_string.c_str());
        emit flushStream(str, msg_type);
        m_string.erase(m_string.begin(), m_string.end());

#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
        if (QRandomGenerator::global()->generate() > m_randWaitThreshold)
        {
            QThread::usleep(1);
        }
#else
        if (qrand() > m_randWaitThreshold)
        {
            QThread::usleep(1);
        }
#endif
    }
    else
    {
        m_string += v;
    }

    return v;
}

} // end namespace ito
