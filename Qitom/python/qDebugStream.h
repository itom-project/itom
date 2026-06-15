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

//################
//# qdebugstream.h  #
//################

#ifndef Q_DEBUG_STREAM_H
#define Q_DEBUG_STREAM_H

#include "../../common/typeDefs.h"
#include <iostream>
#include <qobject.h>

namespace ito {

/*! class inherits from std::basic_streambuf and is able to transform a stream, like std::cout or std::cerr, into emitted signals
*/
class QDebugStream : public QObject, public std::basic_streambuf<char>
{
    Q_OBJECT

public:

    QDebugStream(std::ostream &stream, ito::tStreamMessageType);
    ~QDebugStream();

signals:
    void flushStream(QString, ito::tStreamMessageType); /*!<  signal emits a string which appeared in the observed stream together with indicating the corresponding message type */

protected:

    //! this method overwrites a corresponding method in basic_streambuf class and is invoked, if buffer risks to overflow
    virtual std::basic_streambuf<char>::int_type overflow(int_type v);

    virtual std::streamsize xsputn(const char *p, std::streamsize n);

private:
    std::ostream &m_stream;                     /*!<  standard-ostream which is observed by this instance */
    std::streambuf *m_old_buf;                  /*!<  content of stream at time when this instance starts the observation of the stream is stored here and re-given to the stream, when this instance is destroyed */
    std::string m_string;                       /*!<  buffer string, containing parts of the stream which have not been emitted yet */
    ito::tStreamMessageType msg_type;           /*!<  message type of enumeration ito::tStreamMessageType */

#if (QT_VERSION >= QT_VERSION_CHECK(5, 10, 0))
    // QRandomGenerator produces a value random value between 0 and 2**32-1
    uint32 m_randWaitThreshold;
#else
    // qRand() produces a value between 0 and RAND_MAX
    int m_randWaitThreshold;
#endif
};

}; // namespace ito

#endif
