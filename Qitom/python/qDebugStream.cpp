/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

namespace ito {

//------------------------------------------------------------------------------------------------------------------------
//! constructor
/*!
    initializes this instance and stores actual content of stream in m_old_buf

    \param stream Stream of type std::ostream which should be observed
    \param type message type of enumeration tMsgType which corresponds to the stream
    \param lineBreak string representation of line break, default: \n
    \return description
    \sa tMsgType
*/
QDebugStream::QDebugStream(std::ostream &stream, ito::tStreamMessageType type) :
    m_stream(stream),
    msg_type(type)
{
    m_old_buf = stream.rdbuf();
    stream.rdbuf(this);
}

//------------------------------------------------------------------------------------------------------------------------
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
        QString str = QString::fromLatin1(m_string.c_str()); //Python stdout and stderr streams as well as std::cout streams in itom and plugins are encoded with latin1.
        emit flushStream(str,msg_type); //the c_str will be converted into QString using the codec set by QTextCodec::setCodecForCStrings(textCodec) in MainApplication
    }
    m_stream.rdbuf(m_old_buf);
}

//------------------------------------------------------------------------------------------------------------------------
//! method invoked if new content has been added to stream
std::streamsize QDebugStream::xsputn(const char *p, std::streamsize n)
{
    m_string.append(p, p + n);

    QString str = QString::fromLatin1(m_string.c_str()); //Python stdout and stderr streams as well as std::cout streams in itom and plugins are encoded with latin1.

    emit flushStream(str, msg_type); //the c_str will be converted into QString using the codec set by QTextCodec::setCodecForCStrings(textCodec) in MainApplication
    m_string.clear();

    return n;
}

//------------------------------------------------------------------------------------------------------------------------
//! this method overwrites a corresponding method in basic_streambuf class and is invoked, if buffer risks to overflow
std::basic_streambuf<char>::int_type QDebugStream::overflow(int_type v)
{
    if (v == '\n')
    {
        QString str = QString::fromLatin1(m_string.c_str()); //Python stdout and stderr streams as well as std::cout streams in itom and plugins are encoded with latin1.
        emit flushStream(str, msg_type);
        m_string.erase(m_string.begin(), m_string.end());
    }
    else
    {
        m_string += v;
    }

    return v;
}

} //end namespace ito

