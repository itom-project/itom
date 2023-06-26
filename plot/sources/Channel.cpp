/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

//#define ITOM_IMPORT_API
//#define ITOM_IMPORT_PLOTAPI
#include "../AbstractNode.h"

#include <qdebug.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
class ChannelPrivate
{
public:
    ChannelPrivate() :
        m_pSender(NULL),
        m_pReceiver(NULL),
        m_pSenderParam(NULL),
        m_pReceiverParam(NULL),
        m_updateState(Channel::StateIdle),
        m_hashVal(0)
    {}

    AbstractNode* m_pSender; /*!> The node, which is the sender of information over this channel to the receiver */
    AbstractNode* m_pReceiver; /*!> The receiver node*/
    Param* m_pSenderParam; //!> the parameter-connector on the parent's side (can be sending or receiving)
    Param* m_pReceiverParam; //!> the parameter-connector on the child's side (can be sending or receiving)
    Channel::UpdateState m_updateState; //!> state of channel: default is idle
    uint m_hashVal;
};


//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ uint Channel::calcChannelHash(const AbstractNode *sender,
    const ito::Param *senderParam,
    const AbstractNode *receiver,
    const ito::Param *receiverParam)
{
    return qHash(QString("%1%2%3%4").arg((size_t)sender, 8).arg((size_t)receiver, 8).arg((size_t)senderParam, 8).arg((size_t)receiverParam, 8));
}


//----------------------------------------------------------------------------------------------------------------------------------
Channel::Channel() :
    d_ptr(new ChannelPrivate())
{
    qDebug() << "channel created " << this;
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel::Channel(AbstractNode *sender, ito::Param *senderParam,
    AbstractNode *receiver, ito::Param *receiverParam) :
    d_ptr(new ChannelPrivate())
{
    Q_D(Channel);

    qDebug() << "channel created " << this;

    d->m_pSender = sender;
    d->m_pReceiver = receiver;
    d->m_pSenderParam = senderParam;
    d->m_pReceiverParam = receiverParam;
    d->m_updateState = Channel::StateIdle;
    d->m_hashVal = Channel::calcChannelHash(sender, senderParam, receiver, receiverParam);
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel::~Channel()
{
    qDebug() << "channel deleted " << this;
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel::Channel(const Channel& cpy) :
    d_ptr(new ChannelPrivate())
{
    *this = cpy;
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel& Channel::operator=(const Channel& rhs)
{
    Q_D(Channel);

    d->m_pSender = rhs.getSender();
    d->m_pReceiver = rhs.getReceiver();
    d->m_pSenderParam = rhs.getSenderParam();
    d->m_pReceiverParam = rhs.getReceiverParam();
    d->m_updateState = rhs.getUpdateState();
    d->m_hashVal = rhs.getHash();

    return *this;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal Channel::propagateUpdatePending()
{
    Q_D(Channel);

    if (d->m_updateState != StateIdle)
    {
        return ito::retOk; //this channel is already flagged to wait for an update. However there can be another way to also inform this channel about an update. Therefore it can occur multiple times.
    }

    d->m_updateState = StateUpdatePending;

    return d->m_pReceiver->setUpdatePending();
}


//----------------------------------------------------------------------------------------------------------------------------------
AbstractNode* Channel::getSender() const
{
    Q_D(const Channel);

    return d->m_pSender;
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractNode* Channel::getReceiver() const
{
    Q_D(const Channel);

    return d->m_pReceiver;
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel::UpdateState Channel::getUpdateState() const
{
    Q_D(const Channel);

    return d->m_updateState;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal Channel::setUpdateReceived()
{
    Q_D(Channel);

    if (d->m_updateState == StateUpdatePending)
    {
        d->m_updateState = StateUpdateReceived;
        return ito::retOk;
    }

    return ito::retError;
}

//----------------------------------------------------------------------------------------------------------------------------------
QString Channel::getSenderParamName() const
{
    Q_D(const Channel);

    return QLatin1String(d->m_pSenderParam->getName());
}

//----------------------------------------------------------------------------------------------------------------------------------
QString Channel::getReceiverParamName() const
{
    Q_D(const Channel);

    return QLatin1String(d->m_pReceiverParam->getName());
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!> The hash identifying this channel in the connection list of the participating nodes*/
uint Channel::getHash() const
{
    Q_D(const Channel);

    return d->m_hashVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::Param* Channel::getSenderParam() const
{
    Q_D(const Channel);

    return d->m_pSenderParam;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::Param* Channel::getReceiverParam() const
{
    Q_D(const Channel);

    return d->m_pReceiverParam;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!> Sets the updatePending flag of this channel back to false. Note: This reset is NOT propagated down the node tree*/
void Channel::resetUpdateState(void)
{
    Q_D(Channel);

    d->m_updateState = StateIdle;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!> returns true, if the node given as the argument fills the role of sender in this channel. Useful, since parent/child do not have predefined roles */
bool Channel::isSender(const AbstractNode *node) const
{
    Q_D(const Channel);

    return (d->m_pSender == node);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*!> returns true, if the node given as the argument fills the role of receiver in this channel. Useful, since parent/child do not have predefined roles */
bool Channel::isReceiver(const AbstractNode *node) const
{
    Q_D(const Channel);

    return (d->m_pReceiver == node);
}

} //end namespace
