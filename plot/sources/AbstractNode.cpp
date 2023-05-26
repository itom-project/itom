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

#include "../AbstractNode.h"
#include "../../common/addInInterface.h"

#include <qsharedpointer.h>
#include <qlist.h>

namespace ito
{

unsigned int AbstractNode::UID = 1;

//----------------------------------------------------------------------------------------------------------------------------------
class AbstractNodePrivate
{
public:
    AbstractNodePrivate(rttiNodeType nodeType, unsigned int UID) :
        m_nodeType(nodeType),
        m_uniqueID(UID)
    {}

    rttiNodeType m_nodeType;            //!> the type of the actual node inheriting this abstract node
    QHash<QString, ito::Param*> m_inputParams;        //!> the node's input parameter, given as ito::Param. The name of the parameter is mapped to its pointer of ito::Param.
    QHash<QString, ito::Param*> m_outputParams;       //!> the node's output parameter, given as ito::Param. The name of the parameter is mapped to its pointer of ito::Param.
    QList<QSharedPointer<Channel> > m_channels;  //!>

    unsigned int m_uniqueID;

    QSharedPointer<Channel> getChannel(const AbstractNode *parent, const ito::Param* parentParam,
        const AbstractNode *child, const ito::Param* childParam) const;

    bool channelExists(const QSharedPointer<Channel> &other) const;
};


//----------------------------------------------------------------------------------------------------------------------------------
QSharedPointer<Channel> AbstractNodePrivate::getChannel(const AbstractNode *parent, const ito::Param* parentParam,
    const AbstractNode *child, const ito::Param* childParam) const
{
    uint channelHash = Channel::calcChannelHash(parent, parentParam, child, childParam);

    foreach(const QSharedPointer<ito::Channel> channel, m_channels)
    {
        if ((channel->getHash() == channelHash))
        {
            return channel;
        }
    }

    return QSharedPointer<Channel>();
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractNodePrivate::channelExists(const QSharedPointer<Channel> &other) const
{
    foreach(const QSharedPointer<ito::Channel> channel, m_channels)
    {
        if ((channel->getHash() == other->getHash()))
        {
            return true;
        }
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------------------
void dumpChannels(const QList<QSharedPointer<Channel> > &channels, AbstractNode* an, const QString &prefix)
{
    qDebug() << prefix << " Channel dump for node " << an;
    for (int i = 0; i < channels.size(); ++i)
    {
        Channel* c = channels[i].data();
        qDebug() << i << ". " << c->getSender() << "->" << c->getSenderParamName() << "::" << c->getReceiver() << "->" << c->getReceiverParam()->getName() << " pending: " << c->getUpdateState();
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
AbstractNode::AbstractNode() :
    d_ptr(new AbstractNodePrivate(rttiUnknown, UID++))
{
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractNode::~AbstractNode()
{
    Q_D(AbstractNode);

    foreach(QSharedPointer<Channel> channel, d->m_channels)
    {
        removeChannel(channel);
    }
    d->m_channels.clear();

    ito::Param *delParam;

    foreach(delParam, d->m_inputParams)
    {
        DELETE_AND_SET_NULL(delParam);
    }

    d->m_inputParams.clear();

    foreach(delParam, d->m_outputParams)
    {
        DELETE_AND_SET_NULL(delParam);
    }

    d->m_outputParams.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::createChannel(const QString &senderParamName, AbstractNode *receiver, const QString &receiverParamName, bool replaceChannelIfExists /*= false*/)
{
    Q_D(AbstractNode);

    ito::RetVal retVal;

    ito::Param *senderParam = this->getOutputParam(senderParamName);
    ito::Param *receiverParam = receiver->getInputParam(receiverParamName);

    if (!senderParam)
    {
        return ito::RetVal::format(ito::retError, 0, "The parameter '%s' is no output parameter if the sender node", senderParamName.toLatin1().data());
    }
    else if (!receiverParam)
    {
        return ito::RetVal::format(ito::retError, 0, "The parameter '%s' is no input parameter if the receiver node", receiverParamName.toLatin1().data());
    }

    QSharedPointer<Channel> existingChannel = d->getChannel(this, senderParam, receiver, receiverParam);

    if (!existingChannel.isNull())
    {
        if (replaceChannelIfExists)
        {
            retVal += removeChannel(existingChannel);
        }
        else
        {
            //channel exists already, return
            return retVal;
        }
    }

    if (apiCompareParam(*receiverParam, *senderParam, retVal) == ito::tCmpFailed)
    {
        retVal += ito::RetVal(ito::retError, 0, QObject::tr("The output parameter of the sender and the input parameter of the receiver are not compatible").toLatin1().data());
    }

    if (retVal.containsError())
    {
        return retVal;
    }

    QSharedPointer<ito::Channel> newChannel(new Channel(this, senderParam, receiver, receiverParam));

    qDebug() << "create channel " << this << ". sender: " << newChannel->getSender()
        << "->" << newChannel->getSenderParam()->getName() << ", receiver: "
        << newChannel->getReceiver() << "->" << newChannel->getReceiverParam()->getName();

    d->m_channels << newChannel;

    receiver->attachChannel(newChannel);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::removeChannel(QSharedPointer<Channel> channel)
{
    Q_D(ito::AbstractNode);

    qDebug() << "remove channel " << this << ". sender: " << channel->getSender()
        << "->" << channel->getSenderParam()->getName() << ", receiver: "
        << channel->getReceiver() << "->" << channel->getReceiverParam()->getName();

    channel->getSender()->detachChannel(channel);
    channel->getReceiver()->detachChannel(channel);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::attachChannel(QSharedPointer<Channel> channel)
{
    Q_D(AbstractNode);

    qDebug() << "attach channel " << this << ". sender: " << channel->getSender()
        << "->" << channel->getSenderParam()->getName() << ", receiver: "
        << channel->getReceiver() << "->" << channel->getReceiverParam()->getName();


    if (d->channelExists(channel))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("The given channel is already attached.").toLatin1().data());
    }

    d->m_channels << channel;

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::detachChannel(QSharedPointer<Channel> channel)
{
    Q_D(AbstractNode);

    qDebug() << "detach channel " << this << ". sender: " << channel->getSender()
        << "->" << channel->getSenderParam()->getName() << ", receiver: "
        << channel->getReceiver() << "->" << channel->getReceiverParam()->getName();


    QList<QSharedPointer<Channel> >::iterator it = d->m_channels.begin();

    while (it != d->m_channels.end())
    {
        if (it->data()->getHash() == channel->getHash())
        {
            it = d->m_channels.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::addOutputParam(ito::Param* param)
{
    Q_D(AbstractNode);

    QString name = QLatin1String(param->getName());

    if (d->m_outputParams.contains(name))
    {
        return ito::RetVal(ito::retError, 0, "The same parameter name already exists");
    }
    else
    {
        d->m_outputParams[name] = param;
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> removes the parameter 'paramName' from the list of output parameters and deletes it (if it exists).
//!> Also removes all channels that have this output parameter as sender.
RetVal AbstractNode::removeAndDeleteOutputParam(const QString &paramName)
{
    Q_D(AbstractNode);

    ito::RetVal retVal;

    if (d->m_outputParams.contains(paramName))
    {
        QList<QSharedPointer<Channel> > channels = d->m_channels;
        ito::Param *p = d->m_outputParams[paramName];

        foreach(QSharedPointer<Channel> channel, channels)
        {
            if (channel->getSenderParam() == p)
            {
                retVal += removeChannel(channel);
            }
        }

        d->m_outputParams.remove(paramName);
        DELETE_AND_SET_NULL(p);
    }
    else
    {
        return ito::RetVal::format(ito::retWarning, 0, "An output parameter with name '%s' does not exist.", paramName.toLatin1().data());
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::addInputParam(ito::Param* param)
{
    Q_D(AbstractNode);

    QString name = QLatin1String(param->getName());

    if (d->m_inputParams.contains(name))
    {
        return ito::RetVal(ito::retError, 0, "The same parameter name already exists");
    }
    else
    {
        d->m_inputParams[name] = param;
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!> removes the parameter 'paramName' from the list of output parameters and deletes it (if it exists).
//!> Also removes all channels that have this output parameter as sender.
RetVal AbstractNode::removeAndDeleteInputParam(const QString &paramName)
{
    Q_D(AbstractNode);

    ito::RetVal retVal;

    if (d->m_inputParams.contains(paramName))
    {
        QList<QSharedPointer<Channel> > channels = d->m_channels;
        ito::Param *p = d->m_inputParams[paramName];

        foreach(QSharedPointer<Channel> channel, channels)
        {
            if (channel->getReceiverParam() == p)
            {
                retVal += removeChannel(channel);
            }
        }

        d->m_inputParams.remove(paramName);
        DELETE_AND_SET_NULL(p);
    }
    else
    {
        return ito::RetVal::format(ito::retWarning, 0, "An input parameter with name '%s' does not exist.", paramName.toLatin1().data());
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!>
/*
@param isSource indicates if this update is a first trigger for updating the parameter (true), or if the update is a propagation
       from another source trigger of a parent node (false).
*/
RetVal AbstractNode::inputParamChanged(const ito::ParamBase *updatedInputParam)
{
    Q_D(AbstractNode);

    ito::RetVal retval = ito::retOk;

    QString paramName = QLatin1String(updatedInputParam->getName());

    ito::Param *inputParam = getInputParam(paramName);

    if (!inputParam)
    {
		qWarning("Input parameter '%s' does not exist in input parameters of node. Call 'addInputParam' for this parameter first.", paramName.toLatin1().data());
        return ito::RetVal(ito::retError, 0, QObject::tr("Parameter name '%1' does not exist in input parameters").arg(paramName).toLatin1().data());
    }

    //recursively set the update-pending to all outgoing channels. The receiver nodes recursively set the update-pending flag of their outgoing channels etc.
    retval += setUpdatePending();

    if (retval.containsError())
    {
        return retval;
    }

    // only copy parameter if the update is not called with the parameter of this node, otherwise arrays inside the parameter will be deleted
    if (inputParam != updatedInputParam)
    {
        retval += inputParam->copyValueFrom(updatedInputParam);
    }

    if (retval.containsError())
    {
        return retval;
    }

    //check if all necessary input is available
    foreach(const QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        if (channel->isReceiver(this) && channel->getUpdateState() == Channel::StateUpdatePending)
        {
            return retval;
        }
    }

    retval += update(); //do what you have to do

    if (retval.containsError())
    {
        return retval;
    }

    foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        if (channel->isReceiver(this))
        {
            channel->resetUpdateState();
        }
    }

    //dumpChannels(d->m_channels, this, "Update input param. After reset.");

    if (retval.containsError())
    {
        return retval;
    }

    foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        if (channel->isSender(this))
        {
            ito::AbstractNode *receiver = channel->getReceiver();
            retval += receiver->updateChannelData(channel);

            if (retval.containsError())
            {
                return retval;
            }
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::updateChannelData(QSharedPointer<Channel> updatedChannel)
{
    Q_D(AbstractNode);

    ito::RetVal retval = ito::retOk;
    QString paramName = updatedChannel->getReceiverParamName();
    ito::Param *inputParam = updatedChannel->getReceiverParam();

    if (!inputParam)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Parameter name '%1' does not exist in input parameters").arg(paramName).toLatin1().data());
    }
    else if (!updatedChannel->isReceiver(this))
    {
        return ito::RetVal(ito::retError, 0, "updateChannelData has to be called from the receiver of the channel.");
    }

    if (updatedChannel->getUpdateState() == Channel::StateIdle)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Updated input data can only be propagated through a connected channel, if its 'update pending' flag has been set before.").toLatin1().data());
    }

    if (updatedChannel->getUpdateState() == Channel::StateUpdateReceived)
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Channel is already updating").toLatin1().data());
    }

    //now set the flag that this input channel is being updated right now.
    updatedChannel->setUpdateReceived();

    // only copy parameter if the update is not called with the parameter of this node, otherwise arrays inside the parameter will be deleted
    if (inputParam != updatedChannel->getSenderParam())
    {
        retval += inputParam->copyValueFrom(updatedChannel->getSenderParam());
    }

    if (retval.containsError())
    {
        return retval;
    }

    //now check if all input channels of this node are already updated, meaning that their update pending flags
    //are all reset to false. If this is the case, it is assumed that the last input has been updated and
    //the update method can then be called.
    foreach(const QSharedPointer<ito::Channel> channel, d->m_channels) //check if all necessary input is available
    {
        if (channel->isReceiver(this) && channel->getUpdateState() == Channel::StateUpdatePending)
        {
            return retval;
        }
    }

    retval += update(); //do what you have to do

    if (retval.containsError())
    {
        return retval;
    }

    foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        if (channel->isReceiver(this))
        {
            channel->resetUpdateState();
        }
    }

    //dumpChannels(d->m_channels, this, "Update input param. After reset.");

    if (retval.containsError())
    {
        return retval;
    }

    //propagate the update to
    foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        if (channel->isSender(this))
        {
            ito::AbstractNode *receiver = channel->getReceiver();
            retval += receiver->updateChannelData(channel);

            if (retval.containsError())
            {
                return retval;
            }
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QSharedPointer<Channel> > AbstractNode::getConnectedInputChannels(const QString &inputParamName) const
{
    ito::Param *p = getInputParam(inputParamName);

    return getConnectedInputChannels(p);
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QSharedPointer<Channel> > AbstractNode::getConnectedInputChannels(const ito::Param *inputParam) const
{
    Q_D(const AbstractNode);

    QList<QSharedPointer<Channel> > inputChannels;

    foreach(const QSharedPointer<Channel> &channel, d->m_channels)
    {
        if (channel->isReceiver(this) && channel->getReceiverParam() == inputParam)
        {
            inputChannels << channel;
        }
    }

    return inputChannels;
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QSharedPointer<Channel> > AbstractNode::getConnectedOutputChannels(const QString &outputParamName) const
{
    ito::Param *p = getOutputParam(outputParamName);

    return getConnectedOutputChannels(p);
}

//----------------------------------------------------------------------------------------------------------------------------------
QList<QSharedPointer<Channel> > AbstractNode::getConnectedOutputChannels(const ito::Param *outputParam) const
{
    Q_D(const AbstractNode);

    QList<QSharedPointer<Channel> > outputChannels;

    foreach(const QSharedPointer<Channel> &channel, d->m_channels)
    {
        if (channel->isSender(this) && channel->getSenderParam() == outputParam)
        {
            outputChannels << channel;
        }
    }

    return outputChannels;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::removeAllChannelsToReceiver(const AbstractNode *receiver, QList<ParamNamePair> excludedConnections /*= QList<ParamNamePair>()*/)
{
    Q_D(AbstractNode);

    RetVal retVal;

    if (excludedConnections.size() == 0)
    {
        QList<QSharedPointer<Channel> > allChannels = d->m_channels;

        foreach(QSharedPointer<Channel> channel, allChannels)
        {
            if (channel->getReceiver() == receiver)
            {
                retVal += removeChannel(channel);
            }
        }
    }
    else
    {
        QList<uint> excludedHashes;
        ito::Param *s;
        ito::Param *r;

        foreach(const ParamNamePair &item, excludedConnections)
        {
            s = getOutputParam(item.first);
            r = receiver->getInputParam(item.second);

            if (s && r)
            {
                excludedHashes.append(Channel::calcChannelHash(this, s, receiver, r));
            }
        }

        QList<QSharedPointer<Channel> > allChannels = d->m_channels;

        foreach(QSharedPointer<Channel> channel, allChannels)
        {
            if (channel->getReceiver() == receiver)
            {
                if (!excludedHashes.contains(channel->getHash()))
                {
                    retVal += removeChannel(channel);
                }
            }
        }
    }

    return retVal;
}



//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::updateChannels(const QList<QString> &outputParamNames)
{
    Q_D(AbstractNode);

    ito::RetVal retVal;
    QList<QString> copyParamNames = outputParamNames;
    QString thisName;
    QList<QSharedPointer<ito::Channel> > channelList;
    int nrProcessedChannels = 0;

    //dumpChannels(d->m_channels, this, "update channels start.");

    // Iterate over all channels and check if they are outputs from this node and whose senderParam name is within the given list
    foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
    {
        // CHANGE / CHECK TODO by Wolfram Lyda on 22.05.2015 to avoid missing update if multiple channels are attached to same output parameter
        // Changed copyParamNames to paramNames
        if ((channel->isSender(this)) && (outputParamNames.contains(channel->getSenderParamName())))
        {
            // If they are in the list, we trigger an update and remove them from the temp list
            channelList.append(channel);
            // If we have at least one channel with this param, we remove its name from the copied list to check wether all params were found
            copyParamNames.removeOne(channel->getSenderParamName());
            retVal += setUpdatePending(channel);
        }
    }

    //dumpChannels(d->m_channels, this, "update channels after update");

    if (retVal.containsError())
    {
        return retVal;
    }

    if (copyParamNames.length() != 0)
    {
        // even if we have not found every parameter in the channel list, we should update the rest anyway!
        retVal += ito::RetVal(ito::retWarning, 0,
            QObject::tr("Not all parameters in list could not be found in channels, in updateChannels").toLatin1().data());
    }

    foreach(QSharedPointer<ito::Channel> thisChannel, channelList)
    {
        ito::AbstractNode *receiver = thisChannel->getReceiver();
        retVal += receiver->updateChannelData(thisChannel);
    }

    //check if there are still some unhandled updates
    foreach(const QSharedPointer<ito::Channel> &channel, d->m_channels)
    {
        if (channel->getUpdateState() == Channel::StateUpdatePending)
        {
            return ito::retError;
        }
    }

    return retVal;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::setUpdatePending(QSharedPointer<ito::Channel> singleOutputChannel /*= QSharedPointer<ito::Channel>()*/)
{
    Q_D(AbstractNode);

    RetVal retval = ito::retOk;
    if (singleOutputChannel.isNull())
    {
        foreach(QSharedPointer<ito::Channel> channel, d->m_channels)
        {
            if (channel->isSender(this))
            {
                retval += channel->propagateUpdatePending();
            }
        }
    }
    else
    {
        bool found = false;

        foreach(QSharedPointer<Channel> channel, d->m_channels)
        {
            if (channel.data() == singleOutputChannel.data())
            {
                found = true;

                if (channel->isSender(this))
                {
                    retval += channel->propagateUpdatePending();
                }
                else
                {
                    retval += RetVal(ito::retError, 0, QObject::tr("channel is not a sender in setUpdatePending").toLatin1().data());
                }
                break;
            }
        }

        if (!found)
        {
            retval += RetVal(ito::retError, 0, QObject::tr("unknown channel in setUpdatePending").toLatin1().data());
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
rttiNodeType AbstractNode::getType() const
{
    Q_D(const AbstractNode);

    return d->m_nodeType;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractNode::isConnected() const
{
    Q_D(const AbstractNode);

    return !(d->m_channels.isEmpty());
}

//----------------------------------------------------------------------------------------------------------------------------------
unsigned int AbstractNode::getUniqueID(void) const
{
    Q_D(const AbstractNode);

    return d->m_uniqueID;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::Param* AbstractNode::getInputParam(const QString &paramName) const
{
    Q_D(const AbstractNode);

    if (d->m_inputParams.contains(paramName))
    {
        return d->m_inputParams[paramName];
    }

    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::Param* AbstractNode::getOutputParam(const QString &paramName) const
{
    Q_D(const AbstractNode);

    if (d->m_outputParams.contains(paramName))
    {
        return d->m_outputParams[paramName];
    }

    return NULL;
}

} //end namespace ito
