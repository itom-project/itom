/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
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

#include "AbstractNode.h"
#include "../common/addInInterface.h"

using namespace ito;

unsigned int ito::AbstractNode::UID = 1;
unsigned int ito::Channel::UID = 1;

//----------------------------------------------------------------------------------------------------------------------------------
uint ito::calculateChannelHash(AbstractNode *parent, ito::Param *parentParam, AbstractNode *child, ito::Param *childParam)
{
    return qHash(QString("%1%2%3%4").arg((size_t)parent, 8).arg((size_t)child, 8).arg((size_t)parentParam, 8).arg((size_t)childParam, 8));
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel::Channel(AbstractNode *parent, ito::Param *parentParam, bool deleteOnParentDisconnect, AbstractNode *child, ito::Param *childParam, bool deleteOnChildDiconnect, ChanDirection direction, bool update) :
    m_uniqueID(UID++), m_pParent(parent), m_pChild(child), m_direction(direction), m_pParentParam(parentParam), 
    m_pChildParam(childParam), m_deleteParentOnDisconnect(deleteOnParentDisconnect), m_deleteChildOnDisconnect(deleteOnChildDiconnect), m_updatePending(false), m_channelBuffering(false)
{
    m_hashVal = ito::calculateChannelHash(parent, parentParam, child, childParam);
    if (update)
    {
        childParam->copyValueFrom(parentParam);
        child->updateParam(childParam);
//                propagateUpdatePending(); // do we need this?
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal Channel::propagateUpdatePending()
{
    if( m_updatePending == true)
        return -1; //something went badly wrong, this channel has already been blocked

    m_updatePending = true;

    switch(m_direction)
    {
        case(parentToChild):
            return m_pChild->setUpdatePending();
        break;

        case(childToParent):
            return m_pParent->setUpdatePending();
        break;

        default:
            return ito::retError;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractNode::~AbstractNode()
{
    ito::Param *delParam;

    foreach(delParam, m_pInput)
    {
        delete delParam;
    }
    m_pInput.clear();

    foreach(delParam, m_pOutput)
    {
        delete delParam;
    }
    m_pOutput.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::updateParam(ito::ParamBase *input, int isSource /*=0*/)
{
    ito::RetVal retval = ito::retOk;
    Channel *thisChannel = NULL;

    if (!m_pInput.contains(input->getName()))
    {
        return ito::RetVal(ito::retError, 0, QObject::tr("Parameter: does not exist in updateParam").toLatin1().data()); //Todo: add parameter name in error string
    }

    Channel *inpChannel = getInputChannel(input->getName());

    if ((!isSource) && (inpChannel != NULL))
    {
        if (inpChannel->getUpdatePending() == false)
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("Running update on a locked input channel, i.e. updatePending flag is not set").toLatin1().data());
        }

        if (inpChannel->getChannelBuffering())
        {
            return ito::RetVal(ito::retError, 0, QObject::tr("Channel is already updating").toLatin1().data());
        }

        //now set the flag that this input channel is being updated right now.
        inpChannel->setChannelBuffering(true);
    }
    else
    {
        retval += setUpdatePending();
        if (retval.containsError())
        {
            return retval;
        }
    }

    // only copy parameter if the update is not called with the parameter of this node, otherwise arrays inside the parameter will be deleted
    if (m_pInput[input->getName()] != input)
    {
        retval += m_pInput.value(input->getName())->copyValueFrom(input); 
    }

    if (retval.containsError())
    {
        return retval;
    }

    foreach (thisChannel, m_pChannels) //check if all necessary input is available
    {
        if (thisChannel->isReceiver(this) && thisChannel->getUpdatePending() && thisChannel->getChannelBuffering()!=true)
        {
            return retval;
        }
    }

    retval += update(); //do what you have to do

    if (retval.containsError())
    {
        return retval;
    }

    if (inpChannel != NULL)
    {
        foreach (thisChannel, m_pChannels) 
        {
            if (thisChannel->isReceiver(this))
            {
                thisChannel->setChannelBuffering(false);
                thisChannel->resetUpdatePending();
            }
        }
    }
    if (retval.containsError())
    {
        return retval;
    }

    foreach(thisChannel, m_pChannels)
    {
        if (thisChannel->isSender(this))
        {
            thisChannel->getPartnerParam(this)->copyValueFrom(thisChannel->getOwnParam(this));
            retval += thisChannel->getReceiver()->updateParam(thisChannel->getPartnerParam(this));
            if (retval.containsError())
                return retval;
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*
bool AbstractNode::isRefreshPending()
{
    bool temp = m_refreshPending;
    AbstractNode* node;

    foreach(node,m_pChildren)
    {
        temp |= node->isRefreshPending();
    }
    return temp;
}
*/

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::getUpdateStatus(void) const
{
    ito::Channel *thisChannel;

    foreach(thisChannel, m_pChannels)
    {
        if (thisChannel->getUpdatePending() && !thisChannel->getChannelBuffering())
        {
            return ito::retError;
        }
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::updateChannels(QList<QString> paramNames)
{
    ito::RetVal retval = ito::retOk;
    ito::Channel *thisChannel;
    QList<QString> copyParamNames = paramNames;
    QString thisName;
    QList<ito::Channel *> channelList;

    foreach (thisChannel, m_pChannels)
    {
        if ((thisChannel->isSender(this)) && (copyParamNames.contains(thisChannel->getSenderParamName())))
        {
            channelList.append(thisChannel);
            copyParamNames.removeOne(thisChannel->getSenderParamName());
            retval += setUpdatePending(thisChannel->getUniqueID());
        }
    }
    if (retval.containsError()) 
        return retval;
    if (copyParamNames.length() != 0)
        return ito::RetVal(ito::retError, 0, QObject::tr("parameters in list could not be found in channels, in updateChannels").toLatin1().data());

    foreach (thisChannel, channelList)
    {
        ito::Param *partnerParam = thisChannel->getPartnerParam(this);
        partnerParam->copyValueFrom(m_pOutput[thisChannel->getSenderParamName()]);
        retval += thisChannel->getReceiver()->updateParam(partnerParam);
    }
    retval += getUpdateStatus();

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal AbstractNode::setUpdatePending(int uniqueID /*= -1*/)
{
    RetVal retval = 0;
    Channel *iterChannel;
    if (uniqueID == -1)
    {
        foreach(iterChannel, m_pChannels)
        {
            if (iterChannel->isSender(this))
            {            
                retval += iterChannel->propagateUpdatePending();
            }
        }
    }
    else
    {
        if (m_pChannels.contains(uniqueID))
        {
            if (m_pChannels[uniqueID]->isSender(this))
            {
                retval += m_pChannels[uniqueID]->propagateUpdatePending();
            }
            else
            {
                retval += RetVal(ito::retError, 0, QObject::tr("channel is not a sender in setUpdatePending").toLatin1().data());
            }
        }
        else
        {
            retval += RetVal(ito::retError, 0, QObject::tr("unknown channel in setUpdatePending").toLatin1().data());
        }
    }
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
Channel * AbstractNode::getInputChannel(const char *inpParamName)
{
    Channel *thisChannel;
    foreach(thisChannel, m_pChannels)
    {
        if (((thisChannel->getDirection() == Channel::parentToChild) && (strcmp(thisChannel->getChildParam()->getName(), inpParamName) == 0))
            || ((thisChannel->getDirection() == Channel::childToParent) && (strcmp(thisChannel->getParentParam()->getName(), inpParamName) == 0)))
        {
            return thisChannel;
        }
    }
    return NULL;
}