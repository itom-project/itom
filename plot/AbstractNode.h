/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef AbstractNode_H
#define AbstractNode_H

#include "plotCommon.h"
#include "../common/sharedStructures.h"

#include <qlist.h>
#include <qobject.h>
#include <qhash.h>

namespace ito {

class Channel; // forward declaration
class AbstractNode;

//----------------------------------------------------------------------------------------------------------------------------------
ITOMCOMMONPLOT_EXPORT uint calculateChannelHash(AbstractNode *parent, ito::Param *parentParam, AbstractNode *child, ito::Param *childParam);

//----------------------------------------------------------------------------------------------------------------------------------
typedef enum 
{
    rttiUnknown             = 0x0000, //default node type
    rttiPlotNode            = 0x1000, //arbitray nodes in a tree of graphic nodes
    rttiFilterNode          = 0x2000,
    rttiPlotNodeDObj        = 0x0100,
    rttiPlotNodePCL         = 0x0200,
    rttiPlotNode1D          = 0x0001,
    rttiPlotNode2D          = 0x0002,
    rttiPlotNode25D         = 0x0003,
    rttiPlotNode3D          = 0x0004
} rttiNodeType;


//----------------------------------------------------------------------------------------------------------------------------------
class ITOMCOMMONPLOT_EXPORT Channel
{    
    public: 
       enum ChanDirection {
            undefined,
            parentToChild,
            childToParent
        };
    private:
        static unsigned int UID; /*!> Running counter for nodes created in the UiOrganizer */
        unsigned int m_uniqueID; /*!> Unique indentifier used in the UiOrganizer */
        AbstractNode* m_pParent; /*!> The node calling the constructor of the channel. Note: the parent is NOT necessarily the sender in the channel*/
        AbstractNode* m_pChild; /*!> The node that is not the parent. Note: the child is NOT necessarily the receiver in the channel*/
        ChanDirection m_direction; /*!> Defines whether the parent or the child are the sender/receiver in this channel */
        ito::Param* m_pParentParam; //!> the parameter-connector on the parent´s side (can be sending or receiving)
        ito::Param* m_pChildParam; //!> the parameter-connector on the child´s side (can be sending or receiving)
        bool m_deleteParentOnDisconnect, m_deleteChildOnDisconnect; /*!> Flags defining the behaviour of the connected nodes upon a disconnect. If one of 
                                                                    these flags is set to TRUE, the corresponding node is deleted, when the connection is cut.
                                                                    Note: */
        bool m_updatePending; //!> tells the receiver to wait for updates before proceeding
        bool m_channelBuffering; //!> marks, if the Receiver has received and is buffering data from a new source
        uint m_hashVal;

    public:
        Channel() : m_uniqueID(UID++), m_pParent(NULL), m_pChild(NULL), m_direction(Channel::undefined), m_pParentParam(NULL), m_pChildParam(NULL), m_deleteParentOnDisconnect(false), m_deleteChildOnDisconnect(false), m_updatePending(false), m_channelBuffering(false), m_hashVal(0) {}

        Channel(AbstractNode *parent, ito::Param *parentParam, bool deleteOnParentDisconnect, AbstractNode *child, ito::Param *childParam, bool deleteOnChildDiconnect, ChanDirection direction, bool update = false);
        
        ~Channel() {}
       
        inline AbstractNode* getParent() const { return m_pParent; }
        
        inline AbstractNode* getChild() const { return m_pChild; }
        
        inline ChanDirection getDirection() const { return m_direction; }
        
        inline bool getUpdatePending() const { return m_updatePending; }
        
        inline unsigned int getUniqueID() const { return m_uniqueID; }

        /*!> returns the own node's parameter participating in this channel. Note: The roles of sender/receiver are not relevant in this context*/
        inline ito::Param* getOwnParam(AbstractNode* self) const
        {
            if(self == m_pParent)
                return m_pParentParam;
            else if(self == m_pChild)
                return m_pChildParam;
            else 
                return NULL;
        }

        /*!> returns the other node's parameter participating in this channel. Note: The roles of sender/receiver are not relevant in this context*/
        inline ito::Param* getPartnerParam(AbstractNode* self) const
        {
            if(self == m_pParent)
                return m_pChildParam;
            else if(self == m_pChild)
                return m_pParentParam;
            else 
                return NULL;
        }

        inline QString getSenderParamName() const
        {
            if (m_direction == Channel::childToParent)
                return QString(m_pChildParam->getName());
            else if (m_direction == Channel::parentToChild)
                return QString(m_pParentParam->getName());
            else
                return QString();
        }

        inline signed char getDeleteBehaviour(AbstractNode* query) const 
        { 
            if(query == m_pParent)
                return m_deleteParentOnDisconnect;
            else if(query == m_pChild)
                return m_deleteChildOnDisconnect;
            else 
                return -1;
        }

        /*!> prepares the whole sub tree of nodes rooted at the sending node of this channel for an incoming update. Note: this function does NOT propagate any actual data */ 
        RetVal propagateUpdatePending();

        inline bool getChannelBuffering() const { return m_channelBuffering; }

        inline void setChannelBuffering(bool buffering) { m_channelBuffering = buffering; return; }

        /*!> The hash identifying this channel in the connection list of the participating nodes*/
        inline uint getHash() const { return m_hashVal; }

        inline ito::Param * getParentParam() const { return m_pParentParam; }

        inline ito::Param * getChildParam() const { return m_pChildParam; }

        /*!> Sets the updatePending flag of this channel back to false. Note: This reset is NOT propagated down the node tree*/
        inline void resetUpdatePending(void){ m_updatePending = false;}

        /*!> returns true, if the node given as the argument fills the role of sender in this channel. Useful, since parent/child do not have predefined roles */
        inline bool isSender(AbstractNode *query) 
        { 
            if (((m_direction == Channel::childToParent) && (m_pChild == query)) 
                || ((m_direction == Channel::parentToChild) && (m_pParent == query)))
                return true;
            else
                return false;
        }

        /*!> returns true, if the node given as the argument fills the role of receiver in this channel. Useful, since parent/child do not have predefined roles */
        inline bool isReceiver(AbstractNode *query) const
        { 
            if (((m_direction == Channel::childToParent) && (m_pParent == query)) 
                || ((m_direction == Channel::parentToChild) && (m_pChild == query)))
                return true;
            else
                return false;
        }

        /*!> returns a pointer to the node filling the role of sender in this channel. Useful, since parent/child do not have predefined roles */
        inline AbstractNode* getSender(void) const
        {
            if(m_direction ==  Channel::childToParent)
                return m_pChild;
            else if (m_direction ==  Channel::parentToChild)
                return m_pParent;
            else return NULL;
        }

        /*!> returns a pointer to the node filling the role of receiver in this channel. Useful, since parent/child do not have predefined roles */
        inline AbstractNode* getReceiver(void) const
        {
            if(m_direction ==  Channel::childToParent)
                return m_pParent;
            else if (m_direction ==  Channel::parentToChild)
                return m_pChild;
            else return NULL;
        }
};

//----------------------------------------------------------------------------------------------------------------------------------
class ITOMCOMMONPLOT_EXPORT AbstractNode
{
    public:
        AbstractNode();

        virtual ~AbstractNode(); //>! will delete the node and all data associated with it, except m_pCurrentData if m_isCurrent is set to TRUE

        virtual RetVal applyUpdate(void) = 0; /*!> Performs the in INTERNAL operations necessary in updating the node and displaying the data. This has to implemented
                                                   in each final successor in the inheritance structure. Anyway the "displayed" parameter MUST be filled adequately as
                                                   this can only be done by the node itself. */
        virtual RetVal update(void) = 0; /*!> Calls apply () and updates all children*/

        RetVal updateParam(const ito::ParamBase *input, int isSource = 0); /*!> Updates the input param of the associated node and attempts to propagate the update down the node tree. 
                                                        It DOES NOT/CANNOT, however, ensure that the output parameters of the node are updated correctly, this functionality has to be a part of update(). */

        RetVal updateParam(const QHash<QString, ito::ParamBase*> *inputs) //!> this function must ONLY be used for the root of a tree. Sets several input parameters at once
        {
            ito::RetVal retval = ito::retOk;

            foreach (const ito::ParamBase *thisParam, *inputs)
            {
                retval += updateParam(thisParam);
                if (retval.containsError())
                    return retval;
            }
            return retval;
        }

        inline rttiNodeType getType() const { return m_NodeType; }

        //!> the addChannel and especially removeChannel are virtual since the Node might have to delete itself depending on the deleteOnDisconnect flag.
        //!> As delete(this) is frowned upon, we need QObject::deleteLater() unavailable here as the class is not derived from QObject
        virtual RetVal addChannel(AbstractNode *child, ito::Param* parentParam, ito::Param* childParam, Channel::ChanDirection direction, bool deleteOnParentDisconnect, bool deleteOnChildDisconnect) = 0;
        virtual RetVal addChannel(Channel *newChannel) = 0;

        virtual RetVal removeChannelFromList(unsigned int UniqueID) = 0;
        virtual RetVal removeChannel(Channel *delChannel) = 0;

        RetVal getUpdateStatus(void) const;
        
        RetVal updateChannels(const QList<QString> &paramNames);    

        inline bool isConnected() const { return !(m_pChannels.isEmpty()); }

        inline int getUniqueID(void) const { return m_uniqueID; }

        RetVal setUpdatePending(int uniqueID = -1);

        Channel * getInputChannel(const char *inpParamName) const;

        inline ito::Param* getInputParam(const QString &paramName) const
        { 
            if (m_pInput.contains(paramName))
            {
                return m_pInput[paramName];
            }
            return NULL;
        }

        inline ito::Param* getOutputParam(const QString &paramName) const
        { 
            if (m_pOutput.contains(paramName))
            {
                return m_pOutput[paramName];
            }
            return NULL;
        }

    protected:
        rttiNodeType m_NodeType;            //!> the type of the actual node inheriting this abstract node
        QHash<QString, ito::Param*> m_pInput;        //!> the node's input parameters
        QHash<QString, ito::Param*> m_pOutput;       //!> the node's output parameters
        QHash<unsigned int, Channel*> m_pChannels;  //!> 

        int m_uniqueID;
        static unsigned int UID;
};

//----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito

#endif
