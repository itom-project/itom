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

#ifndef ABSTRACTNODE_H
#define ABSTRACTNODE_H

#include "plotCommon.h"
#include "../common/sharedStructures.h"

#include <qlist.h>
#include <qobject.h>
#include <qhash.h>
#include <qpair.h>
#include <qsharedpointer.h>
#include <qscopedpointer.h>

namespace ito {

class Channel; // forward declaration
class AbstractNode;

//----------------------------------------------------------------------------------------------------------------------------------
/* runtime type information enumeration for type of plot

*/
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

class ChannelPrivate;      //forward declaration
class AbstractNodePrivate; //forward declaration

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class Channel
    \brief A channel defines a propagation pipline between two parameters of two nodes. The nodes are instances
    of the class AbstractNode, which is for instance a base class of AbstractFigure, and therefore each plot designer
    plugin in itom.

    Every instance of AbstractNode can define a set of input and output parameters as list of ito::Param.

    This channel is a directed connection between one output parameter of the sender node and one input parameter of a
    receiver node. Both parameters must be compatible, hence, have the same type.
*/
class ITOMCOMMONPLOT_EXPORT Channel
{
public:

    /* the UpdateState enumeration defines possible states during the
       of the sender parameter of a channel.

       Usually, the update state is idle. If a parameter has been changed,
       its connected channels are set to StateUpdatePending, such that the
       channel is informed about the upcoming propagation of this changed input
       parameter. This update information is then recursively propagated to
       all child nodes (and their child nodes) of the affected node.

       Then, the value of the changed sender parameter is copied via all
       affected channels to their respective receiver parameters. The state
       is then changed to StateUpdateReceived.

       The view of affected nodes is then updated once all input channels,
       whose state has been set to StateUpdatePending is changed to StateUpdateReceived.
       This guarantees that the real update of the view is only done once all
       input channels have finished propagating their changed parameters.
    */
    enum UpdateState
    {
        StateIdle,          //! the channel is in an idle state (default)
        StateUpdatePending, //! the channel is informed to receive an update of the sender soon
        StateUpdateReceived //! the channel was in the update pending state and received an update of its sender and propagated it to its receiver param.
    };

    //! default constructor of a channel
    /*!
    This constructor creates a channel with an invalid sender as well as receiver (both NULL).
    */
    Channel();

    //! constructor of a properly initialized channel
    /*!
    This method constructs an instance of class Channel, that connects a given output parameter (senderParam)
    of a sender node to a given input parameter (receiverParam) of another receiver node.
    */
    Channel(AbstractNode *sender, ito::Param *senderParam,
        AbstractNode *receiver, ito::Param *receiverParam);

    //! destructor of channel.
    virtual ~Channel();

    //! copy constructor
    Channel(const Channel& cpy);

    //! assign operator
    Channel &operator=(const Channel& rhs);

    //! returns the sender node of this channel (or NULL if not available)
    AbstractNode* getSender() const;

    //! returns the receiver node of this channel (or NULL if not available)
    AbstractNode* getReceiver() const;

    //! returns the current update state of this channel (see enumeration UpdateState)
    UpdateState getUpdateState() const;

    //! return parameter name as QLatin1String of output parameter of sender, connected to this channel.
    /* This is equivalent to QLatin1String(getSender()->getName())
    */
    QString getSenderParamName() const;

    //! return parameter name as QLatin1String of input parameter of receiver, connected to this channel
    /* This is equivalent to QLatin1String(getReceiver()->getName())
    */
    QString getReceiverParamName() const;

    //! this method sets the updateState of this channel to StateUpdatePending and calls setUpdatePending of the receiver node.
    /*
    By propagating updates, the whole sub tree of nodes, which has a direct or indirect channel connection to the sender
    of this channel is informed about an incoming update.

    Note: this function does not propagate any updated data. It only informs about a future update of data.

    \return ito::retOk if the update pending flag has been set or if it was already set. Returns ito::retError if an error occurred.
    */
    RetVal propagateUpdatePending();

    //! informs this channel that changed value of the sender parameter has been copied / set to its receiver parameter
    RetVal setUpdateReceived();

    /*!> The hash identifying this channel in the connection list of the participating nodes */
    uint getHash() const;

    //! returns the input parameter of this channel, which is an output parameter of the sender node
    ito::Param * getSenderParam() const;

    //! returns the output parameter of this channel, which is an input parameter of the receiver node
    ito::Param * getReceiverParam() const;

    /*!> Resets the state to StateIdle. Note: This reset is NOT propagated down the node tree */
    void resetUpdateState(void);

    /*!> returns true, if the node given as the argument fills the role of sender in this channel. */
    bool isSender(const AbstractNode *node) const;

    /*!> returns true, if the node given as the argument fills the role of receiver in this channel. */
    bool isReceiver(const AbstractNode *node) const;

    //----------------------------------------------------------------------------------------------------------------------------------
    //! static helper method to calculate a hash value based on the sender, the receiver and both participating parameters.
    static uint calcChannelHash(const AbstractNode *sender,
        const ito::Param *senderParam,
        const AbstractNode *receiver,
        const ito::Param *receiverParam);

private:
    QScopedPointer<ChannelPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed). pointer to private class of Channel defined in Channel.cpp. This container is used to allow flexible changes in the interface without destroying the binary compatibility
    Q_DECLARE_PRIVATE(Channel);
};

//----------------------------------------------------------------------------------------------------------------------------------
/*!
    \class AbstractNode
    \brief Every plot designer plugin in itom, that should be able to open dependent sub-plots (e.g. a 1d line cut is a subplot of
    its 2d parent plot), must be inherited from this class AbstractNode. However, AbstractNode is no widget class of Qt. Therefore
    it is also possible to create non-visible classes derived from AbstractNode. This allows creating a directed net of nodes
    where changes of some input parameters are propagated through all relevant sub-nodes.

    Every node has a set of input parameters (as list of ito::Param) and a set of output parameters. One output parameter of one
    node can be connected with another input parameter of another node, using the class ito::Channel (see methods createChannel,
    removeChannel, ...). Whenever one or multiple output parameters of one node are changed, the changes are propagated through
    all connected channels to the corresponding parameters of the receiver node(s).

    Since the real update of the view of a plot needs some computing power, a mechanism (called 'pending update') is integrated
    in the channels, such that the view update is done as soon as the announced changes of all affected input channels have been
    propagated to all affected input parameters of one node.

    This is realized by a two-step update mechanism: If one or multiple output parameters have been changed, an update of all
    possible channels is triggered, which are connected to these output parameters. Then the states of these channels is set to
    Channel::StateUpdatePending. This is repeated recursively in all receiving nodes of these affected channels, such that all
    output channels of these receivers are also notified about the upcoming update (update pending).

    If the pending update is reported recursively through the entire sub-tree below the changed initial parameters, the changed
    parameters are started to be copied from sender to receiver in every affected channel. Once done, the state of the channel
    switches from Channel::StateUpdatePending to Channel::StateUpdateReceived. As soon as the last input channel of a receiver node
    switched its state from StateUpdatePending to StateUpdateReceived, the real update() method of the node is called, which calls
    the pure virtual method applyUpdate, that calculates the real view update of the plot (or similar node element).
*/
class ITOMCOMMONPLOT_EXPORT AbstractNode
{
    public:

        typedef QPair<QString, QString> ParamNamePair;

        //! constructor
        AbstractNode();

        //! destructor
        virtual ~AbstractNode(); //>! will delete the node and all data associated with it, except m_pCurrentData if m_isCurrent is set to TRUE

        virtual RetVal applyUpdate(void) = 0; /*!> Performs the in INTERNAL operations necessary in updating the node and displaying the data. This has to implemented
                                                   in each final successor in the inheritance structure. Anyway the "displayed" parameter MUST be filled adequately as
                                                   this can only be done by the node itself. */

        //! update of this node if the values of all changed input channels are propgated to the corresponding input parameters of this node
        /* if all channels, whose receiver is this node and that are marked for updates (pending updates),
        are updated with new values, this method is internally called. It calls applyUpdate, that has to be overwritten
        by the real plot/figure class, and starts to update all outgoing channels afterwards.
        */
        virtual RetVal update(void) = 0; /*!> Calls applyUpdate() and updates all children*/

        //! creates a new channel from this node as sender and a specific sender parameter (output) to a receiver node and its specific receiver (input) parameter.
        /*
        If the channel is created, it is appended to the channel list of this node and also attached to the channel list of the receiver node.

        \param senderParamName is the output parameter name of this node, that is used as sending parameter
        \param receiver is the receiver node
        \param receiverParamName is the input parameter of the receiver, that is used as destination for the channel.
        \param replaceChannelIfExists defines if an existing channel with the same sender / receiver combination should be destroyed and created again or if
                                      the existing channel is kept without creating the new one.

        \seealso attachChannel
        */
        RetVal createChannel(const QString &senderParamName, AbstractNode *receiver, const QString &receiverParamName, bool replaceChannelIfExists = false);

        //!> removes the channel from the list of channels, forces the partner node to detach the channel and destroys the channel object
        /*
        \seealso detachChannel
        */
        RetVal removeChannel(QSharedPointer<Channel> channel);

        //!>
        /*
        @excludedConnections can be an optional pair of two strings, where the first string is the name of the sender parameter and the 2nd string the name of the receiver parameter.
                             If channels to the given receiver are found, that match to one entry in excludedConnections, they will not be removed.
        */
        RetVal removeAllChannelsToReceiver(const AbstractNode *receiver, QList<ParamNamePair> excludedConnections = QList<ParamNamePair>());

        //!> returns a list of channels, that are connected to an input param with a given name. This node must be the receiver of the returned channels.
        QList<QSharedPointer<Channel> > getConnectedInputChannels(const QString &inputParamName) const;

        //!> returns a list of channels, that are connected to a given input parameter. This node must be the receiver of the returned channels.
        QList<QSharedPointer<Channel> > getConnectedInputChannels(const ito::Param *inputParam) const;

        //!> returns a list of channels, that are connected to an ouput param with a given name. This node must be the sender of the returned channels.
        QList<QSharedPointer<Channel> > getConnectedOutputChannels(const QString &outputParamName) const;

        //!> returns a list of channels, that are connected to a given output parameter. This node must be the sender of the returned channels.
        QList<QSharedPointer<Channel> > getConnectedOutputChannels(const ito::Param *outputParam) const;

        //!> This method is called, if the sender parameter of the channel changed its value.
        /* This method has to copy the new value to the receiver and continue the recursive update to other child nodes.
        */
        RetVal updateChannelData(QSharedPointer<Channel> updatedChannel);

        //!>propagates information about a soon, future update to all (or one selected) output channel(s) of this node
        /*
        if the parameter singleOutputChannel is a null object, this method calls Channel::propagateUpdatePending to
        all channels, whose sender is this node. Else, the propagateUpdatePending method is only called for the
        one singleOutputChannel.

        \return ito::retError if singleOutputChannel is valid, however not part of the current output channels of this node. Else: ito::retOk.
        */
        RetVal setUpdatePending(QSharedPointer<ito::Channel> singleOutputChannel = QSharedPointer<ito::Channel>());

        //! returns the input parameter whose name is equal to paramName or returns NULL if the parameter is not found.
        ito::Param* getInputParam(const QString &paramName) const;

        //! returns the output parameter whose name is equal to paramName or returns NULL if the parameter is not found.
        ito::Param* getOutputParam(const QString &paramName) const;

        //! adds the given parameter to the list of output parameters of this node.
        /* This node takes the ownership of the parameter and deletes it during the destructor.
        */
        RetVal addOutputParam(ito::Param* param);

        //!> removes the parameter 'paramName' from the list of output parameters and deletes it (if it exists).
        //!> Also removes all channels that have this output parameter as sender.
        RetVal removeAndDeleteOutputParam(const QString &paramName);

        //! adds the given parameter to the list of input parameters of this node.
        /* This node takes the ownership of the parameter and deletes it during the destructor.
        */
        RetVal addInputParam(ito::Param* param);

        //!> removes the parameter 'paramName' from the list of output parameters and deletes it (if it exists).
        //!> Also removes all channels that have this output parameter as sender.
        RetVal removeAndDeleteInputParam(const QString &paramName);

        //! triggers updates of all channels that are connected to the given output parameter names of the sender.
        /* If one or multiple output parameters of a node have been changed, call this method to let the changes
        be propagated to the receiver parameters of all channels, whose sender is this node and whose sender parameter
        is equal to the given names.
        */
        RetVal updateChannels(const QList<QString> &outputParamNames);

        //! returns the type of this node
        rttiNodeType getType() const;

        //! returns true if at least one channel is connected to this node
        bool isConnected() const;

        //! return a unique, auto-incremented UID of this node (beginning with 1)
        unsigned int getUniqueID(void) const;

    protected:
        //! removes the given channel from the list of chnanels of this node.
        /*
        This method does not detach this channel from the other node, that is sender or receiver of the channel.
        */
        RetVal detachChannel(QSharedPointer<Channel> channel);

        //!> attaches the given channel to the list of channels (either input or output, depending if this is sender or receiver)
        RetVal attachChannel(QSharedPointer<Channel> channel);

        //!> indicates that an input parameter of the node has been changed and initializes the update pipline to child nodes if necessary. This is the source of an update.
        RetVal inputParamChanged(const ito::ParamBase *updatedInputParam); /*!> Updates the input param of the associated node and attempts to propagate the update down the node tree.
                                                        It DOES NOT/CANNOT, however, ensure that the output parameters of the node are updated correctly, this functionality has to be a part of update(). */

        //! highest used UID of any node. This value is auto-incremented upon every
        static unsigned int UID;

    private:
        QScopedPointer<AbstractNodePrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed). pointer to private class of AbstractNode defined in AbstractNode.cpp. This container is used to allow flexible changes in the interface without destroying the binary compatibility
        Q_DECLARE_PRIVATE(AbstractNode);
};

//----------------------------------------------------------------------------------------------------------------------------------
} // namespace ito

#endif //ABSTRACTNODE_H
