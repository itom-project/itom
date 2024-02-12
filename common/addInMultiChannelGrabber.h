/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#pragma once

#include "addInInterface.h"
#include "abstractAddInGrabber.h"

#include "../DataObject/dataobj.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"

// only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib
#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

namespace ito
{
    class AddInMultiChannelGrabberPrivate;

    class ITOMCOMMONQT_EXPORT AddInMultiChannelGrabber : public AbstractAddInGrabber
    {
        Q_OBJECT

    protected:

        //! enumeration values, that can be combined
        enum class DataStatus
        {
            //! idle = no new data in object
            Idle = 0,

            //! newData = new data in object, not emitted via newData
            //! yet, not fetched via getVal/copyVal yet
            NewData = 1,

            //! same than newData, however data is already emitted,
            //! but not fetched via getVal/copyVal yet
            NewDataAndEmitted = 2
        };

        class ITOMCOMMONQT_EXPORT ChannelContainer
        {
        public:
            ChannelContainer();
            ~ChannelContainer() = default;
            ChannelContainer(const ChannelContainer&) = default;
            ChannelContainer(
                const ito::Param& roi,
                const ito::Param& pixelFormat,
                const ito::Param& sizex,
                const ito::Param& sizey,
                const ito::Param& axisOffsets,
                const ito::Param& axisScales,
                const ito::Param& axisDescriptions,
                const ito::Param& axisUnits,
                const ito::Param& valueDescription,
                const ito::Param& valueUnit);
            ChannelContainer(
                const ito::Param& roi,
                const ito::Param& pixelFormat,
                const ito::Param& sizex,
                const ito::Param& sizey);

            void addChannelParam(const ito::Param& param);

            //!< dataObject container with currently acquired data for this channel
            ito::DataObject m_data;

            //!< state of the dataObject container
            DataStatus m_dataStatus;

            //!< map of individual parameters for this channel. Every channel has a set
            //! of default parameters, namely: pixelFormat, roi, sizex, sizey, axisOffset, axisScale.
            //! axisDescription, axisUnit, valueDescription, valueUnit. Further parameters
            //! can be added. It is recommended to have the same parameter types and names
            //! in every channel, since they are mirrored to the m_params map of the main
            //! plugin class.
            ParamMap m_channelParams;

        protected:
            void addDefaultMetaParams();
        };

        typedef QMap<QString, ChannelContainer> ChannelContainerMap;
        typedef ChannelContainerMap::iterator ChannelContainerMapIterator;
        typedef ChannelContainerMap::const_iterator ChannelContainerMapConstIterator;

        ChannelContainerMap m_channels; /*!< Map for recently grabbed images of various channels */

        virtual ito::RetVal checkData(ito::DataObject* externalDataObject = nullptr);
        virtual ito::RetVal checkData(const QString& channelName, ito::DataObject* externalDataObject = nullptr);

        //!< calls checkData for the internal data caches for all channels
        virtual ito::RetVal checkDataFromAllChannels();

        virtual ito::RetVal sendDataToListeners(int waitMS); /*!< sends m_data to all registered listeners. */
        ito::RetVal adaptDefaultChannelParams(); /*!< adaptes the params after changing the defaultChannel param*/

        void addChannel(QString name);

        virtual ito::RetVal switchChannelSelector();/*!< synchronizes m_params with the params of default channel container */
        virtual ito::RetVal applyParamsToChannelParams(const QStringList& keyList = QStringList());

        //! registers all channel containers, initializes their parameters as well as all common, global parameters
        ito::RetVal initChannelsAndGlobalParameters(
            const ChannelContainerMap& channelContainerMap,
            const QString& defaultChannelName,
            const QList<ito::Param>& globalParameters = QList<ito::Param>());

        ito::RetVal setParamMeta(const QByteArray& paramName, ito::ParamMeta* meta, bool takeOwnerShip, const QList<QByteArray>& channelList = QList<QByteArray>());
        ito::RetVal setParamFlags(const QByteArray& paramName, const unsigned int& flags, const QList<QByteArray>& channelList = QList<QByteArray>());

        //! This method fetches the latest acquired images or datasets from the grabber and store it into the channel.
        /*
        This method has to be overloaded by the specific plugin instance. It should
        fetch the latest acquired images or datasets for all given channels and store
        it into the internal buffers (m_data) of all corresponding channels.

        The new data is not emitted yet. Its state has to be set to DataStatus::NewData.

        \param channels are the desired channels. If empty, all registered channels should
            be fetched.
        \return ito::retError if at least one dataset of at least one desired channel
            is not available.

        */
        virtual ito::RetVal retrieveData(const QStringList& channels = QStringList()) = 0;

        //! This method fetches the latest acquired image of the current default channel.
        /*
        This method gets the name of the current default channel and calls the overloaded
        method ``retrieveData(channels)`` to fetch the latest acquired image of the default
        channel.

        If the ``externalDataObject`` is given, it must either be empty or have the same
        size and dtype than the latest image of the default channel. If given, the latest
        image is copied into the given externalDataObject.

        Usually, this method has not to be overloaded, since it is implemented in a useable
        way. Nevertheless, it can be overloaded.
        */
        virtual ito::RetVal retrieveData(ito::DataObject* externalDataObject = nullptr);

        //! Specific function to set the parameters in the respective plugin class
        /*!
        This function is a specific implementation of setParam. Overload this function to
        process parameters individually in the plugin class. This function is called
        by setParam after the parameter has been parsed and checked .

        \param [in] val parameter to be processed
        \param [in] it ParamMapIterator iterator to the parameter in m_params
        \param [in] suffix possible suffix of the parameter
        \param [in] key of the parameter
        \param [in] index of the parameter
        \param [in] hasIndex is set to true if parameter has an index
        \param [in] set ok to true if parameter was processed
        \param [in] add key of changed channel specific parameters to pendingUpdate.
        \return retOk if everything was ok, else retError
        */
        virtual ito::RetVal setParameter(
            QSharedPointer<ito::ParamBase>& val,
            const ParamMapIterator& it,
            const QString& suffix,
            const QString& key,
            int index,
            bool hasIndex,
            bool& ok,
            QStringList& pendingUpdate) = 0;

        //! Specific function to get a parameter in the respective plugin class.
        /*
        This method has to be overwritten in every plugin class and is called within
        the ``getParam`` method, which is final in this base class. Overload this method
        instead of the ``getParam`` method to deliver the value of the requested parameter.
        It is either possible, that this method delivers the content of the requested parameter,
        or that the ``ok`` argument is set to true, the return value is retOk. In this case,
        the calling ``getParam`` method will deliver the corresponding value in m_params as
        it is.
        */
        virtual ito::RetVal getParameter(
            QSharedPointer<ito::Param> val,
            const ParamMapIterator& it,
            const QString& key,
            const QString& suffix,
            int index,
            bool hasIndex,
            bool& ok) = 0;

        void updateSizeXY(); /*!< updates sizex und sizey*/

        ChannelContainerMapConstIterator getCurrentDefaultChannel() const;
        ChannelContainerMapIterator getCurrentDefaultChannel();

    public:
        AddInMultiChannelGrabber(const QByteArray& grabberName);
        ~AddInMultiChannelGrabber();

    private:
        QScopedPointer<AddInMultiChannelGrabberPrivate> d_ptr;
        Q_DECLARE_PRIVATE(AddInMultiChannelGrabber);

    public slots:
        ito::RetVal setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore* waitCond = nullptr) final;

        //! returns the current value of a selected parameter
        /*
        Approach: This method calls the abstract method ``getParameter`` that has to be overloaded
        by the plugin in order to obtain the parameter. If this method sets its argument ``ok`` to false,
        the current value of the parameter is returned per default by this base implementation.

        This method can either be invoked as ansychronous call, while the argument ``waitCond`` can
        be used to handle a notification about end of the call or a timeout, or as a direct call. Then,
        ``waitCond`` can also be set to a nullptr.

        getParam will always return the values of the parameters on the top level. If a parameter
        is individual per channel, the value of the current channel is returned. This current channel
        can be set via the parameter ``channelSelector``.

        \params val is the parameter, whose value is requested. It is a shared pointer of an ito::Param
            object, whose name is the only thing that has to be given when calling this method. If this
            method succeeds, this value is filled with the full parameter information including its current
            value.
        \params waitCond is the wait condition, that is used to observe the call state of an asychronous call.
        \returns a RetVal object, that indicates the success of this call. The same return value is also
            contained in the argument ``waitCond`` if given.
        */
        ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore* waitCond) final;

        virtual ito::RetVal getVal(void* vpdObj, ItomSharedSemaphore* waitCond);
        virtual ito::RetVal copyVal(void* vpdObj, ItomSharedSemaphore* waitCond);

        virtual ito::RetVal getVal(QSharedPointer<QMap<QString, ito::DataObject> > channelDatasets, ItomSharedSemaphore* waitCond);
        virtual ito::RetVal copyVal(QSharedPointer<QMap<QString, ito::DataObject> > channelDatasets, ItomSharedSemaphore* waitCond);

        ito::RetVal changeChannelForListener(QObject* listener, const QString& newChannel);

    signals:
        /*!< Signals that a new image or set of images is available. Connect to this signal to obtain a shallow copy of the new images */
        void newData(QSharedPointer<const QMap<QString, ito::DataObject> > channelDatasets);
    };

}

#endif
