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

#include "AddInMultiChannelGrabber.h"

#include <qmetatype.h>
#include <qcoreapplication.h>
#include <qmetaobject.h>
#include <qmap.h>
#include <qlist.h>
#include "common/helperCommon.h"


namespace ito
{
    //---------------------------------------------------------------------------------
    class AddInMultiChannelGrabberPrivate
    {
    public:
        AddInMultiChannelGrabberPrivate() : m_channelParamsProxyInitialized(false)
        {

        }

        //!< true if an appropriate parameter is created in the m_params map of the
        //! plugin for every parameter of the channels. The m_params value is a
        //! proxy for the currently selected channel.
        bool m_channelParamsProxyInitialized;


        QMap<QString, QStringList> m_paramChannelAvailabilityMap;
    };

    //---------------------------------------------------------------------------------
    AddInMultiChannelGrabber::ChannelContainer::ChannelContainer()
    {
        ito::Param paramVal;
        int roi[] = { 0, 0, 1, 1 };
        paramVal = ito::Param("roi", ito::ParamBase::IntArray, 4, roi, "roi");
        m_channelParam.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 1, 1, "sizex");
        m_channelParam.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 1, 1, "sizey");
        m_channelParam.insert(paramVal.getName(), paramVal);

        paramVal = ito::Param("pixelFormat", ito::ParamBase::String, "mono8", "pixelFormat");
        m_channelParam.insert(paramVal.getName(), paramVal);

        addDefaultMetaParams();
    };

    //---------------------------------------------------------------------------------
    AddInMultiChannelGrabber::ChannelContainer::ChannelContainer(
        const ito::Param& roi,
        const ito::Param& pixelFormat,
        const ito::Param& sizex,
        const ito::Param& sizey)
    {
        Q_ASSERT(QByteArray(roi.getName()) == "roi");
        Q_ASSERT(QByteArray(pixelFormat.getName()) == "pixelFormat");
        Q_ASSERT(QByteArray(sizex.getName()) == "sizex");
        Q_ASSERT(QByteArray(sizey.getName()) == "sizey");

        addChannelParam(roi);
        addChannelParam(pixelFormat);
        addChannelParam(sizex);
        addChannelParam(sizey);
        addDefaultMetaParams();
    }

    //---------------------------------------------------------------------------------
    AddInMultiChannelGrabber::ChannelContainer::ChannelContainer(
        const ito::Param& roi,
        const ito::Param& pixelFormat,
        const ito::Param& sizex,
        const ito::Param& sizey,
        const ito::Param& axisOffsets,
        const ito::Param& axisScales,
        const ito::Param& axisDescriptions,
        const ito::Param& axisUnits,
        const ito::Param& valueDescription,
        const ito::Param& valueUnit)
    {
        Q_ASSERT(QByteArray(roi.getName()) == "roi");
        Q_ASSERT(QByteArray(pixelFormat.getName()) == "pixelFormat");
        Q_ASSERT(QByteArray(sizex.getName()) == "sizex");
        Q_ASSERT(QByteArray(sizey.getName()) == "sizey");
        Q_ASSERT(QByteArray(axisOffsets.getName()) == "axisOffsets");
        Q_ASSERT(QByteArray(axisScales.getName()) == "axisScales");
        Q_ASSERT(QByteArray(axisDescriptions.getName()) == "axisDescriptions");
        Q_ASSERT(QByteArray(axisUnits.getName()) == "axisUnits");
        Q_ASSERT(QByteArray(valueDescription.getName()) == "valueDescription");
        Q_ASSERT(QByteArray(valueUnit.getName()) == "valueUnit");

        addChannelParam(roi);
        addChannelParam(pixelFormat);
        addChannelParam(sizex);
        addChannelParam(sizey);
        addChannelParam(axisOffsets);
        addChannelParam(axisScales);
        addChannelParam(axisDescriptions);
        addChannelParam(axisUnits);
        addChannelParam(axisDescriptions);
        addChannelParam(axisUnits);
        addChannelParam(valueDescription);
        addChannelParam(valueUnit);
    }

    //---------------------------------------------------------------------------------
    void AddInMultiChannelGrabber::ChannelContainer::addDefaultMetaParams()
    {
        ito::Param paramVal;

        double axisOffsets[] = { 0.0, 0.0 };
        paramVal = ito::Param("axisOffsets", ito::ParamBase::DoubleArray, 2, axisOffsets, "the offset values of the y- and x-axis of this channel image.");
        paramVal.setMeta(new ito::DoubleArrayMeta(-DBL_MAX, DBL_MAX, 0.0, 2, 2, 1, "MetaInformation"), true);
        addChannelParam(paramVal);

        double axisScales[] = { 1.0, 1.0 };
        paramVal = ito::Param("axisScales", ito::ParamBase::DoubleArray, 2, axisOffsets, "the scale values of the y- and x-axis of this channel image.");
        paramVal.setMeta(new ito::DoubleArrayMeta(-DBL_MAX, DBL_MAX, 0.0, 2, 2, 1, "MetaInformation"), true);
        addChannelParam(paramVal);

        ito::ByteArray axisUnits[] = { "", "" };
        paramVal = ito::Param(
            "axisUnits",
            ito::ParamBase::StringList, 2,
            axisUnits, "The unit strings for the y- and x-axis of the grabber image");
        paramVal.setMeta(new ito::StringListMeta(ito::StringMeta::Wildcard, "*", 2, 2, 1, "MetaInformation"), true);
        addChannelParam(paramVal);

        ito::ByteArray axisDescriptions[] = { "", "" };
        paramVal = ito::Param(
            "axisDescriptions",
            ito::ParamBase::StringList, 2,
            axisDescriptions,
            "axis description");
        paramVal.setMeta(new ito::StringListMeta(ito::StringMeta::Wildcard, "*", 2, 2, 1, "MetaInformation"), true);
        addChannelParam(paramVal);

        paramVal = ito::Param(
            "valueDescription",
            ito::ParamBase::String,
            "",
            "value description");
        addChannelParam(paramVal);

        paramVal = ito::Param(
            "valueUnit", ito::ParamBase::String, "", "The unit string of the values of this channel image");
        addChannelParam(paramVal);
    }

    //---------------------------------------------------------------------------------
    void AddInMultiChannelGrabber::ChannelContainer::addChannelParam(const ito::Param& param)
    {
        m_channelParam.insert(param.getName(), param);
    }

    //---------------------------------------------------------------------------------
    //! constructor
    AddInMultiChannelGrabber::AddInMultiChannelGrabber(const QByteArray& grabberName) :
        AbstractAddInGrabber(),
        d_ptr(new AddInMultiChannelGrabberPrivate())
    {
        ito::Param paramVal("name", ito::ParamBase::String | ito::ParamBase::Readonly, grabberName.data(), "GrabberName");
        paramVal.setMeta(new ito::StringMeta(ito::StringMeta::String, "General"), true);
        insertParam(paramVal);

        paramVal = ito::Param("defaultChannel", ito::ParamBase::String, "", tr("indicates the default channel name, that is for instance used in plots if not otherwise stated.").toLatin1().data());
        insertParam(paramVal);

        paramVal = ito::Param("channelSelector", ito::ParamBase::String, "", tr("The channel dependent parameters (like sizex, sizey, roi, pixelFormat...) are related to this channel.").toLatin1().data());
        insertParam(paramVal);

        paramVal = ito::Param("availableChannels", ito::ParamBase::StringList | ito::ParamBase::Readonly, {}, tr("names of the channels provided by the plugin").toLatin1().data());
        insertParam(paramVal);
    }

    //---------------------------------------------------------------------------------
    //! destructor
    AddInMultiChannelGrabber::~AddInMultiChannelGrabber()
    {
    }

    //---------------------------------------------------------------------------------
    void AddInMultiChannelGrabber::initChannelsAndGlobalParameters(
        const QMap<QString, ChannelContainer>& channelContainerMap,
        const QString& defaultChannelName,
        const QList<ito::Param>& globalParameters /*= QList<ito::Param>()*/)
    {
        Q_D(AddInMultiChannelGrabber);

        ito::RetVal retValue(ito::retOk);

        Q_ASSERT_X(
            channelContainerMap.contains(defaultChannelName),
            "initChannelsAndGlobalParameters",
            "a channel with the defaultChannelName must exist");

        Q_ASSERT_X(!d->m_channelParamsProxyInitialized,
            "initChannelsAndGlobalParameters",
            "channels and global parameters must not be initialized yet");

        // initialize default global parameters
        m_params["channelName"].setVal<const char*>(defaultChannelName.toLatin1().constData());
        m_params["channelSelector"].setVal<const char*>(defaultChannelName.toLatin1().constData());

        QList<ito::ByteArray> channelNames;

        foreach(const QString & channelName, channelContainerMap.keys())
        {
            channelNames << ito::ByteArray(channelName.toLatin1().constData());
        }

        m_params["availableChannels"].setVal<ito::ByteArray*>(channelNames.data(), channelNames.size());

        /*
        * todo:
        *
        bool channelSpecificParam = false;

        assert(channelContainerMap.size() != 0);
        QMapIterator<QString, ChannelContainer> channel(channelContainerMap);
        QVector<ByteArray> channelList;
        QMap<QString, ito::Param>initialParams = m_params;

        while (channel.hasNext())
        {
            channel.next();
            QMapIterator<QString, ito::Param> channelParamIterator(channel.value().m_channelParam);

            while (channelParamIterator.hasNext())//iterate through channel params
            {
                channelParamIterator.next();

                if (!m_params.contains(channelParamIterator.key()))
                {
                    m_params.insert(channelParamIterator.key(), channelParamIterator.value());
                }

                if (!d->m_paramChannelAvailabilityMap.contains(channelParamIterator.key()))
                {
                    d->m_paramChannelAvailabilityMap.insert(channelParamIterator.key(), QStringList(channel.key()));
                }
                else
                {
                    d->m_paramChannelAvailabilityMap[channelParamIterator.key()].append(channel.key());
                }

            }

            channelList.append(channel.key().toLatin1().data());
        }

        m_params["availableChannels"].setVal<ByteArray*>(channelList.data(), channelList.length());
        m_params["defaultChannel"].setVal<const char*>(channelContainerMap.firstKey().toLatin1().data());
        m_params["channelSelector"].setVal<const char*>(channelContainerMap.firstKey().toLatin1().data());
        m_channels = channelContainerMap;

        QMapIterator<QString, ito::Param> nonChannelSpecificParamsIt(nonChannelSpecificParams);

        while (nonChannelSpecificParamsIt.hasNext())
        {
            nonChannelSpecificParamsIt.next();
            assert(!m_params.contains(nonChannelSpecificParamsIt.key()));
            assert(!d->m_paramChannelAvailabilityMap.contains(nonChannelSpecificParamsIt.key()));
            d->m_paramChannelAvailabilityMap.insert(nonChannelSpecificParamsIt.key(), QStringList());
            m_params.insert(nonChannelSpecificParamsIt.key(), nonChannelSpecificParamsIt.value());

        }

        QMapIterator<QString, ito::Param> initialParamsIt(initialParams);

        while (initialParamsIt.hasNext())
        {
            initialParamsIt.next();
            assert(!d->m_paramChannelAvailabilityMap.contains(initialParamsIt.key()));
            d->m_paramChannelAvailabilityMap.insert(nonChannelSpecificParamsIt.key(), QStringList());
        }
        */

        d->m_channelParamsProxyInitialized = true;
        switchChannelSelector();
    }

    ////-------------------------------------------------------------------------------
    ////! sends m_image to all registered listeners.
    ///*!
    //This method is continuously called from timerEvent. Also call this method from your getVal-Method (usually with 0-timeout). The function adds axis scale and axis unit to the dataObject.

    //\param [in] waitMS indicates the time (in ms) that should be waiting until every registered live image source node received m_image. 0: no wait, -1: infinit waiting time, else time in milliseconds
    //\return retOk if everything was ok, retWarning if live image could not be invoked
    //*/
    ito::RetVal AddInMultiChannelGrabber::sendDataToListeners(int waitMS)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retValue = ito::retOk;
        int size = m_autoGrabbingListeners.size();
        if (waitMS == 0)
        {
            QMultiMap<QString, QObject*>::iterator it = m_autoGrabbingListeners.begin();
            while (it != m_autoGrabbingListeners.end())
            {
                const ChannelContainer& container = m_channels[it.key().toLatin1().data()];
                QSharedPointer<ito::DataObject> pDObj(new ito::DataObject(container.m_data));

                if (!QMetaObject::invokeMethod(it.value(), "setSource", Q_ARG(QSharedPointer<ito::DataObject>, pDObj), Q_ARG(ItomSharedSemaphore*, NULL)))
                {
                    retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                }
                it++;
            }
        }
        else if (m_autoGrabbingListeners.size() > 0)
        {
            ItomSharedSemaphore** waitConds = new ItomSharedSemaphore * [size];
            int i = 0;
            QMultiMap<QString, QObject*>::iterator it = m_autoGrabbingListeners.begin();
            while (it != m_autoGrabbingListeners.end())
            {
                waitConds[i] = new ItomSharedSemaphore();
                // \todo On Linux a crash occurs here when closing the liveImage ... maybe the same reason why we get an error message on windows?
                if (it.value())
                {
                    if (!QMetaObject::invokeMethod(it.value(), "setSource", Q_ARG(QSharedPointer<ito::DataObject>, QSharedPointer<ito::DataObject>(new ito::DataObject(m_channels[it.key().toLatin1().data()].m_data))), Q_ARG(ItomSharedSemaphore*, waitConds[i])))
                    {
                        retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
                    }
                }
                it++;
                i++;
            }

            for (i = 0; i < size; i++)
            {
                if (!waitConds[i]->wait(waitMS))
                {
                    qDebug() << "timeout in number: " << i << "number of items: " << size;
                }
                waitConds[i]->deleteSemaphore();
                waitConds[i] = NULL;
            }

            delete[] waitConds;
            waitConds = NULL;
        }


        return retValue;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal ito::AddInMultiChannelGrabber::checkData(QMap<QString, ito::DataObject*>& externalDataObject)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retVal(ito::retOk);
        unsigned int futureType = 0;
        bool ok;
        ito::float64 axisOffset[] = { 0.0, 0.0 };
        ito::float64 axisScale[] = { 1.0, 1.0 };
        QString axisUnit[] = { "<auto>", "<auto>" };
        QString axisDescription[] = { "<auto>", "<auto>" };
        QString valueDescription = "<auto>";
        QString valueUnit = "<auto>";

        QMap<QString, ito::DataObject*>::const_iterator it = externalDataObject.constBegin();
        while (it != externalDataObject.constEnd())
        {
            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("axisOffset"))
            {
                axisOffset[0] = m_channels[it.key()].m_channelParam["axisOffset"].getVal<ito::float64*>()[0];
                axisOffset[1] =
                    m_channels[it.key()].m_channelParam["axisOffset"].getVal<ito::float64*>()[1];
            }

            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("axisScale"))
            {
                axisScale[0] =
                    m_channels[it.key()].m_channelParam["axisScale"].getVal<ito::float64*>()[0];
                axisScale[1] =
                    m_channels[it.key()].m_channelParam["axisScale"].getVal<ito::float64*>()[1];
            }

            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("axisDescription"))
            {
                axisDescription[0] = QString::fromUtf8(m_channels[it.key()]
                    .m_channelParam["axisDescription"]
                    .getVal<ito::ByteArray*>()[0]
                    .data());
                axisDescription[1] = QString::fromUtf8(m_channels[it.key()]
                    .m_channelParam["axisDescription"]
                    .getVal<ito::ByteArray*>()[1]
                    .data());
            }

            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("axisUnit"))
            {
                axisUnit[0] = QString::fromUtf8(m_channels[it.key()]
                    .m_channelParam["axisUnit"]
                    .getVal<ito::ByteArray*>()[0]
                    .data());
                axisUnit[1] = QString::fromUtf8(m_channels[it.key()]
                    .m_channelParam["axisUnit"]
                    .getVal<ito::ByteArray*>()[1]
                    .data());
            }

            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("valueDescription"))
            {
                valueDescription = QString::fromLatin1(
                    m_channels[it.key()].m_channelParam["valueDescription"].getVal<char*>());
            }

            // only if exists in plugin
            if (m_channels[it.key()].m_channelParam.contains("valueUnit"))
            {
                valueUnit = QString::fromLatin1(
                    m_channels[it.key()].m_channelParam["valueUnit"].getVal<char*>());
            }

            futureType = itoDataTypeFromPixelFormat(m_channels[it.key()].m_channelParam["pixelFormat"].getVal<const char*>(), &ok);

            if (ok)
            {
                int* roi = m_channels[it.key()].m_channelParam["roi"].getVal<int*>();
                int width = roi[2];
                int height = roi[3];
                if (it.value()->getDims() == 0)
                {
                    *(it.value()) = ito::DataObject(height, width, futureType);

                    m_channels[it.key()].m_data.setAxisScale(0, axisScale[0]);
                    m_channels[it.key()].m_data.setAxisScale(1, axisScale[1]);
                    m_channels[it.key()].m_data.setAxisOffset(0, axisOffset[0]);
                    m_channels[it.key()].m_data.setAxisOffset(1, axisOffset[1]);
                    m_channels[it.key()].m_data.setAxisDescription(
                        0, axisDescription[0].toLatin1().data());
                    m_channels[it.key()].m_data.setAxisDescription(
                        1, axisDescription[1].toLatin1().data());
                    m_channels[it.key()].m_data.setAxisUnit(0, axisUnit[0].toLatin1().data());
                    m_channels[it.key()].m_data.setAxisUnit(1, axisUnit[1].toLatin1().data());
                    m_channels[it.key()].m_data.setValueDescription(
                        valueDescription.toLatin1().data());
                    m_channels[it.key()].m_data.setValueUnit(valueUnit.toLatin1().data());
                }
                else if (it.value()->calcNumMats() != 1)
                {
                    return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject for channel %1 is invalid. Object has more or less than 1 plane. It must be of right size and type or an uninitilized image.").arg(it.key()).toLatin1().data());
                }
                else if (it.value()->getSize(it.value()->getDims() - 2) != height || it.value()->getSize(it.value()->getDims() - 1) != width || it.value()->getType() != futureType)
                {
                    return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject for channel %1 is invalid. Object must be of right size and type or an uninitilized image.").arg(it.key()).toLatin1().data());
                }
            }
            ++it;
        }
        return retVal;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal ito::AddInMultiChannelGrabber::checkData(ito::DataObject* externalDataObject)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retVal(ito::retOk);
        bool ok;
        ito::float64 axisOffset[] = { 0.0, 0.0 };
        ito::float64 axisScale[] = { 1.0, 1.0 };
        QString axisUnit[] = { "<auto>", "<auto>" };
        QString axisDescription[] = { "<auto>", "<auto>" };
        QString valueDescription = "<auto>";
        QString valueUnit = "<auto>";
        unsigned int futureType;

        if (!externalDataObject)
        {
            QMutableMapIterator<QString, ChannelContainer> it(m_channels);
            while (it.hasNext()) {
                it.next();

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("axisOffset"))
                {
                    axisOffset[0] = m_channels[it.key()]
                        .m_channelParam["axisOffset"]
                        .getVal<ito::float64*>()[0];
                    axisOffset[1] = m_channels[it.key()]
                        .m_channelParam["axisOffset"]
                        .getVal<ito::float64*>()[1];
                }

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("axisScale"))
                {
                    axisScale[0] =
                        m_channels[it.key()].m_channelParam["axisScale"].getVal<ito::float64*>()[0];
                    axisScale[1] =
                        m_channels[it.key()].m_channelParam["axisScale"].getVal<ito::float64*>()[1];
                }

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("axisDescription"))
                {
                    axisDescription[0] = QString::fromUtf8(m_channels[it.key()]
                        .m_channelParam["axisDescription"]
                        .getVal<ito::ByteArray*>()[0]
                        .data());
                    axisDescription[1] = QString::fromUtf8(m_channels[it.key()]
                        .m_channelParam["axisDescription"]
                        .getVal<ito::ByteArray*>()[1]
                        .data());
                }

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("axisUnit"))
                {
                    axisUnit[0] = QString::fromUtf8(m_channels[it.key()]
                        .m_channelParam["axisUnit"]
                        .getVal<ito::ByteArray*>()[0]
                        .data());
                    axisUnit[1] = QString::fromUtf8(m_channels[it.key()]
                        .m_channelParam["axisUnit"]
                        .getVal<ito::ByteArray*>()[1]
                        .data());
                }

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("valueDescription"))
                {
                    valueDescription = QString::fromLatin1(
                        m_channels[it.key()].m_channelParam["valueDescription"].getVal<char*>());
                }

                // only if exists in plugin
                if (m_channels[it.key()].m_channelParam.contains("valueUnit"))
                {
                    valueUnit = QString::fromLatin1(
                        m_channels[it.key()].m_channelParam["valueUnit"].getVal<char*>());
                }

                futureType = itoDataTypeFromPixelFormat(it.value().m_channelParam["pixelFormat"].getVal<const char*>(), &ok);
                if (ok)
                {
                    int* roi = it.value().m_channelParam["roi"].getVal<int*>();
                    int height = roi[3];
                    int width = roi[2];
                    if (it.value().m_data.getDims() < 2 || it.value().m_data.getSize(0) != height || it.value().m_data.getSize(1) != width || it.value().m_data.getType() != futureType)
                    {
                        it.value().m_data = ito::DataObject(height, width, futureType);
                        it.value().m_data.setAxisScale(0, axisScale[0]);
                        it.value().m_data.setAxisScale(1, axisScale[1]);
                        it.value().m_data.setAxisOffset(0, axisOffset[0]);
                        it.value().m_data.setAxisOffset(1, axisOffset[1]);
                        it.value().m_data.setAxisDescription(0, axisDescription[0].toLatin1().data());
                        it.value().m_data.setAxisDescription(1, axisDescription[1].toLatin1().data());
                        it.value().m_data.setAxisUnit(0, axisUnit[0].toLatin1().data());
                        it.value().m_data.setAxisUnit(1, axisUnit[1].toLatin1().data());
                        it.value().m_data.setValueDescription(valueDescription.toLatin1().data());
                        it.value().m_data.setValueUnit(valueUnit.toLatin1().data());
                    }
                }
                else
                {
                    retVal += ito::RetVal(ito::retError, 0, tr("invalid pixel format").toLatin1().data());
                }

            }
        }
        else
        {
            const char* channel = m_params["defaultChannel"].getVal<const char*>();

            if (m_channels.contains(channel))
            {
                futureType = itoDataTypeFromPixelFormat(m_channels[channel].m_channelParam["pixelFormat"].getVal<const char*>(), &ok);
                if (ok)
                {
                    int* roi = m_channels[channel].m_channelParam["roi"].getVal<int*>();
                    int width = roi[2];
                    int height = roi[3];
                    if (externalDataObject->getDims() == 0)
                    {
                        *externalDataObject = ito::DataObject(height, width, futureType);
                    }
                    else if (externalDataObject->calcNumMats() != 1)
                    {
                        return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object has more or less than 1 plane. It must be of right size and type or an uninitilized image.").toLatin1().data());
                    }
                    else if (externalDataObject->getSize(externalDataObject->getDims() - 2) != height || externalDataObject->getSize(externalDataObject->getDims() - 1) != width || externalDataObject->getType() != futureType)
                    {
                        return ito::RetVal(ito::retError, 0, tr("Error during check data, external dataObject invalid. Object must be of right size and type or an uninitilized image.").toLatin1().data());
                    }
                }
                else
                {
                    retVal += ito::RetVal(ito::retError, 0, tr("invalid pixel format").toLatin1().data());
                }
            }
        }
        return retVal;
    }

    //---------------------------------------------------------------------------------
    void AddInMultiChannelGrabber::addChannel(QString name)
    {
        ChannelContainer container(
            m_params["roi"],
            m_params["pixelFormat"],
            m_params["sizex"],
            m_params["sizey"],
            m_params["axisOffset"],
            m_params["axisScale"],
            m_params["axisDescription"],
            m_params["axisUnit"],
            m_params["valueDescription"],
            m_params["valueUnit"]);
        m_channels[name] = container;
        const ByteArray* channelList = m_params["availableChannels"].getVal<const ByteArray*>();
        int len = 0;
        m_params["availableChannels"].getVal<const ByteArray*>(len);

        QVector<ByteArray> qVectorList(len, *channelList);
        qVectorList.append(ByteArray(name.toLatin1().data()));
        m_params["availableChannels"].setVal<ByteArray*>(qVectorList.data(), qVectorList.length());
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::adaptDefaultChannelParams()
    {
        ito::RetVal retVal(ito::retOk);
        char* channel = m_params["defaultChannel"].getVal<char*>();
        return retVal;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore* waitCond)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue;
        QString key;
        bool hasIndex = false;
        int index;
        QString suffix;
        ParamMapIterator it;
        bool ok = false;

        //parse the given parameter-name (if you support indexed or suffix-based parameters)
        retValue += apiParseParamName(val->getName(), key, hasIndex, index, suffix);

        if (!retValue.containsError())
        {
            retValue += apiGetParamFromMapByKey(m_params, key, it, false);
        }
        if (!retValue.containsError())
        {
            retValue += getParameter(val, it, suffix, key, index, hasIndex, ok);
        }
        if (!retValue.containsError() && !ok)//the parameter was not processed by the plugin, so it is done here
        {
            *val = it.value();
        }

        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }

        return retValue;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::changeChannelForListeners(const QString& newChannel, QObject* obj)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retValue(ito::retOk);
        bool found = false;

        if (obj)
        {
            QMultiMap<QString, QObject*>::iterator i = m_autoGrabbingListeners.begin();

            while (i != m_autoGrabbingListeners.end())
            {
                if (i.value() == obj)
                {
                    found = true;

                    if (i.key() != newChannel)
                    {
                        // the channel has changed
                        m_autoGrabbingListeners.remove(i.key(), obj);
                        m_autoGrabbingListeners.insert(newChannel, obj);
                    }

                    break;
                }
            }

            if (!found)
            {
                retValue += ito::RetVal(ito::retError, 0, tr("Could not find plot in m_autoGrabbingListeners").toLatin1().data());
            }
        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, "QObject not callable.");
        }
        return retValue;
    }

    ////-------------------------------------------------------------------------------
    ////! Sets a new value to a parameter.
    ///*!
    // This function parses the given parameter and calls setParameter. If the bool parameter ok in the setParameter (to be implemented in the individual plugins)
    // function returns false, it gets assumed that the plugin didn't process the parameter. In this case the value of the parameter gets copied here.
    // If the parameter name is "roi" sizex and sizey gets updated by setParam. If the key of the parameter is "defaultChannel" the function "switchDefaultChannel" gets called.
    // Both happens also if the "ok" value of setParameter is true.
    // "applyParamsToChannelParams" is called to synchronize the parameters of the channel container follwed by a call of checkData.

    //\param [in] val is a QSharedPOinter of type ParamBase containing the paremeter to be set.
    //\param [in] waitCond
    //\return retOk if everything was ok, else retError
    //*/
    ito::RetVal AddInMultiChannelGrabber::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore* waitCond/* = NULL*/)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retValue;
        bool hasIndex, ok;
        int index;
        QString suffix, key;
        QStringList paramUpdateList;
        ParamMapIterator it;
        int cntStartedDevices = grabberStartedCount();
        retValue += ito::parseParamName(val->getName(), key, hasIndex, index, suffix);
        retValue += apiGetParamFromMapByKey(m_params, key, it, true);
        retValue += apiValidateParam(*it, *val, false, true);
        if (!retValue.containsError())
        {
            retValue += setParameter(val, it, suffix, key, index, hasIndex, ok, paramUpdateList);
            if (ok && !paramUpdateList.contains(val->getName()))
            {
                paramUpdateList << val->getName();
            }
            if (!retValue.containsError() && !ok)
            {


                if (!retValue.containsError())
                {
                    if (key != "defaultChannel")
                    {
                        retValue += it->copyValueFrom(&(*val));
                        paramUpdateList << val->getName();
                    }
                    else
                    {
                        if (m_channels.find(val->getVal<char*>()) != m_channels.end())
                        {
                            retValue += it->copyValueFrom(&(*val));
                            paramUpdateList << val->getName();
                        }
                        else
                        {
                            retValue += retValue += ito::RetVal(ito::retError, 0, tr("could not switch to channel %1. The channel is unknown.").arg(val->getVal<const char*>()).toLatin1().data());
                        }
                    }
                }
            }
            if (!retValue.containsError())
            {
                if (key == "roi" || paramUpdateList.contains("roi"))
                {
                    updateSizeXY();
                    paramUpdateList << "sizex" << "sizey";
                }
                if (key == "defaultChannel")
                {
                    retValue += switchChannelSelector();
                }
                retValue += applyParamsToChannelParams(paramUpdateList);
                retValue += checkData();
            }
            if (!retValue.containsError())
            {
                emit parametersChanged(m_params);
            }
        }
        if (cntStartedDevices < grabberStartedCount())
        {
            if (cntStartedDevices != 0)
            {
                retValue += startDevice(NULL);
                setGrabberStarted(cntStartedDevices);
            }
        }
        if (waitCond)
        {
            waitCond->returnValue = retValue;
            waitCond->release();
        }
        return retValue;
    }

    ////-------------------------------------------------------------------------------
    ////! synchronizes m_params with the params of default channel container
    ///*!
    //This method synchronizes the parameters from the current selected channel container with m_params. Call this function after changing the defaultChannel parameter.Parameters which are not available for the current default channel are set to readonly

    //\return retOk if everything was ok, else retError
    //*/
    ito::RetVal AddInMultiChannelGrabber::switchChannelSelector()
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retValue(ito::retOk);
        unsigned int flag = 0;
        QString selectedChannel = QLatin1String(m_params["channelSelector"].getVal<const char*>());

        if (m_channels.contains(selectedChannel))
        {
            QMutableMapIterator<QString, ito::Param> itParam(m_params);

            while (itParam.hasNext())
            {
                itParam.next();

                if (d->m_paramChannelAvailabilityMap[itParam.key()].contains(selectedChannel))
                {
                    itParam.value() = m_channels[selectedChannel].m_channelParam[itParam.key()];
                }
                else if (!d->m_paramChannelAvailabilityMap[itParam.key()].isEmpty())
                {
                    flag = itParam.value().getFlags();
                    flag |= ito::ParamBase::Readonly;
                    itParam.value().setFlags(flag);
                }
            }
        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, tr("could not switch to channel %1. The channel is not registered.").arg(selectedChannel).toLatin1().data());
        }
        return retValue;
    }

    ////-------------------------------------------------------------------------------
    ////! copies value m_params to the channel params of the current default channel
    ///*!
    //This method copies params of m_params to the params of the channel container if the param is contained in the channel container . This function is usally called after setParam to apply the changed entries of m_params to the corresponding channel container.
    //If a parameter is not found in the channel container nothing happens.

    //\param [in] keyList indicates which params are copied. If the List is empty all Parameters of the current channel are updated.
    //\return retOk if everything was ok, else retError
    //*/
    ito::RetVal AddInMultiChannelGrabber::applyParamsToChannelParams(const QStringList& keyList)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retVal(ito::retOk);
        QString channelSelector = QLatin1String(m_params["channelSelector"].getVal<const char*>());

        if (!keyList.isEmpty())
        {
            if (m_channels.contains(channelSelector))
            {
                QString tmp;
                foreach(tmp, keyList)
                {
                    if (m_channels[channelSelector].m_channelParam.contains(tmp))
                    {
                        if (m_params.contains(tmp))
                        {
                            m_channels[channelSelector].m_channelParam[tmp] = m_params[tmp];
                        }
                        else
                        {
                            retVal = ito::RetVal(ito::retError, 0, tr("unknown parameter %1 in m_params").arg(tmp).toLatin1().data());
                        }
                    }
                }
            }

            else
            {
                retVal = ito::RetVal(ito::retError, 0, tr("unknown channel %1").arg(channelSelector).toLatin1().data());
            }
        }
        else
        {
            QMapIterator<QString, ito::Param> it(m_channels[m_params["channelSelector"].getVal<const char*>()].m_channelParam);

            while (it.hasNext())
            {
                it.next();
                const_cast<ito::Param&>(it.value()) = m_params[it.key()];
            }
        }

        return retVal;
    }

    ////-------------------------------------------------------------------------------
    ////! updates sizex and sizey
    ///*!
    //Call this function to update sizex and sizey. If the roi is changed via setParam this function will be called automatically.
    //Note: Do not forget to apply the changes to the channel parameters by calling applyParamsToChannelParams after calling this function.

    //\return retOk if everything was ok, else retError
    //*/
    void AddInMultiChannelGrabber::updateSizeXY()
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        const int* roi = m_params["roi"].getVal<const int*>();
        int height = roi[3];
        int width = roi[2];
        m_params["sizex"].setVal<int>(width);
        m_params["sizey"].setVal<int>(height);
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::setParamMeta(const QByteArray& paramName, ito::ParamMeta* meta, bool takeOwnerShip, const QList<QByteArray>& channelList/* = QList<ByteArray>()*/)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retValue(ito::retOk);
        if (d->m_paramChannelAvailabilityMap.contains(paramName) && m_params.contains(paramName))
        {
            if (channelList.isEmpty()) //we want to update the param for all channels even if it is a global one
            {
                m_params[paramName].setMeta(meta, takeOwnerShip);
                QStringListIterator it(d->m_paramChannelAvailabilityMap[paramName]);
                while (it.hasNext()) // update param for all channels
                {
                    m_channels[it.next()].m_channelParam[paramName].setMeta(meta, takeOwnerShip);
                }
            }
            else // we only want to update the param for a list of channels
            {
                for (int i = 0; i < channelList.length(); i++)
                {
                    if (m_channels.contains(channelList[i]))
                    {
                        if (d->m_paramChannelAvailabilityMap[paramName].contains(channelList[i]))
                        {
                            m_channels[channelList[i]].m_channelParam[paramName].setMeta(meta, takeOwnerShip);
                        }
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, QString("unknown channel %1").arg(QString(paramName)).toLatin1().data());
                    }

                }
                if (!retValue.containsError())
                {
                    //update m_params if the current default channel is listed in channelList or if the current default channel does not support the param (the param in m_params then is set to readonly)
                    if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()) || !d->m_paramChannelAvailabilityMap[paramName].contains(m_params["defaultChannel"].getVal<const char*>()))
                    {
                        m_params[paramName].setMeta(meta, takeOwnerShip);
                    }
                }
            }

        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, QString("could not find parameter %1. Maybe the parameter is not registered").arg(QString(paramName)).toLatin1().data());
        }

        return retValue;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::setParamFlags(const QByteArray& paramName, const unsigned int& flags, const QList<QByteArray>& channelList/* = QList<QByteArray>()*/)
    {
        Q_D(AddInMultiChannelGrabber);

        assert(d->m_channelParamsProxyInitialized);
        ito::RetVal retValue(ito::retOk);
        if (d->m_paramChannelAvailabilityMap.contains(paramName) && m_params.contains(paramName))
        {
            if (channelList.isEmpty()) //we want to update the param for all channels even if it is a global one
            {
                m_params[paramName].setFlags(flags);
                QStringListIterator it(d->m_paramChannelAvailabilityMap[paramName]);
                while (it.hasNext()) // update param for all channels
                {
                    m_channels[it.next()].m_channelParam[paramName].setFlags(flags);
                }
            }
            else // we only want to update the param for a list of channels
            {
                for (int i = 0; i < channelList.length(); i++)
                {
                    if (m_channels.contains(channelList[i]))
                    {
                        if (d->m_paramChannelAvailabilityMap[paramName].contains(channelList[i]))
                        {
                            m_channels[channelList[i]].m_channelParam[paramName].setFlags(flags);
                        }
                    }
                    else
                    {
                        retValue += ito::RetVal(ito::retError, 0, QString("unknown channel %1").arg(QString(paramName)).toLatin1().data());
                    }

                }
                if (!retValue.containsError())
                {
                    //update m_params if the current default channel is listed in channelList or if the current default channel does not support the param (the param in m_params then is set to readonly)
                    if (channelList.contains(m_params["defaultChannel"].getVal<const char*>()) || !d->m_paramChannelAvailabilityMap[paramName].contains(m_params["defaultChannel"].getVal<const char*>()))
                    {
                        m_params[paramName].setFlags(flags);
                    }
                }
            }

        }
        else
        {
            retValue += ito::RetVal(ito::retError, 0, QString("could not find parameter %1. Maybe the parameter is not registered").arg(QString(paramName)).toLatin1().data());
        }

        return retValue;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::getVal(QSharedPointer<QMap<QString, ito::DataObject*>> dataObjMap, ItomSharedSemaphore* waitCond)
    {
        ito::RetVal retval(ito::retOk);
        ItomSharedSemaphoreLocker locker(waitCond);
        QMap<QString, ito::DataObject*>::const_iterator it = (*dataObjMap).constBegin();
        bool validChannelNames = true;

        while (it != (*dataObjMap).constEnd())
        {
            if (!m_channels.contains(it.key()))
            {
                retval += ito::RetVal(ito::retError, 0, tr("The following channel is not a valid channel of the dataIO instance: %1").arg(it.key()).toLatin1().data());
            }
            ++it;
        }

        if (!retval.containsError())
        {
            retval = getValByMap(dataObjMap);
        }

        if (waitCond)
        {
            waitCond->returnValue = retval;
            waitCond->release();
        }

        return retval;
    }

    //---------------------------------------------------------------------------------
    ito::RetVal AddInMultiChannelGrabber::copyVal(QSharedPointer<QMap<QString, ito::DataObject*>> dataObjMap, ItomSharedSemaphore* waitCond)
    {

        ito::RetVal retval(ito::retOk);
        ItomSharedSemaphoreLocker locker(waitCond);
        QMap<QString, ito::DataObject*>::const_iterator it = (*dataObjMap).constBegin();
        bool validChannelNames = true;
        while (it != (*dataObjMap).constEnd())
        {
            if (!m_channels.contains(it.key()))
            {
                retval += ito::RetVal(ito::retError, 0, tr("The following channel is not a valid channel of the dataIO instance: %1").arg(it.key()).toLatin1().data());
            }
            ++it;
        }
        if (!retval.containsError())
        {
            retval = copyValByMap(dataObjMap);
        }
        if (waitCond)
        {
            waitCond->returnValue = retval;
            waitCond->release();
        }
        return retval;
    }

}//end namespace ito
