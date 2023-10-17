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

//-------------------------------------------------------------------------------------
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

    //!< this map contains all parameter names in m_param, that belong to a
    //! channel parameter. The value string list contains all channel names,
    //! where this parameter is explicitely contained. Global parameters,
    //! that are independent on channels, are not listed here.
    QMap<QString, QStringList> m_paramChannelAvailabilityMap;
};

//-------------------------------------------------------------------------------------
AddInMultiChannelGrabber::ChannelContainer::ChannelContainer() :
    m_dataStatus(DataStatus::Idle)
{
    ito::Param paramVal;
    int roi[] = { 0, 0, 1, 1 };
    paramVal = ito::Param("roi", ito::ParamBase::IntArray, 4, roi, "roi");
    m_channelParams.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("sizex", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 1, 1, "sizex");
    m_channelParams.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("sizey", ito::ParamBase::Int | ito::ParamBase::Readonly, 1, 1, 1, "sizey");
    m_channelParams.insert(paramVal.getName(), paramVal);

    paramVal = ito::Param("pixelFormat", ito::ParamBase::String, "mono8", "pixelFormat");
    m_channelParams.insert(paramVal.getName(), paramVal);

    addDefaultMetaParams();
};

//-------------------------------------------------------------------------------------
AddInMultiChannelGrabber::ChannelContainer::ChannelContainer(
    const ito::Param& roi,
    const ito::Param& pixelFormat,
    const ito::Param& sizex,
    const ito::Param& sizey) :
    m_dataStatus(DataStatus::Idle)
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

//-------------------------------------------------------------------------------------
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
    const ito::Param& valueUnit) :
    m_dataStatus(DataStatus::Idle)
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

//-------------------------------------------------------------------------------------
void AddInMultiChannelGrabber::ChannelContainer::addDefaultMetaParams()
{
    ito::Param paramVal;

    double axisOffsets[] = { 0.0, 0.0 };
    paramVal = ito::Param("axisOffsets", ito::ParamBase::DoubleArray, 2, axisOffsets, "the offset values of the y- and x-axis of this channel image.");
    paramVal.setMeta(new ito::DoubleArrayMeta(-DBL_MAX, DBL_MAX, 0.0, 2, 2, 1, "MetaInformation"), true);
    addChannelParam(paramVal);

    double axisScales[] = { 1.0, 1.0 };
    paramVal = ito::Param("axisScales", ito::ParamBase::DoubleArray, 2, axisScales, "the scale values of the y- and x-axis of this channel image.");
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

//-------------------------------------------------------------------------------------
void AddInMultiChannelGrabber::ChannelContainer::addChannelParam(const ito::Param& param)
{
    m_channelParams.insert(param.getName(), param);
}

//-------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------
//! destructor
AddInMultiChannelGrabber::~AddInMultiChannelGrabber()
{
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::initChannelsAndGlobalParameters(
    const ChannelContainerMap& channelContainerMap,
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

    Q_ASSERT_X(channelContainerMap.size() > 0,
        "initChannelsAndGlobalParameters",
        "at least one channel container must be given");

    // initialize default global parameters
    m_params["defaultChannel"].setVal<const char*>(defaultChannelName.toLatin1().constData());
    m_params["channelSelector"].setVal<const char*>(defaultChannelName.toLatin1().constData());

    QList<ito::ByteArray> channelNames;

    foreach(const QString & channelName, channelContainerMap.keys())
    {
        channelNames << ito::ByteArray(channelName.toLatin1().constData());
    }

    m_params["availableChannels"].setVal<ito::ByteArray*>(channelNames.data(), channelNames.size());

    m_channels = channelContainerMap;

    // iterate over all channels and register their parameters as global parameter,
    // however store a flag, that this is a channel parameter.

    QString paramName;
    ChannelContainerMapConstIterator channelIter = channelContainerMap.constBegin();

    while (channelIter != channelContainerMap.constEnd())
    {
        ParamMapConstIterator channelParamIter = channelIter->m_channelParams.constBegin();

        while (channelParamIter != channelIter->m_channelParams.constEnd())
        {
            //iterate through channel params

            paramName = channelParamIter.key();

            if (m_params.contains(paramName))
            {
                // if another channel already registered this key, it must have the same type
                Q_ASSERT_X(m_params[paramName].getType() == channelParamIter->getType(),
                    "initChannelsAndGlobalParameters",
                    "The type of channel parameters of the same name must be equal.");
            }
            else
            {
                // add a deep copy of the channel parameter to m_params
                m_params.insert(paramName, Param(channelParamIter.value()));
            }

            if (!d->m_paramChannelAvailabilityMap.contains(paramName))
            {
                d->m_paramChannelAvailabilityMap[paramName] = QStringList(channelIter.key());
            }
            else
            {
                d->m_paramChannelAvailabilityMap[paramName].append(channelIter.key());
            }

            channelParamIter++;
        }

        channelIter++;
    }

    // add all global parameters to m_params, unless they exist already: error

    foreach(const ito::Param& p, globalParameters)
    {
        paramName = QLatin1String(p.getName());
        Q_ASSERT_X(!m_params.contains(paramName),
            "initChannelsAndGlobalParameters",
            "the globalParameters must not contain a parameter whose name is already contained in at least one channel.");
        m_params[paramName] = p;
    }

    d->m_channelParamsProxyInitialized = true;

    // call switchChannelSelector to synchronize the values in the channel specific parameters with
    // to these of the current channel (channelSelector).
    retValue += switchChannelSelector();

    return retValue;
}

//-------------------------------------------------------------------------------
//! sends m_image to all registered listeners.
/*
This method is continuously called from timerEvent. Also call this method from
your getVal-Method (usually with 0-timeout). The function adds axis scale and
axis unit to the dataObject.

\param [in] waitMS indicates the time (in ms) that should be waiting until every
            registered live image source node received m_image. 0: no wait, -1:
            infinit waiting time, else time in milliseconds
\return retOk if everything was ok, retWarning if live image could not be invoked
*/
ito::RetVal AddInMultiChannelGrabber::sendDataToListeners(int waitMS)
{
    Q_D(AddInMultiChannelGrabber);

    assert(d->m_channelParamsProxyInitialized);
    ito::RetVal retValue = ito::retOk;
    int size = m_autoGrabbingListeners.size();

    if (waitMS == 0)
    {
        auto it = m_autoGrabbingListeners.constBegin();

        while (it != m_autoGrabbingListeners.constEnd())
        {
            const ChannelContainer& container = m_channels[it.key().toLatin1().data()];
            QSharedPointer<ito::DataObject> pDObj(new ito::DataObject(container.m_data));

            if (!QMetaObject::invokeMethod(it.value(), "setSource", Q_ARG(QSharedPointer<ito::DataObject>, pDObj), Q_ARG(ItomSharedSemaphore*, nullptr)))
            {
                retValue += ito::RetVal(ito::retWarning, 1001, tr("slot 'setSource' of live source node could not be invoked").toLatin1().data());
            }

            it++;
        }
    }
    else if (m_autoGrabbingListeners.size() > 0)
    {
        ItomSharedSemaphore** waitConds = new ItomSharedSemaphore*[size];
        int i = 0;
        auto it = m_autoGrabbingListeners.constBegin();

        while (it != m_autoGrabbingListeners.constEnd())
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
            waitConds[i] = nullptr;
        }

        DELETE_AND_SET_NULL_ARRAY(waitConds);
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::checkData(ito::DataObject* externalDataObject /*= nullptr*/)
{
    QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());
    return checkData(defaultChannel, externalDataObject);
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::checkDataFromAllChannels()
{
    auto it = m_channels.constBegin();
    ito::RetVal retval;

    while (it != m_channels.constEnd())
    {
        retval += checkData(it.key());
        ++it;
    }

    return retval;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::checkData(const QString& channelName, ito::DataObject* externalDataObject /*= nullptr*/)
{
    Q_D(AddInMultiChannelGrabber);

    assert(d->m_channelParamsProxyInitialized);
    ito::RetVal retVal(ito::retOk);

    auto channel = m_channels.find(channelName);

    if (channel == m_channels.end())
    {
        return ito::RetVal(ito::retError, 0, tr("channel name does not exist").toLatin1().data());
    }

    const int futureHeight = channel->m_channelParams["sizey"].getVal<int>();
    const int futureWidth = channel->m_channelParams["sizex"].getVal<int>();
    bool futureTypeOk;
    int futureType = itoDataTypeFromPixelFormat(channel->m_channelParams["pixelFormat"].getVal<const char*>(), &futureTypeOk);

    if (!futureTypeOk)
    {
        return ito::RetVal(ito::retError, 0, tr("unsupported or invalid pixelFormat").toLatin1().data());
    }

    ito::float64 axisOffset[] = {
        channel->m_channelParams["axisOffsets"].getVal<const ito::float64*>()[0],
        channel->m_channelParams["axisOffsets"].getVal<const ito::float64*>()[1]
    };

    ito::float64 axisScale[] = {
        channel->m_channelParams["axisScales"].getVal<const ito::float64*>()[0],
        channel->m_channelParams["axisScales"].getVal<const ito::float64*>()[1]
    };

    ito::ByteArray axisUnit[] = {
        channel->m_channelParams["axisUnits"].getVal<const ito::ByteArray*>()[0],
        channel->m_channelParams["axisUnits"].getVal<const ito::ByteArray*>()[1]
    };

    ito::ByteArray axisDescription[] = {
        channel->m_channelParams["axisDescriptions"].getVal<const ito::ByteArray*>()[0],
        channel->m_channelParams["axisDescriptions"].getVal<const ito::ByteArray*>()[1]
    };

    ito::ByteArray valueDescription = channel->m_channelParams["valueDescription"].getVal<const char*>();
    ito::ByteArray valueUnit = channel->m_channelParams["valueUnit"].getVal<const char*>();

    if (!externalDataObject)
    {
        if (channel->m_data.getDims() < 2 || channel->m_data.getSize(0) != futureHeight ||
            channel->m_data.getSize(1) != futureWidth || channel->m_data.getType() != futureType)
        {
            channel->m_data = ito::DataObject(futureHeight, futureWidth, futureType);
        }

        channel->m_data.setAxisScale(0, axisScale[0]);
        channel->m_data.setAxisScale(1, axisScale[1]);
        channel->m_data.setAxisOffset(0, axisOffset[0]);
        channel->m_data.setAxisOffset(1, axisOffset[1]);
        channel->m_data.setAxisDescription(0, axisDescription[0].data());
        channel->m_data.setAxisDescription(1, axisDescription[1].data());
        channel->m_data.setAxisUnit(0, axisUnit[0].data());
        channel->m_data.setAxisUnit(1, axisUnit[1].data());
        channel->m_data.setValueDescription(valueDescription.data());
        channel->m_data.setValueUnit(valueUnit.data());
    }
    else
    {
        int dims = externalDataObject->getDims();

        if (dims == 0)
        {
            *externalDataObject = ito::DataObject(futureHeight, futureWidth, futureType);
            externalDataObject->setAxisScale(0, axisScale[0]);
            externalDataObject->setAxisScale(1, axisScale[1]);
            externalDataObject->setAxisOffset(0, axisOffset[0]);
            externalDataObject->setAxisOffset(1, axisOffset[1]);
            externalDataObject->setAxisDescription(0, axisDescription[0].data());
            externalDataObject->setAxisDescription(1, axisDescription[1].data());
            externalDataObject->setAxisUnit(0, axisUnit[0].data());
            externalDataObject->setAxisUnit(1, axisUnit[1].data());
            externalDataObject->setValueDescription(valueDescription.data());
            externalDataObject->setValueUnit(valueUnit.data());
        }
        else if (externalDataObject->calcNumMats() != 1)
        {
            return ito::RetVal(
                ito::retError, 0,
                tr("Error during check data, external dataObject invalid. Object has more or less "
                    "than 1 plane. It must be of right size and type or an uninitilized image.")
                .toLatin1()
                .data());
        }
        else if (externalDataObject->getSize(dims - 2) != (unsigned int)futureHeight ||
            externalDataObject->getSize(dims - 1) != (unsigned int)futureWidth ||
            externalDataObject->getType() != futureType)
        {
            return ito::RetVal(
                ito::retError, 0,
                tr("Error during check data, external dataObject invalid. Object must be of right "
                    "size and type or an uninitilized image.")
                .toLatin1()
                .data());
        }
    }

    return ito::retOk;
}

//-------------------------------------------------------------------------------------
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
        retValue += getParameter(val, it, key, suffix, index, hasIndex, ok);
    }

    if (!retValue.containsError() && !ok)
    {
        //the parameter was not processed by the plugin, so it is done here
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
ito::RetVal AddInMultiChannelGrabber::changeChannelForListener(QObject* listener, const QString& newChannel)
{
    Q_D(AddInMultiChannelGrabber);

    assert(d->m_channelParamsProxyInitialized);
    ito::RetVal retValue(ito::retOk);
    bool found = false;

    if (listener)
    {
        auto it = m_autoGrabbingListeners.begin();

        while (it != m_autoGrabbingListeners.end())
        {
            if (it.value() == listener)
            {
                found = true;

                if (it.key() != newChannel)
                {
                    // the channel has changed
                    m_autoGrabbingListeners.erase(it);
                    m_autoGrabbingListeners.insert(newChannel, listener);
                }

                break;
            }

            it++;
        }

        if (!found)
        {
            retValue += ito::RetVal(ito::retError, 0, tr("The given listener (e.g. plot) must be registered first.").toLatin1().data());
        }
    }
    else
    {
        retValue += ito::RetVal(ito::retError, 0, "QObject not callable.");
    }

    return retValue;
}

//-------------------------------------------------------------------------------
//! Sets a new value to a parameter.
/*!
    This function parses the given parameter and calls setParameter. If the bool
    parameter ok in the setParameter (to be implemented in the individual plugins)
    function returns false, it gets assumed that the plugin didn't process the
    parameter. In this case the value of the parameter gets copied here.
    If the parameter name is "roi" sizex and sizey gets updated by setParam. If the
    key of the parameter is "defaultChannel" the function "switchDefaultChannel" gets called.
    Both happens also if the "ok" value of setParameter is true.
    "applyParamsToChannelParams" is called to synchronize the parameters of the
    channel container follwed by a call of checkData.

\param [in] val is a QSharedPOinter of type ParamBase containing the paremeter to be set.
\param [in] waitCond
\return retOk if everything was ok, else retError
*/
ito::RetVal AddInMultiChannelGrabber::setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore* waitCond/* = nullptr*/)
{
    Q_D(AddInMultiChannelGrabber);
    assert(d->m_channelParamsProxyInitialized);

    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;
    bool hasIndex;
    bool ok = false;
    int index;
    QString suffix;
    QString key;
    QStringList paramUpdateList;
    ParamMapIterator it;

    int cntStartedDevices = grabberStartedCount();

    retValue += apiParseParamName(val->getName(), key, hasIndex, index, suffix);
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
                        QStringList availableChannels = m_channels.keys();

                        retValue += ito::RetVal(
                            ito::retError,
                            0,
                            tr("Cannot switch to channel \"%1\" since it does not exist. Available channels are %2.")
                            .arg(val->getVal<const char*>())
                            .arg(availableChannels.join(", ")).toLatin1().data()
                        );
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
            else if (key == "channelSelector")
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
            retValue += startDevice(nullptr);
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

//-------------------------------------------------------------------------------
//! synchronizes m_params with the params of default channel container
/*!
This method synchronizes the parameters from the current selected channel container
with m_params. Call this function after changing the defaultChannel parameter.
Parameters which are not available for the current default channel are set to readonly

\return retOk if everything was ok, else retError
*/
ito::RetVal AddInMultiChannelGrabber::switchChannelSelector()
{
    Q_D(AddInMultiChannelGrabber);

    assert(d->m_channelParamsProxyInitialized);
    ito::RetVal retValue(ito::retOk);
    ito::uint32 flag = 0;
    QString selectedChannel = QLatin1String(m_params["channelSelector"].getVal<const char*>());

    if (m_channels.contains(selectedChannel))
    {
        const auto& selChannel = m_channels[selectedChannel];
        auto paramsInChannelIter = d->m_paramChannelAvailabilityMap.constBegin();

        while (paramsInChannelIter != d->m_paramChannelAvailabilityMap.constEnd())
        {
            if (paramsInChannelIter->contains(selectedChannel))
            {
                // this parameters is contained in the selected channel. Copy its value and flags
                // from the channel to m_params
                const ito::Param p = selChannel.m_channelParams[paramsInChannelIter.key()];
                m_params[paramsInChannelIter.key()].copyValueFrom(&p);
                m_params[paramsInChannelIter.key()].setFlags(selChannel.m_channelParams[paramsInChannelIter.key()].getFlags());
            }

            paramsInChannelIter++;
        }
    }
    else
    {
        retValue += ito::RetVal(
            ito::retError,
            0,
            tr("Cannot switch to channel %1. The channel is not registered.").arg(selectedChannel).toLatin1().data()
        );
    }

    emit parametersChanged(m_params);

    return retValue;
}

//-------------------------------------------------------------------------------
//! copies value m_params to the channel params of the current default channel
/*!
This method copies params of m_params to the params of the channel container if the param is contained in the channel container . This function is usally called after setParam to apply the changed entries of m_params to the corresponding channel container.
If a parameter is not found in the channel container nothing happens.

\param [in] keyList indicates which params are copied. If the List is empty all Parameters of the current channel are updated.
\return retOk if everything was ok, else retError
*/
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
                if (m_channels[channelSelector].m_channelParams.contains(tmp))
                {
                    if (m_params.contains(tmp))
                    {
                        m_channels[channelSelector].m_channelParams[tmp] = m_params[tmp];
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
            retVal = ito::RetVal(ito::retError, 0, tr("Unknown channel \"%1\". Available channels are: %2").arg(channelSelector).arg(m_channels.keys().join(", ")).toLatin1().data());
        }
    }
    else
    {
        QMapIterator<QString, ito::Param> it(m_channels[m_params["channelSelector"].getVal<const char*>()].m_channelParams);

        while (it.hasNext())
        {
            it.next();
            const_cast<ito::Param&>(it.value()) = m_params[it.key()];
        }
    }

    return retVal;
}

//-------------------------------------------------------------------------------
//! updates sizex and sizey
/*!
Call this function to update sizex and sizey. If the roi is changed via setParam this function will be called automatically.
Note: Do not forget to apply the changes to the channel parameters by calling applyParamsToChannelParams after calling this function.

\return retOk if everything was ok, else retError
*/
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
                m_channels[it.next()].m_channelParams[paramName].setMeta(meta, takeOwnerShip);
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
                        m_channels[channelList[i]].m_channelParams[paramName].setMeta(meta, takeOwnerShip);
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
                m_channels[it.next()].m_channelParams[paramName].setFlags(flags);
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
                        m_channels[channelList[i]].m_channelParams[paramName].setFlags(flags);
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

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::getVal(void* vpdObj, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue(ito::retOk);
    ito::DataObject* dObj = reinterpret_cast<ito::DataObject*>(vpdObj);
    QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());

    if (!dObj)
    {
        retValue += ito::RetVal(
            ito::retError, 0, tr("Empty dataObject handle retrieved from caller").toLatin1().data());
    }

    if (!retValue.containsError())
    {
        retValue += retrieveData(QStringList(defaultChannel));
    }

    if (!retValue.containsError())
    {
        // don't wait for live image, since user should get the image as fast as possible.
        sendDataToListeners(0);

        (*dObj) = m_channels[defaultChannel].m_data;

        auto channelDatasets = QSharedPointer<QMap<QString, ito::DataObject>>::create();
        channelDatasets->insert(defaultChannel, m_channels[defaultChannel].m_data);

        emit newData(channelDatasets);
        m_channels[defaultChannel].m_dataStatus = DataStatus::NewDataAndEmitted;
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::copyVal(void* vpdObj, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;
    ito::DataObject* dObj = reinterpret_cast<ito::DataObject*>(vpdObj);

    if (!dObj)
    {
        retValue += ito::RetVal(
            ito::retError, 0, tr("Empty dataObject handle retrieved from caller").toLatin1().data());
    }
    else
    {
        retValue += checkData(dObj);
    }

    if (!retValue.containsError())
    {
        retValue += getVal(vpdObj, nullptr);
    }

    if (!retValue.containsError())
    {
        QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());
        m_channels[defaultChannel].m_data.deepCopyPartial(*dObj);
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::getVal(QSharedPointer<QMap<QString, ito::DataObject> > channelDatasets, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;
    QStringList channels;

    auto it = channelDatasets->constBegin();

    while (it != channelDatasets->constEnd())
    {
        retValue += checkData(it.key(), nullptr);
        channels << it.key();
        it++;
    }

    if (!retValue.containsError())
    {
        retValue += retrieveData(channels);
    }

    if (!retValue.containsError())
    {
        // don't wait for live image, since user should get the image as fast as possible.
        sendDataToListeners(0);

        auto it2 = channelDatasets->begin();

        while (it2 != channelDatasets->end())
        {
            it2->operator=(m_channels[it2.key()].m_data);
            m_channels[it.key()].m_dataStatus = DataStatus::NewDataAndEmitted;
            it2++;
        }

        emit newData(channelDatasets);
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::copyVal(QSharedPointer<QMap<QString, ito::DataObject> > channelDatasets, ItomSharedSemaphore* waitCond)
{
    ItomSharedSemaphoreLocker locker(waitCond);
    ito::RetVal retValue;
    QStringList channels;

    auto it = channelDatasets->begin();

    while (it != channelDatasets->end())
    {
        retValue += checkData(it.key(), &(*it));
        channels << it.key();
        it++;
    }

    if (!retValue.containsError())
    {
        retValue += retrieveData(channels);
    }

    if (!retValue.containsError())
    {
        // don't wait for live image, since user should get the image as fast as possible.
        sendDataToListeners(0);

        it = channelDatasets->begin();

        while (it != channelDatasets->end())
        {
            m_channels[it.key()].m_dataStatus = DataStatus::NewDataAndEmitted;
            retValue += m_channels[it.key()].m_data.deepCopyPartial(*it);
            it++;
        }

        emit newData(channelDatasets);
    }

    if (waitCond)
    {
        waitCond->returnValue = retValue;
        waitCond->release();
    }

    return retValue;
}

//-------------------------------------------------------------------------------------
ito::RetVal AddInMultiChannelGrabber::retrieveData(ito::DataObject* externalDataObject /*= nullptr*/)
{
    QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());
    ito::RetVal retval;

    if (externalDataObject)
    {
        retval += checkData(defaultChannel, externalDataObject);
    }

    if (!retval.containsError())
    {
        // fetch all channels. This is necessary for the auto
        // grabbing using the timerEvent in abstractAddInGrabber.
        retval += retrieveData(QStringList());
    }

    if (externalDataObject && !retval.containsError())
    {
        retval += m_channels[defaultChannel].m_data.deepCopyPartial(*externalDataObject);
    }

    return retval;
}


//-------------------------------------------------------------------------------------
AddInMultiChannelGrabber::ChannelContainerMapConstIterator AddInMultiChannelGrabber::getCurrentDefaultChannel() const
{
    QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());
    return m_channels.constFind(defaultChannel);
}

//-------------------------------------------------------------------------------------
AddInMultiChannelGrabber::ChannelContainerMapIterator AddInMultiChannelGrabber::getCurrentDefaultChannel()
{
    QString defaultChannel = QLatin1String(m_params["defaultChannel"].getVal<const char*>());
    return m_channels.find(defaultChannel);
}

} //end namespace ito
