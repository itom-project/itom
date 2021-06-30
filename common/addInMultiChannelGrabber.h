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

#ifndef ADDINMULTICHANNELGRABBER_H
#define ADDINMULTICHANNELGRABBER_H

#include "addInInterface.h"
#include "addInGrabber.h"

#include "../DataObject/dataobj.h"
#include "sharedStructuresQt.h"
#include "sharedStructures.h"

#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{
    class AddInMultiChannelGrabberPrivate;

    class ITOMCOMMONQT_EXPORT AddInMultiChannelGrabber : public AddInAbstractGrabber
    {
        Q_OBJECT
    private:
        AddInMultiChannelGrabberPrivate *dd;
        bool m_defaultConfigReady;
        QMap<QString, QStringList> m_paramChannelAvailabilityMap;
    protected:

        struct ChannelContainer
        {
            ito::DataObject data;
            QMap<QString, ito::Param> m_channelParam;
            ChannelContainer() {};
            ChannelContainer(ito::Param roi, ito::Param pixelFormat, ito::Param sizex, ito::Param sizey)
            {
                m_channelParam.insert("pixelFormat", pixelFormat);
                m_channelParam.insert("roi", roi);
                m_channelParam.insert("sizex", sizex);
                m_channelParam.insert("sizey", sizey);

            }
        };
        QMap<QString, ChannelContainer> m_channels; /*!< Map for recently grabbed images of various channels*/
        virtual ito::RetVal checkData(ito::DataObject *externalDataObject = NULL);
        virtual ito::RetVal sendDataToListeners(int waitMS); /*!< sends m_data to all registered listeners. */
        ito::RetVal adaptDefaultChannelParams(); /*!< adaptes the params after changing the defaultChannel param*/
        void addChannel(QString name);
        virtual ito::RetVal switchDefaultChannel();/*!< synchronizes m_params with the params of default channel container */
        virtual ito::RetVal applyParamsToChannelParams(QStringList keyList = QStringList());
        void initializeDefaultConfiguration(const QMap<QString, ChannelContainer>& channelContainerMap, const QMap<QString, ito::Param>& nonChannelSpecificParams = QMap<QString, ito::Param>());/*!< sets the channel parameters.*/
        
        template <typename T> ito::RetVal setParamVal(const QByteArray& paramName,const T& val, const QList<QByteArray>& channelList = QList<QByteArray>());
        ito::RetVal setParamMeta(const QByteArray& paramName, ito::ParamMeta* meta, bool takeOwnerShip, const QList<QByteArray>& channelList = QList<QByteArray>());
        ito::RetVal setParamFlags(const QByteArray& paramName, const unsigned int& flags, const QList<QByteArray>& channelList = QList<QByteArray>());
        ito::RetVal roiChanged(const QList<QByteArray>& channelList = QList<QByteArray>());


        ////! Specific function to set the parameters in the respective plugin class
        ///*!
        //This function is a specific implementation of setParam. Overload this function to process parameters individually in the plugin class. This function is called by setParam after the parameter has been parsed and checked .

        //\param [in] val parameter to be processed
        //\param [in] it ParamMapIterator iterator to the parameter in m_params
        //\param [in] suffix possible suffix of the parameter
        //\param [in] key of the parameter
        //\param [in] index of the parameter
        //\param [in] hasIndex is set to true if parameter has an index
        //\param [in] set ok to true if parameter was processed
        //\param [in] add key of changed channel specific parameters to pendingUpdate. 
        //\return retOk if everything was ok, else retError
        //*/
        virtual ito::RetVal setParameter(QSharedPointer<ito::ParamBase> val, const ParamMapIterator& it, const QString& suffix, const QString& key, int index, bool hasIndex, bool &ok, QStringList &pendingUpdate) = 0;
        virtual ito::RetVal getParameter(QSharedPointer<ito::Param> val, const ParamMapIterator& it, const QString& suffix, const QString& key, int index, bool hasIndex, bool &ok) = 0;
        void updateSizeXY(); /*!< updates sizex und sizey*/

    public:
        AddInMultiChannelGrabber(const QByteArray &grabberName);
        ~AddInMultiChannelGrabber();
    public slots:
        ito::RetVal setParam(QSharedPointer<ito::ParamBase> val, ItomSharedSemaphore *waitCond = NULL) final;
        ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond) final;
        ito::RetVal changeChannelForListerners(const QString& newChannel, QObject* obj);
    };
}
#endif
#endif