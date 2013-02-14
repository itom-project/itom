/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef PARAMINPUTPARSER_H
#define PARAMINPUTPARSER_H

#include "../../common/addInInterface.h"
#include "../../common/sharedStructures.h"

#include "../global.h"

#include <QtGui>
#include <qdialog.h>
#include <qvector.h>
#include <qsharedpointer.h>

namespace ito {

class ParamInputParser : public QObject
{
    Q_OBJECT

public:
    ParamInputParser(QWidget *canvas);
    ~ParamInputParser();

    ito::RetVal createInputMask(const QVector<ito::Param> &params);
    bool validateInput(bool mandatoryValues, ito::RetVal &retValue, bool showMessages = false);
    ito::RetVal getParameters(QVector<ito::ParamBase> &params);

    inline int getItemSize() const { return m_params.size(); };

protected:

    QWidget* renderTypeInt(const ito::Param &param, int virtualIndex, QWidget *parent = NULL);
    QWidget* renderTypeChar(const ito::Param &param, int virtualIndex, QWidget *parent = NULL);
    QWidget* renderTypeDouble(const ito::Param &param, int virtualIndex, QWidget *parent = NULL);
    QWidget* renderTypeString(const ito::Param &param, int virtualIndex, QWidget *parent = NULL);
    QWidget* renderTypeHWRef(const ito::Param &param, int virtualIndex, QWidget *parent = NULL);

    ito::RetVal getIntValue(ito::ParamBase &param, const ito::Param orgParam, QWidget *contentWidget, void *internalData, bool mandatory);
    ito::RetVal getCharValue(ito::ParamBase &param, const ito::Param orgParam, QWidget *contentWidget, void *internalData, bool mandatory);
    ito::RetVal getDoubleValue(ito::ParamBase &param, const ito::Param orgParam, QWidget *contentWidget, void *internalData, bool mandatory);
    ito::RetVal getStringValue(ito::ParamBase &param, const ito::Param orgParam, QWidget *contentWidget, void *internalData, bool mandatory);
    ito::RetVal getHWValue(ito::ParamBase &param, const ito::Param orgParam, QWidget *contentWidget, void *internalData, bool mandatory);

    QSignalMapper *m_pSignalMapper;
    QVector<ito::Param> m_params;
    QVector<void*> m_internalData;
    QWeakPointer<QWidget> m_canvas;
    QIcon m_iconInfo;

private:

private slots:
    void browsePluginPicker(int i);
};

} //end namespace ito

#endif