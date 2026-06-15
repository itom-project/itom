/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#ifndef ABSTRACTFILTERDIALOG_H
#define ABSTRACTFILTERDIALOG_H

#include "../global.h"

#include <qdialog.h>
#include <qtreewidget.h>

namespace ito {

class AbstractFilterDialog : public QDialog
{
    Q_OBJECT

public:
    AbstractFilterDialog(QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QWidget *parent = 0);
    ~AbstractFilterDialog() {};

    inline QVector<ito::ParamBase> getAutoMand() const { return m_autoMand; }
    inline QVector<ito::ParamBase> getAutoOut() const { return m_autoOut; }

protected:
    QVector<ito::ParamBase> m_autoMand; //mandatory parameters defined by Filter-Interface ito::AddInAlgo::tAlgoInterface
    QVector<ito::ParamBase> m_autoOut;  //output parameters defined by Filter-Interface ito::AddInAlgo::tAlgoInterface

    QList<QTreeWidgetItem*> renderAutoMandAndOutResult() const;
    QTreeWidgetItem* renderParam( const ito::ParamBase &p ) const;

private:
    void presentResult();
    void presentResultItem( const ito::ParamBase &p );
};

} //end namespace ito

#endif
