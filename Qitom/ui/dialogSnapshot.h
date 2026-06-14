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

#ifndef DIALOGSNAPSHOT_H
#define DIALOGSNAPSHOT_H

#include "../global.h"
#include <qdialog.h>
#include <qpointer.h>
#include "../common/addInInterface.h"
#include "../DataObject/dataobj.h"
#include "../../AddInManager/addInManager.h"

#include "ui_dialogSnapshot.h"

class QTimerEvent; //forward declaration
class QCloseEvent; //forward declaration

namespace ito {

class DataIOThreadCtrl; //forward declaration

class DialogSnapshot : public QMainWindow
{
    Q_OBJECT

public:
    DialogSnapshot(QWidget *parent, QPointer<AddInDataIO> cam, ito::RetVal &retval);
    ~DialogSnapshot();

protected:
    void init();

    void checkRetval(const ito::RetVal retval);
    void setBtnOptions(const bool checking);
    void acquisitionStart();
    void acquisitionEnd();

    void timerEvent(QTimerEvent *event);
    void closeEvent(QCloseEvent *event);
    void setGroupTimestampEnabled();

    QString m_path;
    QList<ito::AddInAlgo::FilterDef*> m_filterPlugins;

    bool addComboItem;
    Ui::DialogSnapshot ui;
    ito::DataIOThreadCtrl *m_pCamera;
    QVector<ito::ParamBase> m_paramsOpt;
    QVector<ito::ParamBase> m_paramsMand;
    QVector<ito::ParamBase> m_autoOut;
    QList<int64> m_stamp;
    int m_totalSnaps;
    int m_numSnapsDone;
    int m_timerID;
    bool m_wasAutoGrabbing;
    QList<ito::DataObject> m_acquiredImages;

private slots:
	void on_btnSnap_clicked();
    void on_btnClose_clicked();
    void on_btnFolder_clicked();
    void on_btnOptions_clicked();
    void on_comboType_currentIndexChanged(int index);
    void on_checkMulti_stateChanged(int state);
    void on_checkTimer_stateChanged(int state);
    void on_checkAutograbbing_stateChanged(int state);
    void on_checkSaveAfterSnap_stateChanged(int state);
};

} //end namespace ito

#endif
