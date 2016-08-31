/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "dialogSnapshot.h"
#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../common/pluginThreadCtrl.h"
#include "../common/addInInterface.h"
#include "../ui/dialogSaveFileWithFilter.h"
#include <qmessagebox.h>

#include "../plot/AbstractDObjFigure.h"
#include <qdir.h>
#include <QFileDialog.h>
#include <qpair.h>

namespace ito {

bool cmpStringIntPair(const QPair<QString, int> &a, const QPair<QString, int> &b)
{
    return a.first < b.first;
}



//----------------------------------------------------------------------------------------------------------------------------------
DialogSnapshot::DialogSnapshot(QWidget *parent, QPointer<ito::AddInDataIO> cam, ito::RetVal &retval) :
    QMainWindow(parent),
    m_path(""),
    m_paramsOpt(NULL),
    m_paramsMand(NULL),
    m_pCamera(NULL),
    addComboItem(false),
    m_totalSnaps(0),
    m_numSnapsDone(0),
    m_timerID(-1),
    m_wasAutoGrabbing(true)
{
    retval = ito::retOk;
    ui.setupUi(this);

    DesignerWidgetOrganizer *dwo = qobject_cast<DesignerWidgetOrganizer*>(AppManagement::getDesignerWidgetOrganizer());
    if (!dwo)
    {
        retval += RetVal(retError, 0, tr("designerWidgetOrganizer is not available").toLatin1().data());
    }
    else
    {
        m_pCamera = new DataIOThreadCtrl(cam.data()); //increments the reference to the camera

        QWidget* widget = NULL;
        QString plotClassName = dwo->getFigureClass("DObjLiveImage", "", retval);
        widget = dwo->createWidget(plotClassName, ui.groupPlot, "liveImagePlot", AbstractFigure::ModeStandaloneInUi);
        widget->setVisible(true);
        
        QVBoxLayout *layout = new QVBoxLayout();
        layout->addWidget(widget);
        ui.groupPlot->setLayout(layout);

        ito::AbstractDObjFigure *dObjFigure = NULL;
        if (widget->inherits("ito::AbstractDObjFigure"))
        {
            dObjFigure = (ito::AbstractDObjFigure*)(widget);

            //check if dObjFigure has property "yAxisFlipped" and flip it, if so.
            QVariant yAxisFlipped = dObjFigure->property("yAxisFlipped");
            if (yAxisFlipped.isValid())
            {
                dObjFigure->setProperty("yAxisFlipped", true);
            }

            dObjFigure->setCamera(cam);
            ito::AddInInterfaceBase *aib = cam->getBasePlugin();
            ui.labelSource->setText(aib->objectName() + " (" + cam->getIdentifier() + ")");

            m_path = QDir::currentPath();
            ui.statusbar->showMessage(m_path);
        }

        QList<QPair<QString, int> > list;
        ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
        if (AIM)
        {
            m_filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, QString::Null());
            for (int i = 0; i < m_filterPlugins.size(); ++i)
            {
                QString item = m_filterPlugins[i]->m_interfaceMeta;
                item = item.mid(item.indexOf("(") + 1);
                item = item.mid(0, item.indexOf(")"));
                foreach(const QString &s, item.split(" "))
                {
                    list << QPair<QString, int>(s, i);
                }
            }
        }

        std::sort(list.begin(), list.end(), cmpStringIntPair);

        addComboItem = true;
        ui.comboType->addItem("*.idc", -1);
        ui.comboType->addItem("*.mat", -2);

        for (int i = 0; i < list.size(); ++i)
        {
            ui.comboType->addItem(list[i].first, list[i].second);
        }
        addComboItem = false;

        ui.checkAutograbbing->setChecked(cam->getAutoGrabbing());
    }

    ui.lblProgress->setVisible(false);
    ui.progress->setVisible(false);
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogSnapshot::~DialogSnapshot()
{
    DELETE_AND_SET_NULL(m_pCamera);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::checkRetval(const ito::RetVal retval)
{
    if (retval.containsError())
    {
        QMessageBox msgBox;
        msgBox.setText(QLatin1String(retval.errorMessage()));
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning())
    {
        QMessageBox msgBox;
        msgBox.setText(QLatin1String(retval.errorMessage()));
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::timerEvent(QTimerEvent *event)
{
    ito::RetVal retval = m_pCamera->acquire();
    
    if (!retval.containsError())
    {
        ui.lblProgress->setText(tr("acquire image %1 from %2").arg(m_numSnapsDone+1).arg(m_totalSnaps));
        m_numSnapsDone ++;
        ito::DataObject image;
        retval += m_pCamera->getVal(image);
        ui.progress->setValue(ui.progress->value() + 1);
        m_acquiredImages << image;
    
        if (m_numSnapsDone >= m_totalSnaps)
        {
            acquisitionEnd();
        }
    }

    checkRetval(retval);

    if (retval.containsError())
    {
        acquisitionEnd();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::closeEvent(QCloseEvent *event)
{
    if (m_timerID == -1) 
    {
        event->accept();
    } 
    else 
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Please stop the acquisition before closing the dialog"));
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
        event->ignore();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::acquisitionStart()
{
    if (m_timerID >= 0)
    {
        killTimer(m_timerID);
        m_timerID = -1;
    }

    if (m_wasAutoGrabbing = m_pCamera->getAutoGrabbing())
    {
        m_pCamera->disableAutoGrabbing();
    }

    m_totalSnaps = (ui.checkMulti->isChecked()) ? ui.spinMulti->value() : 1;
    m_numSnapsDone = 0;

    ui.progress->setVisible(m_totalSnaps > 1);
    ui.lblProgress->setVisible(true);
    ui.groupMultishot->setEnabled(false);
    ui.groupSaveData->setEnabled(false);
    ui.checkAutograbbing->setEnabled(false);
    ui.btnClose->setEnabled(false);
    ui.progress->setMaximum(ui.checkSaveAfterSnap->isChecked() ? m_totalSnaps * 2 : m_totalSnaps);
    ui.progress->setValue(0);

    if (m_totalSnaps > 1)
    {
        ui.btnSnap->setText(tr("Stop"));
        int interval = ui.checkTimer->isChecked() ? ui.spinTimer->value() : 0;
        timerEvent(NULL);
        m_timerID = startTimer(interval, Qt::PreciseTimer);
    }
    else
    {
        ui.btnSnap->setEnabled(false);
        timerEvent(NULL);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::acquisitionEnd()
{
    if (m_timerID >= 0)
    {
        killTimer(m_timerID);
        m_timerID = -1;
    }

    if (m_totalSnaps == m_numSnapsDone && ui.checkSaveAfterSnap->isChecked() && m_acquiredImages.size() > 0)
    {
        QDir dir(m_path);
        QStringList filters;
        filters << "pic_" + ui.comboType->currentText();
        dir.setNameFilters(filters);
        dir.setSorting(QDir::Name);
        QStringList list = dir.entryList();
        int fileIndex = 1;
        if (list.size() > 0)
        {
            QString fn = list[list.size() - 1];
            fileIndex = fn.mid(4, fn.indexOf(".") - 4).toInt() + 1;
        }

        int index = ui.comboType->itemData(ui.comboType->currentIndex()).toInt();
        if (index > -1)
        {
            ito::AddInAlgo::FilterDef *filter = m_filterPlugins[index];
            QString fileExt = ui.comboType->currentText();
            fileExt.replace("*", "");
            QString fileNo = QString("%1").arg(fileIndex, 3, 10, QLatin1Char('0'));
            QString fileName = m_path + "/pic_" + fileNo + fileExt;
            m_paramsMand[1].setVal<char*>(fileName.toLatin1().data());
            m_paramsMand[0].setVal<ito::DataObject*>(&(m_acquiredImages[0]));
            ito::RetVal retval = filter->m_filterFunc(&m_paramsMand, &m_paramsOpt, &m_autoOut);
            checkRetval(retval);
        }
        else if (index == -1)
        {
            // idc

        }
        else
        {
            // mat

        }
    }

    ui.lblProgress->setVisible(false);
    ui.progress->setVisible(false);
    ui.groupMultishot->setEnabled(true);
    ui.groupSaveData->setEnabled(true);
    ui.btnSnap->setEnabled(true);
    ui.checkAutograbbing->setEnabled(true);
    ui.btnClose->setEnabled(true);
    ui.btnSnap->setText(tr("Snapshot"));

    if (m_wasAutoGrabbing)
    {
        m_pCamera->enableAutoGrabbing();
    }

    m_acquiredImages.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::setBtnOptions(const bool checking)
{
    bool isOptions = checking;
    if (isOptions)
    {
        int filterIndex = ui.comboType->itemData(ui.comboType->currentIndex()).toInt();
        isOptions = filterIndex > -1;

        if (isOptions)
        {
            ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

            if (AIM)
            {
                const ito::FilterParams *fp = AIM->getHashedFilterParams(m_filterPlugins[filterIndex]->m_paramFunc);
                isOptions = fp->paramsMand.size() + fp->paramsOpt.size() > 2;
            }
            else
            {
                isOptions = false;
            }
        }
    }

    ui.btnOptions->setEnabled(isOptions);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_btnSnap_clicked()
{
    if (m_timerID == -1)
    {
        acquisitionStart();
    }
    else
    {
        acquisitionEnd();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_btnOptions_clicked()
{
    ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    if (AIM)
    {
        ito::AddInAlgo::FilterDef *filter = m_filterPlugins[ui.comboType->itemData(ui.comboType->currentIndex()).toInt()];
        const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);
        const AlgoInterfaceValidator *aiv = AIM->getAlgoInterfaceValidator();

        if (aiv)
        {
            QVector<ito::ParamBase> autoMand;
//            QVector<ito::ParamBase> autoOut;
            QVector<ito::ParamBase> paramsOpt;
            QVector<ito::ParamBase> paramsMand;
            QVector<ito::Param> userMand;
            QVector<ito::Param> userOpt;

            ito::RetVal retval = aiv->getInterfaceParameters(filter->m_interface, autoMand, m_autoOut);

            autoMand[0].setVal<ito::DataObject*>(NULL);
            QString filename = ui.comboType->itemText(ui.comboType->currentIndex());
            filename = filename.replace("*", "a");
            autoMand[1].setVal<char*>(filename.toLatin1().data());

            userOpt = fp->paramsOpt;
            userMand = fp->paramsMand.mid(autoMand.size());

            if (userMand.size() > 0 || userOpt.size() > 0)
            {
                DialogSaveFileWithFilter *dialog = new DialogSaveFileWithFilter(filename, filter, autoMand, m_autoOut, userMand, userOpt, false, this);
                if (dialog->exec() == QDialog::Accepted)
                {
                    dialog->getParameters(paramsMand, paramsOpt);
                    m_paramsMand = autoMand + paramsMand;
                    m_paramsOpt = paramsOpt;
                }

                DELETE_AND_SET_NULL(dialog);
            }

            checkRetval(retval);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_btnClose_clicked()
{
    this->close();
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_btnFolder_clicked()
{
    QString path = QFileDialog::getExistingDirectory(this, tr("Select a directory..."), m_path);
    if (!path.isEmpty())
    {
        m_path = path;
        ui.statusbar->showMessage(m_path);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_comboType_currentIndexChanged(int index)
{
    if (!addComboItem)
    {
        ui.comboColor->setEnabled(ui.comboType->currentText() == "*.idc" && ui.checkMulti->isChecked());
        setBtnOptions(ui.checkSaveAfterSnap->isChecked());

        ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

        if (AIM)
        {
            int filterIndex = ui.comboType->itemData(ui.comboType->currentIndex()).toInt();
            if (filterIndex > -1)
            {
                ito::AddInAlgo::FilterDef *filter = m_filterPlugins[filterIndex];
                const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);

                if (fp)
                {
                    m_paramsMand.clear();
                    foreach(const ito::Param &p, fp->paramsMand)
                    {
                        m_paramsMand.append(ito::ParamBase(p));
                    }

                    m_paramsOpt.clear();
                    foreach(const ito::Param &p, fp->paramsOpt)
                    {
                        m_paramsOpt.append(ito::ParamBase(p));
                    }
                }
            }
/*            else
            {
                m_paramsMand.clear();
                m_paramsOpt.clear();
            }*/
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_checkMulti_stateChanged(int state)
{
    bool checking = ui.checkMulti->isChecked();
    ui.spinMulti->setEnabled(checking);
    ui.comboColor->setEnabled(ui.comboType->currentText() == "*.idc" && checking);
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_checkTimer_stateChanged(int state)
{
    ui.spinTimer->setEnabled(ui.checkTimer->isChecked());
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_checkAutograbbing_stateChanged(int state)
{
    if (ui.checkAutograbbing->isChecked())
    {
        m_pCamera->enableAutoGrabbing();
    }
    else
    {
        m_pCamera->disableAutoGrabbing();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::on_checkSaveAfterSnap_stateChanged(int state)
{
    bool checking = ui.checkSaveAfterSnap->isChecked();
    ui.comboType->setEnabled(checking);
    ui.btnFolder->setEnabled(checking);
    setBtnOptions(checking);
}

//----------------------------------------------------------------------------------------------------------------------------------
} //end namespace ito