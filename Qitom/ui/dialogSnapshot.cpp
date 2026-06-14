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

#include "dialogSnapshot.h"

#include "../AppManagement.h"
#include "../organizer/userOrganizer.h"
#include "../organizer/designerWidgetOrganizer.h"
#include "../common/pluginThreadCtrl.h"
#include "../common/addInInterface.h"
#include "../AddInManager/algoInterfaceValidator.h"
#include "../ui/dialogSaveFileWithFilter.h"

#include <qmessagebox.h>

#include "../plot/AbstractDObjFigure.h"
#include <qdir.h>
#include <qfiledialog.h>
#include <qpair.h>
#include <qregularexpression.h>

namespace ito {

bool cmpStringIntPair(const QPair<QString, int> &a, const QPair<QString, int> &b)
{
    return a.first < b.first;
}

//----------------------------------------------------------------------------------------------------------------------------------
void DialogSnapshot::setGroupTimestampEnabled()
{
    ui.checkDataObjectTag->setEnabled(ui.comboType->itemData(ui.comboType->currentIndex()).toInt() < 0 );
}

//----------------------------------------------------------------------------------------------------------------------------------
DialogSnapshot::DialogSnapshot(QWidget *parent, QPointer<ito::AddInDataIO> cam, ito::RetVal &retval) :
    QMainWindow(parent),
    m_path(""),
    m_pCamera(NULL),
    addComboItem(false),
    m_totalSnaps(0),
    m_numSnapsDone(0),
    m_timerID(-1),
    m_wasAutoGrabbing(true),
    m_stamp()
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
        QString plotClassName = "";
        int bpp = 0;
        int sizex = 0;
        int sizey = 0;
        m_pCamera->getImageParams(bpp, sizex, sizey);

        if (sizey>1)
        {
            plotClassName = dwo->getFigureClass("DObjLiveImage", "", retval);
        }
        else
        {
            plotClassName = dwo->getFigureClass("DObjLiveLine", "", retval);
        }

        widget = dwo->createWidget(plotClassName, ui.groupPlot, AbstractFigure::ModeStandaloneInUi);
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
            m_filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, QString());
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
        ui.comboType->addItem(tr("Workspace"), -3);
        ui.comboType->addItem("*.idc", -1);
        ui.comboType->addItem("*.mat", -2);

        for (int i = 0; i < list.size(); ++i)
        {
            ui.comboType->addItem(list[i].first, list[i].second);
        }
        addComboItem = false;

        ui.checkAutograbbing->setChecked(cam->getAutoGrabbing());
    }

    QRegularExpression regExp("^[a-zA-Z][a-zA-Z0-9_]*$");
    auto *validator = new QRegularExpressionValidator(regExp, ui.leFilename);
    ui.leFilename->setValidator(validator);
    ui.leFilename->setToolTip(tr("The name must start with a letter followed by numbers or letters [a-z] or [A-Z]"));

    ui.btnFolder->setEnabled(false);
    ui.lblProgress->setVisible(false);
    ui.progress->setVisible(false);

    setGroupTimestampEnabled();
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
    m_stamp.append(QDateTime::currentMSecsSinceEpoch());

    if (!retval.containsError())
    {
        bool acquireStack = (ui.comboType->itemData(ui.comboType->currentIndex()).toInt() < 0 && ui.comboSingleStack->currentIndex() == 1);
        ui.lblProgress->setText(tr("acquire image %1 from %2").arg(m_numSnapsDone+1).arg(m_totalSnaps));

        if (!acquireStack)
        {
            ito::DataObject image;
            retval += m_pCamera->copyVal(image);

            m_acquiredImages << image;
            //m_acquiredImages.append(image);
        }
        else
        {
            if (m_acquiredImages.size() == 0)
            {
                //acquire first image to check for size and type
                ito::DataObject image;
                retval += m_pCamera->copyVal(image);

                if (image.getDims() == 2)
                {
                    //create 3D data object
                    ito::DataObject stack(m_totalSnaps, image.getSize(0), image.getSize(1), image.getType());

                    //get shallow copy of first plane in stack
                    ito::Range ranges[] = { ito::Range(0, 1), ito::Range::all(), ito::Range::all() };
                    ito::DataObject plane0 = stack.at(ranges);
                    image.deepCopyPartial(plane0);

                    m_acquiredImages << stack;
                }
                else
                {
                    retval += ito::RetVal(ito::retError, 0, tr("The acquired image must be two-dimensional for a stack-acquisition").toLatin1().data());
                }
            }
            else if (m_acquiredImages.size() == 1 && m_acquiredImages[0].getDims() == 3 && m_acquiredImages[0].getSize(0) > m_numSnapsDone)
            {
                //get shallow copy of first plane in stack
                ito::Range ranges[] = { ito::Range(m_numSnapsDone, m_numSnapsDone + 1), ito::Range::all(), ito::Range::all() };
                ito::DataObject plane = m_acquiredImages[0].at(ranges);
                retval += m_pCamera->copyVal(plane);
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, tr("Acquisition could not be finished. Wrong allocated stack size.").toLatin1().data());
            }
        }

        m_numSnapsDone++;

        ui.progress->setValue(ui.progress->value() + 1);
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
    m_stamp.clear();

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
    ito::RetVal retval = ito::retOk;

    if (m_timerID >= 0)
    {
        killTimer(m_timerID);
        m_timerID = -1;
    }

    if (m_totalSnaps == m_numSnapsDone && ui.checkSaveAfterSnap->isChecked() && m_acquiredImages.size() > 0)
    {
        ui.lblProgress->setText(tr("save image"));
        if (ui.checkDataObjectTag->isChecked() && ui.checkDataObjectTag->isEnabled())
        {
            QList<ito::DataObject>::iterator i;
            int cnt = 0;
            if (m_acquiredImages.length() < m_stamp.length())
            {
                uint64 val;
                foreach(val, m_stamp)
                {
                    m_acquiredImages[0].setTag(tr("UnixTimestamp%1").arg(cnt).toLatin1().data(), val*0.001);
                    cnt += 1;
                }
            }
            else {
                for (i = m_acquiredImages.begin(); i != m_acquiredImages.end(); ++i)
                {
                    i->setTag("UnixTimestamp", m_stamp[cnt]*0.001);
                    cnt += 1;
                }
            }
        }
        QCoreApplication::processEvents();

        int index = ui.comboType->itemData(ui.comboType->currentIndex()).toInt();

        if (index == -3)
        {
            QObject *pyEngine = AppManagement::getPythonEngine();
            if (!pyEngine)
            {
                retval += ito::RetVal(ito::retError, 0, tr("Python was not found").toLatin1().data());
                checkRetval(retval);
                return;
            }
            QSharedPointer<IntList> existing;

            QSharedPointer<QStringList> list(new QStringList());
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QString imageName = ui.leFilename->text();

            QMetaObject::invokeMethod(pyEngine, "getVarnamesListInWorkspace", Q_ARG(bool, true), Q_ARG(QString, imageName + "*"),
                Q_ARG(QSharedPointer<QStringList>, list), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

            if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
            {
                retval += ito::RetVal(ito::retError, 0, tr("Timeout while seaching file name at workspace").toLatin1().data());
                checkRetval(retval);
            }

            if (!retval.containsError())
            {
                list->sort();
                int fileIndex = 1;

                if (list->size() > 0)
                {
                    if (!ui.checkFilename->isChecked())//if there is no timestamp in the name the name might be not unique
                    {
                        //find the highest number after name
                        int prev = 0;
                        for (int i = 0; i < list->length(); ++i)
                        {
                            QString fn = list->at(i);
                            prev = fn.mid(imageName.size(), fn.indexOf(".") - imageName.size()).toInt() + 1;
                            if (prev > fileIndex)
                            {
                                fileIndex = prev;
                            }
                        }
                    }
                }

                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                QStringList varNames;
                QString baseName(imageName);
                QVector<SharedParamBasePointer> values;
                SharedParamBasePointer paramBasePointer;
                for (int i = 0; i < m_acquiredImages.size(); ++i)
                {

                    QString fileNo = QString("%1").arg(fileIndex++, 3, 10, QLatin1Char('0'));
                    if (ui.checkFilename->isEnabled() && ui.checkFilename->isChecked())
                    {
                        imageName = baseName + QString::number(m_stamp[i])+'_';
                    }
                    varNames.append(imageName + fileNo);
                    paramBasePointer = QSharedPointer<ito::ParamBase>(new ito::ParamBase("image", ito::ParamBase::DObjPtr, (char*)&(m_acquiredImages[i])));
                    values.append(paramBasePointer);
                }

                QMetaObject::invokeMethod(pyEngine, "putParamsToWorkspace", Q_ARG(bool, true), Q_ARG(QStringList, varNames),
                    Q_ARG(QVector<SharedParamBasePointer>, values), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

                if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
                {
                    retval += ito::RetVal(ito::retError, 0, tr("Timeout while writing picture to workspace").toLatin1().data());
                    checkRetval(retval);
                }
            }
        }
        else
        {
            QString imageName = ui.leFilename->text();
            QDir dir(m_path);
            QStringList filters;
            filters << imageName + ui.comboType->currentText();
            dir.setNameFilters(filters);
            dir.setSorting(QDir::Name);
            QStringList list = dir.entryList();
            int fileIndex = 1;
            QString baseName(imageName);

            if (list.size() > 0)
            {
                if (!ui.checkFilename->isChecked())//if there is no timestamp in the name the name might be not unique
                {
                    //find the highest number after name
                    int prev = 0;
                    for (int i = 0; i < list.length(); ++i)
                    {
                        QString fn = list.at(i);
                        prev = fn.mid(imageName.size(), fn.indexOf(".") - imageName.size()).toInt() + 1;
                        if (prev > fileIndex)
                        {
                            fileIndex = prev;
                        }
                    }
                }
            }

            QString fileExt = ui.comboType->currentText();
            fileExt.replace("*", "");

            if (index > -1)
            {
                ito::AddInAlgo::FilterDef *filter = m_filterPlugins[index];
                ito::AddInAlgo::FilterDefExt *filterExt = dynamic_cast<ito::AddInAlgo::FilterDefExt*>(filter);

                for (int i = 0; i < m_acquiredImages.size(); ++i)
                {
                    ui.lblProgress->setText(tr("save image %1 from %2").arg(i + 1).arg(m_acquiredImages.size()));

                    QString fileNo = QString("%1").arg(fileIndex++, 3, 10, QLatin1Char('0'));
                    if (ui.checkFilename->isEnabled() && ui.checkFilename->isChecked())
                    {
                        imageName = baseName + QString::number(m_stamp[i]) + '_';
                    }
                    QString fileName = m_path + "/" + imageName + fileNo + fileExt;
                    m_paramsMand[1].setVal<char*>(fileName.toLatin1().data());
                    m_paramsMand[0].setVal<ito::DataObject*>(&(m_acquiredImages[i]));

                    if (filterExt == NULL)
                    {
                        retval = filter->m_filterFunc(&m_paramsMand, &m_paramsOpt, &m_autoOut);
                    }
                    else
                    {
                        QSharedPointer<ito::FunctionCancellationAndObserver> emptyObserver; //no observer initialized
                        retval = filterExt->m_filterFuncExt(&m_paramsMand, &m_paramsOpt, &m_autoOut, emptyObserver);
                    }

                    checkRetval(retval);

                    if (retval.containsError())
                    {
                        break;
                    }
                    ui.progress->setValue(ui.progress->value() + 1);
                    QCoreApplication::processEvents();
                }
            }
            else
            {
                // idc & mat
                QObject *pyEngine = AppManagement::getPythonEngine();
                if (!pyEngine)
                {
                    retval += ito::RetVal(ito::retError, 0, tr("Python was not found").toLatin1().data());
                    checkRetval(retval);
                    return;
                }

                QByteArray funcName;
                if (index == -1)
                {
                    funcName = "pickleSingleParam";
                }
                else if (index == -2)
                {
                    funcName = "saveMatlabSingleParam";
                }

                QSharedPointer<ito::Param> param(new ito::Param("image", ito::ParamBase::DObjPtr, NULL, ""));

                for (int i = 0; i < m_acquiredImages.size(); ++i)
                {
                    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                    QString fileNo = QString("%1").arg(fileIndex++, 3, 10, QLatin1Char('0'));
                    if (ui.checkFilename->isEnabled() && ui.checkFilename->isChecked())
                    {
                        imageName = baseName + QString::number(m_stamp[i]) + '_';
                    }
                    QString fileName = m_path + "/" + imageName + fileNo + fileExt;

                    param->setVal<ito::DataObject*>(&(m_acquiredImages[i]));
                    QMetaObject::invokeMethod(pyEngine, funcName.data(), Q_ARG(QString, fileName), Q_ARG(QSharedPointer<ito::Param>, param),
                        Q_ARG(QString, imageName + fileNo), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

                    if (!locker.getSemaphore()->wait(AppManagement::timeouts.pluginFileSaveLoad))
                    {
                        retval += ito::RetVal(ito::retError, 0, tr("Timeout while saving picture").toLatin1().data());
                        checkRetval(retval);
                        break;
                    }
                }
            }
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
				//try to update userOpt and userMand to the values from a previous call of this function
				for (int i = 0; i < qMin(m_paramsOpt.size(), userOpt.size()); ++i)
				{
					if ((userOpt[i].getType() == m_paramsOpt[i].getType()) && \
						userOpt[i].getName() == m_paramsOpt[i].getName())
					{
						userOpt[i].copyValueFrom(&(m_paramsOpt[i]));
					}
				}

				int offset = autoMand.size();
				for (int i = 0; i < qMin(m_paramsMand.size() - offset, userMand.size()); ++i)
				{
					if ((userMand[i].getType() == m_paramsMand[i + offset].getType()) && \
						userMand[i].getName() == m_paramsMand[i + offset].getName())
					{
						userMand[i].copyValueFrom(&(m_paramsMand[i + offset]));
					}
				}

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
        int index = ui.comboType->itemData(ui.comboType->currentIndex()).toInt();
        ui.comboSingleStack->setEnabled(index < 0 && ui.checkMulti->isChecked());
        ui.btnFolder->setEnabled(index != -3);
        setBtnOptions(ui.checkSaveAfterSnap->isChecked());
        setGroupTimestampEnabled();

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
    ui.comboSingleStack->setEnabled(ui.comboType->itemData(ui.comboType->currentIndex()).toInt() < 0 && checking);
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
