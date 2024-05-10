/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

#include "dialogScriptCharsetEncoding.h"

namespace ito
{
    //---------------------------------------------------------------------------------
    DialogScriptCharsetEncoding::DialogScriptCharsetEncoding(const IOHelper::CharsetEncodingItem &currentEncoding, bool enableReload, QWidget *parent) :
        QDialog(parent)
    {
        ui.setupUi(this);


        setWindowModality(Qt::NonModal);
        setWindowFlag(Qt::WindowStaysOnTopHint, true);
        setAttribute(Qt::WA_DeleteOnClose, true);
        int currentIdx = 0;
        int idx = 0;
        QStringList pythonCodings;

        foreach(const IOHelper::CharsetEncodingItem &item, IOHelper::getSupportedScriptEncodings())
        {
            ui.comboSaveCharsetEncoding->addItem(item.displayName, QVariant::fromValue(item));
            pythonCodings.append(item.aliases);

            if (item.displayName == currentEncoding.displayName)
            {
                currentIdx = idx;
            }

            idx++;
        }

        pythonCodings.removeDuplicates();
        ui.comboPyMagicComment->addItems(pythonCodings);
        ui.comboSaveCharsetEncoding->setCurrentIndex(currentIdx);

        ui.btnReloadFile->setEnabled(enableReload);
    }

    //-------------------------------------------------------------------------------------
    IOHelper::CharsetEncodingItem DialogScriptCharsetEncoding::getSaveCharsetEncoding() const
    {
        return ui.comboSaveCharsetEncoding->currentData(Qt::UserRole).value<IOHelper::CharsetEncodingItem>();
    }

    //-------------------------------------------------------------------------------------
    void DialogScriptCharsetEncoding::on_btnAddPythonEncodingComment_clicked()
    {
        emit addPythonEncodingComment(
                ui.comboPyMagicComment->currentText()
        );
    }

    //-------------------------------------------------------------------------------------
    void DialogScriptCharsetEncoding::on_btnReloadFile_clicked()
    {
        emit reloadWithEncoding(
            ui.comboSaveCharsetEncoding->currentData(Qt::UserRole).value<IOHelper::CharsetEncodingItem>()
        );
    }


} //end namespace ito
