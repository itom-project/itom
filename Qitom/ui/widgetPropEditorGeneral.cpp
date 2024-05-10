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

#include "widgetPropEditorGeneral.h"

#include "../global.h"
#include "../AppManagement.h"
#include "../helper/IOHelper.h"

#include <qmenu.h>
#include <qsettings.h>
#include <qstringlist.h>

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorGeneral::WidgetPropEditorGeneral(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    ui.groupEolMode->setVisible(false);

    // file available text codecs
    auto codecs = IOHelper::getSupportedScriptEncodings();

    foreach(const IOHelper::CharsetEncodingItem &item, codecs)
    {
        ui.comboEncoding->addItem(item.displayName, item.encodingName);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorGeneral::~WidgetPropEditorGeneral()
{
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorGeneral::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // EOL-Mode
    QString eolMode = settings.value("eolMode", "EolUnix").toString();
    ui.radioEOL1->setChecked(eolMode == "EolWindows");
    ui.radioEOL2->setChecked(eolMode == "EolUnix");
    ui.radioEOL3->setChecked(eolMode == "EolMac");

    //// Fold Style
    //QString foldStyle = settings.value("foldStyle", "plus_minus").toString();
    //ui.radioFoldingPM->setChecked(foldStyle == "plus_minus");
    //ui.radioFoldingCirclesTree->setChecked(foldStyle == "circles_tree");
    //ui.radioFoldingCircles->setChecked(foldStyle == "circles");
    //ui.radioFoldingSquares->setChecked(foldStyle == "squares");
    //ui.radioFoldingSquaresTree->setChecked(foldStyle == "squares_tree");
    //ui.radioFoldingNone->setChecked(foldStyle == "none");

    // Indentation
    ui.checkAutoIndent->setChecked(settings.value("autoIndent", true).toBool());
    ui.checkIndentUseTabs->setChecked(settings.value("indentationUseTabs", false).toBool());
    ui.spinIndentWidth->setValue(settings.value("indentationWidth", 4).toInt());

    ui.checkIndentShowGuides->setChecked(settings.value("showIndentationGuides", true).toBool());
    ui.checkShowWhitespace->setChecked(settings.value("showWhitespace", true).toBool());

    ui.checkStripSpacesAfterReturn->setChecked(settings.value("autoStripTrailingSpacesAfterReturn", true).toBool());

    //cut, copy, paste behaviour
    ui.checkSelectLineOnCopyEmpty->setChecked(settings.value("selectLineOnCopyEmpty", true).toBool());
    ui.checkKeepIndentationOnPaste->setChecked(settings.value("keepIndentationOnPaste", true).toBool());

    // encoding
    auto defaultEncodingItem = IOHelper::getDefaultScriptEncoding();
    QString encoding = settings.value("characterSetEncoding", defaultEncodingItem.encodingName).toString();
    auto encodingItem = IOHelper::getEncodingFromAlias(encoding, nullptr);
    int idx = ui.comboEncoding->findData(encodingItem.encodingName, Qt::UserRole);

    if (idx == -1)
    {
        idx = ui.comboEncoding->findData(defaultEncodingItem.encodingName, Qt::UserRole);
    }

    ui.comboEncoding->setCurrentIndex(idx);

    ui.checkAutoDetectEncoding->setChecked(settings.value("characterSetEncodingAutoGuess", true).toBool());

    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorGeneral::writeSettings()
{

    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("CodeEditor");

    // EOL-Mode
    if (ui.radioEOL1->isChecked())
    {
        settings.setValue("eolMode", "EolWindows");
    }
    else if (ui.radioEOL2->isChecked())
    {
        settings.setValue("eolMode", "EolUnix");
    }
    else
    {
        settings.setValue("eolMode", "EolMac");
    }

    // indentation
    settings.setValue("autoIndent", ui.checkAutoIndent->isChecked());
    settings.setValue("indentationUseTabs", ui.checkIndentUseTabs->isChecked());
    settings.setValue("indentationWidth", ui.spinIndentWidth->value());
    settings.setValue("showIndentationGuides", ui.checkIndentShowGuides->isChecked());
    settings.setValue("showWhitespace", ui.checkShowWhitespace->isChecked());

    settings.setValue("autoStripTrailingSpacesAfterReturn", ui.checkStripSpacesAfterReturn->isChecked());

    // cut, copy, paste behaviour
    settings.setValue("selectLineOnCopyEmpty", ui.checkSelectLineOnCopyEmpty->isChecked());
    settings.setValue("keepIndentationOnPaste", ui.checkKeepIndentationOnPaste->isChecked());

    // character set encoding
    QString encoding = ui.comboEncoding->currentData(Qt::UserRole).toString();
    auto encodingItem = IOHelper::getEncodingFromAlias(encoding, nullptr);
    settings.setValue("characterSetEncoding", encodingItem.encodingName);

    settings.setValue("characterSetEncodingAutoGuess", ui.checkAutoDetectEncoding->isChecked());

    settings.endGroup();
}

} //end namespace ito
