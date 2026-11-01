/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

#ifndef WIDGETPROPLLM_H
#define WIDGETPROPLLM_H

#include "abstractPropertyPageWidget.h"
#include "ui_widgetPropLlm.h"

#include "../llm/llmCredentialService.h"

namespace ito
{

class WidgetPropLlm : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropLlm(QWidget *parent = NULL);
    ~WidgetPropLlm();

    void readSettings();
    void writeSettings();

private:
    llm::LlmSettings collectSettingsFromUi() const;

    Ui::WidgetPropLlm ui;
    llm::LlmCredentialService m_credentialService;
};

} //end namespace ito

#endif
