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

#pragma once

#include "llmSettings.h"

#include "../../common/retVal.h"

namespace ito
{
namespace llm
{

/*! \class ito::llm::LlmSettingsManager
    \brief Loads and parses persistent LLM settings from application config.

    The manager maps stored textual config values to strongly typed LLM
    settings fields used by providers and controller code.

    \sa LlmSettings
*/
class LlmSettingsManager
{
    public:
        //! Load LLM settings from persistent configuration.
        /*! \param [out] settings Loaded settings instance.
            \return retOk on success, otherwise retError.
            \sa parseMode
        */
        static RetVal load(LlmSettings &settings);

    private:
        //! Parse textual mode value to enum.
        /*! \param [in] modeText Stored mode text.
            \return Parsed LLM mode value.
            \sa load
        */
        static LlmMode parseMode(const QString &modeText);
};

} // namespace llm
} // namespace ito
