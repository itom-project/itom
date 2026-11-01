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

#include "iOsKeyStore.h"

namespace ito
{
namespace llm
{

/*! \class ito::llm::OsKeyStoreWin
    \brief Windows credential-store backend for secret persistence.

    This implementation stores and retrieves secrets via Windows Credential
    Manager for provider-specific LLM credentials.

    \sa IOsKeyStore
*/
class OsKeyStoreWin : public IOsKeyStore
{
    public:
        //! Store secret value under a key.
        /*! \param [in] key Credential identifier.
            \param [in] value Secret value to persist.
            \return retOk on success, otherwise retError.
            \sa readSecret
        */
        RetVal writeSecret(const QString &key, const QString &value) override;

        //! Read secret value for a key.
        /*! \param [in] key Credential identifier.
            \param [out] valueOut Retrieved secret value.
            \return retOk on success, otherwise retError.
            \sa writeSecret
        */
        RetVal readSecret(const QString &key, QString &valueOut) override;

        //! Remove secret for a key.
        /*! \param [in] key Credential identifier.
            \return retOk on success, otherwise retError.
            \sa writeSecret
        */
        RetVal removeSecret(const QString &key) override;
};

} // namespace llm
} // namespace ito
