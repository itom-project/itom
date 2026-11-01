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

#include "osKeyStoreWin.h"

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <wincred.h>

#pragma comment(lib, "Advapi32.lib")

#endif

namespace ito
{
namespace llm
{

RetVal OsKeyStoreWin::writeSecret(const QString &key, const QString &value)
{
#ifdef _WIN32
    if (key.trimmed().isEmpty())
    {
        return RetVal(retError, 0, "Credential key must not be empty.");
    }

    const QString target = key.trimmed();
    QByteArray targetUtf16(reinterpret_cast<const char*>(target.utf16()), target.size() * 2);
    QByteArray secretUtf16(reinterpret_cast<const char*>(value.utf16()), value.size() * 2);

    CREDENTIALW cred;
    ZeroMemory(&cred, sizeof(CREDENTIALW));

    cred.Type = CRED_TYPE_GENERIC;
    cred.TargetName = reinterpret_cast<LPWSTR>(const_cast<ushort*>(target.utf16()));
    cred.CredentialBlobSize = static_cast<DWORD>(secretUtf16.size());
    cred.CredentialBlob = reinterpret_cast<LPBYTE>(secretUtf16.data());
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
    wchar_t userName[] = L"itom-llm";
    cred.UserName = userName;

    if (!CredWriteW(&cred, 0))
    {
        return RetVal(retError, 0, "Could not store secret in Windows Credential Manager.");
    }

    return RetVal(retOk);
#else
    Q_UNUSED(key)
    Q_UNUSED(value)
    return RetVal(retError, 0, "OS key store backend not implemented for this platform.");
#endif
}

RetVal OsKeyStoreWin::readSecret(const QString &key, QString &valueOut)
{
#ifdef _WIN32
    valueOut.clear();

    if (key.trimmed().isEmpty())
    {
        return RetVal(retError, 0, "Credential key must not be empty.");
    }

    PCREDENTIALW pCred = NULL;
    const QString target = key.trimmed();
    if (!CredReadW(reinterpret_cast<LPCWSTR>(target.utf16()), CRED_TYPE_GENERIC, 0, &pCred))
    {
        return RetVal(retError, 0, "Could not read secret from Windows Credential Manager.");
    }

    if (pCred && pCred->CredentialBlob && pCred->CredentialBlobSize > 0)
    {
        const int wcharCount = static_cast<int>(pCred->CredentialBlobSize / sizeof(wchar_t));
        valueOut = QString::fromWCharArray(
            reinterpret_cast<const wchar_t*>(pCred->CredentialBlob),
            wcharCount);
    }

    if (pCred)
    {
        CredFree(pCred);
    }

    return RetVal(retOk);
#else
    Q_UNUSED(key)
    valueOut.clear();
    return RetVal(retError, 0, "OS key store backend not implemented for this platform.");
#endif
}

RetVal OsKeyStoreWin::removeSecret(const QString &key)
{
#ifdef _WIN32
    if (key.trimmed().isEmpty())
    {
        return RetVal(retError, 0, "Credential key must not be empty.");
    }

    const QString target = key.trimmed();
    if (!CredDeleteW(reinterpret_cast<LPCWSTR>(target.utf16()), CRED_TYPE_GENERIC, 0))
    {
        return RetVal(retError, 0, "Could not remove secret from Windows Credential Manager.");
    }

    return RetVal(retOk);
#else
    Q_UNUSED(key)
    return RetVal(retError, 0, "OS key store backend not implemented for this platform.");
#endif
}

} // namespace llm
} // namespace ito
