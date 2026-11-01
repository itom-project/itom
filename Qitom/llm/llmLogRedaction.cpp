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

#include "llmLogRedaction.h"

#include <QRegularExpression>

namespace ito
{
namespace llm
{

QString LlmLogRedaction::redactBearerToken(const QString &text)
{
    QString result(text);
    result.replace(
        QRegularExpression("(Authorization\\s*:\\s*Bearer\\s+)[^\\s\"']+", QRegularExpression::CaseInsensitiveOption),
        "\\1***");
    return result;
}

QString LlmLogRedaction::redactApiKey(const QString &text)
{
    QString result(text);
    result.replace(
        QRegularExpression("(api[_-]?key\\s*[:=]\\s*)[^\\s\"']+", QRegularExpression::CaseInsensitiveOption),
        "\\1***");
    return result;
}

QString LlmLogRedaction::redactBasicAuth(const QString &text)
{
    QString result(text);
    result.replace(
        QRegularExpression("(Authorization\\s*:\\s*Basic\\s+)[^\\s\"']+", QRegularExpression::CaseInsensitiveOption),
        "\\1***");
    return result;
}

QString LlmLogRedaction::redactGenericSecrets(const QString &text)
{
    QString result(text);
    result.replace(
        QRegularExpression("((token|secret|password)\\s*[:=]\\s*)[^\\s\"']+", QRegularExpression::CaseInsensitiveOption),
        "\\1***");
    return result;
}

QString LlmLogRedaction::redactAll(const QString &text)
{
    return redactGenericSecrets(redactBasicAuth(redactApiKey(redactBearerToken(text))));
}

} // namespace llm
} // namespace ito
