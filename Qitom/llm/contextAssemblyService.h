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

#include "llmTypes.h"

#include <QStringList>

namespace ito
{
namespace llm
{

struct ContextFileContent
{
    QString filePath;
    QString content;
};

struct ContextAssemblyInput
{
    QString selectedText;
    QString activeFilePath;
    QString activeFileContent;
    QList<ContextFileContent> additionalFiles;
};

struct ContextEnvelope
{
    QString selectedText;
    QString activeFilePath;
    QString activeFileContent;
    QList<ContextFileContent> additionalFiles;
    QStringList referencedFilePaths;
    int estimatedTokenCount = 0;
    QString truncationReport;
};

class ContextAssemblyService
{
    public:
        ContextEnvelope build(const ContextAssemblyInput &input) const;

    private:
        QString truncateText(const QString &text, int maxChars, bool &truncated) const;
        int estimateTokenCount(const ContextEnvelope &envelope) const;
};

} // namespace llm
} // namespace ito
