/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef ABSTRACTITOMDESIGNERPLUGIN_H
#define ABSTRACTITOMDESIGNERPLUGIN_H

#include "../common/sharedStructuresGraphics.h"
    
namespace ito {

    class AbstractItomDesignerPlugin : public QObject
    {
        Q_OBJECT

        public:
            AbstractItomDesignerPlugin(QObject *parent = NULL) : QObject(parent), m_plotFeatures(ito::Static) {}
            ~AbstractItomDesignerPlugin() {}

            inline ito::PlotDataTypes getPlotDataTypes(void) const { return m_plotDataTypes; }
            inline ito::PlotDataFormats getPlotDataFormats(void) const { return m_plotDataFormats; }
            inline ito::PlotFeatures getPlotFeatures(void) const { return m_plotFeatures; }

            inline void setItomSettingsFile(const QString &settingsFile) { m_itomSettingsFile = settingsFile; }

        protected:
            ito::PlotDataTypes   m_plotDataTypes;
            ito::PlotDataFormats m_plotDataFormats;
            ito::PlotFeatures    m_plotFeatures;

            QString m_itomSettingsFile;

        signals:

        public slots:

        private slots:

    };
} // namepsace ito

#endif // ABSTRACTITOMDESIGNERPLUGIN_H
