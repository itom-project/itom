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
        
        //! the classinfo ito.AbstractItomDesignerPlugin is the interface number of AbstractItomDesignerPlugin.
        //  increment this number if you changed something in this interface or other abstract classes of the
        //  plot designerPlugin system.
        Q_CLASSINFO("ito.AbstractItomDesignerPlugin", "1.0.0")

        public:
            AbstractItomDesignerPlugin(QObject *parent = NULL) : 
                QObject(parent), 
                m_plotFeatures(ito::Static), 
                m_version(0),
                m_author(""),
                m_description(""),
                m_detaildescription(""),
                m_aboutThis(""),
                m_license("LGPL with ITO itom-exception") {}

            ~AbstractItomDesignerPlugin() {}

            inline ito::PlotDataTypes getPlotDataTypes(void) const { return m_plotDataTypes; }
            inline ito::PlotDataFormats getPlotDataFormats(void) const { return m_plotDataFormats; }
            inline ito::PlotFeatures getPlotFeatures(void) const { return m_plotFeatures; }

            //! returns addIn version
            inline int getVersion(void) const { return m_version; }
            //! returns plugin author
            const QString getAuthor(void) const { return m_author; }
            //! returns a brief description of the plugin
            const QString getDescription(void) const { return m_description; }
            //! returns a detailed description of the plugin
            const QString getDetailDescription(void) const { return m_detaildescription; }
            //! returns a detailed description of the plugin license
            const QString getLicenseInfo(void) const { return m_license; }
            //! returns a detailed description of the plugin compile informations
            const QString getAboutInfo(void) const { return m_aboutThis; }

            inline void setItomSettingsFile(const QString &settingsFile) { m_itomSettingsFile = settingsFile; }

        protected:
            ito::PlotDataTypes   m_plotDataTypes;
            ito::PlotDataFormats m_plotDataFormats;
            ito::PlotFeatures    m_plotFeatures;
            
            int m_version;                        //!< plugin version
            QString m_author;                     //!< the plugin author
            QString m_description;                //!< a brief descrition of the plugin
            QString m_detaildescription;          //!< a detail descrition of the plugin
            QString m_license;                    //!< a short license string for the plugin, default value is "LGPL with ITO itom-exception"
            QString m_aboutThis;                  //!< a short string with compile informations
            QString m_itomSettingsFile;

        signals:

        public slots:

        private slots:

    };
} // namepsace ito

#endif // ABSTRACTITOMDESIGNERPLUGIN_H
