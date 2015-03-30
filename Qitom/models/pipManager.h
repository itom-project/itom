/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
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

#ifndef PIPMANAGER_H
#define PIPMANAGER_H

#include <qabstractitemmodel.h>
#include <qprocess.h>

namespace ito 
{

struct PythonPackage
{
    enum Status {Unknown, Uptodate, Outdated};
    PythonPackage() {};
    PythonPackage(const QString &name, const QString &version, const QString &location, const QString &requires) : 
        m_name(name), m_version(version), m_location(location), m_requires(requires), m_status(Unknown), m_newVersion("") 
    {}
    QString m_name;
    QString m_version;
    QString m_location;
    QString m_requires;
    Status  m_status;
    QString m_newVersion;
};


class PipManager : public QAbstractItemModel
{
    Q_OBJECT

    public:
        PipManager(QObject *parent = 0);
        ~PipManager();

        enum pipModelIndex
        {
            idxName = 0,
            idxVersion = 1,
            idxLocation = 2,
            idxRequires = 3,
            idxStatus = 4
        };

        QVariant data(const QModelIndex &index, int role) const;
        QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
        QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
        QModelIndex parent(const QModelIndex &index) const;
        int rowCount(const QModelIndex &parent = QModelIndex()) const;
        int columnCount(const QModelIndex &parent = QModelIndex()) const;

        void startPipProcess();
        bool isPipStarted() const;


    private:
        QList<QString> m_headers;               //!<  string list of names of column headers
        QList<QVariant> m_alignment;            //!<  list of alignments for the corresponding headers
        QList<PythonPackage> m_pythonPackages;  //!<  list with installed python packages
        QProcess m_pipProcess;

    signals:
        void pipManagerBusy();
        void output(const QString &text, bool isError);
};
}

#endif //PIPMANAGER_H
