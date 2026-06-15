/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#include "abstractFilterDialog.h"

#include "../../DataObject/dataobj.h"

#if ITOM_POINTCLOUDLIBRARY > 0
    #include "../../PointCloud/pclStructures.h"
#endif

namespace ito {

AbstractFilterDialog::AbstractFilterDialog(QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QWidget *parent) :
    QDialog(parent),
    m_autoMand(autoMand),
    m_autoOut(autoOut)
{
}

QList<QTreeWidgetItem*> AbstractFilterDialog::renderAutoMandAndOutResult() const
{
    QList<QTreeWidgetItem*> items;
    foreach( const ito::ParamBase &p, m_autoMand )
    {
        if(p.getFlags() & ito::ParamBase::Out)
        {
            items << renderParam(p);
        }
    }

    foreach( const ito::ParamBase &p, m_autoOut )
    {
        if(p.getFlags() & ito::ParamBase::Out)
        {
            items << renderParam(p);
        }
    }

    return items;
}

QTreeWidgetItem* AbstractFilterDialog::renderParam( const ito::ParamBase &p ) const
{
    QTreeWidgetItem *root = new QTreeWidgetItem();
    QTreeWidgetItem *item;
    QStringList l;

    switch(p.getType())
    {
    case ito::ParamBase::Int:
        {
            root->setData(0, Qt::DisplayRole, "Integer");
            root->setData(1, Qt::DisplayRole, p.getVal<int>() );
        }
        break;

    case ito::ParamBase::Char:
        {
            root->setData(0, Qt::DisplayRole, "Char");
            root->setData(1, Qt::DisplayRole, p.getVal<char>() );
        }
        break;

    case ito::ParamBase::Double:
        {
            root->setData(0, Qt::DisplayRole, "Double");
            root->setData(1, Qt::DisplayRole, p.getVal<double>() );
        }
        break;

    case ito::ParamBase::String:
        {
            root->setData(0, Qt::DisplayRole, "String");
            if( p.getVal<char*>() )
            {
                root->setData(1, Qt::DisplayRole, p.getVal<const char*>() );
            }
            else
            {
                root->setData(1, Qt::DisplayRole, "<NULL>");
            }
        }
        break;

    case ito::ParamBase::CharArray:
        {
            root->setData(0, Qt::DisplayRole, "Char-Array");
            l.clear();
            l << tr("size") << QString::number(p.getLen());
            item = new QTreeWidgetItem( l );
            root->addChild( item );
        }
        break;

    case ito::ParamBase::IntArray:
        {
            root->setData(0, Qt::DisplayRole, "Int-Array");
            l.clear();
            l << tr("size") << QString::number(p.getLen());
            item = new QTreeWidgetItem( l );
            root->addChild( item );
        }
        break;

    case ito::ParamBase::DoubleArray:
        {
            root->setData(0, Qt::DisplayRole, "Double-Array");
            l.clear();
            l << tr("size") << QString::number(p.getLen());
            item = new QTreeWidgetItem( l );
            root->addChild( item );
        }
        break;

    case ito::ParamBase::DObjPtr:
        {

            static const char* types[] = { "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32", "Float32", "Float64", "Complex64", "Complex128", "Rgba32" };

            root->setData(0, Qt::DisplayRole, "DataObject");
            ito::DataObject *dObj = (ito::DataObject*)(p.getVal<void*>());
            if(dObj)
            {
                l.clear();
                l << tr("dims") << QString::number(dObj->getDims());
                item = new QTreeWidgetItem( l );
                root->addChild( item );
                l.clear();
#ifdef _DEBUG
                int maxTypeNum = 10; //number of items in types[] list - 1
                Q_ASSERT(dObj->getType() >= 0 && dObj->getType() <= maxTypeNum);
#endif
                l << tr("type") << types[dObj->getType()];
                item = new QTreeWidgetItem( l );
                root->addChild( item );

                l.clear();
                QString size = "<Empty>";
                for(int i=0;i<dObj->getDims();i++)
                {
                    if(size == "<Empty>")
                    {
                        size = QString::number(dObj->getSize(i));
                    }
                    else
                    {
                        size += "x" + QString::number(dObj->getSize(i));
                    }
                }
                l << tr("size") << size;
                item = new QTreeWidgetItem( l );
                root->addChild( item );
            }
            else
            {
                root->setData(1, Qt::DisplayRole, "<NULL>");
            }


        }
        break;


    case ito::ParamBase::PointCloudPtr:
        {
#if ITOM_POINTCLOUDLIBRARY > 0
            ito::PCLPointCloud *pointCloud = (ito::PCLPointCloud*)(p.getVal<void*>());
            root->setData(0, Qt::DisplayRole, "PointCloud");
            l.clear();
            l << tr("size") << QString::number(pointCloud->size());
            item = new QTreeWidgetItem( l );
            root->addChild( item );
#else
            root->setData(0, Qt::DisplayRole, "PointCloudLibrary not available.");
#endif
        }
        break;


    case ito::ParamBase::PolygonMeshPtr:
        {
            //ito::PCLPolygonMesh *polygonMesh = (ito::PCLPolygonMesh*)(p.getVal<void*>());
#if ITOM_POINTCLOUDLIBRARY > 0
            root->setData(0, Qt::DisplayRole, "PolygonMesh");
#else
            root->setData(0, Qt::DisplayRole, "PointCloudLibrary not available.");
#endif
        }
        break;

    default:
        {
            root->setData(0, Qt::DisplayRole, "Unknown");
        }
    }

    return root;
}




} //end namespace ito
