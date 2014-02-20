#include "helpTreeDockWidget.h"

#include <qdebug.h>
#include "../organizer/addInManager.h"
#include <AppManagement.h>
#include <qdesktopservices.h>
#include <qdiriterator.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qpainter.h>
#include <qregexp.h>
#include <qsortfilterproxymodel.h>
#include <qstandarditemmodel.h>
#include <qstringlistmodel.h>
#include <qtconcurrentrun.h>
#include <qtextdocument.h>
#include <qtextstream.h>
#include <QThread>
#include <qtimer.h>
#include <qtreeview.h>
#include <stdio.h>
#include <qclipboard.h>
#include <common/addInInterface.h>

#include "../widgets/helpDockWidget.h"
#include "../models/leafFilterProxyModel.h"
#include "../AppManagement.h"

//----------------------------------------------------------------------------------------------------------------------------------
// on_start
HelpTreeDockWidget::HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock, Qt::WFlags flags)
    : QWidget(parent, flags),
    m_historyIndex(-1),
    m_pMainModel(NULL),
    m_dbPath(qApp->applicationDirPath()+"/help"),
    m_pParent(dock)
{
    ui.setupUi(this);

    connect(AppManagement::getMainApplication(), SIGNAL(propertiesChanged()), this, SLOT(propertiesChanged()));
    //connect(AppManagement::getMainApplication(), SIGNAL(), this, SLOT());

    // Initialize Variables
    m_treeVisible = false;

    connect(&dbLoaderWatcher, SIGNAL(resultReadyAt(int)), this, SLOT(dbLoaderFinished(int)));

    m_pMainFilterModel = new LeafFilterProxyModel(this);
    m_pMainModel = new QStandardItemModel(this);
    m_pMainFilterModel->setFilterCaseSensitivity(Qt::CaseInsensitive);

    //Install Eventfilter
    ui.commandLinkButton->setVisible(false);
    //ui.commandLinkButton->installEventFilter(this);
    ui.treeView->installEventFilter(this);
    ui.textBrowser->installEventFilter(this);

    m_previewMovie = new QMovie(":/application/icons/loader32x32trans.gif", QByteArray(), this);
    ui.lblProcessMovie->setMovie(m_previewMovie);
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

    ui.treeView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui.treeView->setHeaderHidden(true);

    loadIni();
    m_forced = true;
    propertiesChanged();
    //reloadDB();

    QStringList iconAliasesName;
    QList<int> iconAliasesNumb;
    iconAliasesName << "class" << "const" << "routine" << "module" << "package" << "unknown" << "link_unknown" << "link_class" << "link_const" << "link_module" << "link_package" << "link_routine";
    iconAliasesNumb << 04      << 06      << 05        << 03       << 02        << 00        << 11             << 14           << 16           << 13            << 12             << 15;
    int i = 0;
    foreach (const QString &icon, iconAliasesName)
    {
        m_iconGallery[iconAliasesNumb[i]] = QIcon(":/helpTreeDockWidget/"+icon);
        i++;
    }

    m_iconGallery[100] = QIcon(":/helpTreeDockWidget/filter");
    m_iconGallery[101] = QIcon(":/plugins/icons/pluginAlgo.png");
    m_iconGallery[102] = QIcon(":/plugins/icons/pluginFilter.png");
    m_iconGallery[103] = QIcon(":/plugins/icons/window.png");
    //ui.textBrowser->setLineWrapMode(QTextEdit::NoWrap);
}

//----------------------------------------------------------------------------------------------------------------------------------
// GUI-on_close
HelpTreeDockWidget::~HelpTreeDockWidget()
{
    saveIni();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Get The Filters and put them into a node of the Tree
void HelpTreeDockWidget::createFilterWidgetNode(int fOrW, QStandardItemModel* model, const QMap<int,QIcon> *iconGallery)
{
    // Map der Plugin-Namen und Zeiger auf das Node des Plugins
    QMap <QString, QStandardItem*> plugins;

    // Userrole der nodes in die der typ der Node gespeichert wird
    int urType = Qt::UserRole + 2;
    
    // AddInManager einbinden
    ito::AddInManager *aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    QStandardItem *mainNode = new QStandardItem();

    switch(fOrW)
    {
    case 1: //Filter
        {
            const QHash <QString, ito::AddInAlgo::FilterDef *> *filterHashTable = aim->getFilterList();
            // Main Node zusammenbauen
            mainNode->setText("Algorithms");
            mainNode->setData(typeCategory, urType);
            mainNode->setIcon(iconGallery->value(100));
            QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator i = filterHashTable->constBegin();
            while (i != filterHashTable->constEnd()) 
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das Filter-Node anzuhängen
                    QStandardItem *plugin = new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeFPlugin, urType);
                    plugin->setIcon(iconGallery->value(101));
                    plugin->setToolTip(i.value()->m_pBasePlugin->getFilename() + "; v" + QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhängen
                QStandardItem *filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeFilter, urType);
                filter->setIcon(iconGallery->value(102));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem *test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
        }
        break;
    case 2: //Widgets
        {
            const QHash <QString, ito::AddInAlgo::AlgoWidgetDef *> *widgetHashTable = aim->getAlgoWidgetList();
            // Main Node zusammenbauen
            mainNode->setText("Widgets");
            mainNode->setData(typeCategory, urType);
            mainNode->setIcon(iconGallery->value(100));
            QHash<QString, ito::AddInAlgo::AlgoWidgetDef *>::const_iterator i = widgetHashTable->constBegin();
            while (i != widgetHashTable->constEnd()) 
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das Filter-Node anzuhängen
                    QStandardItem *plugin = new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeWPlugin, urType);
                    plugin->setIcon(iconGallery->value(101));
                    plugin->setToolTip(i.value()->m_pBasePlugin->getFilename() + "; v" + QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhängen
                QStandardItem *filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeWidget, urType);
                filter->setIcon(iconGallery->value(103));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem *test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
        }
        break;
    }
    // Einenteil der oberen Blöcke hierher verschieben
    model->insertRow(0, mainNode);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Get the DocString from a Filter and parse is to html
ito::RetVal HelpTreeDockWidget::showFilterWidgetPluginHelp(const QString &filtername, itemType type)
{
    ito::RetVal retval;
    ito::AddInManager *aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const QHash  <QString, ito::AddInAlgo::FilterDef     *> *filterHashTable = aim->getFilterList();
    const QHash  <QString, ito::AddInAlgo::AlgoWidgetDef *> *widgetHashTable = aim->getAlgoWidgetList();
    ui.textBrowser->clear();
    QFile file(":/helpTreeDockWidget/help_style");
    if (file.open(QIODevice::ReadOnly))
    {
        QByteArray cssData = file.readAll();
        ui.textBrowser->document()->addResource(QTextDocument::StyleSheetResource, QUrl("help_style.css"), QString(cssData));
        file.close();
    }

    QString docString = "";

    if (type != 6)
    {
        // Standard html-Template laden
        // -------------------------------------
        QFile templ(":/helpTreeDockWidget/filter_tmpl");
        templ.open(QIODevice::ReadOnly);
        docString = templ.readAll();
        templ.close();
    
        // Prefix als Navigations-Links einfuegen
        // -------------------------------------
        QString linkNav;
        QStringList splittedLink;

        QString filter;
        // Hier wird der Filtername in Groß und Kleinschreibung übernommen
        //filter = fd->m_name;
        // Hier ist der Name halt fälschlicher weise falsch geschrieben
        filter = filtername;
    
    
        switch(type)
        {

            const ito::AddInAlgo::FilterDef *fd = filterHashTable->value(filtername);
            splittedLink.append("Algorithms");
            splittedLink.append(fd->m_pBasePlugin->objectName());
            break;
        }
        splittedLink.append(filter);
        linkNav.insert(0,">>"+splittedLink[splittedLink.length()-1]);
        for (int i = splittedLink.length() - 2; i > -1; i--)
        {
            QString linkPath;
            for (int j = 0; j <= i; j++)
            {
                linkPath.append(splittedLink.mid(0, i + 1)[j] + ".");
            }
            if (linkPath.right(1) == ".")
            {
                linkPath = linkPath.left(linkPath.length() - 1);
            }
            if (type == typeFilter || type == typeFPlugin)
            {
                linkNav.insert(0, ">> <a id=\"HiLink\" href=\"itom://algorithm.html#" + linkPath.toAscii().toPercentEncoding("",".") + "\">" + splittedLink[i] + "</a>");
            }
            if (type == typeWidget || type == typeWPlugin)
            {
                linkNav.insert(0, ">> <a id=\"HiLink\" href=\"itom://widget.html#" + linkPath.toAscii().toPercentEncoding("",".") + "\">" + splittedLink[i] + "</a>");
            }
        }
        docString.replace("%BREADCRUMB%", linkNav);

        // extract ParameterSection
        // -------------------------------------
        QString parameterSection;
        int start = docString.indexOf("<!--%PARAMETERS_START%-->");
        int end = docString.indexOf("<!--%PARAMETERS_END%-->");

        if (start == -1 && end == -1) //no returns section
        {
            parameterSection = "";
        }
        else if (start == -1 || end == -1) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Parameters section is only defined by either the start or end tag.").toAscii().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of parameters section comes before start tag.").toAscii().data());
        }
        else
        {
            parameterSection = docString.mid(start, end + QString("<!--%PARAMETERS_END%-->").size() - start);
            docString.remove(start, end + QString("<!--%PARAMETERS_END%-->").size() - start);
        }

        // extract ReturnSection 
        // -------------------------------------
        //search for <!--%RETURNS_START%--> and <!--%RETURNS_END%-->
        QString returnsSection;
        start = docString.indexOf("<!--%RETURNS_START%-->");
        end = docString.indexOf("<!--%RETURNS_END%-->");

        if (start == -1 && end == -1) //no returns section
        {
            returnsSection = "";
        }
        else if (start == -1 || end == -1) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Returns section is only defined by either the start or end tag.").toAscii().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of returns section comes before start tag.").toAscii().data());
        }
        else
        {
            returnsSection = docString.mid(start, end + QString("<!--%RETURNS_END%-->").size() - start);
            docString.remove(start, end + QString("<!--%RETURNS_END%-->").size() - start);
        }

        // extract ExampleSection 
        // -------------------------------------
        //search for <!--%EXAMPLE_START%--> and <!--%EXAMPLE_END%-->
        QString exampleSection;
        start = docString.indexOf("<!--%EXAMPLE_START%-->");
        end = docString.indexOf("<!--%EXAMPLE_END%-->");

        if (start == -1 && end == -1) //no returns section
        {
            returnsSection = "";
        }
        else if (start == -1 || end == -1) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Returns section is only defined by either the start or end tag.").toAscii().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of returns section comes before start tag.").toAscii().data());
        }
        else
        {
            exampleSection = docString.mid(start, end + QString("<!--%EXAMPLE_END%-->").size() - start);
            docString.remove(start, end + QString("<!--%EXAMPLE_END%-->").size() - start);
        }


        // Build Parameter and return section
        // -------------------------------------
        if (!retval.containsError())
        {
            switch(type)
            {
                case typeFilter: // Filter
                {
                    if (filterHashTable->contains(filter))
                    {
                        const ito::AddInAlgo::FilterDef *fd = filterHashTable->value(filter);
                        const ito::FilterParams *params = aim->getHashedFilterParams(fd->m_paramFunc); 

                        docString.replace("%NAME%", fd->m_name);
                        docString.replace("%INFO%",parseFilterWidgetContent(fd->m_description));
                
                        // Parameter-Section
                        if ((params->paramsMand.size() + params->paramsOpt.size() == 0) && parameterSection.isNull() == false)
                        {   //remove parameters section
                            parameterSection = "";
                        }
                        else if (parameterSection.isNull() == false)
                        {
                            parseParamVector("PARAMMAND", params->paramsMand, parameterSection);
                            parseParamVector("PARAMOPT", params->paramsOpt, parameterSection);
                        }

                        // Return-Section
                        if (params->paramsOut.size() == 0 && returnsSection.isNull() == false)
                        {   //remove returns section
                            returnsSection = "";
                        }
                        else if (returnsSection.isNull() == false)
                        {
                            parseParamVector("OUT", params->paramsOut, returnsSection);
                        }

                        // Example-Section
                        QString newLink = fd->m_name+"(";
                        foreach(const ito::Param &p, params->paramsMand)
                        {
                            QString para = p.getName();
                            newLink.append(para+", ");
                        }
                        newLink.append(")");
                        newLink.replace(", )",")");
                        QByteArray a = newLink.toAscii();

                        exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                        exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                    }
                    else
                    {
                        retval += ito::RetVal(ito::retError, 0, tr("Unknown filter name '%1'").arg(filter).toAscii().data());
                    }

                    // Example-Section
                    
                    QStringList paramList;
                    foreach(const ito::Param &p, params->paramsMand)
                    {
                        paramList.append(p.getName());
                    }
                    QString newLink = QString("filter(\"%1\",%2)").arg(fd->m_name).arg( paramList.join(", ") );
                    QByteArray a = newLink.toLatin1();

                    exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                    exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                }
                else
                {
                    if (widgetHashTable->contains(filter))
                    {
                        const ito::AddInAlgo::AlgoWidgetDef *awd = widgetHashTable->value(filter);
                        const ito::FilterParams *params = aim->getHashedFilterParams(awd->m_paramFunc);   
                
                        docString.replace("%NAME%", awd->m_name);
                        docString.replace("%INFO%",parseFilterWidgetContent(awd->m_description));
                
                        // Parameter-Section
                        if ((params->paramsMand.size() + params->paramsOpt.size() == 0) && parameterSection.isNull() == false)
                        {
                            //remove parameters section
                            parameterSection = "";
                        }
                        else if (parameterSection.isNull() == false)
                        {
                            parseParamVector("PARAMMAND", params->paramsMand, parameterSection);
                            parseParamVector("PARAMOPT", params->paramsOpt, parameterSection);
                        }

                        //remove returns section (Widgets can´t return something)
                        returnsSection = "";

                    // Example-Section
                    QStringList paramList;
                    foreach(const ito::Param &p, params->paramsMand)
                    {
                        paramList.append(p.getName());
                    }
                    QString newLink = QString("filter(\"%1\",%2)").arg(awd->m_name).arg( paramList.join(", ") );
                    QByteArray a = newLink.toLatin1();

                    exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                    exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                }
                else
                {
                const QList<QObject*> *algoPlugins = aim->getAlgList();
                const ito::AddInInterfaceBase *aib = NULL;

                    foreach(const QObject *obj, *algoPlugins)
                    {
                        if (QString::compare(obj->objectName(), filter, Qt::CaseInsensitive) == 0)
                        {
                            aib = static_cast<const ito::AddInInterfaceBase*>(obj);
                            break;
                        }
                    }

                    if (aib)
                    {
                        docString.replace("%NAME%", aib->objectName());
                        docString.replace("%INFO%", parseFilterWidgetContent(aib->getDescription()));

                        parameterSection = "";
                        returnsSection = "";
                        exampleSection = "";

                    }
                    else
                    {
                        retval += ito::RetVal(ito::retError, 0, tr("Unknown algorithm plugin with name '%1'").arg(filter).toAscii().data());
                    }
                    break;
                }
                default:
                    retval += ito::RetVal(ito::retError, 0, tr("unknown type").toAscii().data());
                    break;
            }

            docString.replace("<!--%PARAMETERS_INSERT%-->", parameterSection);
            docString.replace("<!--%RETURNS_INSERT%-->", returnsSection);
            docString.replace("<!--%EXAMPLE_INSERT%-->", exampleSection);
        }
    }
    else
    {
        ui.textBrowser->clear();
        QFile file(":/helpTreeDockWidget/help_style");
        if (file.open(QIODevice::ReadOnly))
        {
            QByteArray cssData = file.readAll();
            file.close();
            ui.textBrowser->document()->addResource(QTextDocument::StyleSheetResource, QUrl("help_style.css"), QString(cssData));          
        }
        if (filtername == "Algorithms")
        {
            QFile file(":/helpTreeDockWidget/algo_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","Algorithms");
                docString = htmlData;
                file.close();
            }
        }
        else if (filtername == "Widgets")
        {
            QFile file(":/helpTreeDockWidget/widg_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","Widgets");
                docString = htmlData;
                file.close();
            }
        }
        else
        {
            // Load dummy Page
        }
    }

    if (!retval.containsError())
    {   // Create html document
        if (m_plaintext)
        {
            ui.textBrowser->document()->setPlainText(docString);
        }
        else
        {
            ui.textBrowser->document()->setHtml(docString);
        }
    }
    return retval;
}

// changes the strings comming from the system
//----------------------------------------------------------------------------------------------------------------------------------
QString HelpTreeDockWidget::parseFilterWidgetContent(const QString &input)
{
    QString output = input;
    output.replace("\n", "<br>");
    return output;
}

// Creates the Parameter and Return section in  html-Code
//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal HelpTreeDockWidget::parseParamVector(const QString &sectionname, const QVector<ito::Param> &paramVector, QString &content)
{
    ito::RetVal retval;
    QString startString = QString("<!--%%1_START%-->").arg(sectionname);
    QString endString = QString("<!--%%1_END%-->").arg(sectionname);
    QString insertString = QString("<!--%%1_INSERT%-->").arg(sectionname);

    //search for <!--%PARAMETERS_START%--> and <!--%PARAMETERS_END%-->
    int start = content.indexOf(startString);
    int end = content.indexOf(endString);

    if (start == -1 && end == -1) //no returns section
    {
        //pass
    }
    else if (start == -1 || end == -1) //one part is missing
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("Template Error: %s section is only defined by either the start or end tag.").toAscii().data(), sectionname.toAscii().data());
    }
    else if (start > end) //one part is missing
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("Template Error: End tag of %s section comes before start tag.").toAscii().data(), sectionname.toAscii().data());
    }
    else
    {
        QString rowContent = content.mid(start, end + endString.size() - start);
        qDebug() << rowContent;
        content.remove(start, end + endString.size() - start);
        qDebug() << content;
        QString internalContent = "";

        foreach(const ito::Param &p, paramVector)
        {
            internalContent.append(parseParam(rowContent, p));
        }

        content.replace(insertString, internalContent);
    }

    return retval;
}

// Parses every single Parameter
//----------------------------------------------------------------------------------------------------------------------------------
QString HelpTreeDockWidget::parseParam(const QString &tmpl, const ito::Param &param)
{
    QString output = tmpl;
    QString name = param.getName();
    QString info = param.getInfo() ? parseFilterWidgetContent(param.getInfo()) : "";
    QString meta;
    
    QString type;

    switch(param.getType())
    {
    case ito::ParamBase::Int:
        {
            type = "integer";
            if (param.getMeta() != NULL)
            {
                const ito::IntMeta *pMeta = dynamic_cast<const ito::IntMeta*>(param.getMeta());
                meta = tr("Range: min: %1, max: %2").arg(pMeta->getMin()).arg(pMeta->getMax());
            }
        }
        break;
    case ito::ParamBase::Char:
        { // Never tested ... no filter holding metadata as char available
            type = "char";
            if (param.getMeta() != NULL)
            {
                const ito::CharMeta *pMeta = dynamic_cast<const ito::CharMeta*>(param.getMeta());
                meta = tr("Range: min: %1, max: %2").arg(pMeta->getMin()).arg(pMeta->getMax());
            }
        }
        break;
    case ito::ParamBase::Double:
        {
            type = "double";
            if (param.getMeta() != NULL)
            {
                const ito::DoubleMeta *pMeta = dynamic_cast<const ito::DoubleMeta*>(param.getMeta());
                meta = tr("Range: min: %1, max: %2").arg(pMeta->getMin()).arg(pMeta->getMax());
            }
        }
        break;
    case ito::ParamBase::String:
        {
            type = "string";
            if (param.getMeta() != NULL)
            {
                const ito::StringMeta *pMeta = dynamic_cast<const ito::StringMeta*>(param.getMeta());
                QString str = pMeta->getString();
                QString len = QString::number(pMeta->getLen());
                meta = "Default: "+str+"; max. Length: "+len;
            }
        }
        break;
    case ito::ParamBase::CharArray & ito::paramTypeMask:
        {
            type = "char-list";
        }
        break;
    case ito::ParamBase::IntArray & ito::paramTypeMask:
        {
            type = "integer-list";
        }
        break;
    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
        {
            type = "double-list";
        }
        break;
    case ito::ParamBase::DObjPtr & ito::paramTypeMask:
        {
            type = "dataObject";
        }
        break;
    case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
        {
            type = "pointCloud";
        }
        break;
    case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
        {
            type = "polygonMesh";
        }
        break;
    case ito::ParamBase::HWRef & ito::paramTypeMask:
        {
            type = "hardware";
            if (param.getMeta() != NULL)
            {
                const ito::HWMeta *pMeta = dynamic_cast<const ito::HWMeta*>(param.getMeta());
                QString name;
                if (name != "")
                {
                    name = "Only "+QString(pMeta->getHWAddInName())+" is allowed.";
                }
                else
                {
                    if (pMeta->getMinType() & ito::typeDataIO)
                        meta.append("DataIO, ");
                    if (pMeta->getMinType() & ito::typeActuator)
                        meta.append("Actuator, ");
                    if (pMeta->getMinType() & ito::typeAlgo)
                        meta.append("Algo, ");
                    if (pMeta->getMinType() & ito::typeGrabber)
                        meta.append("Grabber, ");
                    if (pMeta->getMinType() & ito::typeADDA)
                        meta.append("ADDA, ");
                    if (pMeta->getMinType() & ito::typeRawIO)
                        meta.append("RawIO, ");

                    meta.insert(0,"Plugin of type: ");
                    meta.append(" are allowed.");
                }
            }
        }
        break;
    }

    output.replace("%PARAMNAME%", name);
    output.replace("%PARAMTYPE%", type);
    output.replace("%PARAMINFO%", info);
    output.replace("%PARAMMETA%", meta);
    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Filter the events for showing and hiding the treeview
bool HelpTreeDockWidget::eventFilter(QObject *obj, QEvent *event)
{
    // = qobject_cast<ito::AbstractDockWidget*>(parent());

    if (obj == ui.commandLinkButton && event->type() == QEvent::Enter)
    {
        showTreeview();
    }
    else if (obj == ui.treeView && event->type() == QEvent::Enter)
    {    
        if (m_pParent && !m_pParent->isFloating())
        {
            showTreeview();
        }
    }
    else if (obj == ui.textBrowser && event->type() == QEvent::Enter)
    {
        if (m_pParent && !m_pParent->isFloating())
        {
            unshowTreeview();
            return true;
        }    
    }
    return QObject::eventFilter(obj, event);
 }

//----------------------------------------------------------------------------------------------------------------------------------
// Save Gui positions to Main-ini-File
void HelpTreeDockWidget::saveIni()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    settings.setValue("percWidthVi", m_percWidthVi);
    settings.setValue("percWidthUn", m_percWidthUn);
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load Gui positions to Main-ini-File
void HelpTreeDockWidget::loadIni()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    m_percWidthVi = settings.value("percWidthVi", "50").toDouble();
    m_percWidthUn = settings.value("percWidthUn", "50").toDouble();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load SQL-DatabasesList in m_ Variable when properties changed
void HelpTreeDockWidget::propertiesChanged()
{ // Load the new list of DBs with checkstates from the INI-File
    
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    // Read the other Options
    m_openLinks = settings.value("OpenExtLinks", true).toBool();
    m_plaintext = settings.value("Plaintext", false).toBool();
    m_showSelection.Filters = settings.value("ShowFilters", true).toBool();
    m_showSelection.Widgets = settings.value("ShowWidgets", true).toBool();
    m_showSelection.Modules = settings.value("ShowModules", true).toBool();

    // if the setting of the loaded DBs has changed:
    // This setting exists only from the time when the property dialog was open till this routine is done!
    if (settings.value("reLoadDBs", false).toBool() | m_forced)
    {
        // Read the List
        m_includedDBs.clear();
        int size = settings.beginReadArray("Databases");
        for (int i = 0; i < size; ++i)
        {
            settings.setArrayIndex(i);
            QString dbName = settings.value("DB", QString()).toString();
            if (dbName.startsWith("$"))    
            {// This was checked and will be used
                dbName.remove(0,2);
                //Add to m_pMainlist
                m_includedDBs.append(dbName);
            }
        }
        settings.endArray();
        reloadDB();
    }
    settings.remove("reLoadDBs");
    settings.endGroup();
    m_forced = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Build Tree - Bekommt das Model, das zuletzt erstellte Item und eine Liste mit dem Pfad
/*static*/ void HelpTreeDockWidget::createItemRek(QStandardItemModel* model, QStandardItem& parent, const QString parentPath, QList<SqlItem> &items, const QMap<int,QIcon> *iconGallery)
{
    SqlItem firstItem;
    int urPath = Qt::UserRole + 1;
    int urType = Qt::UserRole + 2;

    while(items.count() > 0)
    {
        firstItem = items[0];
        //splitt = firstItem.split(':');

        if (firstItem.prefix == parentPath) //first item is direct child of parent
        {    
            items.removeFirst();
            QStandardItem *node = new QStandardItem(firstItem.name);
            if (firstItem.type > 11) //splitt[0].startsWith(1))
            {
                // diese Zeile könnte man auch durch Code ersetzen der das Link Icon automatisch zeichnet... das waere flexibler
                node->setIcon(iconGallery->value(firstItem.type));
            }
            else
            { // Kein Link Normales Bild
                node->setIcon((*iconGallery)[firstItem.type]); //Don't load icons here from file since operations on QPixmap are not allowed in another thread
            }
            node->setEditable(false);
            node->setData(firstItem.path, urPath);
            node->setData(1, urType);
            node->setToolTip(firstItem.path);
            createItemRek(model, *node, firstItem.path, items, iconGallery);
            parent.appendRow(node);
        }
        else if (firstItem.prefix.indexOf(parentPath) == 0) //parentPath is the first part of path
        {
            items.removeFirst();
            int li = firstItem.prefix.lastIndexOf(".");
            QStandardItem *node = new QStandardItem(firstItem.prefix.mid(li+1));
            if (firstItem.type > 11) // Siehe 19 Zeilen vorher
            { //ist ein Link (vielleicht wie oben Icon dynamisch zeichnen lassen
                node->setIcon(iconGallery->value(firstItem.type));
            }
            else
            { // Kein Link Normales Bild
                node->setIcon(iconGallery->value(firstItem.type));
            }
            node->setEditable(false);
            node->setData(firstItem.prefix, urPath); 
            node->setData(1, urType); //typ 1 = docstring wird aus sql gelesen
            createItemRek(model, *node, firstItem.prefix, items, iconGallery);  
            parent.appendRow(node);
        }
        else
        {
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Get Data from SQL File and store it in a table
/*static*/ ito::RetVal HelpTreeDockWidget::readSQL(/*QList<QSqlDatabase> &DBList,*/ const QString &filter, const QString &file, QList<SqlItem> &items)
{
    ito::RetVal retval = ito::retOk;
    QFile f(file);
    SqlItem item;
  
    if (f.exists())
    {
        QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE",file); //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
        database.setDatabaseName(file);
        bool ok = database.open();
        if (ok)
        {
            //QSqlQuery query("SELECT type, prefix, prefixL, name FROM itomCTL ORDER BY prefix", database);
            QSqlQuery query("SELECT type, prefix, name FROM itomCTL ORDER BY prefix", database);
            query.exec();
            while (query.next())
            {
                item.type = query.value(0).toInt();
                item.path = query.value(1).toString();
                int li = query.value(1).toString().lastIndexOf(".");
                if (li >= 0)
                {
                    item.prefix = query.value(1).toString().left(li);
                    item.name = query.value(1).toString().mid(li+1);
                }
                else
                {
                    item.prefix = "";
                    item.name = query.value(1).toString();
                }

                items.append(item);
            }
        }
        else
        {
            retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be opened").toAscii().data(), file.toAscii().data());
        }
        database.close();
    }
    else
    {
        retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be found").toAscii().data(), file.toAscii().data());
    }    
    QSqlDatabase::removeDatabase(file);
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Reload Database and clear search-edit and start the new Thread
void HelpTreeDockWidget::reloadDB()
{
    //Create and Display Mainmodel
    m_pMainModel->clear();
    ui.treeView->reset();
    
    m_pMainFilterModel->setSourceModel(NULL);
    m_previewMovie->start();
    ui.lblProcessMovie->setVisible(true);
    ui.lblProcessText->setVisible(true);
    ui.treeView->setVisible(false);
    ui.splitter->setVisible(false);
    ui.lblProcessText->setText(tr("Help database is loading..."));


    // THREAD START QtConcurrent::run
    QFuture<ito::RetVal> f1 = QtConcurrent::run(loadDBinThread, m_dbPath, m_includedDBs, m_pMainModel/*, m_pDBList*/, &m_iconGallery, m_showSelection);
    dbLoaderWatcher.setFuture(f1);
    //f1.waitForFinished();
    // THREAD END  
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpTreeDockWidget::dbLoaderFinished(int /*index*/)
{
    ito::RetVal retval = dbLoaderWatcher.future().resultAt(0);

    m_pMainFilterModel->setSourceModel(m_pMainModel);

    //model has been 
    ui.treeView->setModel(m_pMainFilterModel);

    //after setModel, the corresponding selectionModel is changed, too
    connect(ui.treeView->selectionModel(), SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(selectedItemChanged(const QModelIndex &, const QModelIndex &)));

    m_previewMovie->stop();
    ui.lblProcessMovie->setVisible(false);

    if ((m_includedDBs.size() > 0 && m_showSelection.Modules) | m_showSelection.Filters | m_showSelection.Widgets)
    {
        ui.lblProcessText->setVisible(false);
        ui.treeView->setVisible(true);
        ui.splitter->setVisible(true);
    }
    else
    {
        ui.lblProcessText->setVisible(true);
        ui.treeView->setVisible(false);
        ui.splitter->setVisible(false);
        ui.lblProcessText->setText(tr("No help database available! \n go to Properties File -> General -> Helpviewer and check the selection"));
    }

    ui.treeView->resizeColumnToContents(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Load the Database in different Thread
/*static*/ ito::RetVal HelpTreeDockWidget::loadDBinThread(const QString &path, const QStringList &includedDBs, QStandardItemModel *mainModel, const QMap<int,QIcon> *iconGallery, const DisplayBool &show)
{
    QList<SqlItem> sqlList;
    ito::RetVal retval;
    if (show.Modules)
    {
        for (int i = 0; i < includedDBs.length(); i++)
        {
            sqlList.clear();
            QString temp;
            temp = path+'/'+includedDBs.at(i);
            retval = readSQL(/*DBList,*/ "", temp, sqlList);
            QCoreApplication::processEvents();
            if (!retval.containsWarningOrError())
            {
                createItemRek(mainModel, *(mainModel->invisibleRootItem()), "", sqlList, iconGallery);
            }
            else
            {/* The Database named: m_pIncludedDBs[i] is not available anymore!!! show Error*/}
        }
    }

    if (show.Filters)
    {
        createFilterWidgetNode(1, mainModel, iconGallery);
    }

    if (show.Widgets)
    {
        createFilterWidgetNode(2, mainModel, iconGallery);
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Highlight (parse) the Helptext to make it nice and readable for non docutils Docstrings
// ERROR decides whether it´s already formatted by docutils (Error = 0) or it must be parsed by this function (Error != 0)
ito::RetVal HelpTreeDockWidget::highlightContent(const QString &prefix, const QString &name, const QString &param, const QString &shortDesc, const QString &helpText, const QString &error, QTextDocument *document)
{
    QString errorS = error.left(error.indexOf(" ", 0));
    int errorCode = errorS.toInt();
    QStringList errorList;

    /*********************************/
    // Allgemeine HTML sachen anfügen /
    /*********************************/ 
    QString rawContent = helpText;
    QString html = "<html><head>"
                   "<link rel='stylesheet' type='text/css' href='itom_help_style.css'>"
                   "</head><body>%1"
                   "</body></html>";
    // Set Label.Text above Textbrowser
    // -------------------------------------
    if (errorCode == 1)
    { // not needed anymore (as long as the Documentation is right)
        ui.label->setText("Parser: ITO-Parser");  
    }
    else if (errorCode == 0)
    {
        ui.label->setText("Parser: docutils");
    }
    else if (errorCode == -1)
    {
        ui.label->setText("Parser: No help-text available");
    }

    // Insert Shortdescription
    // -------------------------------------
    if (shortDesc != "")
        rawContent.insert(0,shortDesc+"");

    // Überschrift (Funktionsname) einfuegen
    // -------------------------------------
    rawContent.insert(0,"<h1 id=\"FunctionName\">"+name+param+"</h1>"+"");

    // Prefix als Navigations-Links einfuegen
    // -------------------------------------
    QStringList splittedLink = prefix.split(".");
    rawContent.insert(0, ">>" + splittedLink[splittedLink.length() - 1]);
    for (int i = splittedLink.length() - 2; i > -1; i--)
    {
        QString linkPath;
        for (int j = 0; j <= i; j++)
            linkPath.append(splittedLink.mid(0, i + 1)[j] + ".");
        if (linkPath.right(1) == ".")
            linkPath = linkPath.left(linkPath.length() - 1);
        rawContent.insert(0, ">> <a id=\"HiLink\" href=\"itom://" + linkPath + "\">" + splittedLink[i] + "</a>");
    }

    // Insert docstring
    // -------------------------------------
    if (m_plaintext)
    {   // Only for debug reasons! Displays the Plaintext instead of the html
        rawContent.replace("<br/>","<br/>\n");
        document->setPlainText(html.arg(rawContent));
    }
    else
    {
        QFile file(":/helpTreeDockWidget/help_style");
        if (file.open(QIODevice::ReadOnly))
        {
            QByteArray cssData = file.readAll();
            document->addResource(QTextDocument::StyleSheetResource, QUrl("itom_help_style.css"), QString(cssData));
            file.close();
        }
        // Remake "See Also"-Section so that the links work
        // -------------------------------------
        // Alte "See Also" Section kopieren
        QRegExp seeAlso("(<div class=\"seealso\">).*(</div>)");
        seeAlso.setMinimal(true);
        seeAlso.indexIn(rawContent);
        QString oldSec = seeAlso.capturedTexts()[0];

        if (oldSec == "") //there are version, where the see-also section is an admonition
        {
            seeAlso.setPattern("(<div class=\"admonition-see-also seealso\">).*(</div>)");
            seeAlso.indexIn(rawContent);
            oldSec = seeAlso.capturedTexts()[0];
        }

        // Extract Links (names) from old Section
        QRegExp links("`(.*)`");
        links.setMinimal(true);
        int offset = 0;
        QStringList texts;
        while (links.indexIn(oldSec, offset) > -1)
        {
            texts.append(links.capturedTexts()[1]);
            offset = links.pos()+links.matchedLength();
        }

        // Build the new Section with Headings, Links, etc
        QString newSection = "<p class=\"rubric\">See Also</p><p>";
        for (int i = 0; i < texts.length(); i++)
        {
            newSection.append("\n<a id=\"HiLink\" href=\"itom://" + prefix.left(prefix.lastIndexOf('.')) + "." + texts[i] + "\">" + texts[i].remove('`') + "</a>, ");
        }
        newSection = newSection.left(newSection.length() - 2);
        newSection.append("\n</p>");

        // Exchange old Section against new one
        rawContent.remove(seeAlso.pos(), seeAlso.matchedLength());
        rawContent.insert(seeAlso.pos(), newSection);

        document->setHtml(html.arg(rawContent));
        
        //dummy output (write last loaded Plaintext into html-File)
        QFile file2("helpOutput.html");
        file2.open(QIODevice::WriteOnly);
        file2.write(html.arg(rawContent).toAscii());
        file2.close();
    }
    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Display the Help-Text
ito::RetVal HelpTreeDockWidget::displayHelp(const QString &path, const int newpage)
{ 
    ito::RetVal retval = ito::retOk;

    ui.textBrowser->clear();
    bool ok = false;
    bool found = false;

    // Das ist ein kleiner workaround mit dem if 5 Zeilen später. Man könnt euahc direkt über die includeddbs list iterieren
    // dann wäre folgende Zeile hinfällig
    QDirIterator it(m_dbPath, QStringList("*.db"), QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

    while(it.hasNext() && !found)
    {
        QString temp = it.next();
        if (m_includedDBs.contains(temp.right(temp.length() - m_dbPath.length() - 1)))
        {
            QFile file(temp);
        
            if (file.exists())
            {
                { //important to have variables database and query in local scope such that removeDatabase (outside of this scope) can securly free all resources! -> see docs about removeDatabase
                    // display the help: Run through all the files in the directory
                    QSqlDatabase database = QSqlDatabase::addDatabase("QSQLITE", temp);
                    database.setDatabaseName(temp);
                    ok = database.open();
                    if (ok)
                    {
                        QSqlQuery query("SELECT type, prefix, name, param, sdesc, doc, htmlERROR  FROM itomCTL WHERE LOWER(prefix) IS '" + path.toUtf8().toLower() + "'", database);
                        query.exec();
                        found = query.next();
                        if (found)
                        {
                            QByteArray docCompressed = query.value(5).toByteArray();
                            QString doc;
                            if (docCompressed.size() > 0)
                            {
                                doc = qUncompress(docCompressed);
                            }

                            highlightContent(query.value(1).toString(), query.value(2).toString(), query.value(3).toString(), query.value(4).toString(), doc, query.value(6).toString(), ui.textBrowser->document());

                            if (newpage == 1)
                            {
                                m_historyIndex++;
                                m_history.insert(m_historyIndex, path.toUtf8());
                                for (int i = m_history.length(); i > m_historyIndex; i--)
                                {
                                    m_history.removeAt(i);
                                }
                            }
                        }
                        database.close();
                    }
                }
                QSqlDatabase::removeDatabase(temp);
            }
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// finds a Modelindex belonging to an Itemname
QModelIndex HelpTreeDockWidget::findIndexByName(const QString &modelName)
{
    QStringList path = modelName.split('.');
    QStandardItem *current = m_pMainModel->invisibleRootItem();
    QStandardItem *temp;
    int counts;
    QString tempString;
    QString firstPath;
    bool found;

    while (path.length() > 0)
    {
        firstPath = path.takeFirst();
        counts = current->rowCount();
        found = false;

        for (int j = 0; j < counts; ++j)
        {
            temp = current->child(j,0);
            tempString = temp->data().toString();

            if (tempString.endsWith(firstPath) && tempString.split(".").last() == firstPath) //fast implementation, first compare will mostly fail, therefore the split is only executed few times
            {
                current = temp;
                found = true;
                break;
            }

            if (!found)
            {
                return QModelIndex(); //nothing found
            }
        }
    }

    return current->index();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Filter the mainmodel
void HelpTreeDockWidget::liveFilter(const QString &filterText)
{
    showTreeview();
    m_pMainFilterModel->setFilterRegExp(filterText);
    expandTree();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Returns a list containing the protocol[0] and the real link[1]
// prot|||....link.....        
QStringList HelpTreeDockWidget::separateLink(const QUrl &link)
{
    QStringList result;
    QByteArray examplePrefix = "example:";
    QString t = link.toString();

    if (link.scheme() == "itom")
    {
        if (link.host() == "widget.html")
        {
            result.append("widget");
            result.append(link.fragment());
        }
        else if (link.host() == "algorithm.html")
        {
            result.append("algorithm");
            result.append(link.fragment());
        }
        else
        {
            result.append("itom");
            result.append(link.host());
        }
    }
    else if (link.scheme() == "mailto")
    {
        result.append("mailto");
        result.append(link.path());
    }
    else if (link.scheme() == "example")
    {
        result.append("example");
        result.append(link.fragment());
    }
    else
    {
        result.append("-1");
    }
    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
// This is the Slot that can be externally called by other widgets to display filter or widget help ... i.a. AIManagerWidget
void HelpTreeDockWidget::showPluginInfo(QString name, int type)
{
    switch(type)
    {
        case 1:
        {
            displayHelp(name , 1);
            break;
        }
        case 2:
        {
            showFilterWidgetPluginHelp(name, typeFilter);
            break;
        }
        case 3:
        {
            showFilterWidgetPluginHelp(name, typeWidget);
            break;
        }
        case 4:
        {
            showFilterWidgetPluginHelp(name, typeFPlugin);
            break;
        }
        case 5:
        {
            showFilterWidgetPluginHelp(name, typeWPlugin);
            break;
        }
        case 6:
        {
            showFilterWidgetPluginHelp(name, typeCategory);
            break;
        }
    }
}

/*************************************************************/
/*****************GUI related methods*************************/
/*************************************************************/

//----------------------------------------------------------------------------------------------------------------------------------
// Expand all TreeNodes
void HelpTreeDockWidget::expandTree()
{
    ui.treeView->expandAll();
    ui.treeView->resizeColumnToContents(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse all TreeNodes
void HelpTreeDockWidget::collapseTree()
{
    ui.treeView->collapseAll();
    ui.treeView->resizeColumnToContents(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Link inside Textbrowser is clicked
void HelpTreeDockWidget::on_textBrowser_anchorClicked(const QUrl & link)
{
    QString t = link.toString();
    QStringList parts = separateLink(link);

    if (parts.size() < 2) return;

    if (parts[1].toLower() == "algorithms")
    {
        showFilterWidgetPluginHelp("Algorithms", typeCategory);
    }
    else if (parts[1].toLower() == "widgets")
    {
        showFilterWidgetPluginHelp("Widgets", typeCategory);
    }
    else if (parts[0] == "itom")
    {//Internal ItomLink
        displayHelp(parts[1], 1);
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(parts[1]));
        //ui.treeView->setCurrentIndex(filteredIndex);
    }
    else if (parts[0] == "algorithm")
    {//Filter
        showFilterWidgetPluginHelp(parts[1].split(".").last(), typeFPlugin);
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(parts[1]));
        //ui.treeView->setCurrentIndex(filteredIndex);
    }
    else if (parts[0] == "widget")
    {//Filter
        showFilterWidgetPluginHelp(parts[1].split(".").last(), typeWPlugin);
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(parts[1]));
        //ui.treeView->setCurrentIndex(filteredIndex);
    }
    else if (parts[0] == "http")
    {//WebLink
        QDesktopServices::openUrl(link);
    }
    else if (parts[0] == "mailto")
    {//MailTo-Link
        QDesktopServices::openUrl(parts[1]);
    }
    else if (parts[0] == "example")
    {//Copy an example to Clipboard
        QClipboard *clip = QApplication::clipboard();
        clip->setText(parts[1], QClipboard::Clipboard);
    }
    else if (parts[0] == "-1")
    {
        ui.label->setText(tr("invalid Link"));
    }
    else
    {
        ui.label->setText(tr("unknown protocol"));
        QMessageBox msgBox;
        msgBox.setText(tr("The protocol of the link is unknown. "));
        msgBox.setInformativeText(tr("Do you want to try with the external browser?"));
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.setDefaultButton(QMessageBox::Yes);
        int ret = msgBox.exec();
        switch (ret) 
        {
            case QMessageBox::Yes:
                QDesktopServices::openUrl(link);
            case QMessageBox::No:
                break;
        }
    }    
}

//----------------------------------------------------------------------------------------------------------------------------------
// Saves the position of the splitter depending on the use of the tree or the textbox
void HelpTreeDockWidget::on_splitter_splitterMoved (int pos, int index)
{
    double width = ui.splitter->width();
    if (m_treeVisible == true)
    {
        m_percWidthVi = pos / width * 100;
    }
    else
    {
        m_percWidthUn = pos / width * 100;
    }
    if (m_percWidthVi < m_percWidthUn)
    {
        m_percWidthVi = m_percWidthUn + 10;
    }
    if (m_percWidthVi == 0)
    {
        m_percWidthVi = 30;
    }
    // Verhaltnis testweise anzeigen lassen
    //ui.label->setText(QString("vi %1 un %2").arg(percWidthVi).arg(percWidthUn));
}

//----------------------------------------------------------------------------------------------------------------------------------
// Show the Help in the right Memo
void HelpTreeDockWidget::selectedItemChanged(const QModelIndex &current, const QModelIndex &/*previous*/)
{
    int urPath = Qt::UserRole + 1;
    int urType = Qt::UserRole + 2;
    int type = current.data(urType).toInt();
    QString t = current.data(urPath).toString();
    if (type == 1) 
    {
        showPluginInfo(current.data(urPath).toString(), type);
    }
    else
    {
        showPluginInfo(current.data(0).toString(), type);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Back-Button
void HelpTreeDockWidget::navigateBackwards()
{
    if (m_historyIndex > 0)
    {
        m_historyIndex--;
        displayHelp(m_history.at(m_historyIndex), 0);

        // Highlight the entry in the tree
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(m_history.at(m_historyIndex)));
        ui.treeView->setCurrentIndex(filteredIndex);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Forward-Button
void HelpTreeDockWidget::navigateForwards()
{
    if (m_historyIndex < m_history.length()-1)
    {
        m_historyIndex++;
        displayHelp(m_history.at(m_historyIndex), 0);

        // Highlight the entry in the tree
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(findIndexByName(m_history.at(m_historyIndex)));
        ui.treeView->setCurrentIndex(filteredIndex);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Show tree
void HelpTreeDockWidget::showTreeview()
{
    m_treeVisible = true;
    QList<int> intList;
    intList  <<  ui.splitter->width()*m_percWidthVi/100  <<  ui.splitter->width() * (100 - m_percWidthVi) / 100;
    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Hide tree
void HelpTreeDockWidget::unshowTreeview()
{
    m_treeVisible = false;
    QList<int> intList;
    intList  <<  ui.splitter->width()*m_percWidthUn/100  <<  ui.splitter->width() * (100 - m_percWidthUn) / 100;
    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Expand Tree
void HelpTreeDockWidget::on_treeView_expanded(const QModelIndex &index)
{
    ui.treeView->resizeColumnToContents(0);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse Tree
void HelpTreeDockWidget::on_treeView_collapsed(const QModelIndex &index)
{
    ui.treeView->resizeColumnToContents(0);
}
