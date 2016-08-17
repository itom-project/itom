#include "helpTreeDockWidget.h"

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

#if QT_VERSION >= 0x050000
#include <QtConcurrent/qtconcurrentrun.h>
#else
#include <qtconcurrentrun.h>
#endif

#include <qtextdocument.h>
#include <qtextstream.h>
#include <QThread>
#include <qtimer.h>
#include <qtreeview.h>
#include <stdio.h>
#include <qclipboard.h>
#include <qsettings.h>
#include <common/addInInterface.h>
#include <QtSql/qsqldatabase.h>
#include <QtSql/qsqlquery.h>

#include "../widgets/helpDockWidget.h"
#include "../models/leafFilterProxyModel.h"
#include "../AppManagement.h"

namespace ito
{

//----------------------------------------------------------------------------------------------------------------------------------
// Constructor
HelpTreeDockWidget::HelpTreeDockWidget(QWidget *parent, ito::AbstractDockWidget *dock, Qt::WindowFlags flags)
    : QWidget(parent, flags),
    m_historyIndex(-1),
    m_pMainModel(NULL),
    m_dbPath(qApp->applicationDirPath()+"/help"),
    m_pParent(dock),
    m_internalCall(false),
    m_doingExpandAll(false)
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
    ui.helpTreeContent->installEventFilter(this);

    m_previewMovie = new QMovie(":/application/icons/loader32x32trans.gif", QByteArray(), this);
    ui.lblProcessMovie->setMovie(m_previewMovie);
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

    ui.treeView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui.treeView->setHeaderHidden(true);

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

    m_iconGallery[iconFilter] = QIcon(":/helpTreeDockWidget/filter");
    m_iconGallery[iconPluginAlgo] = QIcon(":/plugins/icons/pluginAlgo.png");
    m_iconGallery[iconPluginFilter] = QIcon(":/plugins/icons/pluginFilter.png");
    m_iconGallery[iconWidget] = QIcon(":/plugins/icons/window.png");
    m_iconGallery[iconPluginDataIO] = QIcon(":/helpTreeDockWidget/dataIO");
    m_iconGallery[iconPluginGrabber] = QIcon(":/helpTreeDockWidget/pluginGrabber");
    m_iconGallery[iconPluginAdda] = QIcon(":/helpTreeDockWidget/pluginAdda");
    m_iconGallery[iconPluginRawIO] = QIcon(":/helpTreeDockWidget/pluginRawIO");
    m_iconGallery[iconPluginActuator] = QIcon(":/helpTreeDockWidget/pluginActuator");
    //ui.helpTreeContent->setLineWrapMode(QTextEdit::NoWrap);

    loadIni();
    m_forced = true;
    propertiesChanged();
    //reloadDB();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Destructor
HelpTreeDockWidget::~HelpTreeDockWidget()
{
    saveIni();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Get The Filters and put them into a node of the Tree
/*! 

    \param fOrW
    \param model
    \param iconGallery
*/
void HelpTreeDockWidget::createFilterWidgetNode(int fOrW, QStandardItemModel* model, const QMap<int,QIcon> *iconGallery)
{
    // Map der Plugin-Namen und Zeiger auf das Node des Plugins
    QMap <QString, QStandardItem*> plugins;

    // AddInManager einbinden
    ito::AddInManager *aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    QStandardItem *mainNode = new QStandardItem();
    mainNode->setEditable(false);
	QString mainNodeText = "";

    switch(fOrW)
    {
    case 1: //Filter
        {
            const QHash <QString, ito::AddInAlgo::FilterDef *> *filterHashTable = aim->getFilterList();
            // build Main Node
			mainNodeText = "Algorithms";
            mainNode->setText(tr("Algorithms"));
            mainNode->setData(typeCategory, m_urType);
			mainNode->setData(mainNodeText, m_urPath);
            mainNode->setIcon(iconGallery->value(iconPluginAlgo));
            QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator i = filterHashTable->constBegin();
            while (i != filterHashTable->constEnd()) 
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das Filter-Node anzuhaengen
                    QStandardItem *plugin = new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeFPlugin, m_urType);
					plugin->setData(mainNodeText + "." + plugin->text(), m_urPath);
                    plugin->setIcon(iconGallery->value(iconPluginAlgo));
                    plugin->setToolTip(i.value()->m_pBasePlugin->getFilename() + "; v" + QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhaengen
                QStandardItem *filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeFilter, m_urType);
				filter->setData(mainNodeText + "." + i.value()->m_pBasePlugin->objectName() + "." + filter->text(), m_urPath);
                filter->setIcon(iconGallery->value(iconPluginFilter));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem *test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
            break;
        }
    case 2: //Widgets
        {
            const QHash <QString, ito::AddInAlgo::AlgoWidgetDef *> *widgetHashTable = aim->getAlgoWidgetList();
            // Main Node zusammenbauen
			mainNodeText = "Widgets";
            mainNode->setText(tr("Widgets"));
            mainNode->setData(typeCategory, m_urType);
			mainNode->setData(mainNodeText, m_urPath);
            mainNode->setIcon(iconGallery->value(iconWidget));
            QHash<QString, ito::AddInAlgo::AlgoWidgetDef *>::const_iterator i = widgetHashTable->constBegin();
            while (i != widgetHashTable->constEnd()) 
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das Filter-Node anzuhaengen
                    QStandardItem *plugin = new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeWPlugin, m_urType);
					plugin->setData(mainNodeText + "." + plugin->text(), m_urPath);
                    plugin->setIcon(iconGallery->value(iconPluginAlgo));
                    plugin->setToolTip(i.value()->m_pBasePlugin->getFilename() + "; v" + QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhaengen
                QStandardItem *filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeWidget, m_urType);
				filter->setData(mainNodeText + "." + i.value()->m_pBasePlugin->objectName() + "." + filter->text(), m_urPath);
                filter->setIcon(iconGallery->value(iconWidget));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem *test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
            break;
        }
    case 3: //DataIO
        {
            // Main Node zusammenbauen
			mainNodeText = "DataIO";
            mainNode->setText(tr("DataIO"));
            mainNode->setData(typeCategory, m_urType);
			mainNode->setData(mainNodeText, m_urPath);
            mainNode->setIcon(iconGallery->value(iconPluginDataIO));

            // Subcategory Node "Grabber"
            QStandardItem *pluginGrabber = new QStandardItem(tr("Grabber"));
            pluginGrabber->setEditable(false);
            pluginGrabber->setData(typeCategory, m_urType);
			pluginGrabber->setData(mainNodeText + ".Grabber", m_urPath);
            pluginGrabber->setIcon(iconGallery->value(iconPluginGrabber));
            
            // Subcategory Node "ADDA"
            QStandardItem *pluginAdda = new QStandardItem(tr("ADDA"));
            pluginAdda->setEditable(false);
            pluginAdda->setData(typeCategory, m_urType);
			pluginAdda->setData(mainNodeText + ".ADDA", m_urPath);
            pluginAdda->setIcon(iconGallery->value(iconPluginAdda));
            
            // Subcategory Node "Raw IO"
            QStandardItem *pluginRawIO = new QStandardItem(tr("Raw IO"));
            pluginRawIO->setEditable(false);
            pluginRawIO->setData(typeCategory, m_urType);
			pluginRawIO->setData(mainNodeText + ".Raw IO", m_urPath);
            pluginRawIO->setIcon(iconGallery->value(iconPluginRawIO));

            const QList<QObject*> *dataIOList = aim->getDataIOList();
            for(int i = 0; i < dataIOList->length(); i++)
            {
                QObject *obj = dataIOList->at(i);
                const ito::AddInInterfaceBase *aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
                if (aib != NULL)
                {
                    QStandardItem *plugin = new QStandardItem(aib->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeDataIO, m_urType);
                    switch (aib->getType())
                    {
                        case 129:
                        {// Grabber
                            plugin->setIcon(iconGallery->value(iconPluginGrabber));
                            plugin->setData(pluginGrabber->data(m_urPath).toString() + "."+plugin->text(), m_urPath);
                            pluginGrabber->appendRow(plugin);
                            break;
                        }
                        case 257:
                        {// ADDA
                            plugin->setIcon(iconGallery->value(iconPluginAdda));
                            plugin->setData(pluginAdda->data(m_urPath).toString() + "."+plugin->text(), m_urPath);
                            pluginAdda->appendRow(plugin);
                            break;
                        }
                        case 513:
                        {// Raw IO
                            plugin->setIcon(iconGallery->value(iconPluginRawIO));
                            plugin->setData(pluginRawIO->data(m_urPath).toString() + "."+plugin->text(), m_urPath);
                            pluginRawIO->appendRow(plugin);
                            break;
                        }
                    }                   
                }
            }

            mainNode->appendRow(pluginGrabber);
            mainNode->appendRow(pluginAdda);
            mainNode->appendRow(pluginRawIO);
            break;
        }
    case 4: //Actuator
        { 
            // Main Node zusammenbauen
			mainNodeText = "Actuator";
            mainNode->setText(tr("Actuator"));
            mainNode->setData(typeCategory, m_urType);
			mainNode->setData(mainNodeText, m_urPath);
            mainNode->setIcon(iconGallery->value(iconPluginActuator));
            const QList<QObject*> *ActuatorList = aim->getActList();
            for(int i = 0; i < ActuatorList->length(); i++)
            {
                QObject *obj = ActuatorList->at(i);
                const ito::AddInInterfaceBase *aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
                if (aib != NULL)
                {
                    QStandardItem *plugin = new QStandardItem(aib->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeActuator, m_urType);
					plugin->setData(mainNodeText + "." + plugin->text(), m_urPath);
                    plugin->setIcon(iconGallery->value(iconPluginActuator));
                    mainNode->appendRow(plugin);             
                }
            }
            break;
        }
    }
    // MainNode an Model anhaengen
    model->insertRow(0, mainNode);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Get the DocString from a Filter and parse is to html
/*! This function puts all information of a Widget or Plugin together and builds the html help text.

    \param filterpath path with all parents
    \param type the enumeration itemType is defined in the header file helpTreeDockWidget.h
    \return ito::RetVal
*/
ito::RetVal HelpTreeDockWidget::showFilterWidgetPluginHelp(const QString &filterpath, itemType type)
{
    ito::RetVal retval;
    ito::AddInManager *aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const QHash  <QString, ito::AddInAlgo::FilterDef     *> *filterHashTable = aim->getFilterList();
    const QHash  <QString, ito::AddInAlgo::AlgoWidgetDef *> *widgetHashTable = aim->getAlgoWidgetList();
    ui.helpTreeContent->clear();
    QFile file(":/helpTreeDockWidget/help_style");
    if (file.open(QIODevice::ReadOnly))
    {
        QByteArray cssData = file.readAll();
        ui.helpTreeContent->document()->addResource(QTextDocument::StyleSheetResource, QUrl("help_style.css"), QString(cssData));
        file.close();
    }

    QString docString = "";
    QString filter = filterpath.split(".").last();
    
    // needed for breadcrumb and for list of children in algorithms
    QString linkNav;

    if (type != typeCategory)
    {
        // Standard html-Template laden
        // -------------------------------------
        QFile templ(":/helpTreeDockWidget/filter_tmpl");
        templ.open(QIODevice::ReadOnly);
        docString = templ.readAll();
        templ.close();
    
        // Breadcrumb Navigation zusammenstellen
        // -------------------------------------
        QStringList splittedLink = filterpath.split(".");
        QString linkPath = filterpath;
        linkNav.insert(0, ">> " + splittedLink[splittedLink.length() - 1]);
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
            linkNav.insert(0, ">> <a id=\"HiLink\" href=\"itom://algorithm.html#" + linkPath.toLatin1().toPercentEncoding("",".") + "\">" + splittedLink[i] + "</a>");
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
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Parameters section is only defined by either the start or end tag.").toLatin1().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of parameters section comes before start tag.").toLatin1().data());
        }
        else
        {
            parameterSection = docString.mid(start, end + QString("<!--%PARAMETERS_END%-->").size() - start);
            parameterSection.replace("<!--%PARAMETERS_CAPTION%-->", tr("Parameters")); 
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
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Returns section is only defined by either the start or end tag.").toLatin1().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of returns section comes before start tag.").toLatin1().data());
        }
        else
        {
            returnsSection = docString.mid(start, end + QString("<!--%RETURNS_END%-->").size() - start);
            returnsSection.replace("<!--%RETURNS_CAPTION%-->", tr("Returns")); 
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
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: Returns section is only defined by either the start or end tag.").toLatin1().data());
        }
        else if (start > end) //one part is missing
        {
            retval += ito::RetVal(ito::retError, 0, tr("Template Error: End tag of returns section comes before start tag.").toLatin1().data());
        }
        else
        {
            exampleSection = docString.mid(start, end + QString("<!--%EXAMPLE_END%-->").size() - start);
            exampleSection.replace("<!--%EXAMPLE_CAPTION%-->", tr("Example"));
            exampleSection.replace("<!--%EXAMPLELINK_CAPTION%-->", tr("Copy example to clipboard"));
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
                    const ito::AddInAlgo::FilterDef *fd = filterHashTable->value(filter);
                    if (filterHashTable->contains(filter))
                    {
                        const ito::FilterParams *params = aim->getHashedFilterParams(fd->m_paramFunc); 

                        docString.replace("%NAME%", fd->m_name);
                        docString.replace("%INFO%", parseFilterWidgetContent(fd->m_description));
                
                        // Parameter-Section
                        if ((params->paramsMand.size() + params->paramsOpt.size() == 0) && parameterSection.isNull() == false)
                        {   //remove parameters section
                            parameterSection = "";
                        }
                        else if (parameterSection.isNull() == false)
                        {
                            parseParamVector("PARAMMAND", params->paramsMand, parameterSection);
                            parseParamVector("PARAMOPT", params->paramsOpt, parameterSection);
                            parameterSection.replace("<!--%PARAMOPT_CAPTION%-->", tr("optional"));
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
                        QStringList paramList;
                        foreach(const ito::Param &p, params->paramsMand)
                        {
                            paramList.append(QLatin1String(p.getName()));
                        }

                        QString returnString;

                        if (params->paramsOut.size() == 1)
                        {

                            returnString = QString(QLatin1String(params->paramsOut[0].getName())) + " = ";
                        }
                        else if (params->paramsOut.size() > 1)
                        {
                            returnString = "[";
                            QStringList returnList;
                            foreach(const ito::Param &p, params->paramsOut)
                            {
                                returnList.append(QLatin1String(p.getName()));
                            }
                            returnString += returnList.join(", ") + "] = ";
                        }

                        QString newLink = QString("%1filter(\"%2\",%3)").arg(returnString).arg(fd->m_name).arg( paramList.join(", ") );
                        newLink.replace(",)",")");
                        QByteArray a = newLink.toLatin1();

                        exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                        exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                    }
                    else
                    {
                    retval += ito::RetVal(ito::retError, 0, tr("Unknown filter name '%1'").arg(filter).toLatin1().data());
                    }

                    break;
                }
                case typeWidget:
                {
                    const ito::AddInAlgo::AlgoWidgetDef *awd = widgetHashTable->value(filter);
                    if (widgetHashTable->contains(filter))
                    {
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
                            parameterSection.replace("<!--%PARAMOPT_CAPTION%-->", tr("optional"));
                        }

                        //remove returns section (Widgets can´t return something)
                        returnsSection = "";

                        // Example-Section
                        QStringList paramList;
                        foreach(const ito::Param &p, params->paramsMand)
                        {
                            paramList.append(QLatin1String(p.getName()));
                        }
                        QString newLink = QString("ui.createNewPluginWidget(\"%1\",%2)").arg(awd->m_name).arg( paramList.join(", ") );
                        newLink.replace(",)", ")");
                        QByteArray a = newLink.toLatin1();

                        exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                        exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                    }

                    break;
                }
                case typeFPlugin:  // These two lines behave
                case typeWPlugin:  // like an "or" statement
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

                        QString extendedInfo;
                        
                        if (aib->getDescription() != "")
                        {
                            extendedInfo.insert(0, parseFilterWidgetContent(aib->getDescription()));
                            if (aib->getDetailDescription() != "")
                            {
                                extendedInfo.append("<br>");
                            }
                        }
                        if (aib->getDetailDescription() != "")
                        {
                            extendedInfo.append(parseFilterWidgetContent(aib->getDetailDescription()));
                        }

                        if (filterHashTable->size() > 0)
                        {
                            extendedInfo.append("<p class=\"rubric\">This plugin contains the following algorithms:</p>");

                            QHash<QString, ito::AddInAlgo::FilterDef *>::const_iterator i = filterHashTable->constBegin();
                            while (i != filterHashTable->constEnd())
                            {
                                if (aib->objectName() == i.value()->m_pBasePlugin->objectName())
                                {
                                    QString link = "." + i.value()->m_pBasePlugin->objectName() + "." + i.value()->m_name;
                                    extendedInfo.append("<a id=\"HiLink\" href=\"itom://algorithm.html#Algorithms" + link.toLatin1().toPercentEncoding("", ".") + "\">" + i.value()->m_name.toLatin1().toPercentEncoding("", ".") + "</a><br><br>");
                                }
                                ++i;
                            }
                        }

                        if (widgetHashTable->size() > 0)
                        {
                            extendedInfo.append("<p class=\"rubric\">This plugin contains the following widgets:</p>");

                            QHash<QString, ito::AddInAlgo::AlgoWidgetDef *>::const_iterator i = widgetHashTable->constBegin();
                            while (i != widgetHashTable->constEnd())
                            {
                                if (aib->objectName() == i.value()->m_pBasePlugin->objectName())
                                {
                                    QString link = "." + i.value()->m_pBasePlugin->objectName() + "." + i.value()->m_name;
                                    extendedInfo.append("<a id=\"HiLink\" href=\"itom://algorithm.html#Widgets" + link.toLatin1().toPercentEncoding("", ".") + "\">" + i.value()->m_name.toLatin1().toPercentEncoding("", ".") + "</a><br><br>");
                                }
                                ++i;
                            }
                        }

                        docString.replace("%INFO%", extendedInfo);

                        parameterSection = "";
                        returnsSection = "";
                        exampleSection = "";
                    }
                    else
                    {
                    retval += ito::RetVal(ito::retError, 0, tr("Unknown algorithm plugin with name '%1'").arg(filter).toLatin1().data());
                    }

                    break;
                }
                case typeDataIO:
                case typeActuator:
                {
                    QObject *obj;
                    // Lookup the clicked name in the corresponding List
                    if (type == typeActuator)
                    {
                        const QList<QObject*> *ActuatorList = aim->getActList();
                        for(int i = 0; i < ActuatorList->length(); i++)
                        {
                            QString listFilter = ActuatorList->at(i)->objectName();
                            if (listFilter == filter)
                            {
                                obj = ActuatorList->at(i);
                                break;
                            }
                        }
                    }
                    else /*if (type == typeDataIO)*/
                    {
                        const QList<QObject*> *DataIOList = aim->getDataIOList();
                        for(int i = 0; i < DataIOList->length(); i++)
                        {
                            QString listFilter = DataIOList->at(i)->objectName();
                            if (listFilter == filter)
                            {
                                obj = DataIOList->at(i);
                                break;
                            }
                        }
                    }

                    if (obj != NULL)
                    {
                        const ito::AddInInterfaceBase *aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
                        if (aib != NULL)
                        {
                            docString.replace("%NAME%", aib->objectName());        // TODO: should return desc, but returns sdesc
							QString desc = aib->getDescription();
							QString detaileddesc = aib->getDetailDescription();
							if (detaileddesc != NULL)
							{
								desc.append("<br><br>");
								desc.append(detaileddesc);
							}
                            docString.replace("%INFO%", parseFilterWidgetContent(desc));
                
                            // Parameter-Section
                            const QVector<ito::Param> *paramsMand = (qobject_cast<ito::AddInInterfaceBase *>(obj))->getInitParamsMand();
                            const QVector<ito::Param> *paramsOpt = (qobject_cast<ito::AddInInterfaceBase *>(obj))->getInitParamsOpt();
                            if ((paramsMand->size() + paramsOpt->size() == 0) && parameterSection.isNull() == false)
                            {
                                //remove parameters section
                                parameterSection = "";

                            }
                            else if (parameterSection.isNull() == false)
                            {
                                parseParamVector("PARAMMAND", *paramsMand, parameterSection);
                                parseParamVector("PARAMOPT" , *paramsOpt, parameterSection);
                                parameterSection.replace("<!--%PARAMOPT_CAPTION%-->", tr("optional"));
                            }

                            //remove returns section (Widgets can´t return something)
                            returnsSection = "";

                            // Example-Section
                            QStringList paramList;
                            for (int i = 0; i < paramsMand->size(); i++)
                            {
                                const ito::Param &p = paramsMand->at(i);
                                paramList.append(QLatin1String(p.getName()));
                            }

                            QString callName;

                            if (type == typeDataIO)
                            {
                                callName = "dataIO";
                            }
                            else
                            {
                                callName = "actuator";
                            }

                            QString newLink = QString("%1(\"%2\",%3)").arg(callName).arg(aib->objectName()).arg( paramList.join(", ") );
                            newLink.replace(",)",")");
                            QByteArray a = newLink.toLatin1();

                            exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                            exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());
                        }
                    }
                    else
                    {

                    }

                    break;
                }
                default:
                {
                    retval += ito::RetVal(ito::retError, 0, tr("unknown type").toLatin1().data());
                    break;
                }
            }
            docString.replace("<!--%PARAMETERS_INSERT%-->", parameterSection);
            docString.replace("<!--%RETURNS_INSERT%-->", returnsSection);
            docString.replace("<!--%EXAMPLE_INSERT%-->", exampleSection);
        }
    }
    else
    {
        ui.helpTreeContent->clear();
        QFile file(":/helpTreeDockWidget/help_style");
        if (file.open(QIODevice::ReadOnly))
        {
            QByteArray cssData = file.readAll();
            file.close();
            ui.helpTreeContent->document()->addResource(QTextDocument::StyleSheetResource, QUrl("help_style.css"), QString(cssData));
        }

        if (filter == "Algorithms")
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
        else if (filter == "Widgets")
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
        else if (filter == "DataIO")
        {
            QFile file(":/helpTreeDockWidget/dataIO_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","DataIO");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == "Grabber")
        {
            QFile file(":/helpTreeDockWidget/dataGr_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","Grabber");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == "ADDA")
        {
            QFile file(":/helpTreeDockWidget/dataAD_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","ADDA");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == "Raw IO")
        {
            QFile file(":/helpTreeDockWidget/dataRa_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","Raw IO");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == "Actuator")
        {
            QFile file(":/helpTreeDockWidget/actuator_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%","Actuator");
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
            ui.helpTreeContent->document()->setPlainText(docString);
        }
        else
        {
            ui.helpTreeContent->document()->setHtml(docString);
        }
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Reformats all help strings that come from the widgets and plugins
/*! All newLine characters are replaced by the html tag <br> 

    \param input The text that is supposed to be reformated
    \return QString contains the reformated text
*/
QString HelpTreeDockWidget::parseFilterWidgetContent(const QString &input)
{
#if QT_VERSION < 0x050000
    QString output = Qt::escape(input);
#else
    QString output = input.toHtmlEscaped();
#endif
    output.replace("\n", "<br>");
    output.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;");
    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Creates the Parameter- and Return- sections in  html-Code
/*! 

    \param sectionname
    \param paramVector
    \param content
    \return RetVal
*/
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
        retval += ito::RetVal::format(ito::retError, 0, tr("Template Error: %s section is only defined by either the start or end tag.").toLatin1().data(), sectionname.toLatin1().data());
    }
    else if (start > end) //one part is missing
    {
        retval += ito::RetVal::format(ito::retError, 0, tr("Template Error: End tag of %s section comes before start tag.").toLatin1().data(), sectionname.toLatin1().data());
    }
    else
    {
        QString rowContent = content.mid(start, end + endString.size() - start);
        content.remove(start, end + endString.size() - start);
        QString internalContent = "";

        foreach(const ito::Param &p, paramVector)
        {
            internalContent.append(parseParam(rowContent, p));
        }

        content.replace(insertString, internalContent);
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Parses a single Parameter to html code (called by parseParamVector)
/*! 

    \param tmpl
    \param param
    \return QString
*/
QString HelpTreeDockWidget::parseParam(const QString &tmpl, const ito::Param &param)
{
    QString output = tmpl;
    QString name = QLatin1String(param.getName());
    QString info = param.getInfo() ? QLatin1String(param.getInfo()) : QLatin1String("");
    QString meta;
    
    QString type;

    switch(param.getType())
    {
    case ito::ParamBase::Int:
        {
            type = "integer";
            const ito::IntMeta *pMeta = dynamic_cast<const ito::IntMeta*>(param.getMeta());
            if (pMeta)
            {
                if (pMeta->getStepSize() == 1)
                {
                    meta = tr("Range: [%1,%2], Default: %3").arg(minText(pMeta->getMin())).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<ito::int32>()));
                }
                else
                {
                    meta = tr("Range: [%1:%2:%3], Default: %4").arg(minText(pMeta->getMin())).arg(pMeta->getStepSize()).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<ito::int32>()));
                }
            }
            else
            {
                meta = tr("Default: %1").arg(minmaxText(param.getVal<ito::int32>()));
            }
        }
        break;
    case ito::ParamBase::Char:
        { // Never tested ... no filter holding metadata as char available
            type = "char";
            const ito::CharMeta *pMeta = dynamic_cast<const ito::CharMeta*>(param.getMeta());
            if (pMeta)
            {
                if (pMeta->getStepSize() == 1)
                {
                    meta = tr("Range: [%1,%2], Default: %3").arg(minText(pMeta->getMin())).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<char>()));
                }
                else
                {
                    meta = tr("Range: [%1:%2:%3], Default: %4").arg(minText(pMeta->getMin())).arg(pMeta->getStepSize()).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<char>()));
                }
            }
            else
            {
                meta = tr("Default: %1").arg(minmaxText(param.getVal<char>()));
            }
        }
        break;
    case ito::ParamBase::Double:
        {
            type = "double";
            const ito::DoubleMeta *pMeta = dynamic_cast<const ito::DoubleMeta*>(param.getMeta());
            if (pMeta)
            {
                if (pMeta->getStepSize() == 0.0)
                {
                    meta = tr("Range: [%1,%2], Default: %3").arg(minText(pMeta->getMin())).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<ito::float64>()));
                }
                else
                {
                    meta = tr("Range: [%1:%2:%3], Default: %4").arg(minText(pMeta->getMin())).arg(pMeta->getStepSize()).arg(maxText(pMeta->getMax())).arg(minmaxText(param.getVal<ito::float64>()));
                }
            }
            else
            {
                meta = tr("Default: %1").arg(minmaxText(param.getVal<ito::float64>()));
            }
        }
        break;
    case ito::ParamBase::Complex:
        {
            type = "complex";
            ito::float64 real = param.getVal<ito::complex128>().real();
            ito::float64 imag = param.getVal<ito::complex128>().imag();
            if (imag >= 0)
            {
                meta = tr("Default: %1+%2i").arg(real).arg(imag);
            }
            else
            {
                meta = tr("Default: %1-%2i").arg(real).arg(-imag);
            }
        }
        break;
    case ito::ParamBase::String:
        {
            type = "string";
            const ito::StringMeta *pMeta = dynamic_cast<const ito::StringMeta*>(param.getMeta());

            if (pMeta)
            {
                switch (pMeta->getStringType())
                {
                case ito::StringMeta::RegExp:
                    if (pMeta->getLen() == 1)
                    {
                        meta = tr("RegExp: '%1'").arg(QLatin1String(pMeta->getString(0)));
                    }
                    else if (pMeta->getLen() > 1)
                    {
                        QStringList allowed;
                        for (int i = 0; i < pMeta->getLen(); ++i)
                        {
                            allowed += QString("'%1'").arg(QLatin1String(pMeta->getString(i)));
                        }
                        meta = tr("RegExp: [%1]").arg(allowed.join("; "));
                    }
                    else if (pMeta->getLen() == 0)
                    {
                        meta = tr("RegExp: <no pattern given>");
                    }
                    break;
                case ito::StringMeta::String:
                    if (pMeta->getLen() == 1)
                    {
                        meta = tr("Match: '%1'").arg(QLatin1String(pMeta->getString(0)));
                    }
                    else if (pMeta->getLen() > 1)
                    {
                        QStringList allowed;
                        for (int i = 0; i < pMeta->getLen(); ++i)
                        {
                            allowed += QString("'%1'").arg(QLatin1String(pMeta->getString(i)));
                        }
                        meta = tr("Match: [%1]").arg(allowed.join("; "));
                    }
                    else if (pMeta->getLen() == 0)
                    {
                        meta = tr("Match: <no pattern given>");
                    }
                    break;
                case ito::StringMeta::Wildcard:
                    if (pMeta->getLen() == 1)
                    {
                        meta = tr("Wildcard: '%1'").arg(QLatin1String(pMeta->getString(0)));
                    }
                    else if (pMeta->getLen() > 1)
                    {
                        QStringList allowed;
                        for (int i = 0; i < pMeta->getLen(); ++i)
                        {
                            allowed += QString("'%1'").arg(QLatin1String(pMeta->getString(i)));
                        }
                        meta = tr("Wildcard: [%1]").arg(allowed.join("; "));
                    }
                    else if (pMeta->getLen() == 0)
                    {
                        meta = tr("Wildcard: <no pattern given>");
                    }

                    break;
                }
            }
        }
        break;
    case ito::ParamBase::CharArray & ito::paramTypeMask:
        {
            type = "list of characters";
            if (param.getMeta() && param.getMeta()->getType() == ito::ParamMeta::rttiCharArrayMeta)
            {
                ito::CharArrayMeta *m = (ito::CharArrayMeta*)(param.getMeta());
            }
        }

        break;
    case ito::ParamBase::IntArray & ito::paramTypeMask:
        {
            const ito::ParamMeta *m = param.getMeta();
            
            if ((m && m->getType() == ito::ParamMeta::rttiIntArrayMeta) || !m)
            {
                type = "list of integers";
            }
            else if (m && m->getType() == ito::ParamMeta::rttiIntervalMeta)
            {
                type = "interval [first, last] (integers)";
            }
            else if (m && m->getType() == ito::ParamMeta::rttiRangeMeta)
            {
                type = "range [first, last] (integers)";
            }
            else if (m && m->getType() == ito::ParamMeta::rttiRectMeta)
            {
                type = "rectangle [left, top, width, height] (integers)";
            }
        }

        break;
    case ito::ParamBase::DoubleArray & ito::paramTypeMask:
        {
            const ito::ParamMeta *m = param.getMeta();

            if ((m && m->getType() == ito::ParamMeta::rttiDoubleArrayMeta) || !m)
            {
                type = "list of float64";
            }
            else if (m && m->getType() == ito::ParamMeta::rttiDoubleIntervalMeta)
            {
                type = "interval [first, last] (float64)";
            }
        }

        break;
    case ito::ParamBase::ComplexArray & ito::paramTypeMask:
        {
            type = "list of complex128";
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
                ito::ByteArray name = pMeta->getHWAddInName();
                if (name.length() > 0)
                {
                    meta = "Only plugin '"+QString(name.data())+"' is allowed.";
                }
                else
                {
                    meta = "Plugin of type: ";
                    
                    if (pMeta->getMinType() & ito::typeActuator)
                    {
                        meta.append("Actuator, ");
                    }
                    if (pMeta->getMinType() & ito::typeAlgo)
                    {
                        meta.append("Algo, ");
                    }
                    if (pMeta->getMinType() & ito::typeGrabber)
                    {
                        meta.append("DataIO (Grabber), ");
                    }
                    else if (pMeta->getMinType() & ito::typeADDA)
                    {
                        meta.append("DataIO (ADDA), ");
                    }
                    else if (pMeta->getMinType() & ito::typeRawIO)
                    {
                        meta.append("DataIO (RawIO), ");
                    }
                    else if (pMeta->getMinType() & ito::typeDataIO)
                    {
                        meta.append("DataIO, ");
                    }

                    meta.append(" are allowed.");
                }
            }
        }

        break;
    }

    ito::uint32 inOut = param.getFlags();

    // TODO: already tried to avoid the linewrap inside [] bit <td nowrap> didn´t work!
    if ((inOut & ito::ParamBase::In) && (inOut & ito::ParamBase::Out))
    {
        type.append(" [in/out]");
    }
    else if (inOut & ito::ParamBase::In)
    {
        type.append(" [in]");
    }
    else if (inOut & ito::ParamBase::Out)
    {
        type.append(" [out]");
    }

#if QT_VERSION < 0x050000
    output.replace("%PARAMNAME%", Qt::escape(name));
    output.replace("%PARAMTYPE%", Qt::escape(type));
    output.replace("%PARAMMETA%", Qt::escape(meta));
#else
    output.replace("%PARAMNAME%", QString(name).toHtmlEscaped());
    output.replace("%PARAMTYPE%", QString(type).toHtmlEscaped());
    output.replace("%PARAMMETA%", QString(meta).toHtmlEscaped());
#endif
    output.replace("%PARAMINFO%", parseFilterWidgetContent(info));

    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum of a variable is equal to the minimum of the type
/*! For example if a range is min 0 and it´s a byte, this function returns -inf

    \param minimum
    \return QString int as String or -inf
*/
QString HelpTreeDockWidget::minText(int minimum) const
{
    if (minimum == std::numeric_limits<int>::min())
    {
        return "-inf";
    }

    return QString::number(minimum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum of a variable is equal to the minimum of the type
/*! For example if a range is min 0 and it´s a byte, this function returns -inf

    \param minimum
    \return double as String or -inf
*/
QString HelpTreeDockWidget::minText(double minimum) const
{
    if (std::abs(minimum + std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon())
    {
        return "-inf";
    }

    return QString::number(minimum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum of a variable is equal to the minimum of the type
/*! For example if a range is min 0 and it´s a byte, this function returns -inf

    \param minimum
    \return char as String or -inf
*/
QString HelpTreeDockWidget::minText(char minimum) const
{
    if (minimum == std::numeric_limits<char>::min())
    {
        return "-inf";
    }

    return QString::number(minimum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range maximum of a variable is equal to the maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

    \param maximum
    \return maximum as String or inf
*/
QString HelpTreeDockWidget::maxText(int maximum) const
{
    if (maximum == std::numeric_limits<int>::max())
    {
        return "inf";
    }

    return QString::number(maximum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range maximum of a variable is equal to the maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

    \param maximum
    \return maximum as String or inf
*/
QString HelpTreeDockWidget::maxText(double maximum) const
{
    if (std::abs(maximum - std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon())
    {
        return "inf";
    }

    return QString::number(maximum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range maximum of a variable is equal to the maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

    \param maximum
    \return maximum as String or inf
*/
QString HelpTreeDockWidget::maxText(char maximum) const
{
    if (maximum == std::numeric_limits<char>::max())
    {
        return "inf";
    }

    return QString::number(maximum);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum or maximum of a variable is equal to the minimum or maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

\param value
\return maximum as String, -inf or inf
*/
QString HelpTreeDockWidget::minmaxText(int value) const
{
    if (value == std::numeric_limits<int>::max())
    {
        return "inf";
    }
    else if (value == std::numeric_limits<int>::min())
    {
        return "-inf";
    }

    return QString::number(value);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum or maximum of a variable is equal to the minimum or maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

\param value
\return maximum as String, -inf or inf
*/
QString HelpTreeDockWidget::minmaxText(double value) const
{
    if (std::abs(value - std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon())
    {
        return "inf";
    }
    else if (std::abs(value + std::numeric_limits<double>::max()) < std::numeric_limits<double>::epsilon())
    {
        return "-inf";
    }

    return QString::number(value);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This function detects if a range minimum or maximum of a variable is equal to the minimum or maximum of the type
/*! For example if a range is max 255 and it´s a byte, this function returns inf

\param value
\return maximum as String, -inf or inf
*/
QString HelpTreeDockWidget::minmaxText(char value) const
{
    if (value == std::numeric_limits<char>::max())
    {
        return "inf";
    }
    else if (value == std::numeric_limits<int>::min())
    {
        return "-inf";
    }

    return QString::number(value);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Filter the events for showing and hiding the treeview
/*! 
    \param event
    \return bool
*/
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
    else if (obj == ui.helpTreeContent && event->type() == QEvent::Enter)
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
//! Save Gui positions to Main-ini-File
/*! 
*/
void HelpTreeDockWidget::saveIni()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    settings.setValue("percWidthVi", m_treeWidthVisible);
    settings.setValue("percWidthUn", m_treeWidthInvisible);
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Load Gui positions to Main-ini-File
/*! 
*/
void HelpTreeDockWidget::loadIni()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    m_treeWidthVisible = settings.value("percWidthVi", "50").toDouble();
    m_treeWidthInvisible = settings.value("percWidthUn", "50").toDouble();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpTreeDockWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);

    QList<int> intList;
    if (m_treeVisible)
    {
        intList  <<  ui.splitter->width()*m_treeWidthVisible/100  <<  ui.splitter->width() * (100 - m_treeWidthVisible) / 100;
    }
    else
    {
        intList  <<  ui.splitter->width()*m_treeWidthInvisible/100  <<  ui.splitter->width() * (100 - m_treeWidthInvisible) / 100;
    }
    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Load SQL-DatabasesList in m_ Variable when properties changed
/*! 
*/
void HelpTreeDockWidget::propertiesChanged()
{ // Load the new list of DBs with checkstates from the INI-File
    
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    // Read the other Options
    m_openLinks = settings.value("openExtLinks", true).toBool();
    m_plaintext = settings.value("plaintext", false).toBool();
    m_showSelection.Filters = settings.value("showFilters", true).toBool();
    m_showSelection.Widgets = settings.value("showWidgets", true).toBool();
    m_showSelection.DataIO  = settings.value("showDataIO" , true).toBool();
    m_showSelection.Modules = settings.value("showModules", true).toBool();

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
            QString nameID = settings.value("DB", QString()).toString();
            QString name = nameID.left(nameID.indexOf(QChar(0x00, 0xA7) /*section or paragraph sign*/));
            QString dbName = name + ".db";
            //Add to m_pMainlist
            m_includedDBs.append(dbName);
        }
        settings.endArray();
        reloadDB();
    }
    settings.remove("reLoadDBs");
    settings.endGroup();
    m_forced = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Creates the model (tree) from the given data. 
/*! The function is recursiv. It always calles itself with the rest of the list that is not in the tree yet.

    \param parent parent that might have children. This function is going to find them and add them from the list.
    \param parentPath absolute path of the parent
    \param items list of all sql items that are not processed yet
    \param iconGallery Gallery with icons for classes, modules etc.
    
*/
/*static*/ void HelpTreeDockWidget::createItemRek(QStandardItem& parent, const QString &parentPath, QList<SqlItem> &items, const QMap<int,QIcon> *iconGallery)
{
    SqlItem firstItem;
    int m_urPath = Qt::UserRole + 1;
    int m_urType = Qt::UserRole + 2;

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
                // diese Zeile koennte man auch durch Code ersetzen der das Link Icon automatisch zeichnet... das waere flexibler
                node->setIcon(iconGallery->value(firstItem.type));
            }
            else
            { // Kein Link Normales Bild
                node->setIcon(iconGallery->value(firstItem.type)); //Don't load icons here from file since operations on QPixmap are not allowed in another thread
            }
            node->setEditable(false);
            node->setData(firstItem.path, m_urPath);
            node->setData(1, m_urType);
            node->setToolTip(firstItem.path);
            createItemRek(*node, firstItem.path, items, iconGallery);
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
            node->setData(firstItem.prefix, m_urPath); 
            node->setData(1, m_urType); //typ 1 = docstring wird aus sql gelesen
            createItemRek(*node, firstItem.prefix, items, iconGallery);  
            parent.appendRow(node);
        }
        else
        {
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Get Data from SQL File and store it in a table
/*! This function openes a sql database that contains all the static help informations. All these informations are written into a list.
    createItemRek creates the model from this list.

    \param file the path of the sql database
    \param items this parameter is  filled with a list of SqlItems (struct from the header file)
    \return ito::RetVal

    \sa createItemRek
*/
/*static*/ ito::RetVal HelpTreeDockWidget::readSQL(const QString &file, QList<SqlItem> &items)
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
            retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be opened").toLatin1().data(), file.toLatin1().data());
        }
        database.close();
    }
    else
    {
        retval += ito::RetVal::format(ito::retWarning, 0, tr("Database %s could not be found").toLatin1().data(), file.toLatin1().data());
    }

    QSqlDatabase::removeDatabase(file);
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Reload Database and clear search-edit and start the new Thread
/*! This function starts a new thread that loads the database.
  
  \sa dbLoaderFinished
*/
void HelpTreeDockWidget::reloadDB()
{
    if (dbLoaderWatcher.isRunning())
    {
        //a previous reload and QtConcurrent::run is still running, wait for it to be finished
        dbLoaderWatcher.waitForFinished();
    }

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
//! This slot is called when the loading thread is finished
/*! When this slot is called, the database is loaded and the main model created
  
  \sa reloadDB, loadDBinThread
*/
void HelpTreeDockWidget::dbLoaderFinished(int /*index*/)
{
    ito::RetVal retval = dbLoaderWatcher.future().resultAt(0);

    m_pMainFilterModel->setSourceModel(m_pMainModel);

    m_pMainFilterModel->sort(0, Qt::AscendingOrder);

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
//! Load the Databases in different Thread
/*! This function calls createItemRek for each Database. The show parameter is an enumeration from the headerfile.
    It stores which kind of help is included into the model. The options can be set in the properties dialog.

  \param path
  \param includedDBs list of database files (paths)
  \param mainModel pointer to the mainmodel
  \param iconGallery the gallery is passed to createItemRek
  \param show this parameter is an enumeration that is filled by the settings dialog.

  \sa reloadDB, loadDBinThread, createItemRek
*/
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
            retval = readSQL(temp, sqlList);
            QCoreApplication::processEvents();
            if (!retval.containsWarningOrError())
            {
                createItemRek(*(mainModel->invisibleRootItem()), "", sqlList, iconGallery);
            }
            else
            {
				/* The Database named: m_pIncludedDBs[i] is not available anymore!!! show Error*/
			}
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

    if (show.DataIO)
    {
        createFilterWidgetNode(3, mainModel, iconGallery);
        createFilterWidgetNode(4, mainModel, iconGallery);
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
    // Allgemeine HTML sachen anfuegen /
    /*********************************/ 
    QString rawContent = helpText;
    QString html = "<html><head>"
                   "<link rel='stylesheet' type='text/css' href='itom_help_style.css'>"
                   "</head><body>%1"
                   "</body></html>";

    // Insert Shortdescription
    // -------------------------------------
    if (shortDesc != "")
        rawContent.insert(0,shortDesc+"");

    // Ueberschrift (Funktionsname) einfuegen
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
        
        ////dummy output (write last loaded Plaintext into html-File)
        //QFile file2("helpOutput.html");
        //file2.open(QIODevice::WriteOnly);
        //file2.write(html.arg(rawContent).toLatin1());
        //file2.close();
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Displayes the help text in the textbrowser after an element from the tree was clicked.
/*! This function looks for a entry in the database that is specified by the path parameter.
    This text is afterwards passed to all the different parser functions in this class.
  
  \param path Path to the entry, read from the model.
*/
ito::RetVal HelpTreeDockWidget::displayHelp(const QString &path)
{ 
    ito::RetVal retval = ito::retOk;

    ui.helpTreeContent->clear();
    bool ok = false;
    bool found = false;

    // Das ist ein kleiner workaround mit dem if 5 Zeilen spaeter. Man koennt auch direkt ueber die includeddbs list iterieren
    // dann waere folgende Zeile hinfaellig
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

                            highlightContent(query.value(1).toString(), query.value(2).toString(), query.value(3).toString(), query.value(4).toString(), doc, query.value(6).toString(), ui.helpTreeContent->document());
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
//! slot invoked by toolbar 
/*!
    This is the Slot is called by the toolbar when the user enters a filter into the search edit.

    \param filterText the text that the model is filtered with.
*/
void HelpTreeDockWidget::liveFilter(const QString &filterText)
{
    showTreeview();
    m_pMainFilterModel->setFilterRegExp(filterText);
    expandTree();
}

//----------------------------------------------------------------------------------------------------------------------------------
// 
// prot|||....link.....        
//! Returns a list containing the protocol[0] and the real link[1]
/*! This functions looks for different protocols in the links that can be clicked inside the textBrowser
    
    \param link link link that is analysed
    \return returns a list of all parts of the link
*/
QStringList HelpTreeDockWidget::separateLink(const QUrl &link)
{
    QStringList result;
    QByteArray examplePrefix = "example:";

    if (link.scheme() == "itom")
    {
        if (link.host() == "widget.html")
        {
            result.append("widget");
#if QT_VERSION < 0x050000
            result.append(link.fragment());
#else
            result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
#endif     
        }
        else if (link.host() == "algorithm.html")
        {
            result.append("algorithm");
#if QT_VERSION < 0x050000
            result.append(link.fragment());
#else
            result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
#endif      
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
#if QT_VERSION < 0x050000
        result.append(link.fragment());
#else
        result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
#endif
    }
    else
    {
        result.append("-1");
    }

    return result;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by different widgets to display a help page from extern
/*!
    This is the Slot that can be externally called by other widgets to display filter or widget help ... i.a. AIManagerWidget

    \param name name of the function that is supposed to be displayed
    \param type it decides wheather the help is stored in a database (1) or calls showFilterWidgetPluginHelp(...) (2-8)
    \param modelIndex that was clicked. If it´s empty, it´s a call from a link or from extern
    \param fromLink if true, a link called that slot
*/
void HelpTreeDockWidget::showPluginInfo(const QString &name, int type, const QModelIndex &modelIndex, bool fromLink)
{
    // Check if it´s a click by the back or forward button
    if (modelIndex.isValid())
    {
        m_historyIndex++;
        m_history.insert(m_historyIndex, modelIndex);
        for (int i = m_history.length(); i > m_historyIndex; i--)
        {
            m_history.removeAt(i);
        }
    }
    // Check if it´s 
    if (fromLink)
    {
        m_internalCall = true;
        if (modelIndex.isValid())
        {
            ui.treeView->setCurrentIndex(m_pMainFilterModel->mapFromSource(modelIndex));
        }
        else
        {
            if (type == 1)
            {
                ui.treeView->setCurrentIndex(m_pMainFilterModel->mapFromSource(findIndexByPath(1, name.split("."), m_pMainModel->invisibleRootItem())));
            }
            else
            {
                ui.treeView->setCurrentIndex(m_pMainFilterModel->mapFromSource(findIndexByPath(2, name.split("."), m_pMainModel->invisibleRootItem())));
            }
        }
        m_internalCall = false;
    }
    switch(type)
    {
        case 1:
        {
            displayHelp(name);
            break;
        }
        case 2:
        { // 2 Filter
            showFilterWidgetPluginHelp(name, typeFilter);
            break;
        }
        case 3:
        { // 3 Widget
            showFilterWidgetPluginHelp(name, typeWidget);
            break;
        }
        case 4:
        { // 
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
        case 7:
        {
            showFilterWidgetPluginHelp(name, typeDataIO);
            break;
        }
        case 8:
        {
            showFilterWidgetPluginHelp(name, typeActuator);
            break;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! finds a Modelindex related to MainModel (not FilterModel)belonging to an Itemname
/*!

    \param type of the item (for more information see type enumeration in header file)
    \param path path to the item splitted into a list
    \param current item whose children are searched
    \return QModelIndex
*/
QModelIndex HelpTreeDockWidget::findIndexByPath(const int type, QStringList path, const QStandardItem* current)
{
    QStandardItem *temp;
    int counts;
    QString tempString;
    QString firstPath;
    firstPath = path.takeFirst();
    if (current->hasChildren())
    {
        counts = current->rowCount();
        for (int j = 0; j < counts; ++j)
        {
            temp = current->child(j,0);
            if (temp->data(m_urType) == 1)
            {
                if (path.length() == 0 && temp->text().toLower() == firstPath.toLower())
                {
                    return temp->index();
                }
                else if (path.length() > 0 && temp->text().toLower() == firstPath.toLower())
                {
                    return findIndexByPath(1, path, temp);
                }
            }
            else
            {
                if (path.length() == 0 && temp->text().toLower() == firstPath.toLower())
                {
                    return temp->index();
                }
                else if (path.length() > 0 && temp->text().toLower() == firstPath.toLower())
                {
                    return findIndexByPath(2, path, temp);
                }
            }
        }
        return QModelIndex();
    }
    return QModelIndex();
}


/*************************************************************/
/*****************GUI related methods*************************/
/*************************************************************/

//----------------------------------------------------------------------------------------------------------------------------------
// Expand all TreeNodes
void HelpTreeDockWidget::expandTree()
{
    m_doingExpandAll = true;
    ui.treeView->expandAll();
    ui.treeView->resizeColumnToContents(0);
    m_doingExpandAll = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse all TreeNodes
void HelpTreeDockWidget::collapseTree()
{
    m_doingExpandAll = true;
    ui.treeView->collapseAll();
    ui.treeView->resizeColumnToContents(0);
    m_doingExpandAll = false;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Link inside Textbrowser is clicked
void HelpTreeDockWidget::on_helpTreeContent_anchorClicked(const QUrl & link)
{
    QString t = link.toString();
    QStringList parts = separateLink(link);

    if (parts.size() < 2) return;

    QString parts0 = parts[0];
    QStringList parts1 = parts[1].split(".");
        
    if (parts0 == "http")
    {//WebLink
        QDesktopServices::openUrl(link);
    }
    else if (parts0 == "mailto")
    {//MailTo-Link
        QDesktopServices::openUrl(parts[1]);
    }
    else if (parts0 == "example")
    {//Copy an example to Clipboard
        QClipboard *clip = QApplication::clipboard();
        clip->setText(parts[1], QClipboard::Clipboard);
    }
    else if (parts0 == "itom")
    {
        showPluginInfo(parts[1], 1, findIndexByPath(1, parts1, m_pMainModel->invisibleRootItem()), true);
    }
    else if (parts[1].split(".").length() == 1 || (parts1[0] == "DataIO" && parts1.length() == 2))
    {
        showPluginInfo(parts[1], typeCategory, findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()), true);
    }
    else if (parts0 == "algorithm" && parts1.length() < 3)
    {
        //Filter Plugin
        showPluginInfo(parts[1], typeFPlugin, findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()), true);
    }
    else if (parts0 == "algorithm" && parts1.length() >= 3)
    {
        if (parts1[0] == "Widgets")
        {
            //Widget (This is a workaround for the Linklist. Without this else if the links wouldn´t work
            showPluginInfo(parts[1], typeWidget, findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()), true);
        }
        else
        {
            //Filter (This is a workaround for the Linklist. Without this else if the links wouldn´t work
            showPluginInfo(parts[1], typeFilter, findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()), true);
        }
    }
    else if (parts0 == "-1")
    {
        //ui.label->setText(tr("invalid Link"));
    }
    else
    {
        //ui.label->setText(tr("unknown protocol"));
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
        m_treeWidthVisible = pos / width * 100;
    }
    else
    {
        m_treeWidthInvisible = pos / width * 100;
    }

    if (m_treeWidthVisible == 0)
    {
        m_treeWidthVisible = 30;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Show the Help in the right Memo
void HelpTreeDockWidget::selectedItemChanged(const QModelIndex &current, const QModelIndex &/*previous*/)
{
    if (m_internalCall == false)
    {
        int type = current.data(m_urType).toInt();
        QString t = current.data(m_urPath).toString();
        if (type == 1) 
        {
            showPluginInfo(current.data(m_urPath).toString(), type, m_pMainFilterModel->mapToSource(current), false);
        }
        else
        {
            showPluginInfo(current.data(m_urPath).toString(), type, m_pMainFilterModel->mapToSource(current), false);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Back-Button
void HelpTreeDockWidget::navigateBackwards()
{
    if (m_historyIndex > 0)
    {
        m_historyIndex--;
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(m_history.at(m_historyIndex));    
        int type = filteredIndex.data(m_urType).toInt();
        if (type == 1) 
        {
            showPluginInfo(filteredIndex.data(m_urPath).toString(), type, QModelIndex(), true);
        }
        else
        {
            showPluginInfo(filteredIndex.data(m_urPath).toString(), type, QModelIndex(), true);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Forward-Button
void HelpTreeDockWidget::navigateForwards()
{
    if (m_historyIndex < m_history.length()-1)
    {
        m_historyIndex++;
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(m_history.at(m_historyIndex));
        int type = filteredIndex.data(m_urType).toInt();
        if (type == 1) 
        {
            showPluginInfo(filteredIndex.data(m_urPath).toString(), type, QModelIndex(), true);
        }
        else
        {
            showPluginInfo(filteredIndex.data(m_urPath).toString(), type, QModelIndex(), true);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Show tree
void HelpTreeDockWidget::showTreeview()
{
    m_treeVisible = true;
    QList<int> intList;
    intList  <<  ui.splitter->width()*m_treeWidthVisible/100  <<  ui.splitter->width() * (100 - m_treeWidthVisible) / 100;
    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Hide tree
void HelpTreeDockWidget::unshowTreeview()
{
    m_treeVisible = false;
    QList<int> intList;
    intList  <<  ui.splitter->width()*m_treeWidthInvisible/100  <<  ui.splitter->width() * (100 - m_treeWidthInvisible) / 100;
    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
// Expand Tree
void HelpTreeDockWidget::on_treeView_expanded(const QModelIndex &index)
{
    if (!m_doingExpandAll)
    {
        ui.treeView->resizeColumnToContents(0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse Tree
void HelpTreeDockWidget::on_treeView_collapsed(const QModelIndex &index)
{
    if (!m_doingExpandAll)
    {
        ui.treeView->resizeColumnToContents(0);
    }
}

} //end namespace ito
