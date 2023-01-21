#include "helpTreeDockWidget.h"

#include "../../AddInManager/addInManager.h"
#include <AppManagement.h>
#include <qcollator.h>
#include <qdesktopservices.h>
#include <qdiriterator.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qmessagebox.h>
#include <qpainter.h>
#include <qsortfilterproxymodel.h>
#include <qstandarditemmodel.h>
#include <qstringlistmodel.h>
#include <qregularexpression.h>

#include <QtConcurrent/qtconcurrentrun.h>

#include <QThread>
#include <common/addInInterface.h>
#include <qclipboard.h>
#include <qsettings.h>
#include <qtextdocument.h>
#include <qtextstream.h>
#include <qtimer.h>
#include <qtreeview.h>
#include <stdio.h>

#include "../AppManagement.h"
#include "../models/leafFilterProxyModel.h"
#include "../widgets/helpDockWidget.h"
#include "common/helperCommon.h"

namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
// Constructor
HelpTreeDockWidget::HelpTreeDockWidget(QWidget* parent, ito::AbstractDockWidget* dock, Qt::WindowFlags flags)
    : QWidget(parent, flags),
    m_historyIndex(-1),
    m_pMainModel(NULL),
    m_pParent(dock),
    m_internalCall(false),
    m_doingExpandAll(false),
    m_state(stateIdle),
    m_backgroundColorHeading("#efefef"),
    m_textColorHeading("#0c3762"),
    m_linkColor("#dc3c01"),
    m_backgroundParamName("#dcb8aa"),
    m_textColorSection("#dc3c01"),
    m_backgroundColorSection("#eeeeee")
{
    ui.setupUi(this);

    connect(
        AppManagement::getMainApplication(),
        SIGNAL(propertiesChanged()),
        this,
        SLOT(propertiesChanged()));

    // Initialize Variables
    m_treeVisible = false;

    connect(
        &m_loaderWatcher, SIGNAL(resultReadyAt(int)), this, SLOT(loadHelpResourcesFinished(int)));

    m_pMainFilterModel = new LeafFilterProxyModel(this);
    m_pMainModel = new QStandardItemModel(this);
    m_pMainFilterModel->setFilterCaseSensitivity(Qt::CaseInsensitive);

    // Install Eventfilter
    ui.treeView->installEventFilter(this);
    ui.helpTreeContent->installEventFilter(this);

    ui.commandLinkButton->setVisible(false);
    m_previewMovie = new QMovie(":/application/icons/loader32x32trans.gif", QByteArray(), this);
    ui.lblProcessMovie->setMovie(m_previewMovie);
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

    ui.treeView->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    ui.treeView->setHeaderHidden(true);

    QStringList iconAliasesName;
    QList<int> iconAliasesNumb;
    iconAliasesName << "class"
                    << "const"
                    << "routine"
                    << "module"
                    << "package"
                    << "unknown"
                    << "link_unknown"
                    << "link_class"
                    << "link_const"
                    << "link_module"
                    << "link_package"
                    << "link_routine";
    iconAliasesNumb << 04 << 06 << 05 << 03 << 02 << 00 << 11 << 14 << 16 << 13 << 12 << 15;
    int i = 0;

    foreach (const QString& icon, iconAliasesName)
    {
        m_iconGallery[iconAliasesNumb[i]] = QIcon(":/helpTreeDockWidget/" + icon);
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

    restoreSettings();

    // reloadHelpResources();
}

//----------------------------------------------------------------------------------------------------------------------------------
// Destructor
HelpTreeDockWidget::~HelpTreeDockWidget()
{
    storeSettings();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Get The Filters and put them into a node of the Tree
/*!

    \param fOrW
    \param model
    \param iconGallery
*/
void HelpTreeDockWidget::createFilterWidgetNode(
    int fOrW, QStandardItemModel* model, const QMap<int, QIcon>* iconGallery)
{
    // Map der Plugin-Namen und Zeiger auf das Node des Plugins
    QMap<QString, QStandardItem*> plugins;

    // AddInManager einbinden
    ito::AddInManager* aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());

    QStandardItem* mainNode = new QStandardItem();
    mainNode->setEditable(false);
    QString mainNodeText = "";

    switch (fOrW)
    {
    case 1: // Filter
    {
        // build Main Node
        mainNodeText = tr("Algorithms");
        mainNode->setText(mainNodeText);
        mainNode->setData(typeCategory, roleType);
        mainNode->setData(mainNodeText, rolePath);
        mainNode->setIcon(iconGallery->value(iconPluginAlgo));
        if (aim)
        {
            const QHash<QString, ito::AddInAlgo::FilterDef*>* filterHashTable =
                aim->getFilterList();
            QHash<QString, ito::AddInAlgo::FilterDef*>::const_iterator i =
                filterHashTable->constBegin();
            while (i != filterHashTable->constEnd())
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das
                  // Filter-Node anzuhaengen
                    QStandardItem* plugin =
                        new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeFPlugin, roleType);
                    plugin->setData(mainNodeText + "." + plugin->text(), rolePath);
                    plugin->setIcon(iconGallery->value(iconPluginAlgo));
                    plugin->setToolTip(
                        i.value()->m_pBasePlugin->getFilename() + "; v" +
                        QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhaengen
                QStandardItem* filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeFilter, roleType);
                filter->setData(
                    mainNodeText + "." + i.value()->m_pBasePlugin->objectName() + "." +
                        filter->text(),
                    rolePath);
                filter->setIcon(iconGallery->value(iconPluginFilter));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem* test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
        }
        break;
    }
    case 2: // Widgets
    {
        // Main Node zusammenbauen
        mainNodeText = tr("Widgets");
        mainNode->setText(mainNodeText);
        mainNode->setData(typeCategory, roleType);
        mainNode->setData(mainNodeText, rolePath);
        mainNode->setIcon(iconGallery->value(iconWidget));
        if (aim)
        {
            const QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>* widgetHashTable =
                aim->getAlgoWidgetList();
            QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>::const_iterator i =
                widgetHashTable->constBegin();
            while (i != widgetHashTable->constEnd())
            {
                if (!plugins.contains(i.value()->m_pBasePlugin->objectName()))
                { // Plugin existiert noch nicht, erst das Plugin-Node erstellen um dann das
                  // Filter-Node anzuhaengen
                    QStandardItem* plugin =
                        new QStandardItem(i.value()->m_pBasePlugin->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeWPlugin, roleType);
                    plugin->setData(mainNodeText + "." + plugin->text(), rolePath);
                    plugin->setIcon(iconGallery->value(iconPluginAlgo));
                    plugin->setToolTip(
                        i.value()->m_pBasePlugin->getFilename() + "; v" +
                        QString::number(i.value()->m_pBasePlugin->getVersion()));
                    plugins.insert(i.value()->m_pBasePlugin->objectName(), plugin);
                    mainNode->appendRow(plugin);
                }
                // Filter-Node anhaengen
                QStandardItem* filter = new QStandardItem(i.value()->m_name);
                filter->setEditable(false);
                filter->setData(typeWidget, roleType);
                filter->setData(
                    mainNodeText + "." + i.value()->m_pBasePlugin->objectName() + "." +
                        filter->text(),
                    rolePath);
                filter->setIcon(iconGallery->value(iconWidget));
                filter->setToolTip(i.value()->m_pBasePlugin->getAuthor());
                QStandardItem* test = plugins[i.value()->m_pBasePlugin->objectName()];
                test->appendRow(filter);
                ++i;
            }
        }
        break;
    }
    case 3: // DataIO
    {
        // Main Node zusammenbauen
        mainNodeText = tr("DataIO");
        mainNode->setText(mainNodeText);
        mainNode->setData(typeCategory, roleType);
        mainNode->setData(mainNodeText, rolePath);
        mainNode->setIcon(iconGallery->value(iconPluginDataIO));

        // Subcategory Node "Grabber"
        QStandardItem* pluginGrabber = new QStandardItem(tr("Grabber"));
        pluginGrabber->setEditable(false);
        pluginGrabber->setData(typeCategory, roleType);
        pluginGrabber->setData(mainNodeText + "." + tr("Grabber"), rolePath);
        pluginGrabber->setIcon(iconGallery->value(iconPluginGrabber));

        // Subcategory Node "ADDA"
        QStandardItem* pluginAdda = new QStandardItem(tr("ADDA"));
        pluginAdda->setEditable(false);
        pluginAdda->setData(typeCategory, roleType);
        pluginAdda->setData(mainNodeText + "." + tr("ADDA"), rolePath);
        pluginAdda->setIcon(iconGallery->value(iconPluginAdda));

        // Subcategory Node "Raw IO"
        QStandardItem* pluginRawIO = new QStandardItem(tr("Raw IO"));
        pluginRawIO->setEditable(false);
        pluginRawIO->setData(typeCategory, roleType);
        pluginRawIO->setData(mainNodeText + "." + tr("Raw IO"), rolePath);
        pluginRawIO->setIcon(iconGallery->value(iconPluginRawIO));

        if (aim)
        {
            const QList<QObject*>* dataIOList = aim->getDataIOList();
            for (int i = 0; i < dataIOList->length(); i++)
            {
                QObject* obj = dataIOList->at(i);
                const ito::AddInInterfaceBase* aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
                if (aib != NULL)
                {
                    QStandardItem* plugin = new QStandardItem(aib->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeDataIO, roleType);
                    switch (aib->getType())
                    {
                    case 129: { // Grabber
                        plugin->setIcon(iconGallery->value(iconPluginGrabber));
                        plugin->setData(
                            pluginGrabber->data(rolePath).toString() + "." + plugin->text(),
                            rolePath);
                        pluginGrabber->appendRow(plugin);
                        break;
                    }
                    case 257: { // ADDA
                        plugin->setIcon(iconGallery->value(iconPluginAdda));
                        plugin->setData(
                            pluginAdda->data(rolePath).toString() + "." + plugin->text(), rolePath);
                        pluginAdda->appendRow(plugin);
                        break;
                    }
                    case 513: { // Raw IO
                        plugin->setIcon(iconGallery->value(iconPluginRawIO));
                        plugin->setData(
                            pluginRawIO->data(rolePath).toString() + "." + plugin->text(),
                            rolePath);
                        pluginRawIO->appendRow(plugin);
                        break;
                    }
                    }
                }
            }
        }

        mainNode->appendRow(pluginGrabber);
        mainNode->appendRow(pluginAdda);
        mainNode->appendRow(pluginRawIO);
        break;
    }
    case 4: // Actuator
    {
        // Main Node zusammenbauen
        mainNodeText = tr("Actuator");
        mainNode->setText(mainNodeText);
        mainNode->setData(typeCategory, roleType);
        mainNode->setData(mainNodeText, rolePath);
        mainNode->setIcon(iconGallery->value(iconPluginActuator));

        if (aim)
        {
            const QList<QObject*>* ActuatorList = aim->getActList();
            for (int i = 0; i < ActuatorList->length(); i++)
            {
                QObject* obj = ActuatorList->at(i);
                const ito::AddInInterfaceBase* aib = qobject_cast<ito::AddInInterfaceBase*>(obj);
                if (aib != NULL)
                {
                    QStandardItem* plugin = new QStandardItem(aib->objectName());
                    plugin->setEditable(false);
                    plugin->setData(typeActuator, roleType);
                    plugin->setData(mainNodeText + "." + plugin->text(), rolePath);
                    plugin->setIcon(iconGallery->value(iconPluginActuator));
                    mainNode->appendRow(plugin);
                }
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
ito::RetVal HelpTreeDockWidget::showFilterWidgetPluginHelp(
    const QString& filterpath, HelpItemType type)
{
    ito::RetVal retval;
    ito::AddInManager* aim = static_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const QHash<QString, ito::AddInAlgo::FilterDef*>* filterHashTable = aim->getFilterList();
    const QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>* widgetHashTable =
        aim->getAlgoWidgetList();
    ui.helpTreeContent->clear();

    loadAndProcessCssStyleSheet();

    QString docString = "";
    QString filter = filterpath.split(".").last();

    // needed for breadcrumb and for list of children in algorithms
    QString linkNav;

    if (type != typeCategory)
    {
        // Load standard html template
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
            linkNav.insert(
                0,
                ">> <a id=\"HiLink\" href=\"itom://algorithm.html#" +
                    linkPath.toLatin1().toPercentEncoding("", ".") + "\">" + splittedLink[i] +
                    "</a>");
        }
        docString.replace("%BREADCRUMB%", linkNav);

        // extract ParameterSection
        // -------------------------------------
        QString parameterSection;
        int start = docString.indexOf("<!--%PARAMETERS_START%-->");
        int end = docString.indexOf("<!--%PARAMETERS_END%-->");

        if (start == -1 && end == -1) // no returns section
        {
            parameterSection = "";
        }
        else if (start == -1 || end == -1) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: Parameters section is only defined by either the start or end "
                   "tag.")
                    .toLatin1()
                    .data());
        }
        else if (start > end) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: End tag of parameters section comes before start tag.")
                    .toLatin1()
                    .data());
        }
        else
        {
            parameterSection =
                docString.mid(start, end + QString("<!--%PARAMETERS_END%-->").size() - start);
            parameterSection.replace("<!--%PARAMETERS_CAPTION%-->", tr("Parameters"));
            docString.remove(start, end + QString("<!--%PARAMETERS_END%-->").size() - start);
        }

        // extract ReturnSection
        // -------------------------------------
        // search for <!--%RETURNS_START%--> and <!--%RETURNS_END%-->
        QString returnsSection;
        start = docString.indexOf("<!--%RETURNS_START%-->");
        end = docString.indexOf("<!--%RETURNS_END%-->");

        if (start == -1 && end == -1) // no returns section
        {
            returnsSection = "";
        }
        else if (start == -1 || end == -1) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: Returns section is only defined by either the start or end "
                   "tag.")
                    .toLatin1()
                    .data());
        }
        else if (start > end) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: End tag of returns section comes before start tag.")
                    .toLatin1()
                    .data());
        }
        else
        {
            returnsSection =
                docString.mid(start, end + QString("<!--%RETURNS_END%-->").size() - start);
            returnsSection.replace("<!--%RETURNS_CAPTION%-->", tr("Returns"));
            docString.remove(start, end + QString("<!--%RETURNS_END%-->").size() - start);
        }

        // extract ObserverSection
        // -------------------------------------
        // search for <!--%RETURNS_START%--> and <!--%RETURNS_END%-->
        QString observerSection;
        start = docString.indexOf("<!--%OBSERVER_START%-->");
        end = docString.indexOf("<!--%OBSERVER_END%-->");

        if (start == -1 && end == -1) // no returns section
        {
            observerSection = "";
        }
        else if (start == -1 || end == -1) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: Observer section is only defined by either the start or end "
                   "tag.")
                    .toLatin1()
                    .data());
        }
        else if (start > end) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: End tag of observer section comes before start tag.")
                    .toLatin1()
                    .data());
        }
        else
        {
            observerSection =
                docString.mid(start, end + QString("<!--%OBSERVER_END%-->").size() - start);
            observerSection.replace(
                "<!--%OBSERVER_CAPTION%-->", tr("Status observation and cancellation"));
            docString.remove(start, end + QString("<!--%OBSERVER_END%-->").size() - start);
        }

        // extract ExampleSection
        // -------------------------------------
        // search for <!--%EXAMPLE_START%--> and <!--%EXAMPLE_END%-->
        QString exampleSection;
        start = docString.indexOf("<!--%EXAMPLE_START%-->");
        end = docString.indexOf("<!--%EXAMPLE_END%-->");

        if (start == -1 && end == -1) // no returns section
        {
            exampleSection = "";
        }
        else if (start == -1 || end == -1) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: Returns section is only defined by either the start or end "
                   "tag.")
                    .toLatin1()
                    .data());
        }
        else if (start > end) // one part is missing
        {
            retval += ito::RetVal(
                ito::retError,
                0,
                tr("Template Error: End tag of returns section comes before start tag.")
                    .toLatin1()
                    .data());
        }
        else
        {
            exampleSection =
                docString.mid(start, end + QString("<!--%EXAMPLE_END%-->").size() - start);
            exampleSection.replace("<!--%EXAMPLE_CAPTION%-->", tr("Example"));
            exampleSection.replace("<!--%EXAMPLELINK_CAPTION%-->", tr("Copy example to clipboard"));
            docString.remove(start, end + QString("<!--%EXAMPLE_END%-->").size() - start);
        }

        // Build Parameter and return section
        // -------------------------------------
        if (!retval.containsError())
        {
            switch (type)
            {
            case typeFilter: // Filter
            {
                const ito::AddInAlgo::FilterDef* fd = filterHashTable->value(filter);
                if (filterHashTable->contains(filter))
                {
                    const ito::FilterParams* params = aim->getHashedFilterParams(fd->m_paramFunc);

                    docString.replace("%NAME%", fd->m_name);
                    docString.replace("%INFO%", parseFilterWidgetContent(fd->m_description));

                    // Observer-Section
                    const ito::AddInAlgo::FilterDefExt* fdext =
                        dynamic_cast<const ito::AddInAlgo::FilterDefExt*>(fd);

                    QString description;

                    if (fdext)
                    {
                        if (fdext->m_hasStatusInformation)
                        {
                            description +=
                                "<li>" + tr("Filter provides status information") + "</li>\n";
                        }
                        else
                        {
                            description += "<li>" +
                                tr("Filter does not provide status information") + "</li>\n";
                        }

                        if (fdext->m_isCancellable)
                        {
                            description += "<li>" + tr("Filter can be cancelled") + "</li>";
                        }
                        else
                        {
                            description += "<li>" + tr("Filter cannot be cancelled") + "</li>";
                        }
                    }
                    else
                    {
                        description +=
                            "<li>" + tr("No observer can be passed to this filter") + "</li>\n";
                        description +=
                            "<li>" + tr("Filter does not provide status information") + "</li>\n";
                        description += "<li>" + tr("Filter cannot be cancelled") + "</li>";
                    }

                    observerSection.replace("%OBSERVERTEXT%", description);

                    // Parameter-Section
                    if ((params->paramsMand.size() + params->paramsOpt.size() == 0) &&
                        parameterSection.isNull() == false)
                    {
                        // remove parameters section
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
                    { // remove returns section
                        returnsSection = "";
                    }
                    else if (returnsSection.isNull() == false)
                    {
                        parseParamVector("OUT", params->paramsOut, returnsSection);
                    }

                    // Example-Section
                    QStringList paramList;
                    foreach (const ito::Param& p, params->paramsMand)
                    {
                        paramList.append(QLatin1String(p.getName()));
                    }

                    QString returnString;

                    if (params->paramsOut.size() == 1)
                    {
                        returnString =
                            QString(QLatin1String(params->paramsOut[0].getName())) + " = ";
                    }
                    else if (params->paramsOut.size() > 1)
                    {
                        returnString = "[";
                        QStringList returnList;
                        foreach (const ito::Param& p, params->paramsOut)
                        {
                            returnList.append(QLatin1String(p.getName()));
                        }
                        returnString += returnList.join(", ") + "] = ";
                    }

                    // for algorithms, there are two different example strings:
                    // 1. filter("nameOfAlgorithm", arg1, arg2...)
                    // 2. algorithms.nameOfAlgorithm(arg1, arg2, ...)

                    QString example1 = exampleSection;
                    QString newLink = QString("%1filter(\"%2\", %3)")
                                          .arg(returnString)
                                          .arg(fd->m_name)
                                          .arg(paramList.join(", "));
                    newLink.replace(", )", ")");
                    QByteArray a = newLink.toLatin1();

                    example1.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                    example1.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());

                    QString example2 = exampleSection;
                    newLink = QString("%1algorithms.%2(%3)")
                                          .arg(returnString)
                                          .arg(fd->m_name)
                                          .arg(paramList.join(", "));
                    newLink.replace(", )", ")");
                    a = newLink.toLatin1();

                    example2.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                    example2.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());

                    exampleSection = example1 + example2;
                }
                else
                {
                    retval += ito::RetVal(
                        ito::retError,
                        0,
                        tr("Unknown filter name '%1'").arg(filter).toLatin1().data());
                }

                break;
            }
            case typeWidget: {
                const ito::AddInAlgo::AlgoWidgetDef* awd = widgetHashTable->value(filter);
                if (widgetHashTable->contains(filter))
                {
                    const ito::FilterParams* params = aim->getHashedFilterParams(awd->m_paramFunc);

                    docString.replace("%NAME%", awd->m_name);
                    docString.replace("%INFO%", parseFilterWidgetContent(awd->m_description));

                    // Parameter-Section
                    if ((params->paramsMand.size() + params->paramsOpt.size() == 0) &&
                        parameterSection.isNull() == false)
                    {
                        // remove parameters section
                        parameterSection = "";
                    }
                    else if (parameterSection.isNull() == false)
                    {
                        parseParamVector("PARAMMAND", params->paramsMand, parameterSection);
                        parseParamVector("PARAMOPT", params->paramsOpt, parameterSection);
                        parameterSection.replace("<!--%PARAMOPT_CAPTION%-->", tr("optional"));
                    }

                    // remove returns section (Widgets can�t return something)
                    returnsSection = "";

                    // Example-Section
                    QStringList paramList;
                    foreach (const ito::Param& p, params->paramsMand)
                    {
                        paramList.append(QLatin1String(p.getName()));
                    }
                    QString newLink = QString("ui.createNewPluginWidget(\"%1\",%2)")
                                          .arg(awd->m_name)
                                          .arg(paramList.join(", "));
                    newLink.replace(",)", ")");
                    QByteArray a = newLink.toLatin1();

                    exampleSection.replace("<!--%EXAMPLEPLAIN%-->", newLink);
                    exampleSection.replace("<!--%EXAMPLELINK%-->", a.toPercentEncoding());

                    observerSection = "";
                }

                break;
            }
            case typeFPlugin: // These two lines behave
            case typeWPlugin: // like an "or" statement
            {
                const QList<QObject*>* algoPlugins = aim->getAlgList();
                const ito::AddInInterfaceBase* aib = NULL;

                foreach (const QObject* obj, *algoPlugins)
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
                        extendedInfo.append(
                            "<p class=\"rubric\">" + tr("This plugin contains the following") +
                            " " + tr("Algorithms") + ":</p>");

                        QHash<QString, ito::AddInAlgo::FilterDef*>::const_iterator i =
                            filterHashTable->constBegin();
                        QList<QString> algoLinks;
                        while (i != filterHashTable->constEnd())
                        {
                            if (aib->objectName() == i.value()->m_pBasePlugin->objectName())
                            {
                                QString link = "." + i.value()->m_pBasePlugin->objectName() + "." +
                                    i.value()->m_name;
                                algoLinks.append(
                                    "<a id=\"HiLink\" href=\"itom://algorithm.html#Algorithms" +
                                    link.toLatin1().toPercentEncoding("", ".") + "\">" +
                                    i.value()->m_name.toLatin1().toPercentEncoding("", ".") +
                                    "</a><br><br>");
                            }
                            ++i;
                        }

                        QCollator collator;
                        std::sort(algoLinks.begin(), algoLinks.end(), collator);

                        foreach (const QString& algo, algoLinks)
                        {
                            extendedInfo.append(algo);
                        }
                    }

                    if (widgetHashTable->size() > 0)
                    {
                        extendedInfo.append(
                            "<p class=\"rubric\">" + tr("This plugin contains the following") +
                            " " + tr("Widgets") + ":</p>");

                        QHash<QString, ito::AddInAlgo::AlgoWidgetDef*>::const_iterator i =
                            widgetHashTable->constBegin();
                        QList<QString> widgetList;
                        while (i != widgetHashTable->constEnd())
                        {
                            if (aib->objectName() == i.value()->m_pBasePlugin->objectName())
                            {
                                QString link = "." + i.value()->m_pBasePlugin->objectName() + "." +
                                    i.value()->m_name;
                                widgetList.append(
                                    "<a id=\"HiLink\" href=\"itom://algorithm.html#Widgets" +
                                    link.toLatin1().toPercentEncoding("", ".") + "\">" +
                                    i.value()->m_name.toLatin1().toPercentEncoding("", ".") +
                                    "</a><br><br>");
                            }
                            ++i;
                        }

                        QCollator collator;
                        std::sort(widgetList.begin(), widgetList.end(), collator);

                        foreach (const QString& widget, widgetList)
                        {
                            extendedInfo.append(widget);
                        }
                    }

                    docString.replace("%INFO%", extendedInfo);

                    parameterSection = "";
                    returnsSection = "";
                    exampleSection = "";
                    observerSection = "";
                }
                else
                {
                    retval += ito::RetVal(
                        ito::retError,
                        0,
                        tr("Unknown algorithm plugin with name '%1'")
                            .arg(filter)
                            .toLatin1()
                            .data());
                }

                break;
            }
            case typeDataIO:
            case typeActuator: {
                QObject* obj;
                // Lookup the clicked name in the corresponding List
                if (type == typeActuator)
                {
                    const QList<QObject*>* ActuatorList = aim->getActList();
                    for (int i = 0; i < ActuatorList->length(); i++)
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
                    const QList<QObject*>* DataIOList = aim->getDataIOList();
                    for (int i = 0; i < DataIOList->length(); i++)
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
                    const ito::AddInInterfaceBase* aib =
                        qobject_cast<ito::AddInInterfaceBase*>(obj);
                    if (aib != NULL)
                    {
                        docString.replace(
                            "%NAME%",
                            aib->objectName()); // TODO: should return desc, but returns sdesc
                        QString desc = aib->getDescription();
                        QString detaileddesc = aib->getDetailDescription();
                        if (detaileddesc != NULL)
                        {
                            desc.append("\n\n");
                            desc.append(detaileddesc);
                        }
                        docString.replace("%INFO%", parseFilterWidgetContent(desc));

                        // Parameter-Section
                        const QVector<ito::Param>* paramsMand =
                            (qobject_cast<ito::AddInInterfaceBase*>(obj))->getInitParamsMand();
                        const QVector<ito::Param>* paramsOpt =
                            (qobject_cast<ito::AddInInterfaceBase*>(obj))->getInitParamsOpt();
                        if ((paramsMand->size() + paramsOpt->size() == 0) &&
                            parameterSection.isNull() == false)
                        {
                            // remove parameters section
                            parameterSection = "";
                        }
                        else if (parameterSection.isNull() == false)
                        {
                            parseParamVector("PARAMMAND", *paramsMand, parameterSection);
                            parseParamVector("PARAMOPT", *paramsOpt, parameterSection);
                            parameterSection.replace("<!--%PARAMOPT_CAPTION%-->", tr("optional"));
                        }

                        // remove returns and observer section (Widgets cannot return something)
                        returnsSection = "";
                        observerSection = "";

                        // Example-Section
                        QStringList paramList;
                        for (int i = 0; i < paramsMand->size(); i++)
                        {
                            const ito::Param& p = paramsMand->at(i);
                            paramList.append(QLatin1String(p.getName()));
                        }

                        QString callName;

                        if (type == typeDataIO)
                        {
                            callName = "dataIO"; // do not translate
                        }
                        else
                        {
                            callName = "actuator"; // do not translate
                        }

                        QString newLink = QString("%1(\"%2\", %3)")
                                              .arg(callName)
                                              .arg(aib->objectName())
                                              .arg(paramList.join(", "));
                        newLink.replace(", )", ")");
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
            default: {
                retval += ito::RetVal(ito::retError, 0, tr("unknown type").toLatin1().data());
                break;
            }
            }
            docString.replace("<!--%PARAMETERS_INSERT%-->", parameterSection);
            docString.replace("<!--%RETURNS_INSERT%-->", returnsSection);
            docString.replace("<!--%EXAMPLE_INSERT%-->", exampleSection);
            docString.replace("<!--%OBSERVER_INSERT%-->", observerSection);
        }
    }
    else
    {
        ui.helpTreeContent->clear();
        loadAndProcessCssStyleSheet();

        if (filter == tr("Algorithms"))
        {
            QFile file(":/helpTreeDockWidget/algo_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "Algorithms");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("Widgets"))
        {
            QFile file(":/helpTreeDockWidget/widg_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "Widgets");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("DataIO"))
        {
            QFile file(":/helpTreeDockWidget/dataIO_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "DataIO");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("Grabber"))
        {
            QFile file(":/helpTreeDockWidget/dataGr_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "Grabber");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("ADDA"))
        {
            QFile file(":/helpTreeDockWidget/dataAD_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "ADDA");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("Raw IO"))
        {
            QFile file(":/helpTreeDockWidget/dataRa_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "RawIO");
                docString = htmlData;
                file.close();
            }
        }
        else if (filter == tr("Actuator"))
        {
            QFile file(":/helpTreeDockWidget/actuator_page");
            if (file.open(QIODevice::ReadOnly))
            {
                QByteArray htmlData = file.readAll();
                docString.replace("%BREADCRUMB%", "Actuator");
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
    { // Create html document
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
QString HelpTreeDockWidget::parseFilterWidgetContent(const QString& input)
{
    QString output = input.toHtmlEscaped();
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
ito::RetVal HelpTreeDockWidget::parseParamVector(
    const QString& sectionname, const QVector<ito::Param>& paramVector, QString& content)
{
    ito::RetVal retval;
    QString startString = QString("<!--%%1_START%-->").arg(sectionname);
    QString endString = QString("<!--%%1_END%-->").arg(sectionname);
    QString insertString = QString("<!--%%1_INSERT%-->").arg(sectionname);

    // search for <!--%PARAMETERS_START%--> and <!--%PARAMETERS_END%-->
    int start = content.indexOf(startString);
    int end = content.indexOf(endString);

    if (start == -1 && end == -1) // no returns section
    {
        // pass
    }
    else if (start == -1 || end == -1) // one part is missing
    {
        retval += ito::RetVal::format(
            ito::retError,
            0,
            tr("Template Error: %s section is only defined by either the start or end tag.")
                .toLatin1()
                .data(),
            sectionname.toLatin1().data());
    }
    else if (start > end) // one part is missing
    {
        retval += ito::RetVal::format(
            ito::retError,
            0,
            tr("Template Error: End tag of %s section comes before start tag.").toLatin1().data(),
            sectionname.toLatin1().data());
    }
    else
    {
        QString rowContent = content.mid(start, end + endString.size() - start);
        content.remove(start, end + endString.size() - start);
        QString internalContent = "";

        foreach (const ito::Param& p, paramVector)
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
QString HelpTreeDockWidget::parseParam(const QString& tmpl, const ito::Param& param)
{
    QString output = tmpl;
    QString name = QLatin1String(param.getName());
    QString info = param.getInfo() ? QLatin1String(param.getInfo()) : QLatin1String("");
    QString meta;

    QString type;

    meta = ito::getMetaDocstringFromParam(param, true, type);

    ito::uint32 inOut = param.getFlags();

    // TODO: already tried to avoid the linewrap inside [] bit <td nowrap> didn�t work!
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

    output.replace("%PARAMNAME%", QString(name).toHtmlEscaped());
    output.replace("%PARAMTYPE%", QString(type).toHtmlEscaped());
    output.replace("%PARAMMETA%", QString(meta).toHtmlEscaped());
    output.replace("%PARAMINFO%", parseFilterWidgetContent(info));

    return output;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Save Gui positions to Main-ini-File
/*!
 */
void HelpTreeDockWidget::storeSettings()
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
void HelpTreeDockWidget::restoreSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");
    m_treeWidthVisible = settings.value("percWidthVi", "50").toDouble();
    m_treeWidthInvisible = settings.value("percWidthUn", "50").toDouble();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void HelpTreeDockWidget::showEvent(QShowEvent* event)
{
    m_state |= stateVisible;

    QWidget::showEvent(event);

    // load properties and then load the content
    propertiesChanged();

    QList<int> intList;

    if (m_treeVisible)
    {
        intList << ui.splitter->width() * m_treeWidthVisible / 100
                << ui.splitter->width() * (100 - m_treeWidthVisible) / 100;
    }
    else
    {
        intList << ui.splitter->width() * m_treeWidthInvisible / 100
                << ui.splitter->width() * (100 - m_treeWidthInvisible) / 100;
    }

    ui.splitter->setSizes(intList);
}

//----------------------------------------------------------------------------------------------------------------------------------
//! (Re)load the help resources if some properties have changed
/*!
 */
void HelpTreeDockWidget::propertiesChanged()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("HelpScriptReference");

    // Read the other Options
    bool plaintext = settings.value("plaintext", false).toBool();

    settings.endGroup();

    if (plaintext != m_plaintext)
    {
        m_plaintext = plaintext;
        m_state = m_state & (~stateContentLoaded); // invalidate the content loaded flag
    }

    if ((m_state & stateVisible) && (m_state & stateContentLoaded) == 0)
    {
        reloadHelpResources();
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Reload different help resources and clear search-edit and start the new thread
/*! This function starts a new thread that loads the help resources.

  \sa loadHelpResourcesFinished
*/
void HelpTreeDockWidget::reloadHelpResources()
{
    if (m_loaderWatcher.isRunning())
    {
        // a previous reload and QtConcurrent::run is still running, wait for it to be finished
        m_loaderWatcher.waitForFinished();
    }

    bool success =
        m_dbLoaderMutex.tryLock(1); //!< will be unlocked again if QtConcurrent run is finished.

    if (!success)
    {
        // the reloadHelpResources is still active. The finished-slot of the m_loaderWatcher
        // has to be called first to reset the renewed m_pMainModel. This slot can only be called
        // if the main thread, from which this method has also be called, is idle. Therefore
        // initialize a short singleShot timer to recall this method after a short time.
        QTimer::singleShot(50, this, SLOT(reloadHelpResources()));
        return;
    }

    // clear the main model which will be updated in the QtConcurrent run later on...
    m_pMainModel->clear();
    ui.treeView->reset();

    m_pMainFilterModel->setSourceModel(NULL);
    m_previewMovie->start();
    ui.lblProcessMovie->setVisible(true);
    ui.lblProcessText->setVisible(true);
    ui.treeView->setVisible(false);
    ui.splitter->setVisible(false);
    ui.lblProcessText->setText(tr("Help resources are loading..."));

    // THREAD START QtConcurrent::run
    QFuture<ito::RetVal> f1 =
        QtConcurrent::run(loadHelpResources, m_pMainModel /*, m_pDBList*/, &m_iconGallery);
    m_loaderWatcher.setFuture(f1);
    // THREAD END
}

//----------------------------------------------------------------------------------------------------------------------------------
//! This slot is called when the loading thread is finished
/*! When this slot is called, the database is loaded and the main model created

  \sa reloadHelpResources, loadHelpResources
*/
void HelpTreeDockWidget::loadHelpResourcesFinished(int /*index*/)
{
    ito::RetVal retval = m_loaderWatcher.future().resultAt(0);

    m_pMainFilterModel->setSourceModel(m_pMainModel);

    m_pMainFilterModel->sort(0, Qt::AscendingOrder);

    // disconnect earlier connections (if available)
    if (ui.treeView->selectionModel())
    {
        disconnect(
            ui.treeView->selectionModel(),
            SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
            this,
            SLOT(selectedItemChanged(const QModelIndex&, const QModelIndex&)));
    }

    // model has been
    ui.treeView->setModel(m_pMainFilterModel);

    // after setModel, the corresponding selectionModel is changed, too
    connect(
        ui.treeView->selectionModel(),
        SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)),
        this,
        SLOT(selectedItemChanged(const QModelIndex&, const QModelIndex&)));

    m_previewMovie->stop();
    ui.lblProcessMovie->setVisible(false);

    ui.lblProcessText->setVisible(false);
    ui.treeView->setVisible(true);
    ui.splitter->setVisible(true);

    ui.treeView->resizeColumnToContents(0);

    m_dbLoaderMutex.unlock();

    m_state |= stateContentLoaded;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! Load help resources from various plugins in a different thread
/*! This function parses the information pages for both algorithm and
    hardware plugins.

  \param mainModel pointer to the mainModel
  \param iconGallery the gallery is passed to createFilterWidgetNode

  \sa reloadHelpResources, loadHelpResources, createFilterWidgetNode
*/
/*static*/ ito::RetVal HelpTreeDockWidget::loadHelpResources(
    QStandardItemModel* mainModel, const QMap<int, QIcon>* iconGallery)
{
    ito::RetVal retval;

    createFilterWidgetNode(1, mainModel, iconGallery);
    createFilterWidgetNode(2, mainModel, iconGallery);
    createFilterWidgetNode(3, mainModel, iconGallery);
    createFilterWidgetNode(4, mainModel, iconGallery);

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
// Highlight (parse) the Helptext to make it nice and readable for non docutils Docstrings
// ERROR decides whether it's already formatted by docutils (Error = 0) or it must be parsed by this
// function (Error != 0)
ito::RetVal HelpTreeDockWidget::highlightContent(
    const QString& prefix,
    const QString& name,
    const QString& param,
    const QString& shortDesc,
    const QString& helpText,
    const QString& error,
    QTextDocument* document,
    const QMap<QString, QImage>& images)
{
    QString errorS = error.left(error.indexOf(" ", 0));
    int errorCode = errorS.toInt();
    QStringList errorList;

    /*********************************/
    // Allgemeine HTML sachen anfuegen /
    /*********************************/
    QString rawContent = helpText;
    QRegularExpression bodyFinder("<body>(.*)</body>");
    auto bodyFinderMatch = bodyFinder.match(rawContent);

    if (bodyFinderMatch.hasMatch())
    {
        rawContent = bodyFinderMatch.captured(1);
    }

    QString html = "<html><head>"
                   "<link rel='stylesheet' type='text/css' href='itom_help_style.css'>"
                   "</head><body>%1"
                   "</body></html>";

    // Insert Shortdescription
    // -------------------------------------
    if (shortDesc != "")
    {
        rawContent.insert(0, shortDesc);
    }

    // Ueberschrift (Funktionsname) einfuegen
    // -------------------------------------
    rawContent.insert(0, "<h1 id=\"FunctionName\">" + name + param + "</h1>" + "");

    // Prefix als Navigations-Links einfuegen
    // -------------------------------------
    QStringList splittedLink = prefix.split(".");
    rawContent.insert(0, "&gt;&gt;&nbsp;" + splittedLink[splittedLink.length() - 1]);
    for (int i = splittedLink.length() - 2; i > -1; i--)
    {
        QString linkPath;
        for (int j = 0; j <= i; j++)
            linkPath.append(splittedLink.mid(0, i + 1)[j] + ".");
        if (linkPath.right(1) == ".")
            linkPath = linkPath.left(linkPath.length() - 1);
        rawContent.insert(
            0,
            "&nbsp;&gt;&gt;&nbsp;<a id=\"HiLink\" href=\"itom://" + linkPath + "\">" +
                splittedLink[i] + "</a>");
    }

    // Insert docstring
    // -------------------------------------
    if (m_plaintext)
    { // Only for debug reasons! Displays the Plaintext instead of the html
        rawContent.replace("<br/>", "<br/>\n");
        document->setPlainText(html.arg(rawContent));
    }
    else
    {
        loadAndProcessCssStyleSheet();

        QMap<QString, QImage>::const_iterator it = images.constBegin();

        while (it != images.constEnd())
        {
            document->addResource(QTextDocument::ImageResource, it.key(), it.value());
            it++;
        }

        // see if prefix is a leaf or a module / package:
        QModelIndex idx = findIndexByPath(1, prefix.split("."), m_pMainModel->invisibleRootItem());
        bool leaf = (m_pMainModel->rowCount(idx) == 0);


        if (leaf)
        {
            // matches :obj:`test <sdf>` where sdf must not contain a > sign. < and > are written as
            // &lt; or &gt; in html!
            rawContent.replace(
                QRegularExpression(":obj:`([a-zA-Z0-9_-\\.]+) &lt;(((?!&gt;).)*)&gt;`"),
                "<a id=\"HiLink\" href=\"itom://" + prefix.left(prefix.lastIndexOf('.')) +
                    ".\\1\">\\2</a>");

            rawContent.replace(
                QRegularExpression(":obj:`([a-zA-Z0-9_-\\.]+)`"),
                "<a id=\"HiLink\" href=\"itom://" + prefix.left(prefix.lastIndexOf('.')) +
                    ".\\1\">\\1</a>");
        }
        else
        {
            // matches :obj:`test <sdf>` where sdf must not contain a > sign. < and > are written as
            // &lt; or &gt; in html!
            rawContent.replace(
                QRegularExpression(":obj:`([a-zA-Z0-9_-\\.]+) &lt;(((?!&gt;).)*)&gt;`"),
                "<a id=\"HiLink\" href=\"itom://" + prefix + ".\\1\">\\2</a>");

            rawContent.replace(
                QRegularExpression(":obj:`([a-zA-Z0-9_-\\.]+)`"),
                "<a id=\"HiLink\" href=\"itom://" + prefix + ".\\1\">\\1</a>");
        }

        document->setHtml(html.arg(rawContent));

        ////dummy output (write last loaded Plaintext into html-File)
        /*QFile file2("helpOutput.html");
        file2.open(QIODevice::WriteOnly);
        file2.write(html.arg(rawContent).toLatin1());
        file2.close();*/
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
//! slot invoked by toolbar
/*!
    This is the Slot is called by the toolbar when the user enters a filter into the search edit.

    \param filterText the text that the model is filtered with.
*/
void HelpTreeDockWidget::liveFilter(const QString& filterText)
{
    m_filterTextPending = filterText;
    if (m_filterTextPendingTimer >= 0)
    {
        killTimer(m_filterTextPendingTimer);
    }
    m_filterTextPendingTimer = startTimer(250);
}

//---------------------------------------------------------------------------------------------------------------------------------
void HelpTreeDockWidget::timerEvent(QTimerEvent* event)
{
    //    showTreeview();
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    m_pMainFilterModel->setFilterRegularExpression(m_filterTextPending);
#else
    m_pMainFilterModel->setFilterRegExp(m_filterTextPending);
#endif
    expandTree();

    killTimer(m_filterTextPendingTimer);
    m_filterTextPendingTimer = -1;
}

//----------------------------------------------------------------------------------------------------------------------------------
//
// prot|||....link.....
//! Returns a list containing the protocol[0] and the real link[1]
/*! This functions looks for different protocols in the links that can be clicked inside the
   textBrowser

    \param link link link that is analysed
    \return returns a list of all parts of the link
*/
QStringList HelpTreeDockWidget::separateLink(const QUrl& link)
{
    QStringList result;
    QByteArray examplePrefix = "example:";
    QString scheme = link.scheme();

    if (scheme == "itom")
    {
        if (link.host() == "widget.html")
        {
            result.append("widget");
            result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
        }
        else if (link.host() == "algorithm.html")
        {
            result.append("algorithm");
            result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
        }
        else
        {
            result.append("itom");
            result.append(link.host());
        }
    }
    else if (scheme == "mailto")
    {
        result.append("mailto");
        result.append(link.path());
    }
    else if (scheme == "example")
    {
        result.append("example");
        result.append(QUrl::fromPercentEncoding(link.fragment().toLatin1()));
    }
    else if (scheme == "http" || scheme == "https")
    {
        result.append("http");
        result.append(link.toString());
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
    This is the Slot that can be externally called by other widgets to display filter or widget help
   ... i.a. AIManagerWidget

    \param name name of the function that is supposed to be displayed
    \param type it decides wheather the help is stored in a database (1) or calls
   showFilterWidgetPluginHelp(...) (2-8) \param modelIndex that was clicked. If it's empty, it's a
   call from a link or from extern \param fromLink if true, a link called that slot
*/
void HelpTreeDockWidget::showPluginInfo(
    const QString& name, HelpItemType type, const QModelIndex& modelIndex, bool fromLink)
{
    bool ok = false;

    for (int i = 0; i < 1000; ++i)
    {
        if (m_dbLoaderMutex.tryLock(100))
        {
            ok = true;
            break;
        }
        else
        {
            QCoreApplication::processEvents();
        }
    }

    if (!ok)
    {
        return;
    }

    // Check if it is a click by the back or forward button
    if (modelIndex.isValid())
    {
        m_historyIndex++;
        m_history.insert(m_historyIndex, modelIndex);

        for (int i = m_history.length() - 1; i > m_historyIndex; i--)
        {
            m_history.removeAt(i);
        }
    }

    // Check if it is it
    if (fromLink)
    {
        m_internalCall = true;

        if (modelIndex.isValid())
        {
            ui.treeView->setCurrentIndex(m_pMainFilterModel->mapFromSource(modelIndex));
        }
        else
        {
            QModelIndex index = findIndexByPath(
                type == 1 ? 1 : 2, name.split("."), m_pMainModel->invisibleRootItem());
            if (index.isValid() && m_pMainFilterModel->sourceModel())
            {
                ui.treeView->setCurrentIndex(m_pMainFilterModel->mapFromSource(index));
            }
        }

        m_internalCall = false;
    }

    showFilterWidgetPluginHelp(name, type);

    m_dbLoaderMutex.unlock();
}

//----------------------------------------------------------------------------------------------------------------------------------
//! finds a model index related to MainModel (not FilterModel)belonging to an Itemname
/*!

    \param type of the item (for more information see type enumeration in header file)
    \param path path to the item splitted into a list
    \param current item whose children are searched
    \return QModelIndex
*/
QModelIndex HelpTreeDockWidget::findIndexByPath(
    const int type, QStringList path, const QStandardItem* current)
{
    QStandardItem* temp;
    int counts;
    QString tempString;
    QString firstPath;
    firstPath = path.takeFirst();

    if (current->hasChildren())
    {
        counts = current->rowCount();

        for (int j = 0; j < counts; ++j)
        {
            temp = current->child(j, 0);

            if (path.length() == 0 && temp->text().toLower() == firstPath.toLower())
            {
                return temp->index();
            }
            else if (path.length() > 0 && temp->text().toLower() == firstPath.toLower())
            {
                return findIndexByPath(2, path, temp);
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
void HelpTreeDockWidget::on_helpTreeContent_anchorClicked(const QUrl& link)
{
    QString t = link.toString();
    QStringList parts = separateLink(link);

    if (parts.size() < 2)
        return;

    QString parts0 = parts[0];
    QStringList parts1 = parts[1].split(".");

    if (parts0 == "http")
    { // WebLink
        QDesktopServices::openUrl(link);
    }
    else if (parts0 == "mailto")
    { // MailTo-Link
        QDesktopServices::openUrl(parts[1]);
    }
    else if (parts0 == "example")
    { // Copy an example to Clipboard
        QClipboard* clip = QApplication::clipboard();
        clip->setText(parts[1], QClipboard::Clipboard);
    }
    else if (parts0 == "itom")
    {
        // pass
    }
    else if (parts[1].split(".").length() == 1 || (parts1[0] == "DataIO" && parts1.length() == 2))
    {
        showPluginInfo(
            parts[1],
            typeCategory,
            findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()),
            true);
    }
    else if (parts0 == "algorithm" && parts1.length() < 3)
    {
        // Filter Plugin
        showPluginInfo(
            parts[1],
            typeFPlugin,
            findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()),
            true);
    }
    else if (parts0 == "algorithm" && parts1.length() >= 3)
    {
        if (parts1[0] == "Widgets")
        {
            // Widget (This is a workaround for the Linklist. Without this else if the links
            // wouldn�t work
            showPluginInfo(
                parts[1],
                typeWidget,
                findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()),
                true);
        }
        else
        {
            // Filter (This is a workaround for the Linklist. Without this else if the links
            // wouldn�t work
            showPluginInfo(
                parts[1],
                typeFilter,
                findIndexByPath(2, parts1, m_pMainModel->invisibleRootItem()),
                true);
        }
    }
    else if (parts0 == "-1")
    {
        // ui.label->setText(tr("invalid Link"));
    }
    else
    {
        // ui.label->setText(tr("unknown protocol"));
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
void HelpTreeDockWidget::on_splitter_splitterMoved(int pos, int index)
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
void HelpTreeDockWidget::selectedItemChanged(
    const QModelIndex& current, const QModelIndex& previous)
{
    if (m_internalCall == false)
    {
        HelpItemType type = (HelpItemType)current.data(roleType).toInt();
        showPluginInfo(
            current.data(rolePath).toString(),
            type,
            m_pMainFilterModel->mapToSource(current),
            false);
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
        HelpItemType type = (HelpItemType)filteredIndex.data(roleType).toInt();

        showPluginInfo(filteredIndex.data(rolePath).toString(), type, QModelIndex(), true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Forward-Button
void HelpTreeDockWidget::navigateForwards()
{
    if (m_historyIndex < m_history.length() - 1)
    {
        m_historyIndex++;
        QModelIndex filteredIndex = m_pMainFilterModel->mapFromSource(m_history.at(m_historyIndex));
        HelpItemType type = (HelpItemType)filteredIndex.data(roleType).toInt();

        showPluginInfo(filteredIndex.data(rolePath).toString(), type, QModelIndex(), true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Expand Tree
void HelpTreeDockWidget::on_treeView_expanded(const QModelIndex& index)
{
    if (!m_doingExpandAll)
    {
        ui.treeView->resizeColumnToContents(0);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
// Collapse Tree
void HelpTreeDockWidget::on_treeView_collapsed(const QModelIndex& index)
{
    if (!m_doingExpandAll)
    {
        ui.treeView->resizeColumnToContents(0);
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::backgroundColorHeading() const
{
    return m_backgroundColorHeading;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setBackgroundColorHeading(const QColor& color)
{
    if (color != m_backgroundColorHeading)
    {
        m_backgroundColorHeading = color;
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::textColorHeading() const
{
    return m_textColorHeading;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setTextColorHeading(const QColor& color)
{
    if (color != m_textColorHeading)
    {
        m_textColorHeading = color;
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::linkColor() const
{
    return m_linkColor;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setLinkColor(const QColor& color)
{
    if (color != m_linkColor)
    {
        m_linkColor = color;
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::backgroundParamName() const
{
    return m_backgroundParamName;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setBackgroundParamName(const QColor& color)
{
    if (color != m_backgroundParamName)
    {
        m_backgroundParamName = color;
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::backgroundColorSection() const
{
    return m_backgroundColorSection;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setBackgroundColorSection(const QColor& color)
{
    if (color != m_backgroundColorSection)
    {
        m_backgroundColorSection = color;
    }
}

//-------------------------------------------------------------------------------------
QColor HelpTreeDockWidget::textColorSection() const
{
    return m_textColorSection;
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::setTextColorSection(const QColor& color)
{
    if (color != m_textColorSection)
    {
        m_textColorSection = color;
    }
}

//-------------------------------------------------------------------------------------
void HelpTreeDockWidget::loadAndProcessCssStyleSheet()
{
    QFile file(":/helpTreeDockWidget/help_style");

    if (file.open(QIODevice::ReadOnly))
    {
        QString cssData = QLatin1String(file.readAll());

        // replace some colors
        cssData.replace("$backgroundColorHeading$", m_backgroundColorHeading.name());
        cssData.replace("$textColorHeading$", m_textColorHeading.name());
        cssData.replace("$linkColor$", m_linkColor.name());
        cssData.replace("$backgroundParamName$", m_backgroundParamName.name());
        cssData.replace("$textColorSection$", m_textColorSection.name());
        cssData.replace("$backgroundColorSection$", m_backgroundColorSection.name());

        ui.helpTreeContent->document()->addResource(
            QTextDocument::StyleSheetResource, QUrl("help_style.css"), cssData);

        file.close();
    }
}

} // end namespace ito
