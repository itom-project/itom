# -*- coding: utf-8 -*-

from docutils import nodes, utils, statemachine
from docutils.parsers.rst.roles import set_classes
from docutils.parsers.rst import directives
from sphinx.util.compat import Directive
import itom
import __main__

def getPluginInfo(env, plugin):
    
    if (not hasattr(env,"itom_plugin_infos")):
        env.itom_plugin_infos = {}
    
    p = str(plugin)
    if (not p in env.itom_plugin_infos):
        if (itom.pluginLoaded(p)):
            env.itom_plugin_infos[p] = itom.pluginHelp(p,True)
        else:
            dummy = {"license":"unknown license", "author":"unknown author", "description":"no description","type":"unknown type","version":"unknown version", "detaildescription":"no description"}
            if ("pluginOverloads" in __main__.__dict__):
                if p in __main__.__dict__["pluginOverloads"]:
                    dummy.update(__main__.__dict__["pluginOverloads"][p])
                    return dummy
                else:
                    raise RuntimeError("Plugin not available")
    
    return env.itom_plugin_infos[p]

def pluginAuthor_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Author of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPluginInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["author"], pluginInfo["author"])
    return [node], []

def pluginSummary_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Description of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPluginInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["description"], pluginInfo["description"])
    
    return [node], []

def pluginLicense_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """License of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPluginInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["license"], pluginInfo["license"])
    
    return [node], []

def pluginType_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Type of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPluginInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["type"], pluginInfo["type"])
    
    return [node], []

def pluginVersion_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Version of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPluginInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["version"], pluginInfo["version"])
    
    return [node], []

class PluginSummaryExtended(Directive):
    """display detailed description from plugin (the description is inserted into the loaded rst-code such that
    it is also parsed using the rst-parser."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            pluginInfo = getPluginInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plugin information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]

        text = pluginInfo["detaildescription"]
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []

class PluginInitParams(Directive):
    """puts the list with all mandatory and optional parameters for initializing the dataIO or actuator
    plugin into the rst-code for further parsing"""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            pluginInfo = getPluginInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plugin information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]
        
        textlist = ["\n"]
        
        if ("Mandatory Parameters" in pluginInfo):
            for p in pluginInfo["Mandatory Parameters"]:
                text = "* **%s**: {%s} %s\n    %s" % (p["name"],p["type"],self.parseContent(p),p["info"])
                textlist .append(text)
        
        if ("Optional Parameters" in pluginInfo):
            for p in pluginInfo["Optional Parameters"]:
                text = "* **%s**: {%s, optional} %s\n    %s" % (p["name"],p["type"],self.parseContent(p),p["info"])
                textlist .append(text)
        
        textlist.append("\n")
        text = "\n".join(textlist)
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []
    
    def parseContent(self,param):
        if ("min" in param and "max" in param):
            if ("step" in param and not (param["step"] is None)):
                content = "[%s,%s], default: %s, step: %s" % (param["min"],param["max"],param["value"],param["step"])
            else:
                content = "[%s,%s], default: %s" % (param["min"],param["max"],param["value"])
        elif ("value" in param):
            content = "default: %s" % param["value"]
        else:
            content = ""
        return content

class PluginFilterList(Directive):
    """."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required, "overviewonly":directives.flag}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            pluginInfo = getPluginInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plugin information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]
        
        if (not "filter" in pluginInfo):
            return [self.state.document.reporter.warning('given plugin is no algorithm plugin', line=self.lineno)]
        
        textlist = []
        if (pluginInfo["filter"] is None):
            textlist.append("The plugin does not contain any filters")
        else:
            if ("overviewonly" in self.options):
                for f in pluginInfo["filter"]:
                    textlist.append("#. %s" % f)
            else:
                for f in pluginInfo["filter"]:
                    t = ".. py:function:: %s(...)" % f
                    t += "\n    \n    ...\n"
                    textlist.append(t)
        
        text = "\n".join(textlist)
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def getPlotInfo(env, plugin):
    
    if (not hasattr(env,"itom_designerPlugin_infos")):
        env.itom_designerPlugin_infos = {}
    
    p = str(plugin)
    if (not p in env.itom_designerPlugin_infos):
        if (itom.plotLoaded(p)):
            env.itom_designerPlugin_infos[p] = itom.plotHelp(p,True)
        else:
            dummy = {"license":"unknown license", "author":"unknown author", "description":"no description","type":"unknown type","version":"unknown version", "detaildescription":"no description"}
            if ("pluginOverloads" in __main__.__dict__):
                if p in __main__.__dict__["pluginOverloads"]:
                    dummy.update(__main__.__dict__["pluginOverloads"][p])
                    return dummy
                else:
                    raise RuntimeError("designerPlugin not available")
    
    return env.itom_designerPlugin_infos[p]

def plotAuthor_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Author of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["author"], pluginInfo["author"])
    return [node], []

def plotSummary_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Description of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["description"], pluginInfo["description"])
    
    return [node], []

def plotLicense_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """License of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["license"], pluginInfo["license"])
    
    return [node], []

def plotType_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Type of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["type"], pluginInfo["type"])
    
    return [node], []

def plotInputType_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Input type Type of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["inputtype"], pluginInfo["inputtype"])
    
    return [node], []

def plotFormats_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Input type Type of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["dataformats"], pluginInfo["dataformats"])
    
    return [node], []
    
def plotFeatures_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Input type Type of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["features"], pluginInfo["features"])
    
    return [node], []

def plotVersion_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Version of plugin.

    Returns 2 part tuple containing list of nodes to insert into the
    document and a list of system messages.  Both are allowed to be
    empty.

    :param name: The role name used in the document.
    :param rawtext: The entire markup snippet, with role.
    :param text: The text marked with the role.
    :param lineno: The line number where rawtext appears in the input.
    :param inliner: The inliner instance that called us.
    :param options: Directive options for customization.
    :param content: The directive content for customization.
    """
    try:
        pluginInfo = getPlotInfo(inliner.document.settings.env.app.builder.env, text)
    except:
        msg = inliner.reporter.error(
            'Error getting plugin information from plugin "%s".' % text, line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]

    set_classes(options)
    node = nodes.inline(pluginInfo["version"], pluginInfo["version"])
    
    return [node], []

class PlotSummaryExtended(Directive):
    """display detailed description from plugin (the description is inserted into the loaded rst-code such that
    it is also parsed using the rst-parser."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            pluginInfo = getPlotInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plugin information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]

        text = pluginInfo["detaildescription"]
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []

class PlotProperties(Directive):
    """."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required, "overviewonly":directives.flag}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            plotInfo = getPlotInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plot information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]
        
        if (not "properties" in plotInfo):
            return [self.state.document.reporter.warning('given PLOT is not an itom designer plugin', line=self.lineno)]
        
        textlist = []
        if (plotInfo["properties"] is None or len(plotInfo["properties"]) == 0):
            textlist.append("This designerPlugin / plot has no public properties ")
        else:
            if ("overviewonly" in self.options):
                for f in plotInfo["properties"]:
                    textlist.append("#. %s" % f)
            else:
                for f in plotInfo["properties"]:
                    t = ".. py:attribute:: %s" % f
                    t += "\n%s\n    ...\n" % plotInfo["properties"][f]
                    textlist.append(t)
        
        text = "\n".join(textlist)
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []

class PlotSlots(Directive):
    """."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required, "overviewonly":directives.flag}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            plotInfo = getPlotInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plot information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]
        
        if (not "slots" in plotInfo):
            return [self.state.document.reporter.warning('given PLOT is not an itom designer plugin', line=self.lineno)]
        
        textlist = []
        if (plotInfo["slots"] is None or len(plotInfo["slots"]) == 0):
            textlist.append("This designerPlugin / plot has no public slots ")
        else:
            if ("overviewonly" in self.options):
                for f in plotInfo["slots"]:
                    textlist.append("#. %s" % plotInfo["slots"][f])
            else:
                for f in plotInfo["slots"]:
                    t = ".. py:function:: %s(...)" % f
                    t += "\n%s\n\n" % plotInfo["slots"][f]
                    textlist.append(t)
        
        text = "\n".join(textlist)
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []

class PlotSignals(Directive):
    """."""
    has_content = False
    required_arguments = 0
    optional_arguments = 1
    option_spec = {"plugin":directives.unchanged_required, "overviewonly":directives.flag}

    def run(self):
        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
        
        if (not "plugin" in self.options):
            return [self.state.document.reporter.warning('option "plugin" is missing', line=self.lineno)]
        
        try:
            plotInfo = getPlotInfo(self.state.document.settings.env, self.options["plugin"])
        except Exception:
            return [self.state.document.reporter.warning('Error getting plot information from plugin "%s"' % (self.options["plugin"]), line=self.lineno)]
        
        if (not "signals" in plotInfo):
            return [self.state.document.reporter.warning('given PLOT is not an itom designer plugin', line=self.lineno)]
        
        textlist = []
        if (plotInfo["signals"] is None or len(plotInfo["signals"]) == 0):
            textlist.append("This designerPlugin / plot has no public signals ")
        else:
            if ("overviewonly" in self.options):
                for f in plotInfo["signals"]:
                    textlist.append("#. %s" % plotInfo["signals"][f])
            else:
                for f in plotInfo["signals"]:
                    t = ".. py:function:: %s(...)" % f
                    t += "\n%s\n\n" % plotInfo["signals"][f]
                    textlist.append(t)
        
        text = "\n".join(textlist)
        lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
        self.state_machine.insert_input(lines, source)
        return []

def setup(app):
    """Install the plugin.

    :param app: Sphinx application context.
    """
    app.add_role('pluginauthor', pluginAuthor_role)
    app.add_role('pluginsummary', pluginSummary_role)
    app.add_role('pluginlicense', pluginLicense_role)
    #app.add_role('pluginsummaryextended', pluginSummary2_role)
    app.add_role('plugintype', pluginType_role)
    app.add_role('pluginversion', pluginVersion_role)
    
    app.add_directive('pluginsummaryextended', PluginSummaryExtended)
    app.add_directive('plugininitparams', PluginInitParams)
    app.add_directive('pluginfilterlist', PluginFilterList)

    app.add_role('plotauthor', plotAuthor_role)
    app.add_role('plotsummary', plotSummary_role)
    app.add_role('plotlicense', plotLicense_role)
    app.add_role('plottype', plotType_role)
    app.add_role('plotinputtype', plotInputType_role)
    app.add_role('plotdataformats', plotFormats_role)
    app.add_role('plotfeatures', plotFeatures_role)
    app.add_role('plotversion', plotVersion_role)
    
    app.add_directive('plotsummeryextended', PlotSummaryExtended)
    app.add_directive('plotproperties', PlotProperties)
    app.add_directive('plotslots', PlotSlots)
    app.add_directive('plotsignals', PlotSignals)