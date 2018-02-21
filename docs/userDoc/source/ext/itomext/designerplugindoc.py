# -*- coding: utf-8 -*-

from docutils import nodes, utils, statemachine
from docutils.parsers.rst.roles import set_classes
from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.misc import Include

import itom
import __main__
import os
import re

class DesignerPluginDocInclude(Include):
    """."""
    def run(self):
        
        rstPath = os.path.join(itom.getAppPath(), 'designer/docs/%s' % self.arguments[0])
        rstPath = rstPath.replace('\\\\', '/') #include command want's to have slash only
        rstPath = rstPath.replace('\\', '/') #include command want's to have slash only
        rstFileName = os.path.join(rstPath, '%s.rst' % self.arguments[0])
        rstFileName = rstFileName.replace('\\\\', '/') #include command want's to have slash only
        rstFileName = rstFileName.replace('\\', '/') #include command want's to have slash only
    
        if os.path.exists(rstFileName):
            #load the rst file at the build location and check for .. figure:: img or .. image:: img
            #if img seems to be relative, replace it by the absolute location
            with open(rstFileName, 'rt') as f:
                lines = [line for line in f]
                
            pattern = re.compile("(.*)(\.\. figure:: |\.\. image:: )(.*\.[a-zA-Z0-9]{1,4})(.*)")
        
            for i in range(0, len(lines)):
                line = lines[i]
                m = pattern.match(line)
                if m:
                    prefix = m.group(1)
                    directive = m.group(2)
                    file = m.group(3)
                    suffix = m.group(4)
                    if len(file) > 2 and file[1] != ":":
                        #make file absolute
                        if file.startswith('/'):
                            file = rstPath + file
                        else:
                            file = rstPath + '/' + file
                        lines[i] = prefix + directive + file + suffix + '\n'
            
            with open(rstFileName, 'wt') as f:
                for line in lines:
                    f.write(line)
            #end rst file manipulation
            
            self.arguments[0] = rstFileName
            return Include.run(self)
            
        else:
            #not found
            text = "Documentation for designer plugin %s could not be found" % self.arguments[0]
            tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
            source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
    

def setup(app):
    """Install the plugin.

    :param app: Sphinx application context.
    """
    app.add_directive('include-designerplugindoc', DesignerPluginDocInclude)