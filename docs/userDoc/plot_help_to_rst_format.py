import re
import npyhelpformat2basicrst

def indentText(text, indentLevel = 4):
    lines = text.split("\n")
    return "\n    ".join(lines)

def parseProperties(properties):
    '''parses all Q_CLASSINFO("prop://...","...") entries to several rst py:attribute:: sections'''
    tmpl = ".. py:attribute:: %(name)s : %(type)s %(readonly)s\n    :noindex:\n    \n    %(text)s"
    readonlystr = "(readonly)"
    output = []
    for key in properties:
        text = properties[key]
        lines = text.split('\n')
        m = re.match(r"\{(.+)\}( -> (.+))+( \(readonly\))", lines[0])
        if m:
            lines[0] = m.group(3)
            d = {"name":key, "readonly":"(readonly)", "type":m.group(1), "text":indentText('\n'.join(lines))}
        else:
            m = re.match(r"\{(.+)\}( -> (.+))+", lines[0])
            try:
                lines[0] = m.group(3)
                d = {"name":key, "readonly":"", "type":m.group(1), "text":indentText('\n'.join(lines))}
            except Exception:
                d = {"name":key, "readonly":"", "type":"", "text":""}
        
        output.append(tmpl % d)
    return "\n\n".join(output)
    
def parseMethods(methods, appendix = None):
    '''parses all Q_CLASSINFO("prop://...","...") entries to several rst py:attribute:: sections'''
    if appendix is None:
        tmpl = ".. py:function:: %(signature)s\n    :noindex:\n    \n    %(text)s"
    else:
        tmpl = ".. py:function:: %(signature)s [%(appendix)s]\n    :noindex:\n    \n    %(text)s"
    
    output = []
    for key in methods:
        text = methods[key]
        idx = text.find(" -> ")
        if idx >= 0:
            signature = text[0:idx]
            text = text[idx+4:]
        else:
            signature = text
            text = ""
            
        m = re.match(r"(\w+)\s*\((.*)\)", signature)
        name = m.group(1)
        args = m.group(2).split(',')
        args = [arg[0:arg.find('{')].strip() for arg in args]
        signature = m.group(1) + '(' + ', '.join(args) + ')'
        converter = npyhelpformat2basicrst.ItomDocString(text)
        
        d = {"signature":signature, "appendix":appendix, "text":indentText(str(converter))}
        output.append(tmpl % d)
    return "\n\n".join(output)

availablePlots = plotHelp("*", True)
[result, ok] = ui.getItem("Choose a plot plugin", "Choose a plot plugin for which the documentation should be rendered, for", \
    availablePlots, currentIndex = 0, editable = False)
    
if ok:
    doc = plotHelp(result, True)
    
    properties = parseProperties(doc["properties"])
    slots = parseMethods(doc["slots"], "slot")
    signals = parseMethods(doc["signals"], "signal")
    
    if not properties.endswith('\n'):
        properties += '\n'
        
    if not slots.endswith('\n'):
        slots += '\n'
        
    if not signals.endswith('\n'):
        signals += '\n'
    
    clc()
    
    if (properties != ""):
        print("Properties\n-------------------------\n\n")
        print(properties)
    if (slots != ""):
        print("Slots\n-------------------------\n\n")
        print(slots)
    if (signals != ""):
        print("Signals\n-------------------------\n\n")
        print(signals)
    