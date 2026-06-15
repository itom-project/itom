"""
formlayout
==========

Module creating Qt form dialogs/layouts to edit various type of parameters


formlayout License Agreement (MIT License)
------------------------------------------

Copyright (c) 2009 Pierre Raybaut

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

# History:
# 1.0.10: added float validator (disable "Ok" and "Apply" button when not valid)
# 1.0.7: added support for "Apply" button
# 1.0.6: code cleaning

__version__ = "1.0.10"
__license__ = __doc__

import copy
import datetime
import logging
from numbers import Integral, Real

from matplotlib import cbook, colors as mcolors
import itom

_log = logging.getLogger(__name__)

BLACKLIST = {"title", "label"}

__dialogCache__ = (
    []
)  # cache to current instances of DialogEditProperties (as long as the real dialog is visible)


class FormWidget:
    def __init__(self, data, title, comment="", with_margin=False, parentUiItem=None):
        """
        Parameters
        ----------
        data : list of (label, value) pairs
            The data to be edited in the form.
        comment : str, optional

        with_margin : bool, optional, default: False
            If False, the form elements reach to the border of the widget.
            This is the desired behavior if the FormWidget is used as a widget
            alongside with other widgets such as a QComboBox, which also do
            not have a margin around them.
            However, a margin can be desired if the FormWidget is the only
            widget within a container, e.g. a tab in a QTabWidget.
        parent : QWidget or None
            The parent widget.
        """
        self.formWidget = parentUiItem.call(
            "addFormWidget", title, comment, with_margin
        )
        self.data = copy.deepcopy(data)
        self.widgets = []

    def _tuple_to_font(self, tup):
        """
        Create a itom.font from tuple:
            (family [string], size [int], italic [bool], bold [bool])
        """
        if not (
            isinstance(tup, tuple)
            and len(tup) == 4
            and itom.font.isFamilyInstalled(tup[0])
            and isinstance(tup[1], Integral)
            and isinstance(tup[2], bool)
            and isinstance(tup[3], bool)
        ):
            return None

        family, size, italic, bold = tup
        weight = itom.font.Normal
        if bold:
            weigt = itom.font.Bold
        font = itom.font(family, size, weight=weight, italic=italic)
        return font

    def _to_color(self, color):
        """Create a itom.rgba from a matplotlib color"""
        try:
            r, g, b, a = mcolors.to_rgba(color)
        except ValueError:
            cbook._warn_external("Ignoring invalid color %r" % color)
            return itom.rgba(0, 0, 0)  # return invalid QColor
        return itom.rgba(int(r * 255), int(g * 255), int(b * 255), int(a * 255))

    def setup(self):
        for label, value in self.data:
            font = self._tuple_to_font(value)

            if label is None and value is None:
                # Separator: (None, None)
                self.formWidget.call("addSeparator")
                self.widgets.append(None)
                continue
            elif label is None:
                # Comment
                self.formWidget.call("addComment", str(value))
                self.widgets.append(None)
                continue
            elif font is not None:
                field = self.formWidget.call("addFont", label, font)
            elif label.lower() not in BLACKLIST and mcolors.is_color_like(value):
                field = self.formWidget.call("addColor", label, self._to_color(value))
            elif isinstance(value, str):
                field = self.formWidget.call("addText", label, value)
            elif isinstance(value, (list, tuple)):
                if isinstance(value, tuple):
                    value = list(value)
                # Note: get() below checks the type of value[0] in self.data so
                # it is essential that value gets modified in-place.
                # This means that the code is actually broken in the case where
                # value is a tuple, but fortunately we always pass a list...
                selindex = value.pop(0)
                if isinstance(value[0], (list, tuple)):
                    keys = [key for key, _val in value]
                    value = [val for _key, val in value]
                else:
                    keys = value

                if selindex in value:
                    selindex = value.index(selindex)
                elif selindex in keys:
                    selindex = keys.index(selindex)
                elif not isinstance(selindex, Integral):
                    _log.warning(
                        "index '%s' is invalid (label: %s, value: %s)",
                        selindex,
                        label,
                        value,
                    )
                    selindex = 0
                field = self.formWidget.call("addComboBox", label, value, selindex)
            elif isinstance(value, bool):
                field = self.formWidget.call("addBoolean", label, value)
            elif isinstance(value, Integral):
                field = self.formWidget.call("addIntegral", label, value)
            elif isinstance(value, Real):
                field = self.formWidget.call("addReal", label, value)
            elif isinstance(value, datetime.datetime):
                field = self.formWidget.call(
                    "dDatetimeUTC",
                    label,
                    value.year,
                    value.month,
                    value.day,
                    value.hour,
                    value.minute,
                    value.second,
                    value.microsecond,
                )
                field = QtWidgets.QDateTimeEdit(self)
                field.setDateTime(value)
            elif isinstance(value, datetime.date):
                field = self.formWidget.call(
                    "addDate", label, value.year, value.month, value.day
                )
            else:
                field = self.formWidget.call("addText", label, repr(value))
            self.widgets.append(field)

    def get(self):
        valuelist = []
        for index, (label, value) in enumerate(self.data):
            field = self.widgets[index]
            if label is None:
                # Separator / Comment
                continue
            elif self._tuple_to_font(value) is not None:
                font = field["font"]
                value = (
                    font.family,
                    font.pointSize,
                    font.italic,
                    font.weight == itom.font.Bold,
                )
            elif isinstance(value, str) or mcolors.is_color_like(value):
                value = str(field["text"])
            elif isinstance(value, (list, tuple)):
                index = int(field["currentIndex"])
                if isinstance(value[0], (list, tuple)):
                    value = value[index][0]
                else:
                    value = value[index]
            elif isinstance(value, bool):
                value = field["checked"]
            elif isinstance(value, Integral):
                value = field["value"]
            elif isinstance(value, Real):
                value = field["value"]
            elif isinstance(value, datetime.datetime):
                value = field.dateTime().toPyDateTime()
            elif isinstance(value, datetime.date):
                value = field.date().toPyDate()
            else:
                value = eval(str(field["text"]))
            valuelist.append(value)
        return valuelist


class FormComboWidget:
    def __init__(self, datalist, title, comment, parentUiItem):
        self.comboWidget = parentUiItem.call("addFormComboWidget", title, comment)
        self.widgetlist = []
        for data, title, comment in datalist:
            widget = FormWidget(
                data, title, comment=comment, parentUiItem=self.comboWidget
            )
            self.widgetlist.append(widget)

    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    def get(self):
        return [widget.get() for widget in self.widgetlist]


class FormTabWidget:
    def __init__(self, datalist, title, comment, parentUiItem):
        self.tabWidget = parentUiItem.call("addFormTabWidget", title, comment)
        self.widgetlist = []
        for data, title, comment in datalist:
            if len(data[0]) == 3:
                widget = FormComboWidget(
                    data, title, comment=comment, parentUiItem=self.tabWidget
                )
            else:
                widget = FormWidget(
                    data,
                    title,
                    with_margin=True,
                    comment=comment,
                    parentUiItem=self.tabWidget,
                )
            self.widgetlist.append(widget)

    def setup(self):
        for widget in self.widgetlist:
            widget.setup()

    def get(self):
        return [widget.get() for widget in self.widgetlist]


class DialogEditProperties:
    """Form Dialog"""

    def __init__(self, matplotlibplotUiItem, data, title="", comment="", apply=None):

        self.apply_callback = apply

        self.dialog = matplotlibplotUiItem.call(
            "createDialogEditProperties", not apply is None, title
        )
        self.dialog.connect("accepted()", self.accepted)
        self.dialog.connect("applied()", self.applied)
        self.dialog.connect("rejected()", self.rejected)
        # self.dialog["modal"] = True

        self.float_fields = []

        # Form
        if isinstance(data[0][0], (list, tuple)):
            contenttype = "FormTabWidget"
            self.formwidget = FormTabWidget(
                data, title="", comment=comment, parentUiItem=self.dialog
            )
        elif len(data[0]) == 3:
            contenttype = "FormComboWidget"
            self.formwidget = FormComboWidget(
                data, title="", comment=comment, parentUiItem=self.dialog
            )
        else:
            contenttype = "FormWidget"
            self.formwidget = FormWidget(
                data, title="", comment=comment, parentUiItem=self.dialog
            )

        self.formwidget.setup()

    def show(self):
        self.dialog.call("show")

    def register_float_field(self, field):
        self.float_fields.append(field)

    def update_buttons(self):
        valid = True
        for field in self.float_fields:
            if not is_edit_valid(field):
                valid = False
        for btn_type in (
            QtWidgets.QDialogButtonBox.Ok,
            QtWidgets.QDialogButtonBox.Apply,
        ):
            btn = self.bbox.button(btn_type)
            if btn is not None:
                btn.setEnabled(valid)

    def accepted(self):
        self.data = self.formwidget.get()
        self.apply_callback(self.data)
        self._deleteDialog()
        global __dialogCache__
        __dialogCache__ = None

    def rejected(self):
        self.data = None
        self._deleteDialog()
        global __dialogCache__
        __dialogCache__ = None

    def applied(self):
        self.apply_callback(self.formwidget.get())

    def _deleteDialog(self):
        if not self.dialog is None:
            self.dialog.call("deleteLater")
            self.dialog = None

    def get(self):
        """Return form result"""
        return self.data


def fedit(matplotlibplotUiItem, data, title="", comment="", apply=None):
    """
    Create form dialog and return result
    (if Cancel button is pressed, return None)

    data: datalist, datagroup
    title: string
    comment: string
    parent: parent QWidget
    apply: apply callback (function)

    datalist: list/tuple of (field_name, field_value)
    datagroup: list/tuple of (datalist *or* datagroup, title, comment)

    -> one field for each member of a datalist
    -> one tab for each member of a top-level datagroup
    -> one page (of a multipage widget, each page can be selected with a combo
       box) for each member of a datagroup inside a datagroup

    Supported types for field_value:
      - int, float, str, unicode, bool
      - colors: in Qt-compatible text form, i.e. in hex format or name (red,...)
                (automatically detected from a string)
      - list/tuple:
          * the first element will be the selected index (or value)
          * the other elements can be couples (key, value) or only values
    """

    dialog = DialogEditProperties(matplotlibplotUiItem, data, title, comment, apply)
    dialog.show()

    global __dialogCache__
    __dialogCache__ = dialog


if __name__ == "__main__":

    def create_datalist_example():
        return [
            ("str", "this is a string"),
            ("list", [0, "1", "3", "4"]),
            (
                "list2",
                [
                    "--",
                    ("none", "None"),
                    ("--", "Dashed"),
                    ("-.", "DashDot"),
                    ("-", "Solid"),
                    ("steps", "Steps"),
                    (":", "Dotted"),
                ],
            ),
            ("float", 1.2),
            (None, "Other:"),
            ("int", 12),
            ("font", ("Arial", 10, False, True)),
            ("color", "#123409"),
            ("bool", True),
            ("date", datetime.date(2010, 10, 10)),
            ("datetime", datetime.datetime(2010, 10, 10)),
        ]

    def create_datagroup_example():
        datalist = create_datalist_example()
        return (
            (datalist, "Category 1", "Category 1 comment"),
            (datalist, "Category 2", "Category 2 comment"),
            (datalist, "Category 3", "Category 3 comment"),
        )

    # --------- datalist example
    datalist = create_datalist_example()

    def apply_test(data):
        print("data:", data)

    print(
        "result:",
        fedit(
            datalist,
            title="Example",
            comment="This is just an <b>example</b>.",
            apply=apply_test,
        ),
    )

    # --------- datagroup example
    datagroup = create_datagroup_example()
    print("result:", fedit(datagroup, "Global title"))

    # --------- datagroup inside a datagroup example
    datalist = create_datalist_example()
    datagroup = create_datagroup_example()
    print(
        "result:",
        fedit(
            (
                (datagroup, "Title 1", "Tab 1 comment"),
                (datalist, "Title 2", "Tab 2 comment"),
                (datalist, "Title 3", "Tab 3 comment"),
            ),
            "Global title",
        ),
    )
