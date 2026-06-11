"""ANSI escape codes
===================

This demo shows some possibilities of ANSI escape codes."""

###############################################################################
# Change the text color

###############################################################################
# set the text colors to pre-defined values, followed by a reset
print("\u001b[30m A \u001b[31m B \u001b[32m C \u001b[33m D \u001b[0mText after reset")
print("\u001b[34m E \u001b[35m F \u001b[36m G \u001b[37m H \u001b[0mText after reset")

###############################################################################
# use the prefix 1 to choose are more lighter text color
print(
    "\u001b[30;1m A \u001b[31;1m B \u001b[32;1m C \u001b[33;1m D \u001b[0mText after reset"
)
print(
    "\u001b[34;1m E \u001b[35;1m F \u001b[36;1m G \u001b[37;1m H \u001b[0mText after reset"
)

###############################################################################
# now the same with the background color
print("\u001b[40m A \u001b[41m B \u001b[42m C \u001b[43m D \u001b[0mText after reset")
print("\u001b[44m E \u001b[45m F \u001b[46m G \u001b[47m H \u001b[0mText after reset")
print(
    "\u001b[40;1m A \u001b[41;1m B \u001b[42;1m C \u001b[43;1m D \u001b[0mText after reset"
)
print(
    "\u001b[44;1m E \u001b[45;1m F \u001b[46;1m G \u001b[47;1m H \u001b[0mText after reset"
)

###############################################################################
# use text bold, underline, italic or combine them
print(
    "\u001b[1;3;4mText bold, underline and italic \u001b[22mRemove bold only "
    "\u001b[23;1mRemove italic and set bold again \u001b[0mReset all"
)
