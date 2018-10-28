// work around following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5

#pragma push_macro("slots")
#undef slots
#include "Python.h"
#pragma pop_macro("slots")