// work around following: https://stackoverflow.com/questions/23068700/embedding-python3-in-qt-5

#pragma push_macro("slots")
#undef slots
#include "Python.h"
#include "datetime.h"
#pragma pop_macro("slots")

// use this macro if a method of the C-API of the Python datetime module should be used.
#define Itom_PyDateTime_IMPORT if (PyDateTimeAPI == nullptr) {PyDateTime_IMPORT;}