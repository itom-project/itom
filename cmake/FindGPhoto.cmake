# - Find GPhoto
# Find libpghoto includes and library
# This module defines
# LIBGHPOTO_INCLUDE_DIR, where to find libgphotoXXX.h, etc.
# LIBGPHOTO_LIBRARIES, the libraries needed to use libgphoto.
# LIBGPHOTO_FOUND, If false, do not try to use libgphoto.
# also defined, but not for general use are
# LIBGPHOTO_LIBRARY, where to find the libgphoto library.

set(LIBGPHOTO_FOUND false)

find_path(LIBGPHOTO_DIR gphoto2.h PATHS /usr/local/include /usr/local/include/gphoto2 /usr/include /usr/include/gphoto2 /opt/local/lib /opt/local/lib/gphoto2 DOC "Root directory of libgphoto")
find_path(LIBGPHOTO_INCLUDE_DIR gphoto2.h PATHS /usr/local/include /usr/local/include/gphoto /usr/include /usr/include/gphoto2 /opt/local/lib /opt/local/lib/gphoto2 ${LIBGPHOTO_DIR})

find_library(LIBGPHOTO_LIBRARY NAMES gphoto2 libgphoto2 PATHS /usr/lib /usr/local/lib /opt/locala/lib ${LIBGPHOTO_DIR})
find_library(LIBGPHOTO_PORT_LIBRARY NAMES gphoto2_port libgphoto2_port PATHS /usr/lib /usr/local/lib /opt/locala/lib ${LIBGPHOTO_DIR})

if(LIBGPHOTO_LIBRARY AND LIBGPHOTO_PORT_LIBRARY AND LIBGPHOTO_INCLUDE_DIR)
    set(LIBGPHOTO_LIBRARIES ${LIBGPHOTO_LIBRARY} ${LIBGPHOTO_PORT_LIBRARY})
    set(LIBGPHOTO_FOUND true)
else (LIBGPHOTO_LIBRARY AND LIBGPHOTO_PORT_LIBRARY AND LIBGPHOTO_INCLUDE_DIR)
    set(LIBGPHOTO_FOUND false)
    set(LIBGPHOTO_LIBRARIES "")
endif(LIBGPHOTO_LIBRARY AND LIBGPHOTO_PORT_LIBRARY AND LIBGPHOTO_INCLUDE_DIR)


if(LIBGPHOTO_FOUND)
   if(NOT LIBGPHOTO_FIND_QUIETLY)
      message(STATUS "Found libgphoto: ${LIBGPHOTO_LIBRARIES}")
   endif(NOT LIBGPHOTO_FIND_QUIETLY)
else (LIBGPHOTO_FOUND)
   if(LIBGPHOTO_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find libgphoto library")
   endif(LIBGPHOTO_FIND_REQUIRED)
endif(LIBGPHOTO_FOUND)

mark_as_advanced(LIBGPHOTO_LIBRARY)
