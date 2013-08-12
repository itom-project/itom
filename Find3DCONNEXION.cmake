#
# Try to find 3DCONNEXION library and include path.
# Once done this will define
#
# 3DCONNEXION_FOUND
# 3DCONNEXION_INCLUDE_PATH
# 3DCONNEXION_LIBRARIES
# 3DCONNEXION_RUNTIME_LIBRARIES
# 3DCONNEXION_SIAPP
# 3DCONNEXION_SPWMATH
# 3DCONNEXION_SIAPPMT
# 3DCONNEXION_SPWMATHMT
#
FIND_PATH( 3DCONNEXION_DIR include/inc/si.h  )

#set(GLEW_LIBRARIES "")
#set(GLEW_RUNTIME_LIBRARIES "")

set(3DCONNEXION_INCLUDE_PATH "${3DCONNEXION_DIR}/inc" CACHE PATH "3DCONNEXION_DIR include dir")

if(CMAKE_CL_64)
    set(3DCONNEXION_LIBRARIES ${3DCONNEXION_DIR}/lib/x64 CACHE PATH "libraries of 3DCONNEXION")
else(CMAKE_CL_64)
    set(3DCONNEXION_LIBRARIES ${3DCONNEXION_DIR}/lib/x86 CACHE PATH "libraries of 3DCONNEXION")
endif(CMAKE_CL_64)


IF(WIN32)

    FIND_PATH( 3DCONNEXION_INCLUDE_PATH inc/si.h PATHS ${3DCONNEXION_DIR} PATH_SUFFIXES include DOC "The directory where inc/si.h resides")
    
    set(3DCONNEXION_SIAPP optimized ${3DCONNEXION_LIBRARIES}/siapp.lib debug ${3DCONNEXION_LIBRARIES}/siappD.lib)
    #set(3DCONNEXION_SPWMATH optimized ${3DCONNEXION_LIBRARIES}/spwmath.lib debug ${3DCONNEXION_LIBRARIES}/spwmathD.lib)
    #set(3DCONNEXION_SIAPPMT optimized ${3DCONNEXION_LIBRARIES}/siappMT.lib debug ${3DCONNEXION_LIBRARIES}/siappMTD.lib)
    #set(3DCONNEXION_SPWMATHMT optimized ${3DCONNEXION_LIBRARIES}/spwmathMT.lib debug ${3DCONNEXION_LIBRARIES}/spwmathMTD.lib)
    set(3DCONNEXION_SPWMATH optimized ${3DCONNEXION_LIBRARIES}/spwmath.lib)
    set(3DCONNEXION_SIAPPMT optimized ${3DCONNEXION_LIBRARIES}/siappMT.lib)
    set(3DCONNEXION_SPWMATHMT optimized ${3DCONNEXION_LIBRARIES}/spwmathMT.lib)
    #FIND_FILE( 3DCONNEXION_RUNTIME_LIBRARIES glew32.dll PATHS ${GLEW_DIR} ${GLEW_INCLUDE_PATH} PATH_SUFFIXES bin )
    
ELSE (WIN32)
    #FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h /usr/include /usr/local/include /sw/include /opt/local/include DOC "The directory where GL/glew.h resides")
    #FIND_LIBRARY( GLEW_LIBRARY NAMES GLEW glew PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /opt/local/lib DOC "The GLEW library")
    #SET(3DCONNEXION_RUNTIME_LIBRARIES "")
ENDIF (WIN32)

if(EXISTS "${3DCONNEXION_INCLUDE_PATH}")
    set(3DCONNEXION_FOUND true)
else(EXISTS "${3DCONNEXION_INCLUDE_PATH}")
    set(3DCONNEXION_FOUND false)
    set(ERR_MSG "Please specify 3DCONNEXION directory using 3DCONNEXION_DIR env. variable")
endif(EXISTS "${3DCONNEXION_INCLUDE_PATH}")
##====================================================


##====================================================
## Print message
##----------------------------------------------------
if(NOT 3DCONNEXION_FOUND)
    message(STATUS "WARNING: 3DCONNEXION was not found.")
endif(NOT 3DCONNEXION_FOUND)
##====================================================

MARK_AS_ADVANCED(3DCONNEXION_FOUND 3DCONNEXION_SIAPP 3DCONNEXION_SPWMATH 3DCONNEXION_SIAPPMT 3DCONNEXION_SPWMATHMT)
