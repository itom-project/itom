#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARIES
# GLEW_RUNTIME_LIBRARIES
#
FIND_PATH( GLEW_DIR include/GL/glew.h  )

#set(GLEW_LIBRARIES "")
#set(GLEW_RUNTIME_LIBRARIES "")

IF(WIN32)

    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h PATHS ${GLEW_DIR} PATH_SUFFIXES include DOC "The directory where GL/glew.h resides")
                    
    FIND_LIBRARY( GLEW_LIBRARY glew32 PATHS ${GLEW_DIR} ${GLEW_INCLUDE_PATH} PATH_SUFFIXES lib DOC "The GLEW shared library" )
    FIND_FILE( GLEW_RUNTIME_LIBRARIES glew32.dll PATHS ${GLEW_DIR} ${GLEW_INCLUDE_PATH} PATH_SUFFIXES bin )
    
ELSE (WIN32)
    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h /usr/include /usr/local/include /sw/include /opt/local/include DOC "The directory where GL/glew.h resides")
    FIND_LIBRARY( GLEW_LIBRARY NAMES GLEW glew PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /sw/lib /opt/local/lib DOC "The GLEW library")
    SET(GLEW_RUNTIME_LIBRARIES "")
ENDIF (WIN32)
    

IF (EXISTS "${GLEW_INCLUDE_PATH}")
    SET(GLEW_LIBRARIES ${GLEW_LIBRARY})
    SET( GLEW_FOUND true CACHE BOOL "" FORCE)
ELSE ()
    SET(GLEW_LIBRARIES "")
    SET( GLEW_FOUND false CACHE BOOL "" FORCE)
ENDIF ()

MARK_AS_ADVANCED( GLEW_FOUND GLEW_RUNTIME_LIBRARIES GLEW_LIBRARIES )
