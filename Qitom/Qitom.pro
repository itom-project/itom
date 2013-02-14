################################################################
# itom - Qitom
# Institut für Technische Optik, Universität Stuttgart
################################################################

# load constants
include( ../itom_properties.pri )

################################################################
# qmake internal options
################################################################

CONFIG           += warn_on
#CONFIG           += silent


################################################################
# release/debug mode
################################################################

win32 {
    # On Windows you can't mix release and debug libraries.
    # The designer is built in release mode. If you like to use it
    # you need a release version. For your own application development you
    # might need a debug version.
    # Enable debug_and_release + build_all if you want to build both.
}
else {
    #VER_MAJ           = $${QWT_VER_MAJ}
    #VER_MIN           = $${QWT_VER_MIN}
    #VER_PAT           = $${QWT_VER_PAT}
    #VERSION           = $${QWT_VERSION}
}

linux-g++ {
    # CONFIG           += separate_debug_info
}

TEMPLATE = app
TARGET = qitom
QT += core gui sql xml opengl svg uitools designer
CONFIG += help
win32 {
    DEFINES += _WINDOWS QT_LARGEFILE_SUPPORT QT_DLL
}
DEFINES += QT_SQL_LIB QT_XML_LIB QT_HAVE_MMX QT_HAVE_3DNOW QT_HAVE_SSE QT_HAVE_MMXEXT QT_HAVE_SSE2 QT_OPENGL_LIB ITOM_CORE

INCLUDEPATH +=  . \
                ./.. \
                $${ITOM_PYTHONDIR} \
                $${ITOM_OPENCVDIR} \
                $${ITOM_QTDIR}/include/qt4 \
                $${ITOM_QTDIR} \
                $${ITOM_QSCINTILLADIR}/Qt4 \
#                ./../qwtplot \
                $${ITOM_PCLDIR_INCLUDE}

################################################################
# LIBS: It is important to keep the right order of lib-inclusion.
# if lib1 needs methods of lib2, include lib1 first before lib2.
# see: http://stackoverflow.com/questions/1095298/
###############################################################
LIBS += -L$${ITOM_QTLIBDIR}/ -lQtHelp -lQtSvg -lQtSql -lQtXml -lQtOpenGL -lQtGui -lQtCore -lQtUiTools
win32 {
    LIBS += -L$${ITOM_QTDIR}/lib/ -lqscintilla2
    LIBS += -L$${ITOM_PYTHONLIBDIR}/ -lpython32
    LIBS += -L$${ITOM_OPENCVLIBDIR}/ -lopencv_core$${ITOM_OPENCVVER}
}
else {
    LIBS += -L$${ITOM_QSCINTILLADIR}/Qt4/ -lqscintilla2
    LIBS += -L$${ITOM_PYTHONLIBDIR}/ -lpython3.2mu
    LIBS += -L$${ITOM_OPENCVLIBDIR}/ -lopencv_core

    versiontarget.target = version.h
    versiontarget.commands = $$PWD/../tools/subwcrev.sh $$PWD $$PWD/version.tmpl $$PWD/version.h
    versiontarget.depends = FORCE

    PRE_TARGETDEPS += version.h
    QMAKE_EXTRA_TARGETS += versiontarget
}
LIBS += -lgomp
#LIBS += -L$${ITOM_LAPACKLIBDIR}/ -llapack
#LIBS += -L$${ITOM_LAPACKLIBDIR}/ -lblas

DEPENDPATH += .

CONFIG(debug,debug|release) {
    DESTDIR = ./../debug
    MOC_DIR += ./debug/GeneratedFiles
    OBJECTS_DIR += debug
    UI_DIR += ./debug/GeneratedFiles
    RCC_DIR += ./debug/GeneratedFiles

    LIBS += -L./../debug/lib/ -lDataObject
    LIBS += -L./../debug/lib/ -lPointCloud
#    LIBS += -L./../debug/lib/ -lqwtplot
}
CONFIG(release,debug|release) {
    DESTDIR = ./../release
    MOC_DIR += ./release/GeneratedFiles
    OBJECTS_DIR += release
    UI_DIR += ./release/GeneratedFiles
    RCC_DIR += ./release/GeneratedFiles

    LIBS += -L./../release/lib/ -lDataObject
    LIBS += -L./../release/lib/ -lPointCloud
#    LIBS += -L./../release/lib/ -lqwtplot
}

pyfiles.target = itomDebugger.py
pyfiles.commands = cp $$PWD/../*.py $$DESTDIR
pyfiles.depends = FORCE
POST_TARGETDEPS += itomDebugger.py
QMAKE_EXTRA_TARGETS += pyfiles

pyscripts.target = itom-packages
pyscripts.commands = rsync -r -u --exclude='.*' $$PWD/../itom-packages $$DESTDIR
pyscripts.depends = FORCE
POST_TARGETDEPS += itom-packages
QMAKE_EXTRA_TARGETS += pyscripts

settings.target = itomSettings
settings.commands = rsync -r -u --exclude='.*' $$PWD/../itomSettings $$DESTDIR
settings.depends = FORCE
POST_TARGETDEPS += itomSettings
QMAKE_EXTRA_TARGETS += settings

QMAKE_LFLAGS += -Wl,-R./:./lib/:../lib/
QMAKE_CFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp

#QMAKE_POST_LINK += copy References\*.dll  Debug\ &
#QMAKE_POST_LINK += copy References\*.dll ..\bin\ &
#QMAKE_POST_LINK += $${QMAKE_COPY} ./readme.txt $${DESTDIR}  #$$escape_expand(\n\t)

message($${OUT_PWD}/$${DESTDIR}   $${PWD})
resources.path = $${OUT_PWD}/$${DESTDIR}
resources.files += $${PWD}/../itom-packages/
resources.files += $${PWD}/../lib/
resources.files += $${PWD}/../*.py
#resources.files += /path/to/resources/dir2
#resources.files += /path/to/resources/dir3
#resources.files += /path/to/resources/dirn # and so on...
#resources.files += /path/to/resources/*    # don't forget the files in the root



INSTALLS += resources

include(Qitom.pri)
