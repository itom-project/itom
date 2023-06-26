Add the following section to your settings ini file:

[ApplicationStyle]
style=plastique
cssFile=Qitom/styles/darkStyleSheet/style.qss
rccFile=Qitom/styles/darkStyleSheet/style.rcc

Before create the rcc file using the tool rcc from Qt with the following command:
rcc -o style.rcc -binary style.qrc
