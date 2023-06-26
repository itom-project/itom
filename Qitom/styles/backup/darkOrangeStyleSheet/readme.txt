Add this to your setting ini file

[ApplicationStyle]
style=plastique
cssFile=Qitom/styles/darkOrangeStyleSheet/darkorange.qss
rccFile=Qitom/styles/darkOrangeStyleSheet/darkorange.rcc

Before create the rcc file using the tool rcc from Qt with the following command:
rcc -o style.rcc -binary style.qrc
