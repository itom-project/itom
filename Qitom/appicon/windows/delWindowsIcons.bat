taskkill /F /IM explorer.exe
cd /d %userprofile%\AppData\Local
attrib –h IconCache.db
del IconCache.db
start explorer
