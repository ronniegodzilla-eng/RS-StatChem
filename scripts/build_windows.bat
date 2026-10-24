@echo off
rem Build the Windows portable folder: dist\RS-StatChem\RS-StatChem.exe
cd /d "%~dp0\.."
python -m PyInstaller packaging\rsstatchem.spec --noconfirm --clean || exit /b 1
copy /y packaging\PORTABLE-README.txt dist\RS-StatChem\README.txt >nul
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\RS-StatChem -DestinationPath dist\RS-StatChem-Windows-x64-portable.zip"
echo Built dist\RS-StatChem-Windows-x64-portable.zip
