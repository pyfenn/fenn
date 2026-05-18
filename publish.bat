@echo off
setlocal

REM Enable conda in this batch session, then activate env
call conda activate fenn || exit /b 1

REM Project root = directory of this script
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%" || exit /b 1

REM Clean previous build artefacts
if exist dist rmdir /s /q dist
if exist src\fenn.egg-info rmdir /s /q src\fenn.egg-info

REM python -m pip install build
python -m build || exit /b 1

REM python -m pip install twine
python -m twine upload dist\* --verbose || exit /b 1

endlocal