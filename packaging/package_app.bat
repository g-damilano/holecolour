@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0package_app.ps1" %*
