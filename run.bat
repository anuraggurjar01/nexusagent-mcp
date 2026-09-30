@echo off
REM NexusAgent MCP - Quick Launcher
REM Author: Anurag Gurjar (github.com/anuraggurjar01)
echo =======================================================
echo          NexusAgent MCP - Launch Selector
echo   Engineered by: Anurag Gurjar (github.com/anuraggurjar01)
echo =======================================================
echo.
echo [1] Launch Interactive Web Dashboard (Streamlit)
echo [2] Launch CLI Console (Human-in-the-Loop ReAct Agent)
echo.
set /p choice="Enter your choice (1 or 2, default is 1): "
if "%choice%"=="2" (
    python -X utf8 agent.py %*
) else (
    python -m streamlit run app.py
)
