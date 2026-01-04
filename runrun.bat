@echo off
chcp 65001 >nul
title 自动安装并启动

echo 检查 Streamlit 是否可用...
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit 未安装，正在安装依赖...
    
    REM 升级 pip 并静默安装所有包
    python -m pip install --upgrade pip >nul
    pip install streamlit pandas numpy yfinance plotly pymoo >nul
    
    echo 依赖安装完成！
)

REM 启动应用
echo 正在启动应用...
streamlit run app.py

REM 防止窗口闪退（如 app.py 不存在或报错）
pause