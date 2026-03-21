@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Training Data Generation - Full Pipeline
echo  Started: %date% %time%
echo ============================================================

set CONCURRENCY=5

:: Step 1: Generate training data from all expanded prompts
for %%f in (fsharp_core fsharp_libraries dotnet_aspnet general_coding svelte_typescript cross_domain long_context docker_kubernetes agentic_swe) do (
    echo.
    echo [%date% %time%] Generating: %%f
    python generate_data.py --config ../prompts/expanded/%%f_expanded.yaml --output ../../data/raw/%%f.jsonl --concurrency %CONCURRENCY%
    if errorlevel 1 (
        echo [WARNING] %%f generation had errors, continuing...
    ) else (
        echo [OK] %%f generation complete
    )
)

echo.
echo ============================================================
echo  Step 1 Complete: Data Generation
echo  %date% %time%
echo ============================================================

:: Step 2: Verify F# samples
for %%f in (fsharp_core fsharp_libraries dotnet_aspnet cross_domain) do (
    echo.
    echo [%date% %time%] Verifying F#: %%f
    python verify_fsharp.py --input ../../data/raw/%%f.jsonl --output ../../data/verified/%%f.jsonl
    if errorlevel 1 (
        echo [WARNING] %%f verification had errors, continuing...
    ) else (
        echo [OK] %%f verification complete
    )
)

:: Step 3: Copy non-F# files to verified
echo.
echo Copying non-F# files to verified...
for %%f in (svelte_typescript general_coding long_context docker_kubernetes agentic_swe) do (
    copy /Y "../../data/raw/%%f.jsonl" "../../data/verified/%%f.jsonl" >nul 2>&1
    echo [OK] Copied %%f
)

:: Step 4: Format for training
echo.
echo [%date% %time%] Formatting dataset...
python format_dataset.py --input ../../data/verified/ --output ../../data/formatted/ --format chatml --split-by-length

echo.
echo ============================================================
echo  PIPELINE COMPLETE
echo  %date% %time%
echo ============================================================
pause
