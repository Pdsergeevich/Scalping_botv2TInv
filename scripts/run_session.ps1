# Запуск боевой paper-сессии Bot A на живом стакане T-Invest (реальные данные, без денег).
#
# Использование:
#   .\scripts\run_session.ps1                       # 1 час paper на VTBR/SBER, Premium
#   .\scripts\run_session.ps1 -Minutes 30
#   .\scripts\run_session.ps1 -Preset SBERP_SBER -Tariff Trader
#   .\scripts\run_session.ps1 -Record               # сначала записать L2 в файл, потом торговать
#
# Токен берётся из secrets\tinvest_token.txt (или $env:TINVEST_TOKEN). В лог не печатается.

param(
    [string]$Preset = "VTBR_SBER",
    [string]$Tariff = "Premium",
    [int]$Minutes = 60,
    [int]$IntervalMs = 200,
    [int]$Depth = 20,
    [switch]$Record
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path "secrets\tinvest_token.txt") -and -not $env:TINVEST_TOKEN) {
    Write-Host "Токен не найден. Запишите его один раз:" -ForegroundColor Yellow
    Write-Host '  Set-Content -Path secrets\tinvest_token.txt -Value $env:TINVEST_TOKEN -NoNewline' -ForegroundColor Yellow
    exit 3
}

Write-Host "== 1/3 Preflight (счёт, ГО, окна входа) ==" -ForegroundColor Cyan
py -3 -m bot_a --mode preflight --preset $Preset --tariff $Tariff

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

if ($Record) {
    Write-Host "== 2/3 Запись L2 в recordings (10 мин) ==" -ForegroundColor Cyan
    py -3 -m bot_a --mode record --preset $Preset --duration 600 `
        --record-interval-ms $IntervalMs --record-depth $Depth
}

Write-Host "== 3/3 Paper на живом стакане ($Minutes мин, интервал $IntervalMs мс) ==" -ForegroundColor Cyan
py -3 -m bot_a --mode paper --preset $Preset --tariff $Tariff `
    --duration ($Minutes * 60) --record-interval-ms $IntervalMs --record-depth $Depth `
    --report-stem "paper_live_${Preset}_$stamp"

Write-Host ""
Write-Host "Отчёт: reports\charts\paper_live_${Preset}_$stamp\index.html" -ForegroundColor Green
