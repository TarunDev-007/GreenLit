$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = $null
$bundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$candidates = @()
if (Test-Path -LiteralPath $bundled) { $candidates += $bundled }
$installed = Get-Command python -ErrorAction SilentlyContinue
if ($installed) { $candidates += $installed.Source }
foreach ($candidate in $candidates) {
    & $candidate -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $pythonPath = $candidate; break }
}
if (-not $pythonPath) { throw 'Install Python 3.11 or newer, then run setup.ps1 again.' }
& $pythonPath -m venv --without-pip .venv
if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $pythonPath -c 'import pip'
if ($LASTEXITCODE -eq 0) {
    & $pythonPath -m pip --python $venvPython install -r requirements.lock.txt
} else {
    & $venvPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Python pip is unavailable. Install Python with pip enabled.' }
    & $venvPython -m pip install -r requirements.lock.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your internet connection.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
Set-Content -LiteralPath '.venv\greenlit-ready' -Value 'ready'
Write-Host 'GreenLit is ready. Double-click Start GreenLit.cmd.' -ForegroundColor Green
