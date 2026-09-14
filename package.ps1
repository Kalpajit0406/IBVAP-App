<#
.SYNOPSIS
  Builds the IBVAP Command Client as a standalone product: the Flutter
  Windows release build with its own Python backend (server.py, ibvap/,
  trained models) and a self-contained Python + CUDA/torch/ultralytics
  runtime bundled right next to the .exe. The output folder needs nothing
  else installed on the target machine.

.DESCRIPTION
  1. `flutter build windows --release`
  2. Mirrors backend/ (the vendored IBVAP source + models, and the portable
     Python venv built by setup instructions in backend/README.md) into
     build\windows\x64\runner\Release\backend\
  3. Reports the final package size.

  Re-run any time backend/ or the Dart source changes — step 2 is a mirror
  (robocopy /MIR), so it only copies what's different.
#>

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$backendSrc = Join-Path $root 'backend'
$releaseDir = Join-Path $root 'build\windows\x64\runner\Release'
$backendDst = Join-Path $releaseDir 'backend'

if (-not (Test-Path (Join-Path $backendSrc 'app\server.py'))) {
    Write-Error "backend\app\server.py not found — vendor the IBVAP source into backend\app\ and build the portable venv into backend\runtime\ first (see backend\README.md)."
}
if (-not (Test-Path (Join-Path $backendSrc 'runtime\Scripts\python.exe'))) {
    Write-Error "backend\runtime\Scripts\python.exe not found — build the portable venv first (see backend\README.md)."
}

# Anything still running from the package (the console, or a backend it left
# behind after a crash / force-close) locks the runtime's DLLs. robocopy would
# then retry those files indefinitely and the script appears to hang.
$running = Get-CimInstance Win32_Process |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($releaseDir, [StringComparison]::OrdinalIgnoreCase) }
if ($running) {
    $list = ($running | ForEach-Object { "  PID $($_.ProcessId)  $($_.Name)" }) -join "`n"
    throw "Processes are running from the release folder - close the console and stop its backend first:`n$list"
}

Write-Host "== flutter build windows --release ==" -ForegroundColor Cyan
flutter build windows --release
if ($LASTEXITCODE -ne 0) { throw "flutter build failed" }

Write-Host "== staging backend/ into the release folder ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $backendDst | Out-Null
# app\data is the packaged product's own record: event database, evidence hash
# chain, snapshots, fences. Mirroring the dev copy over it would overwrite or
# delete that evidence, so it is excluded (robocopy never purges excluded dirs)
# and only seeded on the first package.
$dataSrc = Join-Path $backendSrc 'app\data'
$dataDst = Join-Path $backendDst 'app\data'
robocopy $backendSrc $backendDst /MIR /XD '__pycache__' '.git' $dataSrc $dataDst /XF '*.pyc' /R:3 /W:2 /NFL /NDL /NJH /NJS /nc /ns /np
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }
if ((Test-Path $dataSrc) -and -not (Test-Path $dataDst)) {
    robocopy $dataSrc $dataDst /E /R:3 /W:2 /NFL /NDL /NJH /NJS /nc /ns /np
    if ($LASTEXITCODE -ge 8) { throw "robocopy (data seed) failed with code $LASTEXITCODE" }
}

$sizeGB = [math]::Round(((Get-ChildItem $releaseDir -Recurse -File | Measure-Object -Property Length -Sum).Sum) / 1GB, 2)
Write-Host ""
Write-Host "== Done ==" -ForegroundColor Green
Write-Host "Standalone package: $releaseDir"
Write-Host "Total size: $sizeGB GB"
Write-Host "Copy that whole 'Release' folder anywhere (another machine, a USB drive) - it needs nothing else installed."
