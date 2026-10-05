<#
.SYNOPSIS
    Reproducible Windows onedir packaging build for halftone-playground.

.DESCRIPTION
    Produces an onedir / windowed PyInstaller bundle:

        <OutputRoot>\dist\HalftonePlayground\
            HalftonePlayground.exe
            _internal\

    The script is intentionally strict: it verifies the build environment and
    stops on any mismatch instead of upgrading or downgrading anything. It never
    deletes an existing output directory.

    Both the repository root and the build environment root are derived at
    runtime -- no machine-specific paths are hardcoded.

.PARAMETER OutputRoot
    A NEW directory OUTSIDE the repository that will hold dist\, build\ and
    spec\. It must not exist yet, or must be empty.

.EXAMPLE
    powershell.exe -ExecutionPolicy Bypass -File .\packaging\build_windows.ps1 `
        -OutputRoot "P:\DevProjects\halftone-playground-packaging\p002-verify"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------
# Expectations -- frozen for the reproducible build chain.
# --------------------------------------------------------------------------
$ExpectedPython           = '3.11.17'
$ExpectedPyInstaller      = '6.22.3'
$ExpectedHooksContrib     = '2026.8'
$ExpectedNumpy            = '2.4.6'
$ExpectedPillow           = '12.3.0'
$ExpectedOpenCv           = '4.14.0.94'

function Stop-Fail {
    param([string]$Message)
    Write-Host ''
    Write-Host "STOP: $Message" -ForegroundColor Red
    Write-Host ''
    exit 1
}

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    OK  $Message" -ForegroundColor Green
}

# --------------------------------------------------------------------------
# 1. Locate repository root from this script's own location.
# --------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot 'pyproject.toml'))) {
    Stop-Fail "Repository root not found. Expected pyproject.toml at '$RepoRoot'."
}
Write-Step "Repository root: $RepoRoot"

$EntryScript = Join-Path $PSScriptRoot 'launch_gui.py'
if (-not (Test-Path $EntryScript)) {
    Stop-Fail "Entry script not found: $EntryScript"
}

# --------------------------------------------------------------------------
# 2. Resolve the active Python interpreter and its environment prefix.
# --------------------------------------------------------------------------
Write-Step 'Resolving build interpreter'

$PythonExe = $null
$Candidates = @()

# Prefer an explicitly activated / ambient interpreter, then fall back to
# discovering one via `python` on PATH.
$Cmd = Get-Command python -ErrorAction SilentlyContinue
if ($Cmd) { $Candidates += $Cmd.Source }

foreach ($cand in $Candidates) {
    if (Test-Path $cand) { $PythonExe = $cand; break }
}

if (-not $PythonExe) {
    Stop-Fail 'No Python interpreter found. Activate the build environment first.'
}

$EnvPrefix = (& $PythonExe -c 'import sys; print(sys.prefix)').Trim()
if ($LASTEXITCODE -ne 0 -or -not $EnvPrefix) {
    Stop-Fail "Could not determine sys.prefix from '$PythonExe'."
}
if (-not (Test-Path $EnvPrefix)) {
    Stop-Fail "Derived environment prefix does not exist: '$EnvPrefix'"
}

Write-Ok "interpreter : $PythonExe"
Write-Ok "env prefix  : $EnvPrefix"

# --------------------------------------------------------------------------
# 3. Verify build environment versions.
# --------------------------------------------------------------------------
Write-Step 'Verifying build environment versions'

# Write the probe to a temporary .py file rather than passing it via
# `python -c`. PowerShell does not preserve multi-token quoting reliably when
# forwarding arguments to native executables, so a file is the robust option.
$ProbeLines = @(
    'import sys'
    'import importlib.metadata as md'
    ''
    'def ver(name):'
    '    try:'
    '        return md.version(name)'
    '    except Exception:'
    '        return ""'
    ''
    'print("python", sys.version.split()[0])'
    'print("pyinstaller", ver("pyinstaller"))'
    'print("pyinstaller-hooks-contrib", ver("pyinstaller-hooks-contrib"))'
    'print("numpy", ver("numpy"))'
    'print("Pillow", ver("Pillow"))'
    'print("opencv-python-headless", ver("opencv-python-headless"))'
)
$ProbeFile = Join-Path ([System.IO.Path]::GetTempPath()) ("halftone_env_probe_{0}.py" -f ([System.Guid]::NewGuid().ToString('N')))

try {
    Set-Content -LiteralPath $ProbeFile -Value $ProbeLines -Encoding UTF8
    $ProbeOut = & $PythonExe $ProbeFile
    $ProbeExit = $LASTEXITCODE
} finally {
    if (Test-Path $ProbeFile) { Remove-Item -LiteralPath $ProbeFile -Force -ErrorAction SilentlyContinue }
}

if ($ProbeExit -ne 0) {
    Stop-Fail "Version probe failed to run under '$PythonExe'."
}

$Vers = @{}
foreach ($line in $ProbeOut) {
    $parts = $line -split ' ', 2
    if ($parts.Count -eq 2) { $Vers[$parts[0].Trim()] = $parts[1].Trim() }
}

$Checks = @(
    @{ Name = 'python';                  Actual = $Vers['python'];                  Expected = $ExpectedPython },
    @{ Name = 'pyinstaller';             Actual = $Vers['pyinstaller'];             Expected = $ExpectedPyInstaller },
    @{ Name = 'pyinstaller-hooks-contrib'; Actual = $Vers['pyinstaller-hooks-contrib']; Expected = $ExpectedHooksContrib },
    @{ Name = 'numpy';                   Actual = $Vers['numpy'];                   Expected = $ExpectedNumpy },
    @{ Name = 'Pillow';                  Actual = $Vers['Pillow'];                  Expected = $ExpectedPillow },
    @{ Name = 'opencv-python-headless';  Actual = $Vers['opencv-python-headless'];  Expected = $ExpectedOpenCv }
)

$Mismatch = @()
foreach ($c in $Checks) {
    if ($c.Actual -ne $c.Expected) {
        $Mismatch += ("  {0,-26} expected {1,-12} found {2}" -f $c.Name, $c.Expected, ($(if ($c.Actual) { $c.Actual } else { '<missing>' })))
    } else {
        Write-Ok ("{0,-26} {1}" -f $c.Name, $c.Actual)
    }
}
if ($Mismatch.Count -gt 0) {
    Write-Host ''
    Write-Host 'Environment version mismatches:' -ForegroundColor Yellow
    $Mismatch | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Stop-Fail 'Build environment does not match the frozen toolchain. Nothing was installed or changed.'
}

# --------------------------------------------------------------------------
# 4. Verify the Tcl/Tk shared libraries that the default hooks miss.
# --------------------------------------------------------------------------
Write-Step 'Verifying Tcl/Tk runtime DLLs in the build environment'

$TclDll = Join-Path $EnvPrefix 'Library\bin\tcl86t.dll'
$TkDll  = Join-Path $EnvPrefix 'Library\bin\tk86t.dll'

foreach ($dll in @(
        @{ Path = $TclDll; Label = 'tcl86t.dll' },
        @{ Path = $TkDll;  Label = 'tk86t.dll'  })) {
    if (-not (Test-Path $dll.Path)) {
        Stop-Fail "Required DLL missing: $($dll.Label) (expected at '$($dll.Path)')."
    }
    Write-Ok "$($dll.Label) -> $($dll.Path)"
}

# --------------------------------------------------------------------------
# 5. Validate OutputRoot: outside the repo, new or empty.
# --------------------------------------------------------------------------
Write-Step 'Validating output root'

$OutputRootFull = [System.IO.Path]::GetFullPath($OutputRoot)

$RepoFull = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\', '/')
$OutFull  = $OutputRootFull.TrimEnd('\', '/')

$repoWithSep = $RepoFull + [System.IO.Path]::DirectorySeparatorChar
if ($OutFull -eq $RepoFull -or $OutFull.StartsWith($repoWithSep, [System.StringComparison]::OrdinalIgnoreCase)) {
    Stop-Fail "OutputRoot must be OUTSIDE the repository. Got: '$OutputRootFull'"
}
Write-Ok "outside repository: $OutputRootFull"

if (Test-Path $OutputRootFull) {
    $existing = @(Get-ChildItem -LiteralPath $OutputRootFull -Force -ErrorAction SilentlyContinue)
    if ($existing.Count -gt 0) {
        Stop-Fail "OutputRoot already exists and is not empty: '$OutputRootFull'. Refusing to touch it -- choose a new directory."
    }
    Write-Ok 'output root exists but is empty'
} else {
    Write-Ok 'output root does not exist yet'
}

# --------------------------------------------------------------------------
# 6. Create the dist / build / spec layout.
# --------------------------------------------------------------------------
$DistPath = Join-Path $OutputRootFull 'dist'
$WorkPath = Join-Path $OutputRootFull 'build'
$SpecPath = Join-Path $OutputRootFull 'spec'

Write-Step 'Creating output layout'
foreach ($p in @($OutputRootFull, $DistPath, $WorkPath, $SpecPath)) {
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
    }
    Write-Ok $p
}

# --------------------------------------------------------------------------
# 7. Invoke PyInstaller with the frozen parameter set.
# --------------------------------------------------------------------------
Write-Step 'Running PyInstaller'

$SrcPath = Join-Path $RepoRoot 'src'

$PyInstallerArgs = @(
    '-m', 'PyInstaller'
    '--noconfirm'
    '--onedir'
    '--windowed'
    '--name', 'HalftonePlayground'
    '--contents-directory', '_internal'
    '--paths', $SrcPath
    '--distpath', $DistPath
    '--workpath', $WorkPath
    '--specpath', $SpecPath
    '--add-binary', "$TclDll;."
    '--add-binary', "$TkDll;."
    $EntryScript
)

Write-Host ''
Write-Host "    $PythonExe $($PyInstallerArgs -join ' ')" -ForegroundColor DarkGray
Write-Host ''

& $PythonExe @PyInstallerArgs
$PyInstallerExit = $LASTEXITCODE

if ($PyInstallerExit -ne 0) {
    Stop-Fail "PyInstaller exited with code $PyInstallerExit."
}
Write-Ok "PyInstaller exit code: $PyInstallerExit"

# --------------------------------------------------------------------------
# 8. Verify the produced bundle.
# --------------------------------------------------------------------------
Write-Step 'Verifying build output'

$BundleDir = Join-Path $DistPath 'HalftonePlayground'
$ExePath   = Join-Path $BundleDir 'HalftonePlayground.exe'
$Internal  = Join-Path $BundleDir '_internal'

if (-not (Test-Path $ExePath)) {
    Stop-Fail "Expected EXE not found: '$ExePath'"
}
if (-not (Test-Path $Internal)) {
    Stop-Fail "Expected _internal directory not found: '$Internal'"
}
Write-Ok "EXE         : $ExePath"
Write-Ok "_internal   : $Internal"

$ExeHash = (Get-FileHash -LiteralPath $ExePath -Algorithm SHA256).Hash

$SizeBytes = 0
Get-ChildItem -LiteralPath $BundleDir -Recurse -File -Force -ErrorAction SilentlyContinue |
    ForEach-Object { $SizeBytes += $_.Length }
$SizeMb = [math]::Round($SizeBytes / 1MB, 1)

# --------------------------------------------------------------------------
# 9. Report.
# --------------------------------------------------------------------------
Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' BUILD COMPLETE' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Write-Host " candidate path : $BundleDir"
Write-Host " exe path       : $ExePath"
Write-Host " exe SHA-256    : $ExeHash"
Write-Host " onedir size    : $SizeMb MB ($SizeBytes bytes)"
Write-Host ''
Write-Host ' The bundle has not been archived. No ZIP was created.'
Write-Host ''

exit 0
