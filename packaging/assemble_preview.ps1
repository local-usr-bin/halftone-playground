<#
.SYNOPSIS
    Reproducible Preview Release assembly for halftone-playground.

.DESCRIPTION
    Turns a clean source checkout into a distributable Windows Preview ZIP.

    The script performs the FULL assembly chain:

        build  ->  release-folder assembly  ->  gates  ->  ZIP
               ->  ZIP round-trip extraction  ->  round-trip gates
               ->  SHA-256 sidecar

    It does NOT create a Git tag, a GitHub Release, a push or any signing.
    Those are separate, explicit steps.

    Division of labour (strict):

      * build_windows.ps1 is the ONLY production runtime build entry point.
        This script calls it unchanged and inherits every one of its properties:
        onedir, windowed, the Tcl/Tk DLL fix, Conda PATH hardening, the six
        transitive DLL resolution and the OpenCV FFmpeg prune. The PyInstaller
        command line is NEVER re-stated here.

      * assemble_preview.ps1 (this script) only assembles the release folder,
        runs the gates, builds the ZIP and verifies it.

    The repository root is derived from $PSScriptRoot. No machine-specific
    paths (no drive letters, no user home, no Conda prefix) are hardcoded --
    everything is resolved at runtime.

.PARAMETER OutputRoot
    A NEW directory OUTSIDE the repository that will hold the assembly. It must
    not exist yet, or must be empty. The script NEVER recursively deletes an
    existing directory.

.EXAMPLE
    powershell.exe -ExecutionPolicy Bypass -File .\packaging\assemble_preview.ps1 `
        -OutputRoot "D:\build\halftone-r4a-verify"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --------------------------------------------------------------------------
# Frozen release naming (Preview). Not to be changed without a new round.
# --------------------------------------------------------------------------
$ReleaseDisplayName = 'Halftone Playground v0.0.1 Preview'
$ReleaseFolderName  = 'HalftonePlayground-Preview-0.0.1-win64'
$ZipFileName        = 'HalftonePlayground-Preview-0.0.1-win64.zip'

# --------------------------------------------------------------------------
# Gate expectations (frozen for the Preview toolchain).
# --------------------------------------------------------------------------
$ExpectedNumPyLicenseCount = 17

$RequiredTransitiveDlls = @(
    'LIBBZ2.dll'
    'ffi.dll'
    'libcrypto-3-x64.dll'
    'libexpat.dll'
    'liblzma.dll'
    'libssl-3-x64.dll'
)

# Top-level legal docs that must exist in the release folder.
$RequiredLegalDocs = @(
    'README.md'
    'LICENSE'
    'THIRD-PARTY-NOTICES.txt'
)

# Individual files that must exist directly under licenses\.
$RequiredLicenseFiles = @(
    'CPython-PSF-2.0.txt'
    'Tcl-Tk-license.terms'
    'Pillow-MIT-CMU.txt'
    'OpenCV-Apache-2.0.txt'
    'OpenCV-LICENSE-3RD-PARTY.txt'
    'OpenSSL-Apache-2.0.txt'
    'libexpat-MIT.txt'
    'libffi-MIT.txt'
    'liblzma-0BSD.txt'
    'zlib.txt'
    'bzip2.txt'
)

function Stop-Fail {
    param([string]$Message)
    Write-Host ''
    Write-Host "STOP: $Message" -ForegroundColor Red
    Write-Host ''
    exit 1
}

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    OK  $Message" -ForegroundColor Green
}

function Get-DirSizeBytes {
    param([string]$Path)
    $sum = 0
    Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
        ForEach-Object { $sum += $_.Length }
    return $sum
}

function Get-FileCount {
    param([string]$Path)
    return @(Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue).Count
}

# --------------------------------------------------------------------------
# 1. Locate the repository root from this script's own location.
# --------------------------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot 'pyproject.toml'))) {
    Stop-Fail "Repository root not found. Expected pyproject.toml at '$RepoRoot'."
}
Write-Step 'Repository root'
Write-Ok $RepoRoot

$BuildScript = Join-Path $PSScriptRoot 'build_windows.ps1'
if (-not (Test-Path $BuildScript)) {
    Stop-Fail "Production build script not found: '$BuildScript'."
}
Write-Ok "build script : $BuildScript"

# --------------------------------------------------------------------------
# 2. Validate OutputRoot: outside the repo, new or empty.
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
# 3. Create the assembly layout.
# --------------------------------------------------------------------------
$BuildDir     = Join-Path $OutputRootFull 'build'
$StageDir     = Join-Path $OutputRootFull 'stage'
$ArtifactsDir = Join-Path $OutputRootFull 'artifacts'
$VerifyDir    = Join-Path $OutputRootFull 'verify'

Write-Step 'Creating assembly layout'
foreach ($p in @($OutputRootFull, $BuildDir, $StageDir, $ArtifactsDir, $VerifyDir)) {
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p -Force | Out-Null }
    Write-Ok $p
}

# --------------------------------------------------------------------------
# 4. Invoke the production build (build_windows.ps1) into an isolated dir.
#
#    Pass <OutputRoot>\build as the build's OutputRoot. build_windows.ps1
#    creates its own dist\ / build\ / spec\ underneath it.
# --------------------------------------------------------------------------
Write-Step 'Running production build (build_windows.ps1)'

$BuildOutputRoot = Join-Path $BuildDir 'runtime'

& $BuildScript -OutputRoot $BuildOutputRoot
$BuildExit = $LASTEXITCODE

if ($BuildExit -ne 0) {
    Stop-Fail "build_windows.ps1 exited with code $BuildExit. Assembly FAIL."
}
Write-Ok "build_windows.ps1 exit code: $BuildExit"

$BuiltBundle = Join-Path $BuildOutputRoot 'dist\HalftonePlayground'
$BuiltExe    = Join-Path $BuiltBundle 'HalftonePlayground.exe'
$BuiltInternal = Join-Path $BuiltBundle '_internal'

if (-not (Test-Path $BuiltExe))    { Stop-Fail "Built EXE not found: '$BuiltExe'" }
if (-not (Test-Path $BuiltInternal)) { Stop-Fail "Built _internal not found: '$BuiltInternal'" }

# --------------------------------------------------------------------------
# 5. Assemble the release staging folder.
# --------------------------------------------------------------------------
Write-Step 'Assembling release folder'

$ReleaseDir = Join-Path $StageDir $ReleaseFolderName
if (Test-Path $ReleaseDir) {
    Stop-Fail "Release staging folder already exists: '$ReleaseDir'."
}
New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

# 5a. runtime payload from the build output
Copy-Item -LiteralPath $BuiltExe -Destination (Join-Path $ReleaseDir 'HalftonePlayground.exe') -Force
Copy-Item -LiteralPath $BuiltInternal -Destination (Join-Path $ReleaseDir '_internal') -Recurse -Force
Write-Ok 'runtime payload copied (HalftonePlayground.exe + _internal\)'

# 5b. legal docs + README from the repository's CURRENT working tree
foreach ($doc in $RequiredLegalDocs) {
    $src = Join-Path $RepoRoot $doc
    if (-not (Test-Path $src)) { Stop-Fail "Required repository file missing: '$src'." }
    Copy-Item -LiteralPath $src -Destination (Join-Path $ReleaseDir $doc) -Force
    Write-Ok "legal doc    : $doc"
}

$RepoLicenses = Join-Path $RepoRoot 'licenses'
if (-not (Test-Path $RepoLicenses)) { Stop-Fail "Repository 'licenses' directory missing: '$RepoLicenses'." }
Copy-Item -LiteralPath $RepoLicenses -Destination (Join-Path $ReleaseDir 'licenses') -Recurse -Force
Write-Ok 'legal docs   : licenses\ (recursive)'

# --------------------------------------------------------------------------
# 6. Gate: release-folder top-level contents must be EXACTLY the expected set.
# --------------------------------------------------------------------------
Write-Step 'Gate: release folder top-level contents'

$ExpectedTop = @('_internal', 'HalftonePlayground.exe', 'LICENSE', 'licenses', 'README.md', 'THIRD-PARTY-NOTICES.txt')
$ActualTop = @(Get-ChildItem -LiteralPath $ReleaseDir -Force | ForEach-Object { $_.Name })
$Missing = @($ExpectedTop | Where-Object { $ActualTop -notcontains $_ })
$Extra   = @($ActualTop | Where-Object { $ExpectedTop -notcontains $_ })
if ($Missing.Count -gt 0) { Stop-Fail "Release folder is missing top-level entries: $($Missing -join ', ')" }
if ($Extra.Count -gt 0)   { Stop-Fail "Release folder has unexpected top-level entries: $($Extra -join ', ')" }
Write-Ok "top-level entries exact ($($ActualTop.Count)): $($ActualTop -join ', ')"

# --------------------------------------------------------------------------
# 7. Gate: legal docs.
# --------------------------------------------------------------------------
Write-Step 'Gate: legal docs'

foreach ($doc in $RequiredLegalDocs) {
    if (-not (Test-Path (Join-Path $ReleaseDir $doc))) { Stop-Fail "Legal doc missing from release folder: '$doc'." }
}
Write-Ok "top-level docs present: $($RequiredLegalDocs -join ', ')"

$LicDir = Join-Path $ReleaseDir 'licenses'
if (-not (Test-Path $LicDir)) { Stop-Fail "licenses\ directory missing from release folder." }

foreach ($f in $RequiredLicenseFiles) {
    if (-not (Test-Path (Join-Path $LicDir $f))) { Stop-Fail "Required license file missing: 'licenses\$f'." }
}
Write-Ok "license files present ($($RequiredLicenseFiles.Count))"

$NumPyLicDir = Join-Path $LicDir 'NumPy'
if (-not (Test-Path $NumPyLicDir)) { Stop-Fail "licenses\NumPy\ missing from release folder." }
$NumPyCount = Get-FileCount -Path $NumPyLicDir
if ($NumPyCount -ne $ExpectedNumPyLicenseCount) {
    Stop-Fail "licenses\NumPy\ file count = $NumPyCount, expected $ExpectedNumPyLicenseCount."
}
Write-Ok "licenses\NumPy\ file count = $NumPyCount"

# --------------------------------------------------------------------------
# 8. Gate: runtime payload.
# --------------------------------------------------------------------------
Write-Step 'Gate: runtime payload'

$IntDir = Join-Path $ReleaseDir '_internal'
$RelExe = Join-Path $ReleaseDir 'HalftonePlayground.exe'
if (-not (Test-Path $RelExe)) { Stop-Fail 'HalftonePlayground.exe missing from release folder.' }
if (-not (Test-Path $IntDir)) { Stop-Fail '_internal\ missing from release folder.' }
Write-Ok 'HalftonePlayground.exe + _internal\ present'

# 8a. FFmpeg videoio plugin must be pruned to zero.
$Ffmpeg = @(Get-ChildItem -LiteralPath $IntDir -Recurse -File -Filter 'opencv_videoio_ffmpeg*.dll' -ErrorAction SilentlyContinue)
if ($Ffmpeg.Count -ne 0) {
    $Ffmpeg | ForEach-Object { Write-Host "    $($_.FullName)" -ForegroundColor Yellow }
    Stop-Fail "opencv_videoio_ffmpeg*.dll count = $($Ffmpeg.Count), expected 0."
}
Write-Ok 'opencv_videoio_ffmpeg*.dll count = 0'

# 8b. Six Conda transitive runtime DLLs.
foreach ($dll in $RequiredTransitiveDlls) {
    if (-not (Test-Path -LiteralPath (Join-Path $IntDir $dll))) {
        Stop-Fail "Conda transitive runtime DLL missing from release folder: '$dll'."
    }
}
Write-Ok "transitive DLLs present ($($RequiredTransitiveDlls.Count)): $($RequiredTransitiveDlls -join ', ')"

# 8c. Tcl/Tk + core runtime artifacts.
foreach ($rel in @('tcl86t.dll', 'tk86t.dll', '_tkinter.pyd', 'cv2\cv2.pyd', 'numpy', 'PIL')) {
    if (-not (Test-Path -LiteralPath (Join-Path $IntDir $rel))) {
        Stop-Fail "Expected runtime artifact missing from release folder: '_internal\$rel'."
    }
    Write-Ok "runtime      : $rel"
}

# --------------------------------------------------------------------------
# 9. Create the ZIP (System.IO.Compression.ZipFile -- no 7-Zip dependency).
#
#    The ZIP root must contain the SINGLE release folder
#    (HalftonePlayground-Preview-0.0.1-win64\). $StageDir holds exactly that
#    one folder, so we zip $StageDir with includeBaseDirectory = $false: its
#    child entry becomes the ZIP root entry -- i.e. the release folder name.
#    (With $true, the literal "stage" folder name would be added as an extra
#     level; zipping $ReleaseDir with $false would drop the folder level.)
# --------------------------------------------------------------------------
Write-Step 'Creating ZIP'

Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

$ZipPath = Join-Path $ArtifactsDir $ZipFileName
if (Test-Path $ZipPath) {
    Stop-Fail "ZIP already exists: '$ZipPath'. Refusing to overwrite."
}

# ZipFile::CreateFromDirectory(sourceDir, destinationZip, level, includeBaseDirectory)
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $StageDir,
    $ZipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

if (-not (Test-Path $ZipPath)) { Stop-Fail "ZIP was not created: '$ZipPath'." }
Write-Ok "ZIP created : $ZipPath"

# --------------------------------------------------------------------------
# 10. ZIP round-trip verification: extract and re-check the same gates.
# --------------------------------------------------------------------------
Write-Step 'ZIP round-trip extraction'

$VerifyExtract = Join-Path $VerifyDir $ReleaseFolderName
if (Test-Path $VerifyExtract) {
    Stop-Fail "Round-trip extract target already exists: '$VerifyExtract'."
}
[System.IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $VerifyDir)
Write-Ok "extracted to: $VerifyDir"

# 10a. exactly one top-level directory in verify\
$VerifyTop = @(Get-ChildItem -LiteralPath $VerifyDir -Force)
$VerifyTopDirs = @($VerifyTop | Where-Object { $_.PSIsContainer })
if ($VerifyTopDirs.Count -ne 1 -or $VerifyTopDirs[0].Name -ne $ReleaseFolderName) {
    $foundNames = ($VerifyTop | ForEach-Object { $_.Name }) -join ', '
    Stop-Fail "ZIP root must contain exactly one folder '$ReleaseFolderName'. Found: $foundNames"
}
Write-Ok "ZIP root folder: $($VerifyTopDirs[0].Name)"

$VE = Join-Path $VerifyDir $ReleaseFolderName
if (-not (Test-Path $VE)) { Stop-Fail "Round-trip release folder missing: '$VE'." }

# 10b. re-check top-level, legal docs, runtime.
$VTop = @(Get-ChildItem -LiteralPath $VE -Force | ForEach-Object { $_.Name })
$VMissing = @($ExpectedTop   | Where-Object { $VTop -notcontains $_ })
$VExtra   = @($VTop | Where-Object { $ExpectedTop -notcontains $_ })
if ($VMissing.Count -gt 0) { Stop-Fail "Round-trip folder missing top-level entries: $($VMissing -join ', ')" }
if ($VExtra.Count -gt 0)   { Stop-Fail "Round-trip folder has unexpected top-level entries: $($VExtra -join ', ')" }
Write-Ok "round-trip top-level entries exact ($($VTop.Count))"

foreach ($doc in $RequiredLegalDocs) {
    if (-not (Test-Path (Join-Path $VE $doc))) { Stop-Fail "Round-trip legal doc missing: '$doc'." }
}
$VLicDir = Join-Path $VE 'licenses'
if (-not (Test-Path $VLicDir)) { Stop-Fail "Round-trip licenses\ missing." }
foreach ($f in $RequiredLicenseFiles) {
    if (-not (Test-Path (Join-Path $VLicDir $f))) { Stop-Fail "Round-trip license file missing: 'licenses\$f'." }
}
$VNumPyCount = Get-FileCount -Path (Join-Path $VLicDir 'NumPy')
if ($VNumPyCount -ne $ExpectedNumPyLicenseCount) {
    Stop-Fail "Round-trip licenses\NumPy\ count = $VNumPyCount, expected $ExpectedNumPyLicenseCount."
}
Write-Ok "round-trip legal docs OK (licenses\NumPy\ = $VNumPyCount)"

$VInt = Join-Path $VE '_internal'
if (-not (Test-Path (Join-Path $VE 'HalftonePlayground.exe'))) { Stop-Fail 'Round-trip EXE missing.' }
if (-not (Test-Path $VInt)) { Stop-Fail 'Round-trip _internal\ missing.' }
$VFfmpeg = @(Get-ChildItem -LiteralPath $VInt -Recurse -File -Filter 'opencv_videoio_ffmpeg*.dll' -ErrorAction SilentlyContinue)
if ($VFfmpeg.Count -ne 0) { Stop-Fail "Round-trip opencv_videoio_ffmpeg*.dll count = $($VFfmpeg.Count), expected 0." }
foreach ($dll in $RequiredTransitiveDlls) {
    if (-not (Test-Path -LiteralPath (Join-Path $VInt $dll))) { Stop-Fail "Round-trip transitive DLL missing: '$dll'." }
}
Write-Ok 'round-trip runtime gates OK (FFmpeg = 0; transitive DLLs present)'

# --------------------------------------------------------------------------
# 11. Hashes.
# --------------------------------------------------------------------------
Write-Step 'Computing hashes'

$ZipHash       = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash
$StagedExeHash = (Get-FileHash -LiteralPath $RelExe -Algorithm SHA256).Hash
$ZipExePath    = Join-Path $VE 'HalftonePlayground.exe'
$ZipExeHash    = (Get-FileHash -LiteralPath $ZipExePath -Algorithm SHA256).Hash

if ($ZipExeHash -ne $StagedExeHash) {
    Stop-Fail "EXE hash mismatch: staged=$StagedExeHash  zip-extracted=$ZipExeHash. Artifact is inconsistent."
}
Write-Ok "staged EXE == ZIP-extracted EXE ($StagedExeHash)"
Write-Ok "ZIP SHA-256 : $ZipHash"

# Sidecar checksum file (NOT placed inside the ZIP).
$SumsFile = Join-Path $ArtifactsDir 'SHA256SUMS.txt'
$SumLines = @(
    "$ZipHash  $ZipFileName"
    "$StagedExeHash  $ReleaseFolderName/HalftonePlayground.exe"
)
Set-Content -LiteralPath $SumsFile -Value $SumLines -Encoding ASCII
Write-Ok "SHA256SUMS.txt : $SumsFile"

# --------------------------------------------------------------------------
# 12. Metrics + report.
# --------------------------------------------------------------------------
$ReleaseSizeBytes = Get-DirSizeBytes -Path $ReleaseDir
$ReleaseFileCount = Get-FileCount -Path $ReleaseDir
$ZipSizeBytes     = (Get-Item -LiteralPath $ZipPath).Length
$SourceCommit     = (& git -C $RepoRoot rev-parse HEAD).Trim()

Write-Host ''
Write-Host '============================================================' -ForegroundColor Green
Write-Host ' PREVIEW ASSEMBLY COMPLETE' -ForegroundColor Green
Write-Host '============================================================' -ForegroundColor Green
Write-Host " release name      : $ReleaseDisplayName"
Write-Host " release folder    : $ReleaseFolderName"
Write-Host " release dir       : $ReleaseDir"
Write-Host " release size      : $([math]::Round($ReleaseSizeBytes / 1MB, 1)) MB ($ReleaseSizeBytes bytes)"
Write-Host " release file count: $ReleaseFileCount"
Write-Host " zip path          : $ZipPath"
Write-Host " zip size          : $([math]::Round($ZipSizeBytes / 1MB, 1)) MB ($ZipSizeBytes bytes)"
Write-Host " zip sha-256       : $ZipHash"
Write-Host " exe sha-256       : $StagedExeHash"
Write-Host " source commit     : $SourceCommit"
Write-Host " verify extract    : $VE"
Write-Host ''
Write-Host ' No tag, no GitHub Release, no push, no signing was performed.'
Write-Host ''

exit 0
