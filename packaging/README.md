# Packaging

Reproducible Windows packaging for the halftone-playground desktop GUI.

## Goal

Produce a self-contained Windows 10/11 x64 bundle with **PyInstaller** in
**onedir / windowed** mode:

```
HalftonePlayground\
├─ HalftonePlayground.exe
└─ _internal\
```

End users need **no Python, no Conda, and no PyInstaller** — they unpack the
folder and double-click the EXE.

## Build environment

Use a dedicated build environment cloned from the verified development
environment, so packaging tools never leak into the runtime environment:

```bash
conda create -n halftone-playground-build --clone halftone-playground
conda activate halftone-playground-build
```

Then install the packaging toolchain:

```bash
python -m pip install -r packaging/requirements-build.txt
```

`requirements-build.txt` pins **only** the PyInstaller toolchain. numpy, Pillow
and opencv-python-headless are application/runtime dependencies that come from
the cloned environment — they are intentionally not listed there.

## Building

From an activated build environment, run:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\packaging\build_windows.ps1 `
    -OutputRoot "<a new directory outside the repository>"
```

Pick any new directory outside the repository, for example a sibling folder of
your dev clone. The script derives the repository root and the environment
prefix at runtime; no machine-specific paths are baked in.

The script will:

1. Verify Python 3.11.17, PyInstaller 6.22.3, pyinstaller-hooks-contrib 2026.8,
   numpy 2.4.6, Pillow 12.3.0, opencv-python-headless 4.14.0.94.
2. Verify `Library\bin\tcl86t.dll` and `Library\bin\tk86t.dll` exist in the
   environment.
3. Refuse to run if `-OutputRoot` is inside the repository, or already exists
   and is not empty. **It never recursively deletes an existing directory.**
4. Create `dist\`, `build\` and `spec\` under the output root.
5. Build the onedir/windowed bundle.
6. Remove the unused OpenCV FFmpeg videoio plugin (see below).
7. Verify the six Conda transitive runtime DLLs were auto-collected (see below).
8. Report the candidate path, the EXE SHA-256, and the total bundle size
   (measured **after** the prune).

Any version mismatch stops the build. The script will not upgrade or downgrade
packages for you.

## Automatic Conda DLL resolution (no manual PATH setup)

The script derives `<env>\Library\bin` from the build interpreter's `sys.prefix`
and **prepends it to the current process PATH for the duration of the build**.
Conda keeps several Windows runtime DLLs there instead of next to the extension
modules that link them, and PyInstaller resolves *transitive* DLL dependencies
through the process PATH. Hardening the PATH inside the script means PyInstaller
reliably collects:

```
LIBBZ2.dll, ffi.dll, libcrypto-3-x64.dll, libexpat.dll,
liblzma.dll, libssl-3-x64.dll
```

(dependencies of `_bz2`, `_ctypes`, `_ssl`, `pyexpat` and `_lzma`).

Scope is strictly the build process and its children. The original PATH is
restored afterwards — **success or failure** — so a failed build never leaves a
modified environment behind. The script never modifies the Windows user or
system PATH, the registry, or Conda configuration, and does not persist any
environment variable.

**You do not need to modify PATH yourself.** The script is self-sufficient: run
it from any shell, with the build environment's Python selected, and it will
establish the complete dependency-resolution environment on its own. (If the
caller *has* already put `Library\bin` on PATH, the script reports that and
still works.)

These six DLLs must be found by **ordinary binary dependency analysis**. Do not
add them as explicit `--add-binary` entries — if that were ever necessary it
would mean the PATH hardening had regressed. After the build the script
verifies all six are present in `_internal\` and stops if any is missing.

## Unused OpenCV FFmpeg videoio plugin is pruned

The frozen runtime uses **opencv-python-headless 4.14.0.94**. Its Windows wheel
bundles a runtime-loaded videoio plugin:

```
_internal\cv2\opencv_videoio_ffmpeg*.dll
```

halftone-playground:

- does **not** read video,
- does **not** write video,
- does **not** use OpenCV `videoio` / FFmpeg,
- uses OpenCV only for the image / geometry capabilities it actually needs —
  for example `cv2.warpPolar` in the Spiral renderer.

Human functional acceptance (R1A) confirmed that removing this DLL leaves
Stripe Generate, Spiral Variable, Spiral Fixed, Spiral Source color and Spiral
PNG export ("Save") all working normally.

The build script therefore removes this unused plugin **after** PyInstaller has
finished collecting. Because the toolchain is frozen, the script expects
**exactly one** match; `0` or `>1` matches stops the build rather than silently
continuing, so a changed release layout is re-audited instead of assumed.

> If video input/output is ever added to the project, this prune decision must
> be re-evaluated.

### Technical basis

`opencv_videoio_ffmpeg*.dll` is a **runtime-loaded** OpenCV `videoio` plugin.
When it is absent, the rest of OpenCV keeps working; only FFmpeg-based video
decode/encode is lost. This is documented upstream in the OpenCV / opencv-python
project (see the `opencv-python` repository's notes on the bundled FFmpeg
plugin).

This section records the engineering rationale for the reproducible build only.
It is **not** a licensing statement.

## Tcl/Tk caveat

Under a Conda Python environment, PyInstaller's default hooks are verified to
collect the Tcl/Tk **scripts**, but the frozen environment **misses the shared
libraries** `tcl86t.dll` and `tk86t.dll`. Without them the GUI fails at
startup.

The build script therefore passes both DLLs explicitly:

```
--add-binary "<env>\Library\bin\tcl86t.dll;."
--add-binary "<env>\Library\bin\tk86t.dll;."
```

## What not to add

Do **not** add preventative `--collect-all` for numpy, cv2 or PIL. The default
hooks handle them correctly; blanket collection bloats the bundle and can
introduce duplicate or conflicting binaries.

Do not add `--onefile`, `--console`, `--icon`, `--upx`, custom hooks or runtime
hooks to the production build.

## Preview accepted limitation

The in-app Result Preview is a **display-scaled preview**. Extremely fine,
high-frequency halftone textures may not render faithfully on screen. The
**saved full-resolution PNG is the authoritative result** and must be judged
from the output file, not the preview.

This note records the packaging acceptance position only; the root README is
not modified here.
