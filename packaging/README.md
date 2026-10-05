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
5. Build the onedir/windowed bundle and report the candidate path, the EXE
   SHA-256, and the total bundle size.

Any version mismatch stops the build. The script will not upgrade or downgrade
packages for you.

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
