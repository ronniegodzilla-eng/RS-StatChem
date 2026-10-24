# PyInstaller spec for RS-StatChem.
#   macOS (Apple Silicon): builds dist/RS-StatChem.app as native arm64
#   Windows:               builds dist/RS-StatChem/ (portable folder, run RS-StatChem.exe)
# Build with:  pyinstaller packaging/rsstatchem.spec --noconfirm
import re
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
RES = ROOT / "rsstatchem" / "resources"
VERSION = re.search(r'__version__ = "([^"]+)"',
                    (ROOT / "rsstatchem" / "__init__.py").read_text()).group(1)

# Every module of the package, found from the files rather than by importing the package:
# PyInstaller's collect_submodules() silently returns nothing when the project folder is not on
# sys.path (as with the plain `pyinstaller` entry point on CI), and the feature modules are only
# loaded by name in ui/menus.py, so the analysis cannot see them.
APP_MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for path in (ROOT / "rsstatchem").rglob("*.py"))

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    datas=[(str(RES), "rsstatchem/resources")],
    hiddenimports=["pyreadstat._readstat_parser", "pyreadstat._readstat_writer",
                   "pyreadstat.worker", "mpl_toolkits.mplot3d",
                   "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_pdf",
                   "matplotlib.backends.backend_svg", "matplotlib.backends.backend_ps",
                   *APP_MODULES],
    excludes=["tkinter", "_tkinter", "IPython", "jupyter", "notebook", "pytest",
              "PyQt5", "PyQt6", "PySide2", "sklearn", "seaborn", "pingouin"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RS-StatChem",
    console=False,
    icon=str(RES / ("icon.ico" if sys.platform == "win32" else "icon.icns")),
    argv_emulation=False,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="RS-StatChem", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="RS-StatChem.app",
        icon=str(RES / "icon.icns"),
        bundle_identifier="id.rseducation.rsstatchem",
        version=VERSION,
        info_plist={
            "CFBundleName": "RS-StatChem",
            "CFBundleDisplayName": "RS-StatChem",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "NSRequiresAquaSystemAppearance": True,
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "RS-StatChem Dataset",
                "CFBundleTypeExtensions": ["rsd"],
                "CFBundleTypeRole": "Editor",
                "CFBundleTypeIconFile": "icon.icns",
            }, {
                "CFBundleTypeName": "RS-StatChem Graph",
                "CFBundleTypeExtensions": ["rsg"],
                "CFBundleTypeRole": "Editor",
            }, {
                "CFBundleTypeName": "RS-StatChem Chemical Drawing",
                "CFBundleTypeExtensions": ["rsc"],
                "CFBundleTypeRole": "Editor",
            }, {
                "CFBundleTypeName": "SPSS Data",
                "CFBundleTypeExtensions": ["sav"],
                "CFBundleTypeRole": "Viewer",
            }],
        },
    )
