# PyInstaller spec for RS-StatChem.
#   macOS (Apple Silicon): builds dist/RS-StatChem.app as native arm64
#   Windows:               builds dist/RS-StatChem/ (portable folder, run RS-StatChem.exe)
# Build with:  pyinstaller packaging/rsstatchem.spec --noconfirm
import re
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent
RES = ROOT / "rsstatchem" / "resources"
_INIT = (ROOT / "rsstatchem" / "__init__.py").read_text(encoding="utf-8")
VERSION = re.search(r'__version__ = "([^"]+)"', _INIT).group(1)
COPYRIGHT = re.search(r'COPYRIGHT = "([^"]+)"', _INIT).group(1)


def windows_version_file() -> str | None:
    """Version resource for RS-StatChem.exe (Properties > Details shows version and
    copyright)."""
    if sys.platform != "win32":
        return None
    nums = (tuple(int(n) for n in re.findall(r"\d+", VERSION)[:3]) + (0, 0, 0, 0))[:4]
    fields = {"CompanyName": "RS Education", "FileDescription": "RS-StatChem",
              "FileVersion": VERSION, "InternalName": "RS-StatChem",
              "LegalCopyright": COPYRIGHT, "OriginalFilename": "RS-StatChem.exe",
              "ProductName": "RS-StatChem", "ProductVersion": VERSION}
    strings = ",\n".join(f"        StringStruct({k!r}, {v!r})" for k, v in fields.items())
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={nums}, prodvers={nums}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
{strings}])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    path = Path(workpath) / "version_info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)

# Every module of the package, found from the files rather than by importing the package:
# PyInstaller's collect_submodules() silently returns nothing when the project folder is not on
# sys.path (as with the plain `pyinstaller` entry point on CI), and the feature modules are only
# loaded by name in ui/menus.py, so the analysis cannot see them.
APP_MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
    for path in (ROOT / "rsstatchem").rglob("*.py"))


def resource_datas():
    """Every resource file (icons, the offline name database, the experimental property
    database and ML models in resources/properties, the OPSIN jar fetched by
    scripts/fetch_opsin.py, the CDK module jars in resources/java/cdk (class path in
    resources/java/cdk-classpath.txt) and the helper built by scripts/fetch_cdk.py) except a
    local Java runtime in resources/opsin/jre: the CI workflow copies the jlink runtime into
    the finished bundle so PyInstaller never rewrites or re-signs the Java binaries. The CDK
    jars must be present for a release build."""
    out = []
    for path in sorted(RES.rglob("*")):
        rel = path.relative_to(RES)
        if not path.is_file() or rel.parts[:2] == ("opsin", "jre") or \
                path.name in (".DS_Store",) or path.suffix == ".part":
            continue
        out.append((str(path), str(Path("rsstatchem/resources") / rel.parent)))
    return out


def rdkit_datas():
    """RDKit's functional-group table (rdkit/Data/FragmentDescriptors.csv): the fr_* descriptors
    used by the Chemical Properties ML models are only defined when it is present (PyInstaller
    does not collect rdkit/Data by itself; found by the packaged --selftest)."""
    import rdkit
    data = Path(rdkit.__file__).parent / "Data" / "FragmentDescriptors.csv"
    return [(str(data), "rdkit/Data")]


a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    datas=resource_datas() + rdkit_datas(),
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
    version=windows_version_file(),
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
            "NSHumanReadableCopyright": COPYRIGHT,
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
                "CFBundleTypeName": "RS-StatChem Design (RSM)",
                "CFBundleTypeExtensions": ["rsx"],
                "CFBundleTypeRole": "Editor",
            }, {
                "CFBundleTypeName": "RS-StatChem Graph Project",
                "CFBundleTypeExtensions": ["rsop"],
                "CFBundleTypeRole": "Editor",
            }, {
                "CFBundleTypeName": "SPSS Data",
                "CFBundleTypeExtensions": ["sav"],
                "CFBundleTypeRole": "Viewer",
            }],
        },
    )
