# RS-StatChem

Aplikasi desktop untuk analisis statistik penelitian (S1–S3), Response Surface Methodology,
grafik ilmiah, dan gambar struktur kimia. Saat dibuka, aplikasi menampilkan layar pemilih
**empat modul**, masing-masing dengan jendela dan cara input datanya sendiri:

| Modul | Mengikuti | Input data |
|---|---|---|
| **Statistics** | SPSS | Data Editor (Data View / Variable View), menu Analyze, Output Viewer |
| **RSM / DOE** | Design-Expert | Wizard New Design, design layout (Std, Run, Factor, Response), panel navigasi Design / Analysis / Optimization |
| **Graph & Analysis** | OriginLab | Workbook (kolom A(X), B(Y), Long Name, Units, F(x)), Project Explorer, grafik dan fitting di area kerja |
| **Chemical Drawing** | ChemDraw | Dokumen gambar dengan palet alat, Analysis Window, Chemical Properties |

Modul lain bisa dibuka kapan saja lewat **Window → Modules**.

Dibangun dengan Python 3.12, PySide6 (Qt), pandas, SciPy, dan statsmodels. Berjalan native di
Apple Silicon dan sebagai aplikasi portable (tanpa instalasi) di Windows.

## Status (v0.9.0)

129 entri menu aktif; yang masih nonaktif hanya Ratio dan Distances. 567 tes otomatis
membandingkan hasil dengan SciPy, statsmodels, pingouin, lavaan (via contoh terpublikasi),
plspm, RDKit, dan contoh buku teks (Montgomery, Miller & Miller).

| Area (setara dengan) | Isi |
|---|---|
| **Data Editor** (SPSS) | Data/Variable View, value labels, missing values, copy–paste Excel, **Undo/Redo**, .rsd/.sav/CSV/Excel |
| **Data** (SPSS) | Sort, Select Cases, Split File, Weight Cases, Identify Duplicates, Merge (Add Cases/Variables), Aggregate, Transpose, Restructure |
| **Transform** (SPSS) | Compute, Recode (Same/Different), Count, Automatic Recode, Visual Binning, Rank Cases, Replace Missing Values, Shift Values |
| **Analyze** (SPSS) | Frequencies, Descriptives, Explore, Crosstabs · t-test, One-Way ANOVA, Means · GLM Univariate, Multivariate, Repeated Measures · Linear Mixed Models · Correlations · Linear, Curve Estimation, Binary/Multinomial/Ordinal Logistic, Nonlinear · Uji asumsi klasik · K-Means, Hierarchical, Discriminant · Factor/PCA · Reliability & validitas · Nonparametrik · ROC · PROCESS Model 1/4/7/14 |
| **Graphs** (Origin) | Graph Studio: 29 jenis plot (line, scatter, column/bar, area, pie, histogram, box, bubble, double-Y, multi-panel, waterfall, contour, heatmap, 3D, polar, ternary), editor objek & properti, tema jurnal, ekspor PNG/TIFF/SVG/PDF/EPS hingga 1200 dpi, file .rsg |
| **Analysis** (Origin) | Fit Linear/Polynomial/Nonlinear (32 fungsi + fungsi sendiri), Calibration Curve (LOD/LOQ), diferensial, integral, interpolasi, normalisasi, smoothing, FFT, baseline, peak finding, multi-peak fit |
| **SEM** (AMOS/lavaan, SmartPLS) | CB-SEM: sintaks lavaan & form builder, ML/GLS/ULS, fit indices, CR/AVE/HTMT, efek tidak langsung (bootstrap), modification indices, diagram jalur · PLS-SEM: PLS/PLSc, reliabilitas & validitas, HTMT, VIF, SRMR, bootstrapping, Q²/PLSpredict, moderasi, diagram |
| **Chemistry** (ChemDraw + kalkulator) | Structure & Reaction Editor native (alat ikatan/cincin/rantai/singkatan, wedge, panah reaksi & mekanisme, Clean, panel Analysis, cek kesetaraan & stoikiometri, template, gaya ACS 1996, ekspor SVG/PNG/PDF/MOL/SMILES, file .rsc) · rumus & massa molar, rumus empiris, properti molekul & Lipinski, deskriptor dari kolom SMILES, penyetaraan reaksi & stoikiometri, kalkulator larutan & pH |
| **RSM** (Design-Expert) | Factorial, fractional, Plackett-Burman, Taguchi, CCD, Box-Behnken, 3-level, D-optimal, mixture · Fit Summary, ANOVA, Box-Cox, diagnostik · contour/3D/perturbation/interaction/cube/ternary · desirability optimization, point prediction, Taguchi S/N |

Rincian modul dan kontrak pengembangan ada di [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Menu yang belum dikerjakan sudah ditampilkan, tetapi dalam keadaan nonaktif.

## Menjalankan dari kode sumber

```bash
.venv/bin/python run.py
```

Menyiapkan lingkungan dari awal (butuh Python 3.12 arm64):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Tes:

```bash
.venv/bin/python -m pytest -q tests
```

## Membuat aplikasi

**macOS (Apple Silicon, native arm64)** → `dist/RS-StatChem.app` dan file zip:

```bash
./scripts/build_mac.sh
```

Aplikasi belum bertanda tangan (unsigned). Jika macOS memblokirnya setelah dipindah dari komputer
lain, klik kanan → Open, atau:

```bash
xattr -dr com.apple.quarantine /Applications/RS-StatChem.app
```

**Windows portable** → `dist/RS-StatChem/RS-StatChem.exe`. PyInstaller tidak bisa membuat
`.exe` dari Mac, jadi pilih salah satu:

- **GitHub Actions (disarankan):** push repo ini ke GitHub, lalu jalankan workflow *Build*
  (tab Actions → Build → Run workflow), atau push tag `v0.1.0`. Hasilnya berupa artifact
  `RS-StatChem-Windows-x64-portable.zip` dan `RS-StatChem-macOS-arm64.zip`.
- **Di komputer Windows:** pasang Python 3.12, `pip install -r requirements.txt -r requirements-dev.txt`,
  lalu jalankan `scripts\build_windows.bat`.

Versi portable menyimpan pengaturan di folder `UserData` di sebelah `RS-StatChem.exe`.

## Struktur kode

```
rsstatchem/
  core/        tanpa Qt: dataset + metadata variabel, model output, laporan, statistik
  io/          .rsd, .sav, CSV, Excel
  ui/          jendela, model tabel, dialog (pola dialog SPSS di ui/dialogs/base.py)
  resources/   logo dan ikon (buat ulang dengan scripts/make_icons.py)
packaging/     spec PyInstaller dan README versi portable
```

Lapisan `core/` sengaja dipisahkan dari UI, supaya kelak bisa dipakai ulang untuk versi web
atau API tanpa menulis ulang analisisnya.

Copyright © 2026 Roni Saputra, M.Si. Untuk penggunaan pribadi/non-komersial.
