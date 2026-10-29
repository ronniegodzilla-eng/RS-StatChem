# Lampiran

## Jenis berkas

Setiap modul menyimpan pekerjaannya dalam berkas miliknya sendiri. Selain itu, setiap modul
juga bisa membuka dan menyimpan format yang lazim dipakai aplikasi lain.

| Modul | Berkas kerja | Bisa dibuka | Bisa disimpan / diekspor |
|---|---|---|---|
| Statistics | `.rsd` (data), `.rso` (output) | SPSS `.sav` dan `.por`, Stata `.dta`, SAS `.sas7bdat` dan `.xpt`, CSV/teks, Excel | `.sav`, `.dta`, `.xpt`, CSV, Excel; output ke Word, Excel, HTML, PDF, teks |
| RSM / DOE | `.rsx` | data historis dari CSV/Excel | layout desain ke CSV/Excel, grafik sebagai gambar |
| Graph & Analysis | `.rsop` (proyek), `.rsg` (grafik) | CSV/ASCII, Excel, JCAMP-DX, SPC, MATLAB, HDF5/NetCDF, WAV, gambar | CSV/ASCII; grafik PNG, TIFF, SVG, PDF, EPS |
| Chemical Drawing | `.rsc` | ChemDraw `.cdx`/`.cdxml`, MOL/SDF, RXN, SMILES, InChI, HELM, FASTA, JCAMP-DX | `.cdx`, `.cdxml`, MOL (V2000/V3000), SMILES, gambar SVG/PNG/PDF |
| Image Analysis | `.rsij` (proyek) | TIFF/ImageJ, PNG, JPEG, GIF, BMP, DICOM, FITS, AVI, raw, ROI ImageJ | TIFF, PNG, JPEG, GIF, BMP, FITS, AVI, raw, teks, ROI/RoiSet |

> [!TIP] Rincian setiap perintah impor dan ekspor ada di bagian Referensi Fitur pada bab modul
> yang bersangkutan, di bawah menu **File**.

## Koneksi internet dan privasi

RS-StatChem bekerja sepenuhnya di komputer Anda. Data, desain, gambar, dan hasil analisis
tidak pernah dikirim ke mana pun. Internet hanya dipakai untuk hal-hal berikut, dan hanya saat
fitur itu digunakan:

| Fitur | Tujuan | Yang dikirim |
|---|---|---|
| Pemeriksaan versi baru | GitHub (halaman rilis RS-StatChem) | Hanya permintaan versi terbaru |
| Nama senyawa yang tidak ada di basis data bawaan (Chemical Drawing) | PubChem (NCBI) | Nama atau struktur yang sedang dicari |
| Contoh citra dan **File ▸ Import ▸ URL** (Image Analysis) | imagej.net atau alamat yang Anda isi | Permintaan berkas |
| Tautan bantuan ImageJ (Image Analysis) | imagej.net, dibuka di browser | Tidak ada data |

Pemeriksaan otomatis versi baru bisa dimatikan di jendela **Software Update** (hapus centang
**Check for updates automatically**).

## Pemecahan masalah

| Masalah | Penyelesaian |
|---|---|
| macOS: "RS-StatChem can't be opened" atau "Apple could not verify..." | Buka **System Settings ▸ Privacy & Security**, lalu klik **Open Anyway** (lihat Bab 2). |
| macOS: aplikasi disebut rusak (*damaged*) | Jalankan di Terminal: `xattr -dr com.apple.quarantine /Applications/RS-StatChem.app` |
| Windows: muncul layar biru SmartScreen | Klik **More info**, lalu **Run anyway**. |
| Windows: aplikasi tidak mau terbuka atau langsung tertutup | Pastikan zip sudah diekstrak semua (jangan dijalankan dari dalam zip) dan folder tidak berada di lokasi yang dibatasi, misalnya langsung di `C:\Program Files`. |
| Tabel output berantakan saat ditempel di Word | Salin lewat **Edit ▸ Copy** di Output Viewer, lalu tempel di Word dengan ++Ctrl+V++ biasa. Tabel akan masuk sebagai tabel Word dengan lebar kolom yang sudah disesuaikan. |
| Nama senyawa tidak ditemukan | Coba nama IUPAC atau nama umum dalam bahasa Inggris. Untuk nama yang jarang dipakai, sambungkan internet agar PubChem bisa dicari. |
| Pemeriksaan versi baru gagal | Periksa koneksi internet. Batas akses anonim GitHub kadang tercapai, jadi coba lagi beberapa saat kemudian. |
| Ingin mengembalikan pengaturan awal | Tutup aplikasi, lalu hapus folder pengaturannya. Di macOS: `~/Library/Application Support/RS-StatChem`. Di Windows: folder **UserData** di sebelah **RS-StatChem.exe**. |

## Lisensi

RS-StatChem © 2026 Roni Saputra, M.Si. Aplikasi ini boleh dipakai, disalin, dan dibagikan untuk
**keperluan pribadi dan non-komersial** sesuai lisensi PolyForm Noncommercial 1.0.0. Teks
lengkap lisensi (`LICENSE.md`) dan daftar pustaka pihak ketiga beserta lisensinya
(`THIRD_PARTY_NOTICES.md`) ikut di dalam aplikasi.

SPSS adalah merek dagang IBM; Origin dan OriginPro milik OriginLab; Design-Expert milik
Stat-Ease; ChemDraw milik Revvity. ImageJ dan Fiji adalah nama proyek masing-masing.
RS-StatChem tidak berafiliasi dengan, ataupun didukung oleh, pihak-pihak tersebut.
