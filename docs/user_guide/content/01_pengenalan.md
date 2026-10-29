# Mengenal RS-StatChem

RS-StatChem adalah aplikasi desktop untuk kebutuhan penelitian: analisis statistik, rancangan
percobaan dan *Response Surface Methodology*, grafik ilmiah dan *curve fitting*, gambar
struktur kimia, serta analisis citra. Kelima bidang itu tersedia sebagai **modul** terpisah.
Setiap modul punya jendela dan cara memasukkan data sendiri, dengan alur kerja yang mengikuti
aplikasi yang sudah lazim dipakai di bidangnya.

## Lima modul dalam satu aplikasi

| Modul | Alur kerja mengikuti | Untuk apa |
|---|---|---|
| **Statistics** | SPSS | Data Editor (Data View dan Variable View), menu Analyze, hasil di Output Viewer yang bisa disalin rapi ke Word |
| **RSM / DOE** | Design-Expert | Wizard rancangan (faktorial, CCD, Box-Behnken, mixture, optimal), analisis ANOVA, grafik model, optimasi |
| **Graph & Analysis** | OriginLab | Workbook berkolom X/Y, grafik publikasi, *curve fitting*, analisis puncak dan sinyal |
| **Chemical Drawing** | ChemDraw | Menggambar struktur dan reaksi, nama ke struktur, sifat kimia, spektrum NMR prediksi |
| **Image Analysis** | ImageJ | Membuka, mengolah, dan mengukur citra serta *stack*: filter, threshold, partikel, ROI |

> [!NOTE] Nama SPSS, OriginLab, Design-Expert, ChemDraw, dan ImageJ hanya menunjukkan alur
> kerja yang diikuti setiap modul. RS-StatChem tidak berafiliasi dengan pemilik nama-nama
> tersebut.

## Module Launcher

Saat aplikasi dibuka, yang pertama muncul adalah **Module Launcher**. Klik salah satu kartu
untuk membuka modul tersebut di jendelanya sendiri. Anda juga bisa menekan angka ++1++ sampai
++5++ di keyboard sesuai nomor di pojok kartu.

![intro.launcher]

Setiap modul memakai jenis berkasnya sendiri. Saat Anda membuka berkas, misalnya dengan tombol
**Open File...** atau klik dua kali di Finder/Explorer, RS-StatChem memilihkan modul yang
sesuai secara otomatis:

| Jenis berkas | Dibuka di modul |
|---|---|
| `.rsd`, `.sav`, `.por`, `.dta`, `.sas7bdat`, `.xpt`, `.csv`, `.txt`, `.xlsx`, `.rso` | Statistics |
| `.rsx` | RSM / DOE |
| `.rsop`, `.rsg` | Graph & Analysis |
| `.rsc`, `.mol`, `.sdf`, `.rxn`, `.cdxml` | Chemical Drawing |
| `.rsij`, `.tif`, `.png`, `.jpg`, `.bmp`, `.gif`, `.dcm`, `.fits`, `.roi` | Image Analysis |

## Berpindah antarmodul

Beberapa modul bisa dibuka sekaligus. Dari jendela modul mana pun, pilih
[[Window > Modules]] untuk kembali ke Module Launcher atau membuka jendela baru dari modul lain.
Aplikasi baru tertutup setelah jendela terakhir ditutup.

![intro.window_modules]

## Cara membaca panduan ini

- Nama menu ditulis seperti ini: [[File > Save As]], artinya buka menu **File** lalu pilih
  **Save As**.
- Tombol keyboard ditulis seperti ++Ctrl+S++. Di macOS, ++Ctrl++ diganti ++Cmd++. Gambar di
  panduan ini diambil di macOS, jadi pintasan di menu tampil dengan simbol ⌘ (= ++Ctrl++ di
  Windows).
- Gambar diberi **nomor merah**. Penjelasan setiap nomor ada di kotak di bawah gambar.
  Label merah dengan panah menunjuk langsung ke bagian yang dijelaskan.
- Setiap bab modul ditutup dengan **Referensi Fitur**, yaitu daftar lengkap semua perintah di
  menu modul itu beserta fungsinya. Daftar ini dibuat otomatis dari aplikasi, sehingga selalu
  sesuai dengan versi yang Anda pakai.

> [!TIP] Panduan ini dibuat ulang setiap kali ada versi baru, jadi isinya selalu sesuai dengan
> versi aplikasi Anda. Dari modul mana pun, panduan untuk versi yang sedang Anda pakai bisa
> dibuka lewat [[Help > User Guide (PDF)]], atau diunduh dari halaman rilis bersama aplikasinya.
