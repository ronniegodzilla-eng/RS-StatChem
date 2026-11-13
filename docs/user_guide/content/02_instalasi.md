# Instalasi dan Pembaruan

RS-StatChem tersedia untuk **macOS** 14 Sonoma atau lebih baru, baik Mac dengan chip Apple (M1,
M2, M3, M4 dan penerusnya) maupun Mac dengan prosesor Intel, serta untuk **Windows** 64-bit. Aplikasi ini tidak memerlukan instalasi Python, Java, atau program
lain, karena semua yang dibutuhkan sudah ada di dalam paketnya.

## Mengunduh

Buka halaman rilis RS-StatChem:

`https://github.com/ronniegodzilla-eng/RS-StatChem-releases/releases/latest`

Di bagian **Assets**, unduh berkas yang sesuai dengan komputer Anda:

| Berkas | Untuk |
|---|---|
| `RS-StatChem-macOS-arm64.zip` | Mac dengan chip Apple (M1 atau lebih baru), macOS 14 Sonoma ke atas |
| `RS-StatChem-macOS-x86_64.zip` | Mac dengan prosesor Intel, macOS 14 Sonoma ke atas |
| `RS-StatChem-Windows-x64-portable.zip` | Windows 10 atau 11, 64-bit |
| `RS-StatChem-User-Guide.pdf` | Panduan ini |

## Instalasi di macOS

> [!TIP] Tidak yakin jenis Mac Anda? Klik menu Apple ▸ **About This Mac**. Jika tertulis
> **Chip** (misalnya Apple M2), unduh versi `arm64`. Jika tertulis **Processor** (misalnya
> Intel Core i5), unduh versi `x86_64`. Fitur kedua versi sama persis.

1. Klik dua kali berkas zip yang sesuai (`RS-StatChem-macOS-arm64.zip` atau
   `RS-StatChem-macOS-x86_64.zip`) untuk mengekstraknya. Akan muncul **RS-StatChem.app**.
2. Seret **RS-StatChem.app** ke folder **Applications**.
3. Buka **Applications**, lalu klik dua kali **RS-StatChem**.
4. Pada pembukaan pertama, macOS mungkin menolak membukanya karena aplikasi ini belum
   didaftarkan ke Apple. Klik **Done** (atau **OK**), lalu buka
   **System Settings ▸ Privacy & Security**. Gulir ke bagian **Security**, lalu klik
   **Open Anyway** di samping pesan tentang RS-StatChem dan konfirmasikan dengan kata sandi Mac
   Anda.
5. Setelah itu RS-StatChem bisa dibuka seperti aplikasi biasa.

> [!TIP] Di macOS 14 atau lebih lama, cara cepatnya: klik kanan **RS-StatChem** di
> Applications, pilih **Open**, lalu klik **Open** sekali lagi.

> [!NOTE] Jika macOS menyebut aplikasi "rusak" (*damaged*), jalankan perintah ini satu kali di
> aplikasi Terminal, lalu buka lagi RS-StatChem:
> `xattr -dr com.apple.quarantine /Applications/RS-StatChem.app`

## Instalasi di Windows (portable)

Versi Windows bersifat *portable*. Aplikasi ini tidak perlu di-*install*, tidak mengubah
registry, dan bisa dijalankan dari flashdisk.

1. Klik kanan `RS-StatChem-Windows-x64-portable.zip`, pilih **Extract All...**, lalu pilih
   lokasinya, misalnya **Documents** atau **Desktop**.
2. Buka folder **RS-StatChem** hasil ekstrak, lalu klik dua kali **RS-StatChem.exe**.
3. Jika muncul layar biru **Windows protected your PC** (SmartScreen), klik **More info**, lalu
   **Run anyway**. Peringatan ini muncul karena aplikasi belum memiliki sertifikat penerbit.
4. Untuk membuatnya mudah dibuka, klik kanan **RS-StatChem.exe** lalu pilih
   **Show more options ▸ Send to ▸ Desktop (create shortcut)**.

> [!WARNING] Jangan menjalankan **RS-StatChem.exe** langsung dari dalam berkas zip tanpa
> mengekstraknya. Aplikasi akan gagal menemukan berkas pendukungnya.

### Jika diblokir oleh Smart App Control

Di sebagian komputer Windows 11, yang muncul bukan layar biru SmartScreen, melainkan kotak
pesan **Smart App Control blocked an app that may be unsafe** (di Windows berbahasa Indonesia:
**Kontrol Aplikasi Pintar memblokir aplikasi yang mungkin tidak aman**) dengan tombol **OK**
(**Oke**) saja. Pesan ini muncul karena RS-StatChem belum memiliki tanda tangan digital
penerbit, bukan karena aplikasinya rusak atau berbahaya. Berbeda dengan SmartScreen, pesan ini
tidak memiliki tombol **Run anyway**, jadi RS-StatChem baru bisa dibuka setelah
Smart App Control dimatikan:

1. Klik **OK** untuk menutup pesan.
2. Buka **Windows Security** (**Keamanan Windows**) dari menu Start.
3. Pilih **App & browser control** (**Kontrol aplikasi & browser**), lalu klik
   **Smart App Control settings** (**Pengaturan Kontrol Aplikasi Pintar**).
4. Pilih **Off** (**Nonaktif**). Langkah ini memerlukan akun administrator.
5. Klik dua kali **RS-StatChem.exe** lagi. Jika kemudian muncul layar biru SmartScreen, klik
   **More info**, lalu **Run anyway**, seperti pada langkah 3 di atas.

> [!WARNING] Smart App Control adalah salah satu lapis perlindungan Windows. Matikan hanya di
> laptop milik sendiri, dan jangan di komputer kantor atau laboratorium yang dikelola orang
> lain (mintalah bantuan pengelolanya). Di Windows 11 yang sudah mendapat pembaruan April 2026
> atau lebih baru, fitur ini bisa dinyalakan kembali dari menu yang sama. Di Windows yang
> lebih lama, fitur yang sudah dimatikan tidak bisa dinyalakan lagi tanpa memasang ulang
> Windows.

Pengaturan aplikasi tersimpan di folder **UserData** di sebelah **RS-StatChem.exe**. Karena
itu, seluruh folder RS-StatChem bisa dipindah ke komputer lain bersama pengaturannya.

## Pembukaan pertama

Setelah aplikasi terbuka, Module Launcher langsung muncul dan Anda bisa memilih modul. Tidak
perlu koneksi internet, kecuali untuk tiga hal berikut:

- pemeriksaan versi baru;
- pencarian nama senyawa yang tidak ada di basis data bawaan (modul Chemical Drawing);
- contoh citra ImageJ yang diunduh dari internet ([[File > Open Samples]] di modul Image
  Analysis).

## Memperbarui ke versi baru

RS-StatChem memeriksa versi baru secara otomatis, kira-kira sekali sehari saat aplikasi
dibuka. Pemeriksaan juga bisa dilakukan kapan saja dari [[Help > Check for Updates]]. Jika ada
versi baru, muncul jendela berisi catatan perubahannya:

![intro.update]

1. Klik **Download**. Browser akan mengunduh zip versi baru.
2. **macOS:** ekstrak zip, lalu seret **RS-StatChem.app** yang baru ke **Applications** dan
   pilih **Replace**. Pengaturan Anda tetap tersimpan.
3. **Windows:** ekstrak zip ke lokasi baru, lalu salin folder **UserData** dari folder lama ke
   folder baru jika ingin membawa pengaturan. Setelah itu folder lama boleh dihapus.

Semua berkas kerja Anda (data, desain, proyek, gambar) tetap bisa dibuka di versi baru.

## Menghapus aplikasi

- **macOS:** seret **RS-StatChem.app** dari Applications ke Trash. Pengaturannya tersimpan di
  `~/Library/Application Support/RS-StatChem` dan boleh ikut dihapus.
- **Windows:** hapus folder **RS-StatChem**. Tidak ada yang tertinggal di sistem.

## Versi dan lisensi

Nomor versi, hak cipta, dan lisensi bisa dilihat di [[Help > About RS-StatChem]]. RS-StatChem bebas
dipakai untuk **keperluan pribadi dan non-komersial** di bawah lisensi PolyForm Noncommercial
1.0.0. Daftar pustaka pihak ketiga beserta lisensinya ikut di dalam aplikasi.

![intro.about]
