# Konektor Claude

RS-StatChem menyediakan **konektor** untuk aplikasi **Claude** di komputer Anda. Setelah
konektor dipasang, Anda cukup menulis permintaan di Claude, misalnya "bandingkan nilai post-test
kelompok kontrol dan eksperimen", lalu Claude membuka berkas data Anda dan mengerjakannya dengan
RS-StatChem. Konektor menjangkau **kelima modul**: Statistics, RSM / DOE, Graph & Analysis,
Chemical Drawing, dan Image Analysis. Angka yang keluar sama dengan yang tampil di aplikasi, dan
Claude membantu menjelaskannya.

Konektor ini memakai standar terbuka *Model Context Protocol* (MCP). RS-StatChem tidak pernah
meminta login Claude dan tidak memerlukan API key. Yang terhubung ke Claude adalah aplikasi
Claude milik Anda sendiri, dengan akun Anda sendiri.

## Yang dibutuhkan

- **Aplikasi Claude untuk desktop** (macOS atau Windows) yang sudah login dengan akun Anda.
  Konektor ini tidak bisa dipakai dari claude.ai di browser maupun dari aplikasi Claude di
  ponsel, karena keduanya tidak dapat menjalankan program di komputer Anda.
- RS-StatChem sudah terpasang dan pernah dibuka minimal satu kali dengan cara biasa (lihat bab
  Instalasi dan Pembaruan).
- Koneksi internet, karena Claude sendiri bekerja lewat internet.

## Memasang konektor

Pemasangan cukup dilakukan satu kali.

1. Di RS-StatChem, klik **Claude Connector...** pada layar pemilih modul, atau pilih
   **Window ▸ Modules ▸ Claude Connector...** di modul mana pun.
2. Klik **Add to Claude**. RS-StatChem menambahkan dirinya ke berkas konfigurasi aplikasi Claude.
   Isi lain di berkas itu tidak diubah, dan salinan berkas lamanya disimpan di folder yang sama.
3. Tutup aplikasi Claude sepenuhnya, lalu buka lagi. Di macOS tekan ++Cmd+Q++. Di Windows,
   keluar juga dari ikon Claude di pojok kanan bawah layar (system tray).
4. Di kotak tulis pesan Claude, klik tombol **+**, lalu buka **Connectors**. Konektor
   **rs-statchem** seharusnya sudah tampil di sana.

Tombol **Remove from Claude** melepas konektor, dan **Copy Configuration** menyalin isi
konfigurasinya jika Anda ingin memasangnya sendiri seperti di bawah ini.

### Isi berkas konfigurasi

Bagian ini hanya diperlukan jika **Add to Claude** tidak dapat dipakai. Di aplikasi Claude, buka
**Settings**, pilih tab **Developer**, klik **Edit Config**, lalu isi berkas
`claude_desktop_config.json` seperti contoh berikut.

Untuk **macOS**, dengan RS-StatChem berada di folder Applications:

```
{
  "mcpServers": {
    "rs-statchem": {
      "command": "/Applications/RS-StatChem.app/Contents/MacOS/RS-StatChem",
      "args": ["--mcp"]
    }
  }
}
```

Untuk **Windows**. Ganti lokasinya dengan tempat Anda mengekstrak folder
RS-StatChem, dan tulis setiap garis miring terbalik dua kali (`\\`):

```
{
  "mcpServers": {
    "rs-statchem": {
      "command": "C:\\Users\\NamaAnda\\Documents\\RS-StatChem\\RS-StatChem.exe",
      "args": ["--mcp"]
    }
  }
}
```

> [!WARNING] Jika berkas itu sudah berisi konektor lain, jangan hapus isinya. Tambahkan bagian
> `"rs-statchem": { ... }` di dalam `"mcpServers"` yang sudah ada, dipisahkan dengan koma dari
> konektor sebelumnya.

> [!NOTE] Berkas konfigurasi itu berada di
> `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) atau
> `%APPDATA%\Claude\claude_desktop_config.json` (Windows). Konektor yang sama juga bisa dipakai
> dari Claude Code dan aplikasi lain yang mendukung MCP: perintahnya adalah program RS-StatChem
> dengan argumen `--mcp`.

## Memakai konektor

Tulis permintaan Anda di Claude dengan bahasa sehari-hari, dan sebutkan **lokasi lengkap berkas
datanya**. Contoh:

- "Buka `/Users/roni/Documents/penelitian.sav` dengan RS-StatChem, lalu tampilkan daftar
  variabelnya."
- "Apakah nilai posttest berbeda antara kelompok kontrol dan eksperimen? Pakai uji yang sesuai
  dan jelaskan hasilnya."
- "Uji pengaruh pretest dan usia terhadap posttest dengan regresi linear, lengkap dengan uji
  asumsi klasik."
- "Jalankan analisis faktor untuk butir p1 sampai p20, lalu simpan hasilnya sebagai Word."
- "Buat rancangan Central Composite untuk suhu 60–80 °C dan waktu 30–60 menit. Ini hasil
  rendemennya: ... Analisis, gambar permukaan responsnya, dan cari kondisi optimumnya."
- "Plot data kinetika ini dan cocokkan dengan y = a·exp(−k·x)."
- "Gambar struktur kafein, lalu tampilkan sifat kimianya."
- "Buka `blobs.tif`, hitung jumlah partikel, dan ukur luasnya."

Setiap kali Claude hendak menjalankan sebuah alat RS-StatChem, aplikasi Claude meminta
persetujuan Anda terlebih dahulu.

### Di latar belakang atau di jendela yang terbuka

Konektor bekerja dengan dua cara.

- **Latar belakang** (bawaan). Jendela RS-StatChem tidak perlu dibuka. Claude mengerjakan semuanya
  tanpa tampilan, lalu menyimpan hasilnya ke berkas yang Anda minta (`.rso`, `.docx`, `.xlsx`,
  `.pdf`, `.rsx`, `.rsop`, dan lainnya) dan dapat membukakannya di RS-StatChem.
- **Di jendela yang terbuka.** Buka **Claude Connector...**, lalu centang **Allow Claude to work in
  the windows that are open**. Claude kemudian membaca data yang sedang Anda buka, dan hasilnya
  langsung muncul di Output Viewer atau jendela modul, sehingga Anda dapat melihat dan
  membatalkannya (Undo). Centang ini tetap berlaku sampai Anda mematikannya. Claude tidak dapat
  mengubahnya sendiri.

> [!WARNING] Simpan pekerjaan Anda sebelum mengizinkan Claude bekerja di jendela yang terbuka. Selama
> Claude bekerja, jangan mengetik atau membuka dialog di jendela yang sama.

> [!TIP] Cara cepat mendapatkan lokasi lengkap berkas: di macOS, klik kanan berkas di Finder
> sambil menahan tombol ++Option++, lalu pilih **Copy ... as Pathname**. Di Windows, tahan
> ++Shift++, klik kanan berkas, lalu pilih **Copy as path**.

Alat yang tersedia bagi Claude:

| Alat | Kegunaan |
|---|---|
| `open_data`, `list_datasets`, `list_variables`, `save_data` | Membuka dan menyimpan data (`.rsd`, `.sav`, `.por`, `.xlsx`, `.csv`, `.txt`, `.dta`, `.sas7bdat`, `.xpt`) serta melihat daftar variabel |
| `descriptives`, `frequencies`, `one_sample_t_test`, `one_sample_z_test`, `independent_samples_t_test`, `paired_samples_t_test`, `one_way_anova`, `correlations`, `linear_regression` | Analisis statistik yang paling sering dipakai, langsung tanpa dialog |
| `find_commands`, `run_command`, `fill_dialog`, `wait` | **Semua perintah menu** di semua modul: mencari perintah, menjalankannya, dan mengisi dialognya |
| `open_module`, `app_status`, `read_window`, `close_window`, `take_screenshot` | Membuka modul dan melihat isi jendelanya |
| `read_table`, `write_table`, `select_in_table` | Membaca dan mengisi tabel: Data View, Variable View, rancangan RSM, lembar kerja |
| `mouse`, `keyboard` | Menggambar dan memilih di kanvas (struktur kimia, ROI, titik pada grafik) |
| `get_output`, `save_output`, `show_in_app` | Membaca hasil di Output Viewer, menyimpannya sebagai `.rso`, Word, Excel, HTML, atau PDF, dan membukanya di RS-StatChem |
| `rsm_create_design`, `rsm_set_response`, `rsm_analyze`, `rsm_graph`, `rsm_predict`, `rsm_optimize`, `rsm_design`, `rsm_save` | Rancangan percobaan, analisis respons, grafik model, prediksi, dan optimasi |
| `graph_data`, `graph_plot`, `graph_fit`, `graph_save` | Lembar kerja, grafik, dan pencocokan kurva |
| `chem_compound`, `chem_draw`, `chem_balance`, `chem_formula`, `chem_save` | Identitas dan sifat senyawa, gambar struktur, penyetaraan reaksi |
| `image_open`, `image_commands`, `image_command`, `image_macro`, `image_save` | Pengolahan dan pengukuran citra, termasuk makro ImageJ |

Select Cases, Split File, dan Weight Cases yang tersimpan di dalam berkas `.rsd` tetap berlaku,
sama seperti di modul Statistics.

## Batasan dan privasi

Bacalah bagian ini sebelum memakai konektor untuk data penelitian.

- **Perlu aplikasi Claude desktop.** Konektor tidak menambahkan jendela obrolan ke dalam
  RS-StatChem. Percakapan tetap berlangsung di aplikasi Claude.
- **Batas pemakaian mengikuti paket Claude Anda.** Setiap permintaan dan setiap tabel hasil
  memakai kuota akun Claude Anda sendiri. RS-StatChem tidak memungut biaya apa pun untuk
  konektor ini dan tidak mengetahui isi percakapan Anda.
- **Apa yang dikirim ke Claude.** Berkas data tetap berada di komputer Anda dan konektor
  RS-StatChem sendiri tidak mengirim apa pun ke internet. Namun, semua yang dikembalikan oleh
  sebuah alat menjadi bagian dari percakapan Anda dengan Claude dan dikirim oleh aplikasi Claude ke
  Anthropic: nama dan label variabel, value label, serta tabel hasil analisis. Tabel frekuensi
  memuat nilai-nilai data itu sendiri, sehingga variabel dengan lebih dari 50 nilai berbeda
  (nama, nomor identitas, skor) tidak ditabelkan kecuali diminta secara khusus. Jangan memakai konektor untuk data rahasia atau data
  pribadi yang tidak boleh keluar dari komputer Anda, atau hapus dulu identitas responden.
- **Mencetak tidak tersedia.** Panel cetak milik sistem tidak dapat dioperasikan oleh konektor.
  Mintalah Claude menyimpan hasilnya sebagai PDF.
- **Gambar ikut dikirim bila diminta.** Grafik, gambar struktur, citra, dan tangkapan jendela yang
  diminta Claude menjadi bagian percakapan, sama seperti tabel.
- **Hasil di latar belakang tidak tampil di jendela Anda.** Mintalah Claude menyimpannya ke berkas,
  atau aktifkan mode jendela yang terbuka.
- **Satu pekerjaan pada satu waktu.** Percakapan Claude yang berbeda memakai sesi latar belakang
  yang terpisah, tetapi mode jendela yang terbuka hanya melayani satu perintah pada satu waktu.
- **Angka dari RS-StatChem, penjelasan dari Claude.** Perhitungannya dikerjakan oleh
  RS-StatChem, tetapi pemilihan uji dan penafsirannya dibuat oleh Claude dan bisa keliru.
  Periksa kembali apakah uji yang dipilih sesuai dengan rancangan penelitian Anda.
- **Menyimpan berkas.** Alat penyimpan dan perintah menu dapat menulis berkas di komputer Anda.
  Alat penyimpan tidak menimpa berkas yang sudah ada kecuali Anda menyetujuinya. Makro dan skrip di
  modul Image Analysis dapat mengubah berkas, jadi baca dulu isinya sebelum menyetujui.
- **Izin tetap di tangan Anda.** Mode jendela yang terbuka hanya dapat dipakai oleh program milik
  pengguna komputer yang sama, tidak dapat dijangkau dari internet, dan selalu mati sampai Anda
  menyalakannya.

## Jika konektor tidak bekerja

| Masalah | Penyelesaian |
|---|---|
| **rs-statchem** tidak tampil di Claude | Periksa kembali isi `claude_desktop_config.json`: tanda kurung kurawal dan koma harus lengkap, dan lokasi RS-StatChem harus benar. Lalu tutup Claude sepenuhnya dan buka lagi. |
| macOS: konektor gagal dijalankan | Buka RS-StatChem satu kali dengan cara biasa sampai jendelanya muncul (lihat bab Instalasi dan Pembaruan), lalu buka ulang Claude. |
| Windows: konektor gagal dijalankan | Pastikan lokasi `RS-StatChem.exe` ditulis lengkap dengan garis miring terbalik ganda (`\\`), dan folder RS-StatChem sudah diekstrak dari berkas zip. |
| Claude menjawab berkas tidak ditemukan | Berikan lokasi lengkap berkas, bukan hanya namanya. |
| Ingin melihat catatan kesalahan | Aplikasi Claude menyimpannya di `~/Library/Logs/Claude` (macOS) atau `%APPDATA%\Claude\logs` (Windows), pada berkas `mcp-server-rs-statchem.log`. |
| Ingin melepas konektor | Klik **Remove from Claude** pada dialog **Claude Connector**, lalu buka ulang Claude. |
| Permintaan pertama terasa lama | Pada pemakaian pertama RS-StatChem menyiapkan daftar huruf untuk grafik, bisa sampai satu atau dua menit. Setelah itu hanya beberapa detik. |
| Claude tidak melihat data yang sedang dibuka | Centang **Allow Claude to work in the windows that are open** pada dialog **Claude Connector**. |
