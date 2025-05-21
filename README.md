# Predict-used-car-price-using-Mechine-Learning-Model
## Project Capstone Modul 3 Purwadhika : Machine Learning

**By: Risdan Kristori**


## Repository Structure
- <b>README.md:</b> Deskripsi Project
- <b>Predict Saudi Used Cars Prise Using Mechine Learning.ipynb:</b> Notebook Project

## Introduction
# **Business Problem**

**Saudi Arabia Market Opportunity**

Kerajaan Arab Saudi menunjukkan pertumbuhan signifikan dalam industri jual beli mobil bekas, didorong oleh berbagai faktor ekonomi, demografis, dan teknologi. Menurut laporan Ken Research, nilai transaksi penjualan mobil bekas di Arab Saudi diproyeksikan mengalami pertumbuhan tahunan sebesar 4,5% pada periode 2019 hingga 2025 .

Faktor-faktor yang mendorong pertumbuhan ini antara lain meningkatnya pendapatan masyarakat, pertumbuhan populasi ekspatriat, serta peningkatan jumlah pengemudi perempuan setelah pelonggaran regulasi. Selain itu, digitalisasi memainkan peran penting dalam transformasi pasar ini. Platform online seperti Syarah.com, Haraj, dan Motory memudahkan konsumen dalam mencari dan membeli mobil bekas dengan menyediakan informasi detail, opsi pembiayaan, dan layanan inspeksi kendaraan .

Peningkatan kualitas kendaraan bekas melalui program sertifikasi juga meningkatkan kepercayaan konsumen. Program ini menawarkan jaminan kualitas dan layanan purna jual, menjadikan mobil bekas pilihan yang menarik bagi konsumen yang mencari nilai lebih dengan harga terjangkau . Secara keseluruhan, kombinasi dari faktor-faktor tersebut menjadikan pasar mobil bekas di Arab Saudi sebagai sektor yang menjanjikan dengan potensi pertumbuhan yang berkelanjutan.

**Used Car Price Problem**

Salah satu permasalahan utama dalam bisnis jual beli mobil bekas di Arab Saudi adalah ketidakpastian harga jual yang wajar. Dikutip dari mordorintelligence.com (Faisal Al-Qasabi, 2024), harga mobil bekas sangat bergantung pada banyak faktor seperti merek, model, tahun pembuatan, jarak tempuh, jenis bahan bakar, serta kondisi fisik dan kelengkapan dokumen. Baik penjual maupun pembeli yang tidak memiliki informasi yang cukup sering kali mengalami kebingungan dalam menentukan harga yang sesuai. Penjual yang menetapkan harga terlalu tinggi berisiko tidak berhasil menjual kendaraannya, sementara penjual yang memasang harga terlalu rendah akan mengalami kerugian. Sebaliknya, pembeli yang tidak memahami nilai pasar kendaraan bisa saja membeli mobil dengan harga jauh di atas nilai sebenarnya, yang tentu merugikan dari sisi finansial. Dikutip dari marketresearch.com (Ken Research, 2024), kurangnya transparansi harga menjadi hambatan besar dalam pertumbuhan pasar mobil bekas secara digital di Arab Saudi.

**Problem Statement**

Penjual dan pembeli mobil bekas yang tidak memiliki pemahaman cukup mengenai harga pasar mobil, akan kesulitan menentukan nilai jual atau beli yang tepat. Hal ini menyebabkan potensi kerugian finansial atau tidak terjadinya transaksi karena adanya ketidakpastian dalam menentukan harga.

**Goals**

Berdasarkan permasalahan tersebut, dibutuhkan pengembangan sebuah model prediktif berbasis data untuk memperkirakan harga mobil bekas berdasarkan spesifikasi kendaraan. Model ini bertujuan memberikan estimasi harga yang akurat sehingga dapat dijadikan acuan oleh penjual dan pembeli saat bertransaksi. Dengan adanya estimasi harga yang objektif, proses pengambilan keputusan menjadi lebih cepat dan efisien, serta meningkatkan kepercayaan terhadap platform digital jual beli mobil bekas seperti Syarah.com. Pada akhirnya, hal ini diharapkan dapat mendorong peningkatan volume transaksi dan mendukung pertumbuhan industri mobil bekas secara keseluruhan.

**Analytic Approach**

Pendekatan yang digunakan adalah analisis data spesifikasi kendaraan yang mempengaruhi harga mobil bekas, seperti merek, tipe, tahun produksi, kapasitas mesin, dan jumlah kilometer yang telah ditempuh. Setelah variabel-variabel tersebut dianalisis, dilakukan pemodelan regresi menggunakan pendekatan machine learning. Beberapa algoritma seperti regresi linier, decision tree, dan random forest akan dibandingkan untuk menentukan model dengan performa terbaik berdasarkan metrik evaluasi seperti RMSE dan MAE. Model dengan performa terbaik akan dijadikan model akhir yang dapat diintegrasikan ke dalam platform jual beli mobil online untuk membantu pengguna menentukan harga jual atau beli yang optimal.

**Matrix Evalution**

Matrix yang akan digunakan untuk model ini adalah:

- Root Mean Squared Error (rmse), atau
- Mean Absolute Error (MAE)

Matrix yang diperlukan akan ditentukan setelah kita melihat kareakteristik data yang akan digunakan.

**source:**

- Ahmed Al-Mutairi. 2024. Used Car Market in Saudi Arabia Expected to Grow as Buyers Go Digital. Riyadh: Zawya.com.

- Ken Research. 2024. KSA Used Cars Market Outlook to 2028. Riyadh: Ken Research.

- Mordor Intelligence. 2024. Used Car Market in Saudi Arabia - Size, Share & Forecast. Riyadh: Mordor Intelligence.

- Verified Market Research. 2025. Saudi Arabia Used Car Market Size By Vehicle Type, Sales Channel, And Region For 2025–2032. Riyadh: Verified Market Research.

- Faisal Al-Qasabi. 2024. Used Car Market Challenges and Digital Transformation in Saudi Arabia. Riyadh: Mordor Intelligence.

- Bonafide Research. 2024. Saudi Arabia Used Car Market Overview, 2029. Riyadh: Bonafide Research.

- Grand View Research. 2024. Saudi Arabia Used Car Market Size & Outlook, 2024-2030. Riyadh: Grand View Research.

# **DATA Understanding**

**Data Information**

- Data yang digunakan merupakan data spesifikasi/fitur mobil bekas beserta beserta harga jualnya yang telah disediakan oleh Purwadhika. Dilihat dari bentuk dan pola data, diasumsikan bahwa data ini diambil dari website Syarah.com yang menjual mobil bekas di Saudi Arabia.
- Data 'price'/harga untuk mobil bekas yang sifatnya 'negotiable' bernilai 0 (nol). Harga berdasarkan kesepakatan antara penjual dan pembeli.

**Feature**
| Attribute | Data Type| Description |
| --- | --- | --- |
| Type | Object | Type of used car |
| Region | Object | The region in which the used car was offered for sale |
| Make | Object | Name of the car company |
| Gear_Type | Object | Automatic / Manual |
| Origin | Object | Country of importer (Gulf / Saudi / Other) |
| Options | Object | Full Options / Semi-Full / Standard |
| Year | Int | Year of Manufacturing |
| Enginee_Size | Float | The engine size of used car |
| Mileage | Int | The distance that used car have travelled, measured in miles |
| Negotiable | Bool | If True, the price is 0. This means the price is negotiable (not set) |
| Price | int | Price of the used car (in SAR) |


