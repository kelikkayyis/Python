# mengatur lokasi folder
import os
from google.colab import drive
os.chdir('/content/drive/MyDrive/KSO SCISI/Absensi 2021/Absensi 07. Juli 2021')
os.getcwd()

# melihat file yang tersimpan di folder
files = os.listdir()
print(files)

# menggabungkan file excel menggunakan pandas dan looping
import pandas as pd
df = pd.DataFrame()
for file in files:
     if file.endswith('.xls'):
         df = df.append(pd.read_excel(file), ignore_index=True)
         
# verifikasi jumlah baris setelah digabungkan
print(df.shape)
print("="*11)
print(414*13)

# menyimpan file hasil penggabungan
df.to_excel('Juli_01to13_2021.xls')