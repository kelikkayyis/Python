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

# merapikan format & menggabungkan dengan id pegawai

# import data id
os.chdir('/content/drive/MyDrive/KSO SCISI/Absensi 2021/')
import pandas as pd
ID = pd.read_excel('Nama & ID Absensi.xlsx', skiprows=[0])

# menggabungkan absen dengan ID
merged_df = pd.merge(df, ID, 
                     left_on = 'Nama', 
                     right_on = 'NOTE', 
                     how='left')

# membuat fitur hari dan shift
import numpy as np
merged_df['Day'] = merged_df['Date'].dt.day_name() # membuat berdarsarkan hari
merged_df['Shift'] = np.where(merged_df['Day']!= 'Sunday', 'ST01', 'NW')  # != berarti tidak sama
merged_df = merged_df.sort_values(by=['ID','Nama','Date'], 
                                  ascending=True) # mengurutkann berdasarkan nama & tanggal

# menyesuaikan dengan format simpeg
df_format = merged_df[['ID','Shift','Day','Date','Time One','Date','Time Off','Nama']]
df_format = df_format[~df_format['ID'].isna()] # membuang pegawai yang tidak memiliki ID (~ 'negasi')
df_format.head(7) # melihat hasil file sudah rapi

# menyimpan file hasil penggabungan
df_format.to_excel('Agustus_01to22_2021.xls') # ganti nama sesuai bulannya
