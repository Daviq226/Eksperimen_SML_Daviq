import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def process_data():
    # 1. Setup Path
    raw_data_path = r"C:\Users\ASUS\Downloads\Repo_Kriteria1_Daviq\insurance_raw\insurance.csv"
    output_path = 'insurance_clean.csv'

    # Cek apakah file raw ada
    if not os.path.exists(raw_data_path):
        print(f"[ERROR] File tidak ditemukan di: {raw_data_path}")
        return

    # Load Data
    df = pd.read_csv(raw_data_path)
    print(f"Data awal: {df.shape}")

    # Preprocessing (Sesuai Notebook)
    
    # Hapus Duplikat
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)

    # Encoding Variabel Kategorikal
    le = LabelEncoder()
    
    # Sex
    df['sex'] = le.fit_transform(df['sex'])
    
    # Smoker
    df['smoker'] = le.fit_transform(df['smoker'])
    
    # Region (One Hot Encoding)
    df = pd.get_dummies(df, columns=['region'], prefix='region')

    # Konversi bool ke int (penting agar bersih)
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # 4. Simpan Data Bersih
    df.to_csv(output_path, index=False)
    print("Preprocessing Selesai!")
    print(f"Data bersih disimpan sebagai: {output_path}")
    print(f"Dimensi akhir: {df.shape}")

if __name__ == "__main__":
    process_data()