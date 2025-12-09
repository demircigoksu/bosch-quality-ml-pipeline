"""
Test verisinden temiz veri çıkarma scripti.
"""
import pandas as pd
import numpy as np
import os

# Dosya yolları
TEST_FILE = 'data/test_numeric.csv'
OUTPUT_FILE = 'data/test_numeric_clean.csv'

print("=" * 60)
print("BOSCH TEST VERİSİ TEMİZLEME SCRIPTI")
print("=" * 60)

# 1. Dosya bilgisi
file_size_mb = os.path.getsize(TEST_FILE) / (1024*1024)
print(f"\n1. Dosya boyutu: {file_size_mb:.2f} MB")

# 2. İlk chunk'ı oku ve yapıyı anla
print("\n2. Veri yapısı analizi...")
sample = pd.read_csv(TEST_FILE, nrows=5000)
print(f"   - Toplam sütun sayısı: {len(sample.columns)}")
print(f"   - İlk 5 sütun: {sample.columns[:5].tolist()}")

# 3. Missing value analizi
missing_per_col = sample.isna().mean()
print(f"   - Ortalama missing value oranı: {missing_per_col.mean()*100:.2f}%")

# 4. Tamamen dolu (eksik değersiz) sütunları bul
complete_cols = missing_per_col[missing_per_col == 0].index.tolist()
print(f"\n3. Tamamen dolu sütun sayısı: {len(complete_cols)}")

# 5. Chunk chunk okuyarak toplam satır sayısını bul
print("\n4. Toplam satır sayısı hesaplanıyor...")
total_rows = 0
for chunk in pd.read_csv(TEST_FILE, chunksize=100000, usecols=['Id']):
    total_rows += len(chunk)
print(f"   - Toplam satır: {total_rows:,}")

# 6. Temiz veri stratejisi
print("\n5. Temiz veri çıkarma stratejileri:")
print("   a) Hiç eksik değeri olmayan SATIRLARI çekme")
print("   b) Hiç eksik değeri olmayan SÜTUNLARI çekme")
print("   c) Düşük missing oranlı verileri çekme (<%50)")

# Strateji: Hiç eksik değeri olmayan satırları çek
print("\n6. Strateji A: Eksik değersiz satırları çekme...")
clean_rows = []
chunk_size = 50000
chunk_num = 0

for chunk in pd.read_csv(TEST_FILE, chunksize=chunk_size):
    chunk_num += 1
    # Eksik değeri olmayan satırları filtrele
    clean_chunk = chunk.dropna()
    if len(clean_chunk) > 0:
        clean_rows.append(clean_chunk)
    print(f"   Chunk {chunk_num}: {len(chunk)} satır, {len(clean_chunk)} temiz satır")

if clean_rows:
    clean_df = pd.concat(clean_rows, ignore_index=True)
    print(f"\n7. SONUÇ - Strateji A:")
    print(f"   - Toplam temiz satır: {len(clean_df):,}")
    print(f"   - Toplam sütun: {len(clean_df.columns)}")
    
    if len(clean_df) > 0:
        clean_df.to_csv(OUTPUT_FILE, index=False)
        print(f"   - Dosya kaydedildi: {OUTPUT_FILE}")
        print(f"   - Yeni dosya boyutu: {os.path.getsize(OUTPUT_FILE)/(1024*1024):.2f} MB")
else:
    print("\n7. UYARI: Strateji A ile temiz satır bulunamadı!")
    print("   Strateji B'ye geçiliyor: Düşük missing oranlı veriler...")

# Eğer temiz satır bulunamadıysa alternatif strateji
if not clean_rows or len(clean_df) < 100:
    print("\n8. Alternatif Strateji: Missing oranı <%50 olan sütunlar ve satırlar")
    
    # Tüm veriyi düşük missing sütunlarla oku
    low_missing_cols = missing_per_col[missing_per_col < 0.5].index.tolist()
    print(f"   - Missing oranı <%50 sütun sayısı: {len(low_missing_cols)}")
    
    # Bu sütunlarla veriyi oku
    alt_clean_rows = []
    for chunk in pd.read_csv(TEST_FILE, chunksize=chunk_size, usecols=low_missing_cols):
        # Bu sefer de missing değeri olmayan satırları al
        clean_chunk = chunk.dropna()
        if len(clean_chunk) > 0:
            alt_clean_rows.append(clean_chunk)
    
    if alt_clean_rows:
        alt_clean_df = pd.concat(alt_clean_rows, ignore_index=True)
        alt_output = 'data/test_numeric_clean_alt.csv'
        alt_clean_df.to_csv(alt_output, index=False)
        print(f"\n   SONUÇ - Alternatif Strateji:")
        print(f"   - Toplam temiz satır: {len(alt_clean_df):,}")
        print(f"   - Toplam sütun: {len(alt_clean_df.columns)}")
        print(f"   - Dosya: {alt_output}")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
