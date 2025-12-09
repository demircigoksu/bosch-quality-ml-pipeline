"""
Train verisinden temiz veri çıkarma scripti.
"""
import pandas as pd
import os

# Dosya yolları
TRAIN_FILE = 'data/train_numeric.csv'
CLEAN_TEST_FILE = 'data/test_numeric_clean_alt.csv'
OUTPUT_TRAIN = 'data/train_numeric_clean.csv'

print("=" * 60)
print("TRAIN VERİSİ TEMİZLEME")
print("=" * 60)

# 1. Temiz test verisindeki sütunları al
clean_cols = pd.read_csv(CLEAN_TEST_FILE, nrows=0).columns.tolist()
print(f"\n1. Temiz test verisindeki sütun sayısı: {len(clean_cols)}")

# 2. Train verisinin yapısını kontrol et
train_sample = pd.read_csv(TRAIN_FILE, nrows=100)
print(f"2. Train verisindeki toplam sütun: {len(train_sample.columns)}")

# Response sütunu var mı?
has_response = 'Response' in train_sample.columns
print(f"3. Response sütunu var mı: {has_response}")

# 3. Kullanılacak sütunları belirle
# Test verisindeki sütunlar + Response
use_cols = clean_cols.copy()
if has_response and 'Response' not in use_cols:
    use_cols.append('Response')

# Train'de olmayanları çıkar
available_cols = [c for c in use_cols if c in train_sample.columns]
print(f"4. Kullanılacak sütun sayısı: {len(available_cols)}")

# 4. Train verisini chunk chunk oku ve temiz satırları çıkar
print("\n5. Train verisi okunuyor ve filtreleniyor...")
clean_rows = []
chunk_size = 50000
chunk_num = 0
total_processed = 0

for chunk in pd.read_csv(TRAIN_FILE, chunksize=chunk_size, usecols=available_cols):
    chunk_num += 1
    total_processed += len(chunk)
    
    # Eksik değeri olmayan satırları filtrele
    clean_chunk = chunk.dropna()
    if len(clean_chunk) > 0:
        clean_rows.append(clean_chunk)
    
    print(f"   Chunk {chunk_num}: {len(chunk)} satır okundu, {len(clean_chunk)} temiz satır")

# 5. Sonuçları birleştir ve kaydet
if clean_rows:
    clean_df = pd.concat(clean_rows, ignore_index=True)
    
    print(f"\n6. SONUÇ:")
    print(f"   - Toplam işlenen satır: {total_processed:,}")
    print(f"   - Temiz satır sayısı: {len(clean_df):,}")
    print(f"   - Sütun sayısı: {len(clean_df.columns)}")
    
    # Response dağılımı
    if 'Response' in clean_df.columns:
        print(f"\n7. Response dağılımı:")
        print(clean_df['Response'].value_counts())
        print(f"   Hatalı parça oranı: {clean_df['Response'].mean()*100:.2f}%")
    
    # Kaydet
    clean_df.to_csv(OUTPUT_TRAIN, index=False)
    print(f"\n8. Dosya kaydedildi: {OUTPUT_TRAIN}")
    print(f"   Boyut: {os.path.getsize(OUTPUT_TRAIN)/(1024*1024):.2f} MB")
else:
    print("\nUYARI: Temiz satır bulunamadı!")

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
