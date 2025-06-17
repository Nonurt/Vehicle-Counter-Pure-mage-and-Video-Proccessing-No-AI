import csv
import os
from config import REPORT_CSV_PATH


def write_metrics_to_csv(method_names, waits, co2s, speeds, throughs, densities):
    """
    Her yöntemin metriklerini CSV dosyasına yazar.
    """

    os.makedirs(os.path.dirname(REPORT_CSV_PATH), exist_ok=True)

    with open(REPORT_CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Yöntem",
            "Ortalama Bekleme Süresi (s)",
            "Toplam CO₂ (mg)",
            "Ortalama Hız (m/s)",
            "Toplam Throughput (araç)",
            "Ortalama Yoğunluk (araç/m)"
        ])

        for name, wait, co2, speed, through, density in zip(method_names, waits, co2s, speeds, throughs, densities):
            writer.writerow([
                name,
                round(wait, 2),
                int(co2),
                round(speed, 2),
                int(through),
                round(density, 6)
            ])

    print(f"📄 Sonuçlar başarıyla kaydedildi: {REPORT_CSV_PATH}")
