from config import HEATMAP_DIR, BARPLOT_DIR, LINEPLOT_DIR, PHASEPLOT_DIR
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from config import HEATMAP_DIR, GRAFİK_ONAY

def draw_heatmap(net_file, lane_data, method_name, metrics=("vehicle_count", "waiting_time")):
    if not GRAFİK_ONAY:
        return

    tree = ET.parse(net_file)
    root = tree.getroot()

    for metric in metrics:
        plt.figure(figsize=(12, 12))

        # --- 1. Tüm yolları altyapı olarak çiz (gri, şeffaf) ---
        for edge in root.findall("edge"):
            if "function" in edge.attrib and edge.attrib["function"] == "internal":
                continue
            for lane in edge.findall("lane"):
                shape = lane.attrib["shape"]
                coords = [tuple(map(float, p.split(','))) for p in shape.strip().split()]
                x, y = zip(*coords)
                plt.plot(x, y, color='gray', linewidth=2, alpha=0.3)  # arka plan çizim

        # --- 2. Heatmap değerlerini üstüne çiz (renkli) ---
        for edge in root.findall("edge"):
            if "function" in edge.attrib and edge.attrib["function"] == "internal":
                continue
            for lane in edge.findall("lane"):
                lane_id = lane.attrib["id"]
                shape = lane.attrib["shape"]
                coords = [tuple(map(float, p.split(','))) for p in shape.strip().split()]
                x, y = zip(*coords)

                value = lane_data.get(lane_id, {}).get(metric, 0)
                color_strength = min(value / 500, 1.0)
                color = (1, 0, 0, color_strength)  # kırmızı ton

                plt.plot(x, y, linewidth=6, color=color)

        plt.title(f"{method_name} - {metric.replace('_', ' ').title()} Haritası")
        plt.axis("equal")
        plt.grid(True)
        plt.tight_layout()

        filename = os.path.join(HEATMAP_DIR, f"{method_name}_{metric}.png")
        plt.savefig(filename)
        plt.close()




def plot_bar_avg_wait(names, avg_waits):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, avg_waits, color='skyblue')
    plt.ylabel("Ortalama Bekleme Süresi (s)")
    plt.title("Yöntemlere Göre Ortalama Bekleme Süresi")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Barların üstüne değerlerini yaz
    for bar, value in zip(bars, avg_waits):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    filename = os.path.join(BARPLOT_DIR, "average_wait_bars.png")
    plt.savefig(filename)
    plt.close()

def plot_stepwise_wait(logs, names):
    plt.figure(figsize=(12, 6))
    for log, name in zip(logs, names):
        times = [entry["Time (s)"] for entry in log]
        waits = [entry["AverageWaitTime (s)"] for entry in log]
        plt.plot(times, waits, label=name)

    plt.xlabel("Simülasyon Zamanı (s)")
    plt.ylabel("Ortalama Bekleme Süresi (s)")
    plt.title("Zamana Göre Ortalama Bekleme Süresi")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(LINEPLOT_DIR, "stepwise_wait.png")
    plt.savefig(filename)
    plt.close()


def plot_phase_change_counts(logs, names):
    plt.figure(figsize=(10, 6))
    for log, name in zip(logs, names):
        tls_phase_pairs = [(entry["TLS"], entry["Phase"]) for entry in log]
        change_count = 0
        last_tls_phase = {}
        for tls, phase in tls_phase_pairs:
            if tls not in last_tls_phase or last_tls_phase[tls] != phase:
                change_count += 1
                last_tls_phase[tls] = phase
        plt.bar(name, change_count)

    plt.title("Faz Değişim Sayısı")
    plt.ylabel("Toplam Değişim")
    plt.tight_layout()

    filename = os.path.join(PHASEPLOT_DIR, "phase_change_counts.png")
    plt.savefig(filename)
    plt.close()


def plot_bar_co2_emission(names, co2_emissions):
    import matplotlib.pyplot as plt
    import os
    from config import BARPLOT_DIR

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, [c / 1000 for c in co2_emissions], color='lightcoral')  # gram cinsinden

    plt.ylabel("Toplam CO₂ Emisyonu (g)")
    plt.title("Yöntemlere Göre Toplam CO₂ Salınımı")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Değerleri barların üstüne yaz
    for bar, co2 in zip(bars, co2_emissions):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{co2 / 1000:.1f} g",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    filename = os.path.join(BARPLOT_DIR, "co2_emission_bars.png")
    plt.savefig(filename)
    plt.close()

def plot_bar_throughput(names, throughputs):
    import matplotlib.pyplot as plt
    import os
    from config import BARPLOT_DIR

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, throughputs, color='steelblue')
    plt.ylabel("Toplam Geçiş (Araç)")
    plt.title("Yöntemlere Göre Toplam Throughput")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, val in zip(bars, throughputs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    filename = os.path.join(BARPLOT_DIR, "throughput_bars.png")
    plt.savefig(filename)
    plt.close()


def plot_bar_speed(names, speeds):
    import matplotlib.pyplot as plt
    import os
    from config import BARPLOT_DIR

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, speeds, color='mediumseagreen')
    plt.ylabel("Ortalama Hız (m/s) | (km/h)")
    plt.title("Yöntemlere Göre Ortalama Hız")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, val in zip(bars, speeds):
        kmh = val * 3.6  # m/s → km/h
        label = f"{val:.2f} m/s\n{kmh:.1f} km/h"
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    filename = os.path.join(BARPLOT_DIR, "average_speed_bars.png")
    plt.savefig(filename)
    plt.close()


def plot_bar_density(names, densities):
    import matplotlib.pyplot as plt
    import os
    from config import BARPLOT_DIR

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, densities, color='darkorange')
    plt.ylabel("Yoğunluk (araç/m)")
    plt.title("Yöntemlere Göre Ortalama Yoğunluk")
    plt.xticks(rotation=45)
    plt.tight_layout()

    for bar, val in zip(bars, densities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.4f}",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    filename = os.path.join(BARPLOT_DIR, "density_bars.png")
    plt.savefig(filename)
    plt.close()

