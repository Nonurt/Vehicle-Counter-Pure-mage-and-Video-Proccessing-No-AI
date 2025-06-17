from config import (
    SEÇİM, GRAFİK_ONAY,
    CONFIG_FILE, SUMO_BINARY, NET_FILE
)
from methods import (
    statik, webster, actuated, max_pressure,
    sotl, sotl_platoon, sotl_multiagent,
    fuzzy_logic, gap_out
)

from rl_runner import run_all_rl_models
from visualizer import (
    draw_heatmap,
    plot_bar_avg_wait,
    plot_bar_co2_emission,
    plot_bar_speed,
    plot_bar_throughput,
    plot_bar_density,
    plot_stepwise_wait,
    plot_phase_change_counts
)
from report_writer import write_metrics_to_csv

# ======================
# YÖNTEMLER & İSİMLERİ
# ======================

methods = [
    statik,
    webster,
    actuated,
    max_pressure,
    sotl,
    sotl_platoon,
    sotl_multiagent,
    fuzzy_logic,
    gap_out
]

names = [
    "Static",
    "Webster",
    "Actuated",
    "Max-Pressure",
    "SOTL-Improved",
    "SOTL-Platoon",
    "SOTL-MultiAgent",
    "Fuzzy-Logic",
    "Gap-Out"
]
# ======================
# ANA PROGRAM
# ======================
if __name__ == "__main__":
    print("============================")
    print("📊 BEKLEME, CO₂, HIZ, YOĞUNLUK, GEÇİŞ")
    print("============================")

    total_waits, logs, lane_maps = [], [], []
    co2_emissions, avg_speeds, throughputs, densities = [], [], [], []
    rl_names, rl_waits, rl_co2s, rl_speeds, rl_throughs, rl_densities = [], [], [], [], [], []

    if SEÇİM == 0:
        # Tüm klasik yöntemleri sırayla çalıştır
        for method, name in zip(methods, names):
            print(f"\n▶️ {name.upper()} BAŞLADI")
            w, l, ld, co2, speed, throughput, density = method(CONFIG_FILE, SUMO_BINARY)
            total_waits.append(w)
            logs.append(l)
            lane_maps.append(ld if GRAFİK_ONAY else None)
            co2_emissions.append(co2)
            avg_speeds.append(speed)
            throughputs.append(throughput)
            densities.append(density)

            if w:
                print(f"{name:<22}: Ortalama {sum(w)/len(w):.2f} sn | CO₂: {co2/1000:.2f} g | Hız: {speed:.2f} m/s | Yoğunluk: {density:.5f} | Geçiş: {throughput}")

        # RL modelleri test et
        rl_results = run_all_rl_models()
        for name, avg_wait, _, co2, speed, throughput, density in rl_results:
            print(f"{name:<22}: Ortalama {avg_wait:.2f} sn | CO₂: {co2/1000:.2f} g | Hız: {speed:.2f} m/s | Yoğunluk: {density:.5f} | Geçiş: {throughput}")
            rl_names.append(name)
            rl_waits.append(avg_wait)
            rl_co2s.append(co2)
            rl_speeds.append(speed)
            rl_throughs.append(throughput)
            rl_densities.append(density)

        # Grafikler
        if GRAFİK_ONAY:
            print("\n📈 GRAFİKLER OLUŞTURULUYOR...")
            all_names = names + rl_names
            all_waits = [sum(w)/len(w) for w in total_waits if w] + rl_waits
            all_co2s = co2_emissions + rl_co2s
            all_speeds = avg_speeds + rl_speeds
            all_throughs = throughputs + rl_throughs
            all_densities = densities + rl_densities

            plot_bar_avg_wait(all_names, all_waits)
            plot_bar_co2_emission(all_names, all_co2s)
            plot_bar_speed(all_names, all_speeds)
            plot_bar_throughput(all_names, all_throughs)
            plot_bar_density(all_names, all_densities)

            plot_stepwise_wait(logs, names)
            plot_phase_change_counts(logs, names)

            for name, lane_map in zip(names, lane_maps):
                if lane_map is not None:
                    draw_heatmap(NET_FILE, lane_map, method_name=name, metrics=["vehicle_count", "waiting_time"])

        # CSV raporu oluştur
        write_metrics_to_csv(
            names + rl_names,
            [sum(w)/len(w) for w in total_waits if w] + rl_waits,
            co2_emissions + rl_co2s,
            avg_speeds + rl_speeds,
            throughputs + rl_throughs,
            densities + rl_densities
        )

    elif SEÇİM == 99:
        # Sadece RL modelleri çalıştır
        rl_results = run_all_rl_models()
        for name, avg_wait, _, co2, speed, throughput, density in rl_results:
            print(f"{name:<22}: Ortalama {avg_wait:.2f} sn | CO₂: {co2/1000:.2f} g | Hız: {speed:.2f} m/s | Yoğunluk: {density:.5f} | Geçiş: {throughput}")
            rl_names.append(name)
            rl_waits.append(avg_wait)
            rl_co2s.append(co2)
            rl_speeds.append(speed)
            rl_throughs.append(throughput)
            rl_densities.append(density)

        if GRAFİK_ONAY:
            plot_bar_avg_wait(rl_names, rl_waits)
            plot_bar_co2_emission(rl_names, rl_co2s)
            plot_bar_speed(rl_names, rl_speeds)
            plot_bar_throughput(rl_names, rl_throughs)
            plot_bar_density(rl_names, rl_densities)

        write_metrics_to_csv(
            rl_names,
            rl_waits,
            rl_co2s,
            rl_speeds,
            rl_throughs,
            rl_densities
        )

    else:
        # Tek bir klasik yöntemi çalıştır
        index = SEÇİM - 1
        if 0 <= index < len(methods):
            name = names[index]
            print(f"\n▶️ {name.upper()} BAŞLADI")
            w, l, ld, co2, speed, throughput, density = methods[index](CONFIG_FILE, SUMO_BINARY)
            if w:
                avg_wait = sum(w)/len(w)
                print(f"{name:<22}: Ortalama {avg_wait:.2f} sn | CO₂: {co2/1000:.2f} g | Hız: {speed:.2f} m/s | Yoğunluk: {density:.5f} | Geçiş: {throughput}")
                if GRAFİK_ONAY:
                    plot_bar_avg_wait([name], [avg_wait])
                    plot_bar_co2_emission([name], [co2])
                    plot_bar_speed([name], [speed])
                    plot_bar_throughput([name], [throughput])
                    plot_bar_density([name], [density])
                    plot_stepwise_wait([l], [name])
                    plot_phase_change_counts([l], [name])
                    draw_heatmap(NET_FILE, ld, method_name=name, metrics=["vehicle_count", "waiting_time"])

                write_metrics_to_csv(
                    [name],
                    [avg_wait],
                    [co2],
                    [speed],
                    [throughput],
                    [density]
                )
