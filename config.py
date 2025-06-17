import os
# ======================
# KULLANICI SEÇİMLERİ
# ======================
SEÇİM = 0  # 0: Hepsi, 1: Webster, 2: Actuated, 3: MaxPressure, vs.
GRAFİK_ONAY = 1  # 1: Açık, 0: Kapalı

# ======================
# ANA KLASÖR
# ======================
#RL_BASE = "E:/python/faltay"  # 🔁 Tüm yollar bu kök klasörden türetiliyor
RL_BASE = "E:/python/trafik4"
# ======================
# SUMO AYARLARI
# ======================
USE_GUI = 0
SUMO_BINARY = "sumo-gui" if USE_GUI else "sumo"

#CONFIG_FILE = os.path.join(RL_BASE, "faltay.sumocfg")
#NET_FILE = os.path.join(RL_BASE, "faltay.net.xml")
#ROUTE_FILE = os.path.join(RL_BASE, "faltay.rou.xml")

CONFIG_FILE = os.path.join(RL_BASE, "custom_config.sumocfg")
NET_FILE = os.path.join(RL_BASE, "mymap.net.xml")
ROUTE_FILE = os.path.join(RL_BASE, "custom_routes.rou.xml")

EXCEL_PATH = os.path.join(RL_BASE, "traffic_results.xlsx")

# ======================
# SIMÜLASYON AYARLARI
# ======================
SIMULATION_DURATION = 700
DATA_COLLECTION_START = 100

# ======================
# RL MODEL YOLLARI
# ======================
MODEL_PATHS = {
    "PPO": os.path.join(RL_BASE, "ppo", "ppo_model"),
    "DQN": os.path.join(RL_BASE, "dqn", "dqn_model"),
    "A2C": os.path.join(RL_BASE, "a2c", "a2c_model")
}

# ======================
# GÖRSEL KAYIT DİZİNLERİ
# ======================
PNG_BASE_DIR = os.path.join(RL_BASE, "png")

HEATMAP_DIR = os.path.join(PNG_BASE_DIR, "heatmaps")
BARPLOT_DIR = os.path.join(PNG_BASE_DIR, "barplots")
LINEPLOT_DIR = os.path.join(PNG_BASE_DIR, "lineplots")
PHASEPLOT_DIR = os.path.join(PNG_BASE_DIR, "phaseplots")

# Klasörleri oluştur
for path in [HEATMAP_DIR, BARPLOT_DIR, LINEPLOT_DIR, PHASEPLOT_DIR]:
    os.makedirs(path, exist_ok=True)

# ======================
# RAPOR DOSYASI
# ======================
REPORT_CSV_PATH = os.path.join(RL_BASE, "reports", "traffic_metrics.csv")
os.makedirs(os.path.dirname(REPORT_CSV_PATH), exist_ok=True)
