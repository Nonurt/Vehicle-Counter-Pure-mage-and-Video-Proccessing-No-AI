import os
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
from stable_baselines3 import PPO, DQN, A2C
from sumo_rl import SumoEnvironment
from config import NET_FILE, ROUTE_FILE, MODEL_PATHS, GRAFİK_ONAY, SIMULATION_DURATION
from visualizer import draw_heatmap

ALGORITHMS = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C
}


def test_rl_model(name, cls, path):
    if not os.path.exists(path + ".zip"):
        print(f"⏭️ {name} modeli bulunamadı, test edilmedi.")
        return None, None, None, None, None, None

    print(f"\n▶️ {name.upper()} MODELİ TESTE ALINIYOR...")

    try:
        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            use_gui=False,
            num_seconds=SIMULATION_DURATION,
            single_agent=True,
            max_depart_delay=1000,
            waiting_time_memory=1000
        )

        model = cls.load(path, env=env)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_wait = 0
        step_count = 0
        total_co2 = 0
        total_speed = 0
        speed_count = 0
        total_throughput = 0
        total_departed = 0

        rl_lane_data = defaultdict(lambda: {"vehicle_count": 0, "waiting_time": 0.0})
        done = False

        while not done:
            action, _ = model.predict(obs)
            result = env.step(action)

            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            total_wait += -reward
            step_count += 1

            vehicle_ids = traci.vehicle.getIDList()
            co2s = [traci.vehicle.getCO2Emission(v) for v in vehicle_ids]
            total_co2 += sum(co2s)

            speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
            total_speed += sum(speeds)
            speed_count += len(speeds)

            total_throughput += traci.simulation.getArrivedNumber()
            total_departed += traci.simulation.getDepartedNumber()

            lane_counts = info.get("lane_vehicles_count", {})
            lane_waits = info.get("lane_waiting_time", {})
            for lane_id in lane_counts:
                rl_lane_data[lane_id]["vehicle_count"] += lane_counts[lane_id]
            for lane_id in lane_waits:
                rl_lane_data[lane_id]["waiting_time"] += lane_waits[lane_id]

        env.close()

        avg_wait = total_wait / step_count if step_count else 0
        avg_speed = total_speed / speed_count if speed_count else 0
        throughput_ratio = (total_throughput / total_departed) * 100 if total_departed else 0

        tree = ET.parse(NET_FILE)
        root = tree.getroot()
        total_road_length = sum(
            float(lane.attrib["length"])
            for edge in root.findall("edge")
            for lane in edge.findall("lane")
            if "function" not in edge.attrib or edge.attrib["function"] != "internal"
        )
        avg_density = (speed_count / SIMULATION_DURATION) / total_road_length if total_road_length else 0

        print(f"{name:<22}: Ortalama {avg_wait:.2f} sn | CO₂: {total_co2/1000:.2f} g | "
              f"Hız: {avg_speed:.2f} m/s | Yoğunluk: {avg_density:.5f} | "
              f"Geçiş: {total_throughput}/{total_departed} ({throughput_ratio:.2f}%)")

        if GRAFİK_ONAY:
            draw_heatmap(NET_FILE, rl_lane_data, method_name=name, metrics=["vehicle_count", "waiting_time"])

        return avg_wait, rl_lane_data, total_co2, avg_speed, total_throughput, avg_density

    except Exception as e:
        print(f"🚨 Ortam başlatılamadı: {e}")
        return None, None, None, None, None, None


def run_all_rl_models():
    results = []
    for name, path in MODEL_PATHS.items():
        cls = ALGORITHMS.get(name)
        if cls:
            result = test_rl_model(name, cls, path)
            if result[0] is not None:
                results.append((name, *result))
    return results
