import math
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
from config import SIMULATION_DURATION, DATA_COLLECTION_START, NET_FILE

import math
import traci
import xml.etree.ElementTree as ET
from collections import defaultdict
from config import SIMULATION_DURATION, DATA_COLLECTION_START, NET_FILE


def run_control_loop(step_fn, CONFIG_FILE, SUMO_BINARY):
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    total_wait, log = [], []
    total_co2 = 0
    total_speed = 0
    speed_count = 0
    total_throughput = 0  # 🚨 Toplam gelen araç sayısını burada topla

    local_lane_data = defaultdict(lambda: {"vehicle_count": 0, "waiting_time": 0.0})

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() <= SIMULATION_DURATION:
        traci.simulationStep()
        t = traci.simulation.getTime()
        step_fn(t)

        total_throughput += traci.simulation.getArrivedNumber()  # ✅ her adımda geleni ekle

        if DATA_COLLECTION_START <= t <= SIMULATION_DURATION:
            vehicle_ids = traci.vehicle.getIDList()
            waits = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
            speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
            co2s = [traci.vehicle.getCO2Emission(v) for v in vehicle_ids]

            total_wait.extend(waits)
            total_co2 += sum(co2s)
            total_speed += sum(speeds)
            speed_count += len(speeds)

            avg = sum(waits) / len(waits) if waits else 0

            for tls_id in traci.trafficlight.getIDList():
                phase = traci.trafficlight.getPhase(tls_id)
                duration = max(0, traci.trafficlight.getNextSwitch(tls_id) - t)
                log.append({
                    "Time (s)": t,
                    "TLS": tls_id,
                    "Phase": phase,
                    "PhaseDuration (s)": duration,
                    "AverageWaitTime (s)": avg
                })

            for lane_id in traci.lane.getIDList():
                local_lane_data[lane_id]["vehicle_count"] += traci.lane.getLastStepVehicleNumber(lane_id)
                local_lane_data[lane_id]["waiting_time"] += traci.lane.getWaitingTime(lane_id)

    traci.close()

    avg_speed = total_speed / speed_count if speed_count else 0

    # Yol uzunluğu → yoğunluk için
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    total_road_length = sum(
        float(lane.attrib["length"])
        for edge in root.findall("edge")
        for lane in edge.findall("lane")
        if "function" not in edge.attrib or edge.attrib["function"] != "internal"
    )
    avg_density = (speed_count / SIMULATION_DURATION) / total_road_length if total_road_length else 0

    return total_wait, log, local_lane_data, total_co2, avg_speed, total_throughput, avg_density

def statik(CONFIG_FILE, SUMO_BINARY):
    return run_control_loop(lambda t: None, CONFIG_FILE, SUMO_BINARY)


def webster(CONFIG_FILE, SUMO_BINARY):
    def step(t):
        if t % 60 == 0:
            for tls_id in traci.trafficlight.getIDList():
                traci.trafficlight.setPhaseDuration(tls_id, 30)
    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def actuated(CONFIG_FILE, SUMO_BINARY):
    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            vehicle_count = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            if vehicle_count < 1:
                traci.trafficlight.setPhaseDuration(tls_id, 5)
    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def max_pressure(CONFIG_FILE, SUMO_BINARY):
    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            pc = len(logic.phases)
            best_phase, best_pressure = 0, -math.inf
            for i in range(pc):
                pressure = 0
                for j, l in enumerate(lanes):
                    if j < len(logic.phases[i].state) and logic.phases[i].state[j] == 'G':
                        incoming = traci.lane.getLastStepVehicleNumber(l)
                        outgoing = sum(traci.lane.getLastStepVehicleNumber(link[0]) for link in traci.lane.getLinks(l)) or 1
                        pressure += incoming - outgoing
                if pressure > best_pressure:
                    best_phase, best_pressure = i, pressure
            dynamic_factor = max(1, min(5, best_pressure / 3))
            duration = max(5, min(60, int(10 * dynamic_factor)))
            traci.trafficlight.setPhase(tls_id, best_phase)
            traci.trafficlight.setPhaseDuration(tls_id, duration)
    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def sotl(CONFIG_FILE, SUMO_BINARY):
    last_switch_time = {}

    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            if tls_id not in last_switch_time:
                last_switch_time[tls_id] = t

            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            pc = len(logic.phases)

            def green_count(idx):
                return sum(traci.lane.getLastStepVehicleNumber(l)
                           for i, l in enumerate(lanes)
                           if i < len(logic.phases[idx].state) and logic.phases[idx].state[i] == 'G')

            total_vehicles = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            interval = max(3, min(15, total_vehicles // 3))

            if t - last_switch_time[tls_id] >= interval:
                best_phase = max(range(pc), key=green_count)
                count = green_count(best_phase)
                duration = max(5, min(60, count * 2))
                traci.trafficlight.setPhase(tls_id, best_phase)
                traci.trafficlight.setPhaseDuration(tls_id, duration)
                last_switch_time[tls_id] = t

    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def sotl_platoon(CONFIG_FILE, SUMO_BINARY):
    last_switch_time = {}

    def is_platoon(lane_id, dynamic_threshold):
        vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
        if len(vehicles) < dynamic_threshold:
            return False
        headways = [
            traci.vehicle.getLanePosition(vehicles[i + 1]) - traci.vehicle.getLanePosition(vehicles[i])
            for i in range(len(vehicles) - 1)
        ]
        return all(h < 10 for h in headways)

    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            if tls_id not in last_switch_time:
                last_switch_time[tls_id] = t

            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            pc = len(logic.phases)

            total_vehicles = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            interval = max(3, min(15, total_vehicles // 3))

            if total_vehicles < 30:
                threshold = 2
            elif total_vehicles < 80:
                threshold = 3
            else:
                threshold = 4

            if t - last_switch_time[tls_id] >= interval:
                best_phase, max_score = 0, -float("inf")
                for i in range(pc):
                    score = 0
                    for j, l in enumerate(lanes):
                        if j < len(logic.phases[i].state) and logic.phases[i].state[j] == 'G':
                            count = traci.lane.getLastStepVehicleNumber(l)
                            if is_platoon(l, threshold):
                                count *= 1.5
                            score += count
                    if score > max_score:
                        best_phase, max_score = i, score

                duration = max(5, min(60, int(max_score * 2)))
                traci.trafficlight.setPhase(tls_id, best_phase)
                traci.trafficlight.setPhaseDuration(tls_id, duration)
                last_switch_time[tls_id] = t

    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def sotl_multiagent(CONFIG_FILE, SUMO_BINARY):
    last_switch_time = {}

    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            if tls_id not in last_switch_time:
                last_switch_time[tls_id] = t

            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            pc = len(logic.phases)

            total_incoming = sum(traci.lane.getLastStepVehicleNumber(l) for l in lanes)
            interval = max(3, min(15, total_incoming // 4))

            if total_incoming < 30:
                score_multiplier = 3
            elif total_incoming < 80:
                score_multiplier = 5
            else:
                score_multiplier = 7

            if t - last_switch_time[tls_id] >= interval:
                best_phase, best_score = 0, -float("inf")
                for i in range(pc):
                    score = 0
                    for j, l in enumerate(lanes):
                        if j < len(logic.phases[i].state) and logic.phases[i].state[j] == 'G':
                            incoming = traci.lane.getLastStepVehicleNumber(l)
                            outgoing = sum(traci.lane.getLastStepVehicleNumber(link[0]) for link in traci.lane.getLinks(l)) or 1
                            score += incoming / outgoing
                    if score > best_score:
                        best_phase = i
                        best_score = score

                duration = max(5, min(60, int(best_score * score_multiplier)))
                traci.trafficlight.setPhase(tls_id, best_phase)
                traci.trafficlight.setPhaseDuration(tls_id, duration)
                last_switch_time[tls_id] = t

    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def gap_out(CONFIG_FILE, SUMO_BINARY):
    last_switch_time = {}
    current_phase = {}

    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            if tls_id not in last_switch_time:
                last_switch_time[tls_id] = t
                current_phase[tls_id] = 0

            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            pc = len(logic.phases)

            gap_time = 5  # saniye içinde araç yoksa geçiş yapılır
            active_lanes = [l for j, l in enumerate(lanes)
                            if j < len(logic.phases[current_phase[tls_id]].state) and
                            logic.phases[current_phase[tls_id]].state[j] == 'G']

            any_vehicle = any(traci.lane.getLastStepVehicleNumber(l) > 0 for l in active_lanes)

            if not any_vehicle and (t - last_switch_time[tls_id]) >= gap_time:
                current_phase[tls_id] = (current_phase[tls_id] + 1) % pc
                traci.trafficlight.setPhase(tls_id, current_phase[tls_id])
                traci.trafficlight.setPhaseDuration(tls_id, 10)
                last_switch_time[tls_id] = t

    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)


def fuzzy_logic(CONFIG_FILE, SUMO_BINARY):
    last_switch_time = {}

    def step(t):
        for tls_id in traci.trafficlight.getIDList():
            if tls_id not in last_switch_time:
                last_switch_time[tls_id] = t

            lanes = traci.trafficlight.getControlledLanes(tls_id)
            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            pc = len(logic.phases)

            if t - last_switch_time[tls_id] >= 5:
                best_phase, best_score = 0, -float("inf")
                for i in range(pc):
                    score = 0
                    for j, l in enumerate(lanes):
                        if j < len(logic.phases[i].state) and logic.phases[i].state[j] == 'G':
                            q = traci.lane.getLastStepHaltingNumber(l)  # Kuyruk uzunluğu
                            v = traci.lane.getLastStepMeanSpeed(l)      # Ortalama hız
                            # Fuzzy benzeri skor: düşük hız ve uzun kuyruk → yüksek öncelik
                            score += (q * (1.0 - v / 15.0))  # 15 m/s = referans hız
                    if score > best_score:
                        best_score = score
                        best_phase = i

                green_time = max(5, min(60, int(best_score * 2)))
                traci.trafficlight.setPhase(tls_id, best_phase)
                traci.trafficlight.setPhaseDuration(tls_id, green_time)
                last_switch_time[tls_id] = t

    return run_control_loop(step, CONFIG_FILE, SUMO_BINARY)

