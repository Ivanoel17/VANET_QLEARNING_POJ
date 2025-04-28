import socket
import json
import logging
import csv
import os
import random
from datetime import datetime
from collections import defaultdict
import numpy as np

"""
train.py – **Batch‑aware Q‑Learning server with full telemetry logging**
=======================================================================
Run this script, point your MATLAB (atau klien lain) ke port 5000, dan kamu
akan mendapatkan:

• Konsol **terminal**  – baris `print()` dan semua log level INFO/DEBUG.
• File  **train.log**   – salinan persis log terminal (rotating manual).
• File  **metrics.csv** – satu baris per *veh_id* per batch berisi:
    `timestamp,veh_id,power,beacon,cbr,snr,reward`

Contoh cuplikan konsol:

    23:00:07 [INFO ] Connected: 127.0.0.1:57814
    [BATCH] Received 2 vehicles' data
    23:00:07 [DEBUG] car‑42 | pwr=15.0 dBm | bc=10 Hz | cbr=0.72 | snr=18.2 dB | rew=‑7.00
    23:00:07 [DEBUG] car‑17 | pwr=10.0 dBm | bc=5 Hz  | cbr=0.58 | snr=24.5 dB | rew=‑2.00
    [DEBUG] Sending RL response to MATLAB:
    23:00:07 [INFO ] Disconnected: 127.0.0.1:57814

Kamu bisa *live‑tail* file‑nya:

    tail -f train.log      # Unix
    Get-Content -Wait train.log  # PowerShell
    tail -f metrics.csv

"""

# ─────────────────────────────────────────────────────────────────────────────
# Config & Hyper‑parameters
# ─────────────────────────────────────────────────────────────────────────────
HOST: str = "0.0.0.0"
PORT: int = 5000

LEARNING_RATE: float = 0.1
DISCOUNT_FACTOR: float = 0.99
EPSILON: float = 0.1             # Exploration probability
CBR_TARGET: float = 0.65         # Target channel‑busy ratio

# Discretisation bins (upper bounds for np.digitize)
POWER_BINS     = [5, 15, 25, 30]       # dBm
BEACON_BINS    = [1, 5, 10, 20]        # Hz
CBR_BINS       = [0.0, 0.3, 0.6, 1.0]  # Ratio
NEIGH_BINS     = [0, 4, 8, 12, 1e9]    # Number of neighbours
SNR_BINS       = [0, 5, 10, 15, 20, 25, 1e9]  # dB

ACTIONS = [0, 1]  # 0 ≡ decrease (‑1), 1 ≡ increase (+1)

# Accept alternative field names from client JSON
FIELD_ALIASES = {
    "power": ["power", "transmissionPower", "current_power"],
    "beacon": ["beacon", "beaconRate", "current_beacon"],
    "cbr": ["cbr", "CBR", "channelBusyRatio"],
    "neighbors": ["neighbors", "neigh", "n_neighbors"],
    "snr": ["snr", "SNR", "signalToNoise"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def discretize(value: float, bins: list[int]) -> int:
    """Return the index (0‑based) of *value* in *bins* for Q‑table lookup."""
    idx = np.digitize([value], bins, right=True)[0]
    return int(max(0, min(idx, len(bins) - 1)))


def adjust_mcs_based_on_snr(snr: float) -> int:
    """Very coarse mapping from SNR (dB) → LTE‑V2X MCS index (0–7)."""
    if snr > 25:
        return 7
    if snr > 20:
        return 6
    if snr > 15:
        return 5
    if snr > 10:
        return 4
    if snr > 5:
        return 3
    if snr > 0:
        return 2
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Q‑learning storage (sparse dictionary instead of huge ndarray)
# ─────────────────────────────────────────────────────────────────────────────
q_table: defaultdict[tuple[int, ...], float] = defaultdict(float)


# ─────────────────────────────────────────────────────────────────────────────
# RL server implementation
# ─────────────────────────────────────────────────────────────────────────────
class QLearningServerBatch:
    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self._setup_logging()
        self._ensure_csv_header()

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        logging.info("Server listening on %s:%s", host, port)

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------
    def _setup_logging(self):
        """Configure root logger to write both console and file."""
        fmt = "%(asctime)s [%(levelname)-5s] %(message)s"
        datefmt = "%H:%M:%S"

        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        # File handler (append)
        file = logging.FileHandler("train.log", mode="a", encoding="utf-8")
        file.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        logging.basicConfig(level=logging.DEBUG, handlers=[console, file])

    def _ensure_csv_header(self):
        """Create metrics.csv with header if it doesn't exist."""
        if not os.path.exists("metrics.csv"):
            with open("metrics.csv", "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "veh_id", "power", "beacon", "cbr", "snr", "reward"
                ])

    def _append_metrics(self, veh_id: str, power: float, beacon: float, cbr: float, snr: float, reward: float):
        with open("metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.utcnow().isoformat(timespec="seconds"),
                veh_id, power, beacon, cbr, snr, reward
            ])

    # ---------------------------------------------------------------------
    # Q‑learning core
    # ---------------------------------------------------------------------
    def _state_indices(self, state: list[float]) -> tuple[int, ...]:
        p, b, c, n, s = state
        return (
            discretize(p, POWER_BINS),
            discretize(b, BEACON_BINS),
            discretize(c, CBR_BINS),
            discretize(n, NEIGH_BINS),
            discretize(s, SNR_BINS),
        )

    @staticmethod
    def _reward(cbr: float) -> float:  # simple absolute error reward
        return -abs(cbr - CBR_TARGET) * 100

    def _select_action(self, idx: tuple[int, ...]) -> int:
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        q_dec = q_table[idx + (0,)]
        q_inc = q_table[idx + (1,)]
        return int(q_inc > q_dec)

    def _update_q(self, old_idx: tuple[int, ...], action: int, reward: float, new_idx: tuple[int, ...]):
        old_q = q_table[old_idx + (action,)]
        max_next = max(q_table[new_idx + (0,)], q_table[new_idx + (1,)])
        q_table[old_idx + (action,)] = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next - old_q)

    # ---------------------------------------------------------------------
    # Networking helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _extract_field(d: dict, canonical: str, default=0):
        for alias in FIELD_ALIASES[canonical]:
            if alias in d:
                return d[alias]
        raise KeyError(canonical)

    # ---------------------------------------------------------------------
    # Client handler
    # ---------------------------------------------------------------------
    def _handle(self, conn: socket.socket, address: tuple[str, int]):
        logging.info("Connected: %s:%s", *address)
        try:
            while True:
                raw = conn.recv(65536)
                if not raw:
                    break

                # 1. Batch decode ------------------------------------------------
                try:
                    batch = json.loads(raw.decode())
                except json.JSONDecodeError as err:
                    logging.error("Bad JSON from %s: %s", address, err)
                    break

                print(f"[BATCH] Received {len(batch)} vehicles' data")
                logging.info("[BATCH] Received %d vehicles' data", len(batch))

                responses: dict[str, dict] = {}

                # 2. Iterate vehicles ------------------------------------------
                for vid, d in batch.items():
                    try:
                        state = [
                            float(self._extract_field(d, "power")),
                            float(self._extract_field(d, "beacon")),
                            float(self._extract_field(d, "cbr")),
                            float(d.get("neighbors", d.get("neigh", 0))),
                            float(self._extract_field(d, "snr")),
                        ]
                    except KeyError as k:
                        logging.warning("Missing field %s for %s -> skip", k, vid)
                        continue

                    idx = self._state_indices(state)
                    action = self._select_action(idx)
                    delta = -1 if action == 0 else 1

                    new_power = float(max(POWER_BINS[0], min(POWER_BINS[-1], state[0] + delta)))
                    new_beacon = float(max(BEACON_BINS[0], min(BEACON_BINS[-1], state[1] + delta)))

                    rew = self._reward(state[2])
                    new_state = [new_power, new_beacon, *state[2:]]
                    new_idx = self._state_indices(new_state)
                    self._update_q(idx, action, rew, new_idx)

                    # DEBUG per‑vehicle -------------------------------------
                    logging.debug(
                        "%s | pwr=%.1f dBm | bc=%.0f Hz | cbr=%.2f | snr=%.1f dB | reward=%.2f",
                        vid, state[0], state[1], state[2], state[4], rew
                    )

                    self._append_metrics(vid, state[0], state[1], state[2], state[4], rew)

                    responses[vid] = {
                        "transmissionPower": new_power,
                        "beaconRate": new_beacon,
                        "MCS": adjust_mcs_based_on_snr(state[4]),
                    }

                # 3. Send back --------------------------------------------------
                print("[DEBUG] Sending RL response to MATLAB:")
                msg = (json.dumps(responses) + "\n").encode()
                conn.sendall(msg)
                logging.info("[DEBUG] Sent RL response (%d bytes)", len(msg))

        except Exception as exc:
            logging.exception("Handler crashed: %s", exc)
        finally:
            conn.close()
            logging.info("Disconnected: %s:%s", *address)

    # ---------------------------------------------------------------------
    # Start/serve
    # ---------------------------------------------------------------------
    def serve_forever(self):
        while True:
            conn, addr = self.server.accept()
            self._handle(conn, addr)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    QLearningServerBatch().serve_forever()
