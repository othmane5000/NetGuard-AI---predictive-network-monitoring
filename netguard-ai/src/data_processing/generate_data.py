"""
NETGUARD AI - Synthetic Network Telemetry Generator
=====================================================

WHY SYNTHETIC DATA?
--------------------
Public datasets that combine (a) continuous device telemetry (CPU, memory,
temperature, packet loss, latency, interface errors...) WITH (b) labeled
failure/degradation events are extremely rare in the open domain. Real
network vendors treat this data as proprietary. Datasets that do exist
(e.g. Kaggle "Network anomaly" sets, NSL-KDD, CICIDS) are focused on
*security intrusion detection* (attack traffic classification), not on
*device health / hardware-performance degradation prediction*, so their
feature schema and label semantics do not match this project's goal.

Decision (explicitly stated assumption):
We therefore build a realistic SYNTHETIC dataset whose feature-to-failure
relationships are grounded in real networking/hardware knowledge (see the
causal rules below), instead of using an ill-fitting public dataset just
for the sake of using "real" data. This is a common and defensible
approach in applied ML when no matching public dataset exists, AS LONG AS
the causal structure is realistic and explicitly documented - which is
what this script does.

DESIGN PRINCIPLES
------------------
1. Data is generated as a TIME SERIES per device (not i.i.d. random rows),
   so temporal features (moving averages, trends) are meaningful.
2. Each device has a baseline "health regime" that can slowly drift into
   a "degrading" regime before a failure event - this mimics real
   hardware/network degradation which is rarely instantaneous.
3. Failure probability at each time step is generated from a DETERMINISTIC
   LOGISTIC FUNCTION of the underlying metrics plus noise - this is
   intentional: it means a logistic regression model has a real, learnable
   signal to recover (this dataset does not pretend logistic regression
   is the "true" generative process for real hardware - it simply gives
   us a controlled, explainable ground truth we can validate against).
4. No feature is generated FROM the failure label (no leakage at
   generation time). The label is generated as a downstream consequence
   of the metrics, as it would be in reality.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
DEVICE_TYPES = ["router", "switch", "firewall", "access_point", "server"]

# Per-device-type baseline ranges (mean, std) reflecting realistic operating
# envelopes. E.g. firewalls tend to run hotter/higher CPU due to deep packet
# inspection; access points have more connection churn; servers have higher
# steady memory usage.
BASELINES = {
    "router":       dict(cpu=35, mem=45, temp=42, conn=120, bw=40),
    "switch":       dict(cpu=25, mem=35, temp=38, conn=300, bw=55),
    "firewall":     dict(cpu=55, mem=60, temp=50, conn=500, bw=65),
    "access_point": dict(cpu=30, mem=40, temp=45, conn=60,  bw=35),
    "server":       dict(cpu=50, mem=70, temp=48, conn=200, bw=50),
}


def _clip(x, lo, hi):
    return np.clip(x, lo, hi)


def generate_device_timeseries(device_id, device_type, n_steps, rng):
    """
    Generate one device's telemetry history as a random-walk process with
    an optional 'degradation episode' injected partway through.

    Returns a DataFrame with n_steps rows (one per hour) for this device.
    """
    base = BASELINES[device_type]

    # --- Baseline signals (mean-reverting random walk = realistic "noise
    # around a operating point" rather than pure white noise) ---
    cpu = np.full(n_steps, base["cpu"], dtype=float)
    mem = np.full(n_steps, base["mem"], dtype=float)
    temp = np.full(n_steps, base["temp"], dtype=float)
    bw = np.full(n_steps, base["bw"], dtype=float)
    conn = np.full(n_steps, base["conn"], dtype=float)

    # Independent slow random walks (mean reverting) for realism
    for arr, vol, lo, hi in [
        (cpu, 1.2, 5, 100), (mem, 0.8, 5, 100),
        (temp, 0.5, 15, 95), (bw, 1.5, 1, 100), (conn, 4.0, 1, 2000),
    ]:
        for t in range(1, n_steps):
            reversion = 0.03 * (arr[0] - arr[t - 1])  # pull back to baseline
            arr[t] = arr[t - 1] + reversion + rng.normal(0, vol)
        arr[:] = _clip(arr, lo, hi)

    # --- Decide if/when this device enters a degradation episode ---
    # ~30% of devices experience at least one meaningful degradation episode
    # in the observation window, which then may or may not culminate in a
    # recorded failure. This mimics real fleets where most devices are
    # healthy most of the time.
    has_episode = rng.random() < 0.30
    episode_start = None
    episode_len = 0
    if has_episode:
        episode_start = rng.integers(int(n_steps * 0.2), int(n_steps * 0.85))
        episode_len = rng.integers(12, 72)  # 12h to 3 days of ramp-up

    # Traffic in/out derived from bandwidth utilization (Mbps), with
    # asymmetry typical of client-server workloads
    traffic_in = bw * rng.uniform(0.8, 1.3, n_steps) * 10
    traffic_out = bw * rng.uniform(0.5, 1.0, n_steps) * 10

    packet_loss = _clip(rng.exponential(0.3, n_steps), 0, 15)  # % baseline low
    latency = _clip(rng.normal(15, 4, n_steps), 1, 300)         # ms baseline
    iface_errors = rng.poisson(0.5, n_steps).astype(float)      # errors/hour baseline

    # --- Inject the degradation ramp (this is the CAUSAL core of the
    # dataset: escalating stress on one or more subsystems) ---
    degrading_mask = np.zeros(n_steps, dtype=bool)
    if has_episode:
        end = min(n_steps, episode_start + episode_len)
        ramp_len = end - episode_start
        ramp = np.linspace(0, 1, ramp_len) ** 1.5  # accelerating stress curve
        degrading_mask[episode_start:end] = True

        # Choose a random "failure mode" that determines which subsystems
        # degrade together (realistic co-movement rather than everything
        # spiking independently):
        mode = rng.choice(
            ["thermal", "overload", "network_congestion", "hardware_wear"]
        )
        if mode == "thermal":
            temp[episode_start:end] += ramp * rng.uniform(20, 35)
            cpu[episode_start:end] += ramp * rng.uniform(10, 25)  # throttling feedback
            iface_errors[episode_start:end] += ramp * rng.uniform(3, 10)
        elif mode == "overload":
            cpu[episode_start:end] += ramp * rng.uniform(30, 45)
            mem[episode_start:end] += ramp * rng.uniform(20, 35)
            latency[episode_start:end] += ramp * rng.uniform(40, 120)
            packet_loss[episode_start:end] += ramp * rng.uniform(2, 8)
        elif mode == "network_congestion":
            bw[episode_start:end] = _clip(bw[episode_start:end] + ramp * rng.uniform(25, 40), 0, 100)
            latency[episode_start:end] += ramp * rng.uniform(60, 180)
            packet_loss[episode_start:end] += ramp * rng.uniform(3, 12)
            traffic_in[episode_start:end] *= (1 + ramp)
            traffic_out[episode_start:end] *= (1 + ramp)
        else:  # hardware_wear
            iface_errors[episode_start:end] += ramp * rng.uniform(5, 20)
            packet_loss[episode_start:end] += ramp * rng.uniform(2, 6)
            temp[episode_start:end] += ramp * rng.uniform(5, 15)

        cpu[:] = _clip(cpu, 0, 100)
        mem[:] = _clip(mem, 0, 100)
        temp[:] = _clip(temp, 10, 105)
        bw[:] = _clip(bw, 0, 100)
        packet_loss[:] = _clip(packet_loss, 0, 40)
        latency[:] = _clip(latency, 1, 500)
        iface_errors[:] = _clip(iface_errors, 0, 100)

    uptime = np.arange(n_steps, dtype=float) + rng.uniform(0, 500)  # hours since boot
    # Long uptime slightly raises baseline risk (maintenance debt), a small,
    # realistic effect - NOT dominant.
    maintenance_days = _clip(
        rng.normal(45, 20, n_steps) - 0.02 * np.arange(n_steps), 0, 400
    )

    df = pd.DataFrame({
        "device_id": device_id,
        "device_type": device_type,
        "step": np.arange(n_steps),
        "cpu_usage": cpu,
        "memory_usage": mem,
        "temperature": temp,
        "bandwidth_usage": bw,
        "packet_loss": packet_loss,
        "latency": latency,
        "interface_errors": iface_errors,
        "active_connections": conn,
        "uptime_hours": uptime,
        "traffic_in": traffic_in,
        "traffic_out": traffic_out,
        "maintenance_days": maintenance_days,
        "_degrading": degrading_mask,
    })
    return df


def assign_failure_labels(df, rng):
    """
    Generate the FAILURE label as a logistic function of the current
    metrics (z-scored), plus noise. This is the ground-truth generative
    process; it is intentionally logistic so that logistic regression has
    real, recoverable signal - while still requiring the model to actually
    learn the right coefficients rather than just detecting the
    `_degrading` flag directly (which is dropped before training).

    IMPORTANT (leakage note): failure at time t is a function of metrics
    at time t (current stress), NOT of future metrics. Historical/lag
    features built later use only data up to t, so no leakage is
    introduced downstream either.
    """
    z = lambda s: (s - s.mean()) / (s.std() + 1e-6)

    logit = (
        -6.5
        + 2.6 * z(df["packet_loss"])
        + 2.2 * z(df["interface_errors"])
        + 1.8 * z(df["temperature"])
        + 1.5 * z(df["cpu_usage"])
        + 1.1 * z(df["latency"])
        + 0.6 * z(df["memory_usage"])
        + 0.3 * z(df["bandwidth_usage"])
        + 0.15 * z(df["maintenance_days"])
        + 0.05 * z(df["uptime_hours"])
        + rng.normal(0, 0.6, len(df))
    )
    prob = 1 / (1 + np.exp(-logit))
    failure = (rng.random(len(df)) < prob).astype(int)

    # Failure type is only assigned when failure == 1, chosen based on the
    # dominant stress factor at that timestep (interpretable, not random).
    factors = df[["packet_loss", "interface_errors", "temperature", "cpu_usage", "latency"]]
    zfactors = factors.apply(z)
    dominant = zfactors.idxmax(axis=1)
    type_map = {
        "packet_loss": "connectivity_failure",
        "interface_errors": "hardware_failure",
        "temperature": "hardware_failure",
        "cpu_usage": "performance_degradation",
        "latency": "performance_degradation",
    }
    failure_type = np.where(failure == 1, dominant.map(type_map), "none")

    df = df.copy()
    df["failure_probability_true"] = prob  # kept ONLY for validation/debug, not a feature
    df["failure"] = failure
    df["failure_type"] = failure_type
    return df


def generate_previous_failures(df):
    """Cumulative count of PAST failures for this device, shifted so that
    the current row never counts itself (no leakage)."""
    df = df.sort_values(["device_id", "step"]).copy()
    df["previous_failures"] = (
        df.groupby("device_id")["failure"].cumsum().shift(1).fillna(0)
    )
    # first row per device has no history
    first_idx = df.groupby("device_id").head(1).index
    df.loc[first_idx, "previous_failures"] = 0
    return df


def generate_dataset(n_devices=40, n_steps=24 * 30, start="2025-06-01", seed=RNG_SEED):
    """
    n_devices: number of distinct devices to simulate
    n_steps: number of hourly readings per device (default = 30 days)
    """
    rng = np.random.default_rng(seed)
    all_devices = []
    type_prefix = {"router": "RTR", "switch": "SW", "firewall": "FW",
                    "access_point": "AP", "server": "SRV"}

    for i in range(n_devices):
        dtype = rng.choice(DEVICE_TYPES)
        device_id = f"{type_prefix[dtype]}-{i:03d}"
        ts = generate_device_timeseries(device_id, dtype, n_steps, rng)
        ts = assign_failure_labels(ts, rng)
        all_devices.append(ts)

    df = pd.concat(all_devices, ignore_index=True)
    df = generate_previous_failures(df)

    timestamps = pd.date_range(start=start, periods=n_steps, freq="h")
    df["timestamp"] = df["step"].map(lambda s: timestamps[s])

    cols = [
        "timestamp", "device_id", "device_type", "cpu_usage", "memory_usage",
        "temperature", "bandwidth_usage", "packet_loss", "latency",
        "interface_errors", "active_connections", "uptime_hours",
        "traffic_in", "traffic_out", "previous_failures", "maintenance_days",
        "failure", "failure_type",
    ]
    df = df[cols].sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    for c in ["cpu_usage", "memory_usage", "temperature", "bandwidth_usage",
              "packet_loss", "latency", "interface_errors", "active_connections",
              "uptime_hours", "traffic_in", "traffic_out", "maintenance_days"]:
        df[c] = df[c].round(2)

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(n_devices=40, n_steps=24 * 30)
    out_path = out_dir / "network_telemetry.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df):,} rows for {df['device_id'].nunique()} devices")
    print(f"Failure rate: {df['failure'].mean():.2%}")
    print(df["failure_type"].value_counts())
    print(f"Saved to: {out_path}")
