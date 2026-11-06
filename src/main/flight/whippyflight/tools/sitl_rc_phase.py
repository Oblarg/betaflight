#!/usr/bin/env python3
"""
SITL flick at every RC-packet phase relative to the PID loop.

Boots the SITL binary sitl_test.py uses, with the same blob ($WFG_PATH,
default test_gains.wfg) and plant ($PLANT_JSON), hovers, and flicks pitch
and roll to 0.5 rot/s (180 deg/s) over 30 ms with the first RC frame landing on each
loop tick in turn.  Prints rise, overshoot, saturation, and the feedforward
target's peak and first frames per phase: the setpoint estimator's output
should not depend on where the RX frames fall.

    python3 sitl_rc_phase.py
"""

import os
import signal
import time

import sitl_test as T
from sim_plant import Plant

PEAK, RISE_S, HOLD_S, SETTLE_S = 1750, 0.030, 0.5, 0.5
BOOT_ATTEMPTS = 5


def boot():
    """Start SITL until the lockstep link answers (it rarely stalls on boot)."""
    for attempt in range(BOOT_ATTEMPTS):
        log = os.path.join(T.OBJ_DIR, f"sitl_rc_phase{attempt}.log")
        proc = T.start_sitl(log)
        msp = T.wait_tcp(5761, 30)
        msp.settimeout(0.1)
        sim = T.Sim(plant=Plant(T.COEFFS))
        sim.start()
        time.sleep(2.0)
        if sim.replies:
            return proc, msp, sim
        print(f"boot {attempt}: no replies (stalled), log {log}; rebooting", flush=True)
        sim.stop()
        msp.close()
        proc.kill()
        proc.wait()
        time.sleep(1)
    raise RuntimeError(f"SITL stalled on {BOOT_ATTEMPTS} boots")


def main():
    os.chdir(T.OBJ_DIR)
    os.system("pkill -9 -f betaflight_2026 2>/dev/null")
    with open(T.WFG, "rb") as f:
        T.install_gains(None, {T.GAIN_FILENAME: f.read()})
    proc, msp, sim = boot()
    try:
        assert T.wait_flags_and_arm(sim, msp), "did not arm"
        centered = [1500, 1500, T.HOVER, 1500, T.ARMED_AUX]
        sim.set_channels(centered)
        sim.wait(2.0)
        print("axis  phase rise90 overshoot sat   target_peak(deg/s) target_first_frames")
        for name, debug_base in (("Pitch", 3), ("Roll", 0)):
            channel, axis, sign = T.STICK_AXES[name]
            setpoint = sign * (PEAK - 1500) / 500.0
            for phase in range(T.RC_EVERY):
                tr = sim.play(T.flick_schedule(channel, PEAK, RISE_S, HOLD_S),
                              2 * RISE_S + HOLD_S + SETTLE_S, rc_phase=phase)
                sim.set_channels(centered)
                m = T.flick_metrics([r[:5] for r in tr], 0.0, channel, axis, sign, setpoint, RISE_S, HOLD_S)
                target = [r[5][debug_base + 2] / 10 for r in tr if r[5]]
                first = [round(r[5][debug_base + 2] / 10)
                         for r in tr[:int(0.012 / T.DT):int(0.002 / T.DT)] if r[5]]
                print(f"{name:5s} {phase}     {m['rise90_ms']:5.1f}  {100 * m['overshoot']:+6.1f}%  "
                      f"{100 * m['sat']:4.1f}%  {max(target, key=abs):7.1f}   {first}", flush=True)
                sim.wait(0.5)
    finally:
        sim.stop()
        msp.close()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=3)


if __name__ == "__main__":
    main()
