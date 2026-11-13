#!/usr/bin/env python3
"""
WhippyFlight SITL end-to-end test.

Phases:
  Phase 0: Gain source selection: boot with combinations of the emulated SD
           card (sdcard.img) and dataflash (dataflash.bin) and check which
           source the gains load from, and the result code
  Phase A: Put the gain blob on the emulated SD card, start SITL, configure
           ARM via CLI, save (triggers exit)
  Phase B: Restart SITL with saved config, check the gains were copied from
           SD to RAM, send fdm+rc, verify motor output; then move the
           sticks and check the setpoint estimate and feedforward target
           (WHIPPYFLIGHT debug mode, per loop in the motor reply) and the
           motor response
  Phase C: Remove all gains, check arming is refused (WHIPPYFLIGHT flag)
  Phase D: Missing ESC telemetry and unsupported settings (self-level
           modes, non-DROP failsafe) are reported as arming blockers
  Phase E: Closed loop: fly the nonlinear plant the gains were solved for
           (sim_plant.py), with its gyro and motor speeds fed back to SITL;
           hover, then flick each axis and score the tracking
  Phase F: The same flights with the feedback gains zeroed
           ($WFG_NO_FEEDBACK_PATH, gen_wfg.py --feedback-scale 0), for the
           plots only; skipped when unset

Traces and plots of the flights go to $SITL_OUT (default out/sitl/ in the
crate): closed_loop.{csv,png}, no_feedback.{csv,png}, and compare.png.

The link is lockstep (see Sim): one fdm_packet per PID loop, carrying ESC
telemetry (motor eRPM, battery voltage) for WhippyFlight's gain scheduling.

Requirements:
  - SITL binary at ../../../obj/betaflight_2026.12.0_SITL (relative to this script)
  - No other process using ports 5761, 9002, 9003, 9004
"""

import json
import math
import socket
import struct
import time
import subprocess
import signal
import os
import sys
import threading

from fat_image import build_image
from sim_plant import Plant, erpm_from_rps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "obj"))
SITL = os.path.join(OBJ_DIR, "betaflight_2026.12.0_SITL")
EEPROM = os.path.join(OBJ_DIR, "eeprom.bin")
WFG = os.environ.get("WFG_PATH", os.path.join(SCRIPT_DIR, "test_gains.wfg"))
WFG_NO_FEEDBACK = os.environ.get("WFG_NO_FEEDBACK_PATH")
OUT_DIR = os.path.realpath(os.environ.get("SITL_OUT", os.path.join(SCRIPT_DIR, "..", "out", "sitl")))

# Emulated dataflash (drivers/flash/flash_sitl.h) and the WHIPPYFLIGHT
# partition carved from its top: WHIPPYFLIGHT_GAIN_STORE_SIZE (common_post.h)
# rounded up to whole 64 KiB sectors.  Must match the firmware.
DATAFLASH = os.path.join(OBJ_DIR, "dataflash.bin")
DATAFLASH_SIZE = 64 * 64 * 1024
GAIN_PARTITION_SIZE = 64 * 1024
GAIN_PARTITION_OFFSET = DATAFLASH_SIZE - GAIN_PARTITION_SIZE

# Emulated SD card (drivers/sdcard_sitl.h) and the gain file on it
# (WHIPPYFLIGHT_GAIN_FILENAME).
SDCARD = os.path.join(OBJ_DIR, "sdcard.img")
GAIN_FILENAME = "WHIPPY.WFG"

# GainLoadResult codes (types.rs / whippyflight_glue.h).
GAIN_OK, GAIN_BAD_MAGIC, GAIN_BAD_CHECKSUM = 0, 2, 3

# Lockstep: one fdm_packet per PID loop of $WFG_PATH (its header dt, which
# must be one of SITL's 8 kHz/denom rates), RC at 500 Hz.
with open(WFG, "rb") as _f:
    _hdr = _f.read(0x20)
DT = struct.unpack_from("<f", _hdr, 0x10)[0]
RC_EVERY = round(0.002 / DT)
# fdm_packet followed by ESC telemetry (fdm_packet_esc in SITL target.h).
FDM_ESC_FMT = "<d3d3d4d3d3dd4dd"
OPEN_LOOP_OMEGA = 500.0  # motor speed (shaft rev/s) reported without a plant
# Plant coefficients: the JSON $WFG_PATH was generated from.
COEFFS = os.environ.get("PLANT_JSON", os.path.join(SCRIPT_DIR, "airframes", "meteor65.json"))
with open(COEFFS) as _f:
    _airframe = json.load(_f)["airframe"]
VBAT = _airframe.get("vbat_v", 4.2)  # V; 4.2 = 1S Meteor65
MOTOR_POLES = _airframe["motor_poles"]


# ---------- MSP helpers ----------

def msp_encode(cmd, payload=b""):
    length = len(payload)
    cs = length ^ cmd
    for b in payload:
        cs ^= b
    return b"$M<" + bytes([length, cmd]) + payload + bytes([cs & 0xFF])


def msp_decode_first(data):
    i = 0
    while i + 5 <= len(data):
        if data[i : i + 3] != b"$M>":
            i += 1
            continue
        length = data[i + 3]
        cmd = data[i + 4]
        if i + 5 + length + 1 > len(data):
            break
        return cmd, data[i + 5 : i + 5 + length]
    return None, None


def crc8_dvb_s2(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def msp2_encode(cmd, payload=b""):
    body = struct.pack("<BHH", 0, cmd, len(payload)) + payload
    return b"$X<" + body + bytes([crc8_dvb_s2(body)])


def msp2_request(sock, cmd, payload=b"", timeout=2.0):
    """Send an MSP v2 request → (ok, reply payload); ok is False for an
    error reply ('!'), None on timeout."""
    sock.send(msp2_encode(cmd, payload))
    data = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data += sock.recv(4096)
        except Exception:
            continue
        i = 0
        while (i := data.find(b"$X", i)) >= 0 and i + 8 <= len(data):
            _, rcmd, length = struct.unpack_from("<BHH", data, i + 3)
            if i + 9 + length > len(data):
                break
            if rcmd == cmd:
                return data[i + 2] == ord(">"), data[i + 8 : i + 8 + length]
            i += 9 + length
    return None, b""


def decode_arming_flags(p):
    """Parse MSP_STATUS_EX payload → dict with armed, flags, blocking."""
    if not p or len(p) < 20:
        return {"error": "payload too short"}
    mode = struct.unpack_from("<I", p, 6)[0]
    off = 15
    ext_n = p[off]
    off += 1 + ext_n + 1  # skip ext bytes + ARMING_DISABLE_FLAGS_COUNT
    aflags = struct.unpack_from("<I", p, off)[0]
    names = []
    for bit, name in [
        (0, "NOGYRO"), (1, "FAILSAFE"), (2, "RXLOSS"), (3, "NOTDIS"),
        (4, "BOXFS"), (7, "THROTTLE"), (8, "ANGLE"), (9, "BOOTGRACE"),
        (12, "CALIB"), (13, "CLI"), (24, "MOTOR_PROTO"), (29, "WHIPPYFLIGHT"),
        (30, "ARMSW"),
    ]:
        if aflags & (1 << bit):
            names.append(name)
    return {"armed": bool(mode & 1), "flags": hex(aflags), "blocking": names}


def query_status(msp_sock, timeout=1.0):
    """Query MSP_STATUS_EX → decode_arming_flags(), or {"error": ...}."""
    timeout_before = msp_sock.gettimeout()
    msp_sock.setblocking(False)
    try:
        while msp_sock.recv(2048):
            pass
    except Exception:
        pass
    msp_sock.settimeout(timeout_before)
    try:
        msp_sock.send(msp_encode(150))
    except Exception as e:
        return {"error": str(e)}
    data = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data += msp_sock.recv(512)
        except Exception:
            continue
        cmd, p = msp_decode_first(data)
        if cmd == 150:
            return decode_arming_flags(p)
    return {"error": "timed out"}


# ---------- Lockstep simulator link (background thread) ----------

class Sim:
    """Lockstep FDM/RC/motor link to SITL.

    WhippyFlight SITL builds run one PID loop per fdm_packet and answer it
    with one servo_packet, so each step sends the sensor state, waits for
    the motor command that loop produced, and advances simulated time by DT.
    With a `plant` the gyro and ESC telemetry come from integrating it under
    those commands; without one the gyro is whatever set_gyro() says and
    the motors report a fixed speed.  SITL's clock follows the packet
    timestamps, so schedules and waits are in simulated time."""

    def __init__(self, plant=None):
        self.fdm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.motor_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.motor_sock.bind(("127.0.0.1", 9002))
        self.plant = plant
        self.t = 0.0
        self.iteration = 0
        self.channels = [1500, 1500, 1000, 1500, 1000] + [1500] * 11
        self.schedule = None
        self.gyro = (0.0, 0.0, 0.0)
        self.duty = None
        self.debug = None
        self.replies = 0
        self.misses = 0
        self.trace = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def set_channels(self, ch):
        with self._lock:
            self.channels = list(ch) + [1500] * (16 - len(ch))
            self.schedule = None

    def set_schedule(self, fn):
        """Drive the channels from fn(simulated seconds since this call)."""
        with self._lock:
            self.schedule = (self.t, fn)

    def set_gyro(self, gyro):
        """Open-loop gyro, fdm_packet imu_angular_velocity_rpy in rad/s."""
        with self._lock:
            self.gyro = tuple(gyro)

    def motors(self):
        """Latest motor command (servo_packet motor_speed), or None."""
        with self._lock:
            return self.duty

    def record(self):
        """Start recording (t, channels[:4], gyro_bf, omega, duty, debug) per
        step; open loop, gyro_bf is from set_gyro() and omega the reported
        speed."""
        with self._lock:
            self.trace = []

    def play(self, fn, duration, rc_phase=0):
        """Drive the channels from fn (as set_schedule) for `duration`
        simulated seconds, recording.  The schedule starts `rc_phase` loops
        after an RC packet, so where its edges fall between RC frames (which
        the setpoint estimator sees) repeats run to run.  Returns the trace
        with t relative to the start; the schedule keeps running after."""
        with self._lock:
            wait = (rc_phase - self.iteration) % RC_EVERY
            start = self.t + wait * DT
            held = list(self.channels[:5])
            self.schedule = (start, lambda t: fn(t) if t >= 0 else held)
            self.trace = []
        self.wait_until(start + duration)
        trace = self.stop_recording()
        return [(r[0] - start, *r[1:]) for r in trace if r[0] - start >= -DT / 2]

    def stop_recording(self):
        with self._lock:
            trace, self.trace = self.trace, None
        return trace

    def wait(self, sim_s):
        """Block until sim_s of simulated time has passed."""
        self.wait_until(self.t + sim_s)

    def wait_until(self, target):
        deadline = time.monotonic() + 20 * (target - self.t) + 5
        while self.t < target and time.monotonic() < deadline:
            time.sleep(0.002)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self.motor_sock.close()

    def _receive(self, timeout):
        """The motor reply (duty, debug) to the packet just sent, plus any
        stragglers.  debug is that PID loop's debug[] (servo_packet_debug),
        None from replies without it."""
        duty = debug = None
        self.motor_sock.settimeout(timeout)
        try:
            while True:
                data, _ = self.motor_sock.recvfrom(256)
                duty = struct.unpack_from("<4f", data)
                debug = struct.unpack_from("<8h", data, 16) if len(data) >= 32 else None
                self.motor_sock.settimeout(0)
        except (BlockingIOError, socket.timeout, OSError):
            pass
        return duty, debug

    def _run(self):
        replying = False
        while self._running:
            with self._lock:
                if self.schedule:
                    start, fn = self.schedule
                    self.channels = list(fn(self.t - start)) + [1500] * 11
                ch = self.channels[:]
                if self.plant:
                    gyro, erpm = self.plant.imu_rpy(), self.plant.erpm()
                else:
                    gyro, erpm = self.gyro, [erpm_from_rps(OPEN_LOOP_OMEGA, MOTOR_POLES)] * 4
            if self.iteration % RC_EVERY == 0:
                self.rc_sock.sendto(struct.pack("<d16H", self.t, *ch[:16]), ("127.0.0.1", 9004))
            pkt = struct.pack(
                FDM_ESC_FMT, self.t, *gyro,
                0, 0, -9.8,  # linear accel
                1, 0, 0, 0,  # quaternion (level)
                0, 0, 0,     # position
                0, 0, 0,     # velocity
                101325,      # pressure
                *erpm, VBAT,
            )
            self.fdm_sock.sendto(pkt, ("127.0.0.1", 9003))
            # Wait out a PID loop while SITL is answering; while it isn't
            # (booting), keep simulated time moving at about real time.
            duty, debug = self._receive(0.005 if replying else 0.0005)
            replying = duty is not None
            with self._lock:
                if duty is not None:
                    self.duty, self.debug = duty, debug
                    self.replies += 1
                else:
                    self.misses += 1
                duty, debug = self.duty, self.debug
            if self.plant:
                gyro_bf, omega = self.plant.gyro_bf(), tuple(self.plant.omega)
                volts = [d * VBAT for d in duty] if duty else [0.0] * 4
                self.plant.step(volts, DT)
            else:
                gyro_bf, omega = gyro, (OPEN_LOOP_OMEGA,) * 4
            self.t += DT
            self.iteration += 1
            if self.trace is not None:
                with self._lock:
                    if self.trace is not None:
                        self.trace.append((self.t, ch[:4], gyro_bf, omega, duty, debug))



# ---------- TCP / SITL helpers ----------

def wait_tcp(port, timeout=15):
    for _ in range(timeout * 10):
        time.sleep(0.1)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect(("127.0.0.1", port))
            return s
        except Exception:
            s.close()
    return None


def start_sitl(log_path="/tmp/sitl_wf.log"):
    return subprocess.Popen(
        ["stdbuf", "-oL", SITL],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=OBJ_DIR,
    )


def install_gains(dataflash_blob=None, sd_files=None):
    """Set up the emulated gain stores.  The dataflash is freshly erased,
    with `dataflash_blob` programmed into the gain partition if given, as an
    uploader would.  `sd_files` (8.3 name → bytes) become a FAT16 card
    image; None means no card."""
    image = bytearray(b"\xff" * DATAFLASH_SIZE)
    if dataflash_blob is not None:
        assert len(dataflash_blob) <= GAIN_PARTITION_SIZE, "blob larger than gain partition"
        image[GAIN_PARTITION_OFFSET:GAIN_PARTITION_OFFSET + len(dataflash_blob)] = dataflash_blob
    with open(DATAFLASH, "wb") as f:
        f.write(image)
    if sd_files is None:
        if os.path.exists(SDCARD):
            os.remove(SDCARD)
    else:
        build_image(SDCARD, sd_files)


def gain_load_line(log_path):
    for line in open(log_path, errors="replace"):
        if "[WHIPPYFLIGHT] gain load" in line:
            return line.strip()
    return None


def boot_gain_load(log_path, timeout=10):
    """Boot SITL just long enough to log the gain load line, then kill it."""
    proc = start_sitl(log_path)
    line = None
    for _ in range(timeout * 10):
        time.sleep(0.1)
        line = gain_load_line(log_path)
        if line:
            break
    proc.kill()
    proc.wait()
    return line


def check_gain_sources(blob):
    """Phase 0: which store the gains come from, and the result code."""
    corrupt = bytearray(blob)
    corrupt[len(blob) // 2] ^= 0xFF
    cases = [
        ("SD file only",                 None, {GAIN_FILENAME: blob},           "sdcard",    GAIN_OK),
        ("SD file wins over dataflash",  blob, {GAIN_FILENAME: blob},           "sdcard",    GAIN_OK),
        ("SD without file → dataflash",  blob, {"README.TXT": b"hi"},           "dataflash", GAIN_OK),
        ("no card → dataflash",          blob, None,                            "dataflash", GAIN_OK),
        ("nothing installed",            None, None,                            "dataflash", GAIN_BAD_MAGIC),
        ("corrupt SD file, no fallback", blob, {GAIN_FILENAME: bytes(corrupt)}, "sdcard",    GAIN_BAD_CHECKSUM),
    ]
    ok = True
    for name, flash_blob, sd_files, source, result in cases:
        install_gains(flash_blob, sd_files)
        line = boot_gain_load("/tmp/sitl_phase0.log")
        expected = f"from {source}: result={result},"
        passed = line is not None and expected in line
        ok &= passed
        print(f"  {'✓' if passed else '✗'} {name}: {line}")
    return ok


# MSP2_WHIPPYFLIGHT_* (msp_protocol_v2_betaflight.h, flight/whippyflight_msp.c).
MSP2_WF_STATUS, MSP2_WF_BEGIN, MSP2_WF_DATA, MSP2_WF_COMMIT, MSP2_WF_ABORT = range(0x3100, 0x3105)
WF_UPLOAD_OK, WF_UPLOAD_REJECTED = 0, 8
WF_SOURCE_DATAFLASH = 2


def wf_status(sock):
    ok, p = msp2_request(sock, MSP2_WF_STATUS)
    if not ok or len(p) < 35:
        return None
    (version, result, source, uploading, size, crc, blob_dt, pid_dt, partition,
     upload_size, received, max_chunk, reason_len) = struct.unpack_from("<4BIIffIIIHB", p)
    return {"version": version, "result": result, "source": source, "uploading": bool(uploading),
            "size": size, "crc": crc, "blob_dt": blob_dt, "pid_dt": pid_dt,
            "partition": partition, "received": received, "max_chunk": max_chunk,
            "reason": p[35 : 35 + reason_len].decode(errors="replace")}


def wf_upload(sock, blob):
    """Upload `blob` over MSP → (upload result, gain load result)."""
    st = wf_status(sock)
    ok, r = msp2_request(sock, MSP2_WF_BEGIN, struct.pack("<I", len(blob)))
    if not ok or r[0] != WF_UPLOAD_OK:
        return (r[0] if r else None), None
    for off in range(0, len(blob), st["max_chunk"]):
        ok, r = msp2_request(sock, MSP2_WF_DATA, struct.pack("<I", off) + blob[off : off + st["max_chunk"]])
        if not ok or r[0] != WF_UPLOAD_OK:
            msp2_request(sock, MSP2_WF_ABORT)
            return (r[0] if r else None), None
    ok, r = msp2_request(sock, MSP2_WF_COMMIT, timeout=30.0)
    return (r[0], r[1]) if ok else (None, None)


def check_msp_upload(blob):
    """Upload over MSP into an erased dataflash, check it survives a reboot,
    and that a corrupt upload is refused without losing the stored gains."""
    install_gains(None, None)
    proc = start_sitl("/tmp/sitl_upload.log")
    sock = wait_tcp(5761, 30)
    if not sock:
        print("  FAILED to connect MSP")
        proc.kill()
        return False
    sock.settimeout(0.1)
    before = wf_status(sock)
    print(f"  before: {before}")
    t0 = time.monotonic()
    upload = wf_upload(sock, blob)
    after = wf_status(sock)
    print(f"  upload {len(blob)} B: result={upload} in {time.monotonic() - t0:.1f}s")
    print(f"  after: {after}")
    crc = struct.unpack_from("<I", blob, len(blob) - 4)[0]
    ok = (before is not None and before["result"] == GAIN_BAD_MAGIC
          and upload == (WF_UPLOAD_OK, GAIN_OK) and after is not None
          and after["result"] == GAIN_OK and after["source"] == WF_SOURCE_DATAFLASH
          and after["size"] == len(blob) and after["crc"] == crc and not after["uploading"])

    corrupt = bytearray(blob)
    corrupt[len(blob) // 2] ^= 0xFF
    rejected = wf_upload(sock, bytes(corrupt))
    kept = wf_status(sock)
    print(f"  corrupt upload: result={rejected}, then {kept}")
    ok = ok and rejected == (WF_UPLOAD_REJECTED, GAIN_BAD_CHECKSUM) and kept == after
    sock.close()
    proc.kill()
    proc.wait()

    image = open(DATAFLASH, "rb").read()
    stored = image[GAIN_PARTITION_OFFSET : GAIN_PARTITION_OFFSET + len(blob)] == blob
    line = boot_gain_load("/tmp/sitl_upload_reboot.log")
    print(f"  partition holds blob={stored}; reboot: {line}")
    return ok and stored and line is not None and f"from dataflash: result=0, {len(blob)} bytes" in line


def cli_command(sock, cmd, settle=0.3):
    sock.send(cmd + b"\r\n")
    time.sleep(settle)
    out = b""
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            out += chunk
    except Exception:
        pass
    return out.decode(errors="replace")


def check_config_blocks():
    """Each unsupported setting must make `status` report its reason, and
    reverting it must clear the report.  Settings are not saved."""
    cases = [
        (b"aux 1 1 0 1700 2100", b"aux 1 0 0 900 900", "self-level mode configured"),
        (b"set failsafe_procedure = AUTO-LAND", b"set failsafe_procedure = DROP",
         "failsafe_procedure is not DROP"),
    ]
    proc = start_sitl("/tmp/sitl_phaseD.log")
    cli = wait_tcp(5761, 30)
    if not cli:
        print("  FAILED to connect CLI")
        proc.kill()
        return False
    cli.settimeout(0.1)
    cli_command(cli, b"#", settle=0.5)
    ok = "WHIPPYFLIGHT: simulator sends no ESC telemetry" in cli_command(cli, b"status")
    print(f"  no fdm packets: ESC telemetry reported missing={ok}")
    sim = Sim()
    sim.start()
    sim.wait(0.5)
    baseline = "WHIPPYFLIGHT:" not in cli_command(cli, b"status")
    print(f"  with ESC telemetry: {'no WHIPPYFLIGHT reason' if baseline else 'unexpected WHIPPYFLIGHT reason'}")
    ok = ok and baseline
    for apply, revert, reason in cases:
        reported = f"WHIPPYFLIGHT: {reason}" in cli_command(cli, apply) + cli_command(cli, b"status")
        cleared = "WHIPPYFLIGHT:" not in cli_command(cli, revert) + cli_command(cli, b"status")
        print(f"  {apply.decode()}: reported={reported} cleared_after_revert={cleared}")
        ok = ok and reported and cleared
    sim.stop()
    cli.close()
    proc.kill()
    proc.wait()
    return ok


HOVER = 1500
ARMED_AUX = 2000


def stick_schedule(channel, ramp_s, hold_s, peak):
    """Centered sticks with one channel ramped 1500 → peak over ramp_s,
    held for hold_s, then snapped back to center."""
    def fn(t):
        ch = [1500, 1500, HOVER, 1500, ARMED_AUX]
        if t < ramp_s:
            ch[channel] = round(1500 + (peak - 1500) * t / ramp_s)
        elif t < ramp_s + hold_s:
            ch[channel] = peak
        return ch
    return fn


def pair_diff(m, up_pair):
    down_pair = tuple(i for i in range(4) if i not in up_pair)
    return sum(m[i] for i in up_pair) / 2 - sum(m[i] for i in down_pair) / 2


def trace_at(trace, t):
    """The last trace record at or before relative time t."""
    rec = trace[0]
    for r in trace:
        if r[0] > t:
            break
        rec = r
    return rec


STICK_SAMPLE_S = 0.01


def stick_input(sim, name, channel, up_pair, debug_base=None,
                ramp_s=0.8, hold_s=0.8, center_s=2.0, peak=1800):
    """Move one stick and check what the controller does with it.

    During the ramp the feedforward target must follow the setpoint estimate;
    during the hold the estimate must settle on the raw RX setpoint and the
    target on the estimate; the gyro stays at zero, so the held rate
    error must drive `up_pair` up, and centering must take most of it back.
    `debug_base` is the WHIPPYFLIGHT debug channel of the axis' raw setpoint
    (raw, estimate, target follow it); None for yaw, which is not logged.
    Everything is read from the per-loop trace at fixed simulated times."""
    print(f"\n--- {name} stick: 1500 → {peak} over {ramp_s:.1f}s, hold {hold_s:.1f}s, center ---")
    res = {}
    release = ramp_s + hold_s
    trace = sim.play(stick_schedule(channel, ramp_s, hold_s, peak), release + center_s)
    m_base = trace[0][4]
    m_hold = trace_at(trace, release - STICK_SAMPLE_S)[4]
    m_center = trace[-1][4]
    print(f"  held motors={tuple(round(x, 3) for x in m_hold) if m_hold else None}")
    print(f"  centered motors={tuple(round(x, 3) for x in m_center) if m_center else None}")
    samples = []
    if debug_base is not None:
        n = round(release / STICK_SAMPLE_S)
        for k in range(1, n):
            r = trace_at(trace, k * STICK_SAMPLE_S)
            if r[5]:
                samples.append((r[0], *r[5][debug_base:debug_base + 3]))
    d_center = trace[-1][5] if debug_base is not None else None

    # Pair differences relative to the centered-stick hover before the input.
    base = pair_diff(m_base, up_pair) if m_base else 0.0
    res["hold_diff"] = pair_diff(m_hold, up_pair) - base if m_hold else 0.0
    res["center_diff"] = pair_diff(m_center, up_pair) - base if m_center else 0.0
    print(f"  motors{up_pair} − others, change from hover: "
          f"held={res['hold_diff']:+.4f} centered={res['center_diff']:+.4f}")

    if debug_base is None:
        return res
    for t, raw, est, tgt in samples[::10]:
        print(f"    t={t:.3f}s raw={raw / 10:7.1f} est={est / 10:7.1f} target={tgt / 10:7.1f} deg/s")

    # Target − estimate during the ramp, skipping its first 0.1 s.
    gaps = [(tgt - est) / 10 for t, _, est, tgt in samples if 0.1 < t < ramp_s]
    if gaps:
        res["ramp_gap"] = sorted(gaps)[len(gaps) // 2]
        print(f"  ramp: median target − estimate = {res['ramp_gap']:+.2f} deg/s")
    held = [s for s in samples if ramp_s + 0.2 < s[0] < release]
    if held:
        res["hold_raw"] = held[-1][1] / 10
        res["hold_est_err"] = max(abs(est - raw) for _, raw, est, _ in held) / 10
        res["hold_gap"] = max(abs(tgt - est) for _, _, est, tgt in held) / 10
        print(f"  hold: raw={res['hold_raw']:.1f} deg/s, |est − raw| ≤ {res['hold_est_err']:.1f}, "
              f"|target − est| ≤ {res['hold_gap']:.1f} deg/s")
    if d_center:
        res["center_est"] = d_center[debug_base + 1] / 10
        print(f"  centered: est={res['center_est']:.1f} deg/s")
    return res


def check_stick(name, res, logged):
    """Print and return the pass/fail lines for one stick_input result."""
    checks = [
        (f"{name} stick drives the expected motor pair (diff={res.get('hold_diff', 0):+.4f})",
         res.get("hold_diff", 0) > 0.001),
        (f"{name} stick centering releases the motors "
         f"(diff {res.get('hold_diff', 0):+.4f} → {res.get('center_diff', 0):+.4f})",
         abs(res.get("center_diff", 1)) < 0.5 * abs(res.get("hold_diff", 0))),
    ]
    if logged:
        gap = res.get("ramp_gap", 99)
        checks += [
            (f"{name} ramp: target follows the estimate ({gap:+.2f} deg/s)", abs(gap) <= 1.0),
            (f"{name} hold: estimate on raw setpoint ({res.get('hold_est_err', 99):.1f} deg/s), "
             f"target on estimate ({res.get('hold_gap', 99):.1f} deg/s)",
             res.get("hold_raw", 0) > 50 and res.get("hold_est_err", 99) <= 1.0
             and res.get("hold_gap", 99) <= 1.0),
            (f"{name} centered: estimate back to {res.get('center_est', 99):.1f} deg/s",
             abs(res.get("center_est", 99)) <= 1.0),
        ]
    ok = True
    for text, passed in checks:
        print(f"  {'✓' if passed else '✗'} {text}")
        ok &= passed
    return ok


def wait_flags_and_arm(sim, msp):
    """Wait (disarmed, throttle low) for the arming flags to clear, then arm
    with throttle low.  True once armed."""
    sim.set_channels([1500, 1500, 1000, 1500, 1000])
    deadline = time.monotonic() + 120
    next_print = 0.0
    while True:
        time.sleep(0.5)
        st = query_status(msp)
        if sim.t >= next_print:
            print(f"  t={sim.t:.1f}s (wall {time.monotonic() - deadline + 120:.1f}s): {st}")
            next_print = sim.t + 2.0
        if "flags" in st and not st["blocking"]:
            print(f"  ✓ All flags clear at t={sim.t:.1f}s (iter={sim.iteration})")
            break
        if time.monotonic() > deadline:
            print("  FAILED: flags never cleared")
            return False
    sim.set_channels([1500, 1500, 1000, 1500, ARMED_AUX])
    sim.wait(1.0)
    st = query_status(msp)
    print(f"  after arming: {st}")
    return isinstance(st, dict) and st.get("armed", False)


# ---------- Closed loop through the nonlinear plant ----------

# Phase A sets linear ACTUAL rates: center sensitivity = max rate = 360 deg/s,
# no expo, so stick x = (ch − 1500)/500 commands x rot/s (BF yaw reversed).
STICK_AXES = {"Roll": (0, 0, 1.0), "Pitch": (1, 1, 1.0), "Yaw": (3, 2, -1.0)}
FLICK_PEAK, FLICK_RISE_S, FLICK_HOLD_S, FLICK_SETTLE_S = 1750, 0.030, 0.5, 0.5
AXIS_NAMES = ("roll", "pitch", "yaw")
AXIS_COLORS = ("tab:blue", "tab:orange", "tab:green")
# WHIPPYFLIGHT debug: per axis (roll, pitch) raw setpoint, estimate and
# feedforward target in 0.1 deg/s, from channel 3·axis.
DEBUG_PER_ROT_S = 3600.0


def stick_setpoint(ch, channel, sign):
    return sign * (ch[channel] - 1500) / 500.0


def flick_schedule(channel, peak, rise_s, hold_s):
    """Raised-cosine flick 1500 → peak over rise_s, held, and back."""
    def s(x):
        return 0.5 * (1 - math.cos(math.pi * min(max(x, 0.0), 1.0)))

    def fn(t):
        ch = [1500, 1500, HOVER, 1500, ARMED_AUX]
        ch[channel] = round(1500 + (peak - 1500) * (s(t / rise_s) - s((t - rise_s - hold_s) / rise_s)))
        return ch
    return fn


def hover_metrics(trace):
    n = len(trace)
    mean = [sum(r[2][a] for r in trace) / n for a in range(3)]
    dev = [math.sqrt(sum((r[2][a] - mean[a]) ** 2 for r in trace) / n) for a in range(3)]
    omega = [sum(r[3][m] for r in trace) / n for m in range(4)]
    return {"gyro_mean": mean, "gyro_dev": dev, "omega": omega}


def flick_metrics(trace, t0, channel, axis, sign, setpoint, rise_s, hold_s):
    """Tracking of one flick, normalised by its setpoint (rot/s)."""
    rel = [(r[0] - t0, stick_setpoint(r[1], channel, sign), r[2], r[3], r[4]) for r in trace]
    y = [g[axis] / setpoint for _, _, g, _, _ in rel]
    r = [sp / setpoint for _, sp, _, _, _ in rel]
    t = [x[0] for x in rel]
    hold_end = rise_s + hold_s
    first_cmd = next(t[i] for i in range(len(t)) if r[i] > 0.01)
    rise90 = next((t[i] - first_cmd for i in range(len(t)) if y[i] >= 0.9), float("inf"))
    held = [y[i] for i in range(len(t)) if hold_end - 0.2 <= t[i] < hold_end]
    settled = [y[i] for i in range(len(t)) if t[i] >= t[-1] - 0.2]
    others = [a for a in range(3) if a != axis]
    return {
        "track_rms": abs(setpoint) * math.sqrt(sum((y[i] - r[i]) ** 2 for i in range(len(t))) / len(t)),
        "rise90_ms": 1e3 * rise90,
        "overshoot": max(y) - 1.0,
        "hold": sum(held) / len(held),
        "after": abs(setpoint) * max(abs(v) for v in settled),
        "cross": max(abs(g[a]) for _, _, g, _, _ in rel for a in others),
        "sat": sum(1 for x in rel if x[4] and (min(x[4]) <= 0.001 or max(x[4]) >= 0.999)) / len(rel),
        "omega": (min(min(x[3]) for x in rel), max(max(x[3]) for x in rel)),
    }


def ff_target(row, axis):
    """The feedforward target (rot/s) logged for this step, None for yaw."""
    if axis > 1 or not row[5]:
        return None
    return row[5][3 * axis + 2] / DEBUG_PER_ROT_S


def save_csv(episodes, path):
    with open(path, "w") as f:
        f.write("episode,t,ch_roll,ch_pitch,ch_throttle,ch_yaw,gyro_roll,gyro_pitch,gyro_yaw,"
                "omega0,omega1,omega2,omega3,duty0,duty1,duty2,duty3,"
                "raw_roll,est_roll,target_roll,raw_pitch,est_pitch,target_pitch\n")
        for name, t0, trace in episodes:
            for t, ch, g, w, d, dbg in trace:
                d = d or (0, 0, 0, 0)
                sp = [v / DEBUG_PER_ROT_S for v in dbg[:6]] if dbg else [""] * 6
                f.write(f"{name},{t - t0:.5f}," + ",".join(str(c) for c in ch) + ","
                        + ",".join(f"{v:.5f}" for v in (*g, *w, *d)) + ","
                        + ",".join(v if v == "" else f"{v:.5f}" for v in sp) + "\n")


def pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (no matplotlib: plots skipped)")
        return None


def flick_windows():
    """(title, x limits) of the plot rows: whole flight, onset, release."""
    release = FLICK_RISE_S + FLICK_HOLD_S
    return (("flick", None), ("onset", (-0.01, 0.12)), ("release", (release - 0.01, release + 0.12)))


def plot_rates(ax, trace, name, style=None, label_prefix="", cross=True, reference=True):
    channel, axis, sign = STICK_AXES[name]
    t = [r[0] for r in trace]
    if reference:
        ax.plot(t, [stick_setpoint(r[1], channel, sign) for r in trace], "k--", lw=1, label="stick")
        target = [ff_target(r, axis) for r in trace]
        if any(v is not None for v in target):
            ax.plot(t, target, color="0.5", lw=1, label="ff target")
    ax.plot(t, [r[2][axis] for r in trace], lw=1.4, label=f"{label_prefix}gyro {AXIS_NAMES[axis]}",
            **(style or {"color": AXIS_COLORS[axis]}))
    if cross:
        for a in range(3):
            if a != axis:
                ax.plot(t, [r[2][a] for r in trace], lw=0.8, color=AXIS_COLORS[a],
                        label=f"{label_prefix}gyro {AXIS_NAMES[a]}")


def shade_saturation(ax, trace):
    """Shade the steps where any motor's duty is at 0 or 1."""
    start = None
    for r in trace + [(None, None, None, None, None, None)]:
        d = r[4]
        sat = bool(d) and (min(d) <= 0.001 or max(d) >= 0.999)
        if sat and start is None:
            start = r[0]
        elif not sat and start is not None:
            ax.axvspan(start, r[0] if r[0] is not None else trace[-1][0], color="red", alpha=0.15, lw=0)
            start = None


def plot_closed_loop(plt, episodes, metrics, title, path):
    """Per flick (columns): for the whole flick and the onset and release
    zooms, the rates and below them the motor speeds, saturated duty shaded."""
    flicks = [e for e in episodes if e[0] in STICK_AXES]
    windows = flick_windows()
    fig, axes = plt.subplots(2 * len(windows), len(flicks), figsize=(5.5 * len(flicks), 22),
                             squeeze=False)
    for col, (name, _, trace) in enumerate(flicks):
        m = metrics.get(name)
        t = [r[0] for r in trace]
        for w, (window, xlim) in enumerate(windows):
            ax = axes[2 * w][col]
            plot_rates(ax, trace, name)
            ax.set_title(f"{name} {window}" + (f" — rise90 {m['rise90_ms']:.0f} ms, "
                                               f"overshoot {100 * m['overshoot']:+.1f}%"
                                               if m and w == 0 else ""), fontsize=10)
            ax.set_ylabel("rot/s")
            motors = axes[2 * w + 1][col]
            for i in range(4):
                motors.plot(t, [r[3][i] for r in trace], lw=1, label=f"motor {i}")
            shade_saturation(motors, trace)
            motors.set_title(f"{name} {window}: motor speeds (duty saturated shaded"
                             + (f", {100 * m['sat']:.1f}%)" if m and w == 0 else ")"), fontsize=10)
            motors.set_ylabel("rev/s")
            for a in (ax, motors):
                if xlim:
                    a.set_xlim(*xlim)
                    autoscale_y(a, trace, xlim)
                a.grid(alpha=0.3)
                a.legend(fontsize=7)
        axes[-1][col].set_xlabel("s from flick start")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def autoscale_y(ax, trace, xlim):
    """Fit y to the lines inside the zoom window."""
    lo, hi = math.inf, -math.inf
    for line in ax.get_lines():
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            if y is not None and math.isfinite(y) and xlim[0] <= x <= xlim[1]:
                lo, hi = min(lo, y), max(hi, y)
    if lo < hi:
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)


def plot_compare(plt, runs, path):
    """On-axis gyro of each run over the same flick, per axis (columns),
    whole flick and onset/release zooms (rows)."""
    windows = flick_windows()
    names = [n for n in STICK_AXES if all(any(e[0] == n for e in r[1]) for r in runs)]
    fig, axes = plt.subplots(len(windows), len(names), figsize=(5.5 * len(names), 9), squeeze=False)
    colors = ("tab:blue", "tab:red", "tab:green")
    for col, name in enumerate(names):
        for row, (window, xlim) in enumerate(windows):
            ax = axes[row][col]
            for i, (label, episodes, _) in enumerate(runs):
                trace = next(e[2] for e in episodes if e[0] == name)
                plot_rates(ax, trace, name, style={"color": colors[i]}, label_prefix=f"{label}: ",
                           cross=False, reference=i == 0)
            if xlim:
                ax.set_xlim(*xlim)
                autoscale_y(ax, trace, xlim)
            ax.set_title(f"{name} {window}", fontsize=10)
            ax.set_ylabel("rot/s")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[-1][col].set_xlabel("s from flick start")
    fig.suptitle("Closed loop flicks, with and without feedback")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_closed_loop(runs):
    """CSV and plots of each closed-loop run [(label, episodes, metrics)] to
    OUT_DIR, and an overlay of the runs."""
    os.makedirs(OUT_DIR, exist_ok=True)
    plt = pyplot()
    for label, episodes, metrics in runs:
        csv = os.path.join(OUT_DIR, f"{label}.csv")
        save_csv(episodes, csv)
        print(f"  trace: {csv}")
        if plt:
            png = os.path.join(OUT_DIR, f"{label}.png")
            plot_closed_loop(plt, episodes, metrics,
                             f"{label} — {os.path.basename(WFG)}, plant {os.path.basename(COEFFS)}", png)
            print(f"  plot: {png}")
    if plt and len(runs) > 1:
        png = os.path.join(OUT_DIR, "compare.png")
        plot_compare(plt, runs, png)
        print(f"  plot: {png}")


def closed_loop(label="closed_loop", peak=FLICK_PEAK, rise_s=FLICK_RISE_S, hold_s=FLICK_HOLD_S,
                settle_s=FLICK_SETTLE_S, rest_each=False):
    """Phase E: fly SITL through the nonlinear plant.  Returns results, with
    the recorded (name, t0, trace) episodes under "episodes".

    rest_each zeroes the plant's body rates before the hover recording and
    each flick: without feedback nothing holds the hover trim, and the rate
    integrates its residual torque from wherever the last segment left it."""
    res = {}
    proc = start_sitl(f"/tmp/sitl_{label}.log")
    msp = wait_tcp(5761, 30)
    if not msp:
        print("  FAILED to connect MSP")
        proc.kill()
        return res
    msp.settimeout(0.1)
    sim = Sim(plant=Plant(COEFFS))
    sim.start()
    try:
        res["armed"] = wait_flags_and_arm(sim, msp)
        if not res["armed"]:
            return res
        print("\n--- Hover: throttle up, settle 2s, record 1s ---")
        sim.set_channels([1500, 1500, HOVER, 1500, ARMED_AUX])
        sim.wait(2.0)
        episodes = []
        if rest_each:
            sim.plant.rate = [0.0, 0.0, 0.0]
        sim.record()
        t0 = sim.t
        sim.wait(1.0)
        trace = sim.stop_recording()
        episodes.append(("Hover", t0, trace))
        res["hover"] = h = hover_metrics(trace)
        print(f"  gyro mean={[round(v, 4) for v in h['gyro_mean']]} rot/s, "
              f"dev={[round(v, 4) for v in h['gyro_dev']]} rot/s")
        print(f"  motor speeds={[round(v, 1) for v in h['omega']]} rev/s")
        for name, (channel, axis, sign) in STICK_AXES.items():
            setpoint = sign * (peak - 1500) / 500.0
            print(f"\n--- {name} flick to {setpoint:+.2f} rot/s over {rise_s * 1e3:.0f} ms, "
                  f"hold {hold_s:.1f}s, back ---")
            t0 = 0.0
            if rest_each:
                sim.plant.rate = [0.0, 0.0, 0.0]
            trace = sim.play(flick_schedule(channel, peak, rise_s, hold_s), 2 * rise_s + hold_s + settle_s)
            sim.set_channels([1500, 1500, HOVER, 1500, ARMED_AUX])
            episodes.append((name, t0, trace))
            res[name] = m = flick_metrics(trace, t0, channel, axis, sign, setpoint, rise_s, hold_s)
            print(f"  track_rms={m['track_rms']:.4f} rot/s  rise90={m['rise90_ms']:.1f} ms  "
                  f"overshoot={100 * m['overshoot']:+.1f}%  hold={100 * m['hold']:.1f}% of setpoint")
            print(f"  after release ≤ {m['after']:.4f} rot/s  cross-axis ≤ {m['cross']:.4f} rot/s  "
                  f"saturated {100 * m['sat']:.1f}%  motor speeds {m['omega'][0]:.0f}–{m['omega'][1]:.0f}")
            sim.wait(0.5)
        print(f"\n  lockstep: {sim.replies} motor replies, {sim.misses} missed steps")
        res["misses"] = sim.misses
        res["episodes"] = episodes
    finally:
        sim.stop()
        msp.close()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=3)
    return res


def check_closed_loop(res):
    """Print and return the pass/fail lines for closed_loop()."""
    if not res.get("armed"):
        print("  ✗ Closed loop: did not arm")
        return False
    h = res["hover"]
    checks = [
        (f"Closed-loop hover holds still (|mean| ≤ {max(abs(v) for v in h['gyro_mean']):.4f}, "
         f"dev ≤ {max(h['gyro_dev']):.4f} rot/s)",
         max(abs(v) for v in h["gyro_mean"]) < 0.05 and max(h["gyro_dev"]) < 0.02),
        (f"Closed-loop hover motor speeds inside the gain grid ({min(h['omega']):.0f}–{max(h['omega']):.0f})",
         50 < min(h["omega"]) and max(h["omega"]) < 500),
    ]
    for name in STICK_AXES:
        m = res.get(name)
        if not m:
            checks.append((f"{name} flick: not flown", False))
            continue
        checks += [
            (f"{name} flick tracks: rise90 {m['rise90_ms']:.1f} ms, overshoot {100 * m['overshoot']:+.1f}%, "
             f"hold {100 * m['hold']:.1f}%",
             m["rise90_ms"] < 80 and m["overshoot"] < 0.25 and abs(m["hold"] - 1) < 0.1),
            (f"{name} flick: back to rest ({m['after']:.4f} rot/s), cross-axis ≤ {m['cross']:.4f} rot/s",
             m["after"] < 0.05 and m["cross"] < 0.1),
        ]
    ok = True
    for text, passed in checks:
        print(f"  {'✓' if passed else '✗'} {text}")
        ok &= passed
    return ok


# ---------- Main test ----------

def main():
    os.chdir(OBJ_DIR)

    os.system("pkill -9 -f betaflight_2026 2>/dev/null")
    time.sleep(1)

    blob = open(WFG, "rb").read()

    # ======= PHASE 0: Gain source selection =======
    print("Phase 0: Gain source selection")
    if os.path.exists(EEPROM):
        os.remove(EEPROM)
    sources_ok = check_gain_sources(blob)

    print("\nPhase 0b: Gain upload over MSP")
    upload_ok = check_msp_upload(blob)

    # ======= PHASE A: Configure eeprom =======
    print("\nPhase A: Configure eeprom")
    if os.path.exists(EEPROM):
        os.remove(EEPROM)
    install_gains(None, {GAIN_FILENAME: blob})
    print(f"  Installed {len(blob)} B gain blob as {GAIN_FILENAME} on the SD card (dataflash erased)")
    proc = start_sitl("/tmp/sitl_phaseA.log")
    cli = wait_tcp(5761, 30)
    if not cli:
        print("  FAILED to connect")
        proc.kill()
        return False

    time.sleep(0.3)
    cli.send(b"#\n")
    time.sleep(0.5)
    try:
        cli.recv(4096)
    except Exception:
        pass

    # motor_idle = 0 puts PWM output low at mincommand, so servo_packet's
    # motor_speed is the controller's duty; rates are linear (STICK_AXES).
    config = [
        b"aux 0 0 0 1700 2100", b"set small_angle = 180", b"set debug_mode = WHIPPYFLIGHT",
        b"set motor_idle = 0", b"set rates_type = ACTUAL",
        b"set motor_poles = %d" % MOTOR_POLES,
    ]
    for axis in (b"roll", b"pitch", b"yaw"):
        config += [b"set " + axis + b"_rc_rate = 36", b"set " + axis + b"_srate = 36",
                   b"set " + axis + b"_expo = 0"]
    for cmd in config:
        cli.send(cmd + b"\r\n")
        time.sleep(0.2)
        try:
            cli.recv(4096)
        except Exception:
            pass
        print(f"  {cmd.decode()}")

    cli.send(b"save\r\n")
    time.sleep(0.5)
    cli.close()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait()
    print(f"  Saved (eeprom={os.path.exists(EEPROM)})")
    time.sleep(1)

    # ======= PHASE B: Run test =======
    print("\nPhase B: Run test")
    proc = start_sitl("/tmp/sitl_phaseB.log")
    print(f"  SITL PID={proc.pid}")

    msp = wait_tcp(5761, 30)
    if not msp:
        print("  FAILED to connect MSP")
        proc.kill()
        return False
    msp.settimeout(0.1)
    print("  MSP connected")
    results = {}

    load = gain_load_line("/tmp/sitl_phaseB.log")
    print(f"  {load}")
    blob_size = os.path.getsize(WFG)
    results["gains_from_sd"] = load is not None and f"from sdcard: result=0, {blob_size} bytes" in load

    # Open loop: the gyro is set by each step, motors report a fixed speed.
    sim = Sim()
    sim.start()

    print("\n--- Steps 1–2: Warmup and arm ---")
    if not wait_flags_and_arm(sim, msp):
        print("  FAILED: not armed")
        sim.stop()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=3)
        return False
    print("  ✓ ARMED")

    # Step 3: Hover (throttle=1500, zero gyro)
    print("\n--- Step 3: Hover (2s) ---")
    sim.set_channels([1500, 1500, HOVER, 1500, ARMED_AUX])
    sim.set_gyro((0, 0, 0))
    sim.wait(2.0)
    m = sim.motors()
    print(f"  motors={m}")
    if m:
        mean = sum(m) / 4
        spread = max(abs(x - mean) for x in m)
        print(f"  mean={mean:.4f} spread={spread:.4f}")
        results["hover_mean"] = mean
        results["hover_spread"] = spread
    results["hover_motors"] = m

    # Steps 4–6: rate disturbances.  Motor pairs follow the regression's
    # wiring (main.R fits raw blackbox motor indices), and the sim → BF gyro
    # map is roll +, pitch −, yaw + (sitlGyroBodyFromSim, Gazebo bridge).
    # Each +0.5 rad/s sim disturbance should drive the listed pair up:
    #   pitch: BF pitch −  → +pitch error → motors 0,2 up
    #   roll:  BF roll  +  → −roll error  → motors 0,1 up
    #   yaw:   BF yaw   +  → −yaw error   → motors 0,3 up
    disturbances = [
        ("pitch", (0, 0.5, 0), (0, 2)),
        ("roll",  (0.5, 0, 0), (0, 1)),
        ("yaw",   (0, 0, 0.5), (0, 3)),
    ]
    for step, (axis, gyro, up_pair) in enumerate(disturbances, start=4):
        print(f"\n--- Step {step}: {axis.capitalize()} disturbance (0.5 rad/s, 1s) ---")
        sim.set_gyro(gyro)
        sim.wait(1.0)
        m = sim.motors()
        print(f"  motors={m}")
        if m:
            down_pair = tuple(i for i in range(4) if i not in up_pair)
            up = sum(m[i] for i in up_pair) / 2
            down = sum(m[i] for i in down_pair) / 2
            diff = up - down
            print(f"  motors{up_pair}={up:.4f} motors{down_pair}={down:.4f} {axis}_diff={diff:+.4f}")
            results[f"{axis}_diff"] = diff
        results[f"{axis}_motors"] = m

    # Stick inputs, gyro at zero.  With rc rate signs (BF yaw stick is
    # reversed) and the motor pairs above, a positive stick should drive:
    #   roll  +  → +roll error  → motors 2,3 up
    #   pitch +  → +pitch error → motors 0,2 up
    #   yaw   +  → −yaw error   → motors 0,3 up
    print("\n--- Sticks: settle centered (2s) ---")
    sim.set_gyro((0, 0, 0))
    sim.set_channels([1500, 1500, HOVER, 1500, ARMED_AUX])
    sim.wait(2.0)
    results["sticks"] = {
        "Roll":  stick_input(sim, "Roll",  0, (2, 3), debug_base=0),
        "Pitch": stick_input(sim, "Pitch", 1, (0, 2), debug_base=3),
        "Yaw":   stick_input(sim, "Yaw",   3, (0, 3)),
    }

    # Cleanup
    sim.stop()
    msp.close()
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=3)

    # ======= PHASE C: No gains → arming refused =======
    print("\nPhase C: No gains installed")
    install_gains(None, None)
    proc = start_sitl("/tmp/sitl_phaseC.log")
    msp = wait_tcp(5761, 30)
    blocked = False
    if msp:
        msp.settimeout(0.1)
        sim = Sim()
        sim.start()
        # Wait for the boot-time flags to clear, leaving only ours.
        st = None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            time.sleep(0.5)
            st = query_status(msp)
            if st.get("blocking") == ["WHIPPYFLIGHT"]:
                break
        print(f"  disarmed: {st}")
        sim.set_channels([1500, 1500, 1000, 1500, ARMED_AUX])
        sim.wait(1.0)
        armed_st = query_status(msp)
        print(f"  after arm attempt: {armed_st}")
        blocked = st.get("blocking") == ["WHIPPYFLIGHT"] and armed_st.get("armed") is False
        sim.stop()
        msp.close()
    else:
        print("  FAILED to connect MSP")
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=3)
    results["no_gains_blocks_arming"] = blocked

    # ======= PHASE D: Unsupported configuration → arming refused =======
    print("\nPhase D: Unsupported configuration")
    install_gains(None, {GAIN_FILENAME: blob})
    results["config_checks"] = check_config_blocks()

    # ======= PHASE E: Closed loop through the nonlinear plant =======
    print("\nPhase E: Closed loop (nonlinear plant)")
    results["closed_loop"] = cl = closed_loop()
    runs = [("closed_loop", cl["episodes"], cl)] if "episodes" in cl else []

    # ======= PHASE F: The same flights without feedback (plots only) =======
    if WFG_NO_FEEDBACK:
        print("\nPhase F: Closed loop without feedback (feedforward only, not scored)")
        with open(WFG_NO_FEEDBACK, "rb") as f:
            install_gains(None, {GAIN_FILENAME: f.read()})
        nf = closed_loop("no_feedback", rest_each=True)
        if "episodes" in nf:
            runs.append(("no_feedback", nf["episodes"], nf))
        install_gains(None, {GAIN_FILENAME: blob})
    if runs:
        print(f"\nTraces and plots ({OUT_DIR}):")
        save_closed_loop(runs)

    # ======= Summary =======
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    passed = True

    if sources_ok:
        print("  ✓ Gain source selection (SD, dataflash fallback, error reporting)")
    else:
        print("  ✗ Gain source selection")
        passed = False

    if upload_ok:
        print("  ✓ Gains uploaded over MSP to dataflash, kept across reboot, corrupt upload refused")
    else:
        print("  ✗ Gain upload over MSP")
        passed = False

    if results.get("gains_from_sd"):
        print("  ✓ Gains copied from SD card to RAM and accepted")
    else:
        print("  ✗ Gains not loaded from SD card")
        passed = False

    if results.get("no_gains_blocks_arming"):
        print("  ✓ Arming refused without gains (WHIPPYFLIGHT)")
    else:
        print("  ✗ Arming not refused without gains")
        passed = False

    if results.get("config_checks"):
        print("  ✓ Arming refused without ESC telemetry, for self-level modes and non-DROP failsafe")
    else:
        print("  ✗ Unsupported configuration not refused")
        passed = False

    # Check hover: motors should be non-zero
    hover_mean = results.get("hover_mean", 0)
    if hover_mean > 0:
        print(f"  ✓ Hover motors active (mean={hover_mean:.4f})")
    else:
        print(f"  ✗ Hover motors zero")
        passed = False

    # Check each disturbance drove the expected motor pair up
    for axis in ("pitch", "roll", "yaw"):
        diff = results.get(f"{axis}_diff", 0)
        if diff > 0.001:
            print(f"  ✓ {axis.capitalize()} response in expected direction (diff={diff:+.4f})")
        else:
            print(f"  ✗ {axis.capitalize()} response missing or reversed (diff={diff:+.4f})")
            passed = False

    for name, res in results.get("sticks", {}).items():
        passed &= check_stick(name, res, logged=name != "Yaw")

    passed &= check_closed_loop(results.get("closed_loop", {}))

    if passed:
        print("\n  ALL CHECKS PASSED ✓")
    else:
        print("\n  SOME CHECKS FAILED ✗")

    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
