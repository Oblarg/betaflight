"""
Betaflight blackbox (.bbl / .bfl) decoder.

Decodes data version 2 logs as written by Betaflight 4.x into numpy arrays,
with no dependency beyond numpy.  Field values are the raw logged integers
(deg/s gyro, centivolt vbat, DShot motor command, eRPM/100), identical to a
Blackbox Explorer CSV export.  A file may hold several logs (one per arm);
each becomes one `Log`.

Usage as a library:

    from bbl import read_logs
    for log in read_logs("btfl_all.bbl"):
        t = log["time"]; gyro = log["gyroADC[0]"]

Usage as a tool (writes one Explorer-style CSV per log):

    python3 bbl.py LOG.bbl [OUT_PREFIX]

Corrupt frames are handled the way blackbox-tools does: a frame is only
accepted if it is followed by a valid frame marker and its iteration/time
advance sanely; after a bad frame, P-frames are dropped until the next I-frame.
"""

import hashlib
import os
import pickle
import struct
import sys

import numpy as np

LOG_START = b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"
LOG_END_MSG = b"End of log\x00"

ENC_SIGNED_VB = 0
ENC_UNSIGNED_VB = 1
ENC_NEG_14BIT = 3
ENC_TAG8_8SVB = 6
ENC_TAG2_3S32 = 7
ENC_TAG8_4S16 = 8
ENC_NULL = 9

PRED_ZERO = 0
PRED_PREVIOUS = 1
PRED_STRAIGHT_LINE = 2
PRED_AVERAGE_2 = 3
PRED_MINTHROTTLE = 4
PRED_MOTOR_0 = 5
PRED_INCREMENT = 6
PRED_1500 = 8
PRED_VBATREF = 9
PRED_MINMOTOR = 11

EVENT_SYNC_BEEP = 0
EVENT_INFLIGHT_ADJUSTMENT = 13
EVENT_LOGGING_RESUME = 14
EVENT_DISARM = 15
EVENT_FLIGHTMODE = 30
EVENT_LOG_END = 255

MAX_TIME_JUMP_US = 10 * 1000000
MAX_ITERATION_JUMP = 500 * 10

FRAME_MARKERS = frozenset(b"IPSEGH")


class CorruptFrame(Exception):
    pass


class Log:
    """One decoded log: header dict, main-frame columns, events."""

    def __init__(self, header, names, data, events, stats):
        self.header = header
        self.names = names
        self.data = data
        self.events = events
        self.stats = stats
        self._index = {n: i for i, n in enumerate(names)}

    def __getitem__(self, name):
        return self.data[:, self._index[name]]

    def __contains__(self, name):
        return name in self._index

    def __len__(self):
        return self.data.shape[0]

    def header_ints(self, key):
        return [int(x) for x in self.header[key].split(",")]

    @property
    def craft(self):
        return self.header.get("Craft name", "")

    @property
    def loop_hz(self):
        """PID loop rate: 1e6 / (gyro looptime · pid_process_denom)."""
        return 1e6 / (int(self.header["looptime"]) * int(self.header.get("pid_process_denom", 1)))

    @property
    def log_hz(self):
        """Mean main-frame rate, from the logged timestamps."""
        t = self["time"]
        return (len(t) - 1) / ((t[-1] - t[0]) * 1e-6) if len(t) > 1 else 0.0


class _Defs:
    def __init__(self, names, signed, predictor, encoding):
        self.names = names
        self.signed = signed
        self.predictor = predictor
        self.encoding = encoding


def _wrap(value, signed):
    value &= 0xFFFFFFFF
    if signed and value & 0x80000000:
        value -= 1 << 32
    return value


def _sign_extend(value, bits):
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


class _Stream:
    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf, pos, end):
        self.buf = buf
        self.pos = pos
        self.end = end

    def byte(self):
        if self.pos >= self.end:
            raise CorruptFrame("eof")
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def uvb(self):
        result = 0
        for shift in range(0, 35, 7):
            b = self.byte()
            result |= (b & 0x7F) << shift
            if b < 0x80:
                return result
        raise CorruptFrame("varint too long")

    def svb(self):
        u = self.uvb()
        return (u >> 1) ^ -(u & 1)

    def tag8_8svb(self, count):
        if count == 1:
            return [self.svb()]
        header = self.byte()
        out = []
        for _ in range(count):
            out.append(self.svb() if header & 1 else 0)
            header >>= 1
        return out

    def tag2_3s32(self):
        lead = self.byte()
        kind = lead >> 6
        if kind == 0:
            return [_sign_extend(lead >> 4, 2), _sign_extend(lead >> 2, 2), _sign_extend(lead, 2)]
        if kind == 1:
            b = self.byte()
            return [_sign_extend(lead, 4), _sign_extend(b >> 4, 4), _sign_extend(b, 4)]
        if kind == 2:
            b1 = self.byte()
            b2 = self.byte()
            return [_sign_extend(lead, 6), _sign_extend(b1, 6), _sign_extend(b2, 6)]
        out = []
        selector = lead
        for _ in range(3):
            width = selector & 3
            if width == 0:
                out.append(_sign_extend(self.byte(), 8))
            elif width == 1:
                b1 = self.byte()
                out.append(_sign_extend(b1 | (self.byte() << 8), 16))
            elif width == 2:
                b1 = self.byte()
                b2 = self.byte()
                out.append(_sign_extend(b1 | (b2 << 8) | (self.byte() << 16), 24))
            else:
                b1 = self.byte()
                b2 = self.byte()
                b3 = self.byte()
                out.append(_sign_extend(b1 | (b2 << 8) | (b3 << 16) | (self.byte() << 24), 32))
            selector >>= 2
        return out

    def tag8_4s16(self):
        selector = self.byte()
        out = []
        nibble = False
        buffer = 0
        for _ in range(4):
            kind = selector & 3
            if kind == 0:
                out.append(0)
            elif kind == 1:
                if not nibble:
                    buffer = self.byte()
                    out.append(_sign_extend(buffer >> 4, 4))
                else:
                    out.append(_sign_extend(buffer, 4))
                nibble = not nibble
            elif kind == 2:
                if not nibble:
                    out.append(_sign_extend(self.byte(), 8))
                else:
                    hi = (buffer << 4) & 0xFF
                    buffer = self.byte()
                    out.append(_sign_extend(hi | (buffer >> 4), 8))
            else:
                if not nibble:
                    b1 = self.byte()
                    out.append(_sign_extend((b1 << 8) | self.byte(), 16))
                else:
                    b1 = self.byte()
                    b2 = self.byte()
                    out.append(_sign_extend(((buffer << 12) | (b1 << 4) | (b2 >> 4)) & 0xFFFF, 16))
                    buffer = b2
            selector >>= 2
        return out

    def raw_values(self, encodings):
        """Read one frame's worth of encoded values (before prediction)."""
        out = []
        n = len(encodings)
        i = 0
        while i < n:
            enc = encodings[i]
            if enc == ENC_SIGNED_VB:
                out.append(self.svb())
                i += 1
            elif enc == ENC_UNSIGNED_VB:
                out.append(self.uvb())
                i += 1
            elif enc == ENC_NEG_14BIT:
                out.append(-_sign_extend(self.uvb(), 14))
                i += 1
            elif enc == ENC_NULL:
                out.append(0)
                i += 1
            elif enc == ENC_TAG8_8SVB:
                j = i + 1
                while j < i + 8 and j < n and encodings[j] == ENC_TAG8_8SVB:
                    j += 1
                out.extend(self.tag8_8svb(j - i))
                i = j
            elif enc == ENC_TAG2_3S32:
                out.extend(self.tag2_3s32())
                i += 3
            elif enc == ENC_TAG8_4S16:
                out.extend(self.tag8_4s16())
                i += 4
            else:
                raise CorruptFrame(f"unsupported encoding {enc}")
        return out[:n]


def _parse_header(buf, pos, end):
    header = {}
    while pos < end and buf[pos:pos + 2] == b"H ":
        nl = buf.find(b"\n", pos, end)
        if nl < 0:
            nl = end
        line = buf[pos + 2:nl].decode("latin-1")
        key, _, value = line.partition(":")
        header[key] = value
        pos = nl + 1
    return header, pos


def _defs(header, kind, names=None, signed=None):
    def ints(key):
        v = header.get(f"Field {kind} {key}")
        return [int(x) for x in v.split(",")] if v else None

    names = header[f"Field {kind} name"].split(",") if names is None else names
    signed = ints("signed") if signed is None else signed
    return _Defs(names, signed or [0] * len(names), ints("predictor"), ints("encoding"))


def _decode_log(buf, pos, end):
    header, pos = _parse_header(buf, pos, end)
    if header.get("Data version") != "2":
        raise ValueError(f"unsupported blackbox data version {header.get('Data version')!r}")

    idef = _defs(header, "I")
    pdef = _defs(header, "P", names=idef.names, signed=idef.signed)
    sdef = _defs(header, "S") if "Field S name" in header else None
    nfields = len(idef.names)

    i_interval = int(header.get("I interval", 1))
    p_interval = header.get("P interval", "1")
    if "/" in p_interval:
        p_num, p_denom = (int(x) for x in p_interval.split("/"))
    else:
        p_num, p_denom = 1, int(p_interval)
    minthrottle = int(header.get("minthrottle", 0))
    vbatref = int(header.get("vbatref", 0))
    minmotor = int(header.get("motorOutput", "0").split(",")[0])
    idx_iter = idef.names.index("loopIteration")
    idx_time = idef.names.index("time")
    idx_motor0 = idef.names.index("motor[0]") if "motor[0]" in idef.names else -1

    def should_have_frame(index):
        return (index % i_interval + p_num - 1) % p_denom < p_num

    def predict(defs, raw, prev, prev2, skipped):
        cur = [0] * nfields
        for i in range(nfields):
            v = raw[i]
            p = defs.predictor[i]
            if p == PRED_ZERO:
                pass
            elif p == PRED_PREVIOUS:
                if prev is not None:
                    v += prev[i]
            elif p == PRED_STRAIGHT_LINE:
                if prev is not None:
                    v += 2 * prev[i] - prev2[i]
            elif p == PRED_AVERAGE_2:
                if prev is not None:
                    if defs.signed[i]:
                        s = _wrap(prev[i] + prev2[i], True)
                        v += int(s / 2)
                    else:
                        v += ((prev[i] & 0xFFFFFFFF) + (prev2[i] & 0xFFFFFFFF)) // 2
            elif p == PRED_MINTHROTTLE:
                v += minthrottle
            elif p == PRED_MOTOR_0:
                v += cur[idx_motor0]
            elif p == PRED_INCREMENT:
                if prev is not None:
                    v += skipped + 1 + prev[i]
            elif p == PRED_1500:
                v += 1500
            elif p == PRED_VBATREF:
                v += vbatref
            elif p == PRED_MINMOTOR:
                v += minmotor
            else:
                raise CorruptFrame(f"unsupported predictor {p}")
            cur[i] = _wrap(v, defs.signed[i])
        return cur

    stream = _Stream(buf, pos, end)
    rows = []
    slow_names = sdef.names if sdef else []
    slow_cur = [0] * len(slow_names)
    slow_rows = []
    events = []
    prev = prev2 = None
    valid = False
    last_iter = -1
    last_time = -1
    stats = {"I": 0, "P": 0, "S": 0, "E": 0, "corrupt": 0, "dropped_p": 0}

    def sane(frame):
        if last_iter < 0:
            return True
        it, t = frame[idx_iter], frame[idx_time]
        return (last_iter <= it <= last_iter + MAX_ITERATION_JUMP
                and last_time <= t <= last_time + MAX_TIME_JUMP_US)

    def next_is_marker():
        return stream.pos >= end or buf[stream.pos] in FRAME_MARKERS

    while stream.pos < end:
        start = stream.pos
        kind = buf[start]
        stream.pos += 1
        try:
            if kind == ord("I"):
                raw = stream.raw_values(idef.encoding)
                frame = predict(idef, raw, prev, prev, 0)
                if not next_is_marker() or not sane(frame):
                    raise CorruptFrame("bad I frame")
                prev = prev2 = frame
                valid = True
                last_iter, last_time = frame[idx_iter], frame[idx_time]
                rows.append(frame)
                slow_rows.append(list(slow_cur))
                stats["I"] += 1
            elif kind == ord("P"):
                raw = stream.raw_values(pdef.encoding)
                if not valid:
                    if not next_is_marker():
                        raise CorruptFrame("bad P frame")
                    stats["dropped_p"] += 1
                    continue
                skipped = 0
                index = last_iter + 1
                while not should_have_frame(index):
                    skipped += 1
                    index += 1
                frame = predict(pdef, raw, prev, prev2, skipped)
                if not next_is_marker() or not sane(frame):
                    raise CorruptFrame("bad P frame")
                prev2, prev = prev, frame
                last_iter, last_time = frame[idx_iter], frame[idx_time]
                rows.append(frame)
                slow_rows.append(list(slow_cur))
                stats["P"] += 1
            elif kind == ord("S") and sdef is not None:
                raw = stream.raw_values(sdef.encoding)
                if not next_is_marker():
                    raise CorruptFrame("bad S frame")
                slow_cur = [_wrap(v, s) for v, s in zip(raw, sdef.signed)]
                if stats["S"] == 0:
                    slow_rows = [list(slow_cur) for _ in slow_rows]
                stats["S"] += 1
            elif kind == ord("E"):
                etype = stream.byte()
                if etype == EVENT_SYNC_BEEP:
                    events.append(("sync_beep", stream.uvb()))
                elif etype == EVENT_INFLIGHT_ADJUSTMENT:
                    func = stream.byte()
                    if func & 0x80:
                        b = bytes(stream.byte() for _ in range(4))
                        events.append(("adjustment", func & 0x7F, struct.unpack("<f", b)[0]))
                    else:
                        events.append(("adjustment", func, stream.svb()))
                elif etype == EVENT_LOGGING_RESUME:
                    it, t = stream.uvb(), stream.uvb()
                    events.append(("logging_resume", it, t))
                    last_iter, last_time = it, t
                    valid = False
                elif etype == EVENT_DISARM:
                    events.append(("disarm", stream.uvb()))
                elif etype == EVENT_FLIGHTMODE:
                    events.append(("flightmode", stream.uvb(), stream.uvb()))
                elif etype == EVENT_LOG_END:
                    if buf[stream.pos:stream.pos + len(LOG_END_MSG)] != LOG_END_MSG:
                        raise CorruptFrame("bad log end")
                    events.append(("log_end",))
                    stats["E"] += 1
                    break
                else:
                    raise CorruptFrame(f"unknown event {etype}")
                if not next_is_marker():
                    raise CorruptFrame("bad E frame")
                stats["E"] += 1
            else:
                raise CorruptFrame(f"unknown frame type {kind!r}")
        except CorruptFrame:
            stats["corrupt"] += 1
            valid = False
            stream.pos = start + 1
            while stream.pos < end and buf[stream.pos] not in FRAME_MARKERS:
                stream.pos += 1

    names = list(idef.names) + list(slow_names)
    if rows:
        data = np.hstack([np.asarray(rows, dtype=np.int64),
                          np.asarray(slow_rows, dtype=np.int64).reshape(len(rows), len(slow_names))])
    else:
        data = np.zeros((0, len(names)), dtype=np.int64)
    return Log(header, names, data, events, stats)


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "whippyflight", "bbl")
CACHE_VERSION = 1


def _cache_path(path):
    st = os.stat(path)
    key = f"{CACHE_VERSION}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".pkl")


def read_logs(path, cache=True):
    """Decode every log in a blackbox file (memoized in ~/.cache by path, size and mtime)."""
    if cache:
        cp = _cache_path(path)
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                return pickle.load(f)
    logs = _read_logs(path)
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f"{cp}.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
    return logs


def _read_logs(path):
    with open(path, "rb") as f:
        buf = f.read()
    starts = []
    pos = buf.find(LOG_START)
    while pos >= 0:
        starts.append(pos)
        pos = buf.find(LOG_START, pos + 1)
    logs = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(buf)
        log = _decode_log(buf, start, end)
        if len(log):
            logs.append(log)
    return logs


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else path
    for n, log in enumerate(read_logs(path), 1):
        out = f"{prefix}.{n:02d}.csv"
        with open(out, "w") as f:
            f.write(",".join(f'"{name}"' for name in log.names) + "\n")
            np.savetxt(f, log.data, fmt="%d", delimiter=",")
        print(f"{out}: {log.craft} {len(log)} frames, {log.log_hz:.0f} Hz log, "
              f"{log.loop_hz:.0f} Hz loop, {log.stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
