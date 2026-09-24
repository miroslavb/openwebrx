"""
Decoder for a proprietary 600 baud audio-FSK dispatch status protocol
(1300 Hz = 0 / idle, 1700 Hz = 1), as heard on a UHF analog repeater pair.

Observed message format (all fields MSB first):

    11 | address (6 bits) | odd parity over address | tail

    tail 'SSSS0000100'  A: unit status report (SSSS = 4-bit status code)
    tail '000100'       B: call from the dispatcher to the unit
    tail '00100'        C: acknowledgement from the unit
    tail '0100'         D: short unit message
    anything longer     DATA: unit/dispatcher data message

The tone is tracked by zero-crossing periods of the band-passed audio, which
is amplitude independent and cheap. Bits are recovered from the run lengths
between tone changes. A burst starts at the first 1700 Hz bit after the
1300 Hz idle tone and ends when the idle tone lasts long enough, or when the
signal stops being tonal (e.g. voice).

This module only needs the Python standard library.
"""

import math

IDLE, MARK = 0, 1


class Fsk600Decoder(object):
    def __init__(self, sampleRate: int = 12000, baudRate: float = 600, low: float = 1300, high: float = 1700):
        self.fs = float(sampleRate)
        self.T = self.fs / baudRate
        self.split = (low + high) / 2
        self.low, self.high = low, high
        self.minIdle = 3
        self.burstIdle = 0.0
        self.tol = 0.4 * (high - low)
        self.midCount = 0
        # 2nd order band-pass around the tone pair (RBJ cookbook, constant 0 dB peak gain)
        f0 = math.sqrt(low * high)
        q = f0 / (2.5 * (high - low))
        w0 = 2 * math.pi * f0 / self.fs
        alpha = math.sin(w0) / (2 * q)
        a0 = 1 + alpha
        self.b0, self.b2 = alpha / a0, -alpha / a0
        self.a1, self.a2 = -2 * math.cos(w0) / a0, (1 - alpha) / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self.dc = 0.0
        self.dcAlpha = 1.0 - math.exp(-2 * math.pi * 50.0 / self.fs)
        self.n = 0
        self.prev = 0.0
        self.lastCross = None
        self.hist = []          # last three half-period frequencies (median filter)
        self.level = None
        self.levelSince = 0.0
        self.runs = []          # (level, duration in samples) of the current burst
        self.inBurst = False
        self.tonalSince = None
        self.badCount = 0

    def process(self, samples) -> list:
        """Feed audio samples; returns a list of (time_in_samples, bit string, idle_bits_before) bursts."""
        out = []
        b0, b2, a1, a2 = self.b0, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        dc, dcAlpha = self.dc, self.dcAlpha
        n, prev = self.n, self.prev
        for x in samples:
            dc += (x - dc) * dcAlpha
            x -= dc
            y = b0 * x + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, x
            y2, y1 = y1, y
            if (y >= 0) != (prev >= 0):
                # sub-sample zero crossing time by linear interpolation
                t = n - 1 + prev / (prev - y) if prev != y else float(n)
                if self.lastCross is not None:
                    half = t - self.lastCross
                    if half > 0:
                        r = self._halfCycle(t, self.fs / (2 * half))
                        if r is not None:
                            out.append(r)
                self.lastCross = t
            prev = y
            n += 1
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.dc, self.n, self.prev = dc, n, prev
        # idle timeout also ends a burst when no crossings arrive (squelch closed)
        if self.inBurst and self.lastCross is not None and n - self.lastCross > 12 * self.T:
            r = self._endBurst(self.lastCross)
            if r is not None:
                out.append(r)
        return out

    def _halfCycle(self, t, f):
        h = self.hist
        h.append(f)
        if len(h) > 3:
            del h[0]
        fm = sorted(h)[len(h) // 2]
        # a half cycle counts as tonal only close to one of the two tones; the
        # crossover region is tolerated once (a tone change), voice wanders
        if abs(fm - self.low) <= self.tol or abs(fm - self.high) <= self.tol:
            self.midCount = 0
        elif self.low < fm < self.high and self.midCount < 1:
            self.midCount += 1
        else:
            fm = None
        if fm is None:
            self.badCount += 1
            if self.badCount >= 2:
                # not a tone any more (voice, noise): drop the current state
                result = self._endBurst(t) if self.inBurst else None
                self.level = None
                self.tonalSince = None
                return result
            return None
        self.badCount = 0
        if self.tonalSince is None:
            self.tonalSince = t
        lvl = MARK if fm > self.split else IDLE
        if self.level is None:
            self.level, self.levelSince = lvl, t
            return None
        if lvl == self.level:
            if self.inBurst and lvl == IDLE and t - self.levelSince > 12 * self.T:
                return self._endBurst(t)
            return None
        # tone change: close the previous run
        dur = t - self.levelSince
        result = None
        if not self.inBurst:
            # start of a message: first mark after at least 3 bits of stable idle tone
            if self.level == IDLE and lvl == MARK and dur >= self.minIdle * self.T:
                self.inBurst = True
                self.runs = []
                self.burstIdle = dur / self.T
        else:
            self.runs.append((self.level, dur))
        self.level, self.levelSince = lvl, t
        return result

    def _endBurst(self, t):
        self.inBurst = False
        runs = self.runs
        self.runs = []
        if self.level == MARK and self.levelSince is not None:
            # burst ended while on mark: close that run too
            runs = runs + [(MARK, t - self.levelSince)]
        bits = ""
        for lvl, dur in runs:
            k = int(round(dur / self.T))
            if k <= 0:
                continue
            bits += ("1" if lvl == MARK else "0") * min(k, 64)
        bits = bits.rstrip("0")
        if not bits:
            return None
        # every observed message ends in '1' followed by two idle bits
        return (t, bits + "00", self.burstIdle)


ROLES = {"A": "status", "B": "call", "C": "ack", "D": "short", "DATA": "data"}


def parseMessages(bits: str, called=frozenset()) -> list:
    """
    Split a recovered burst into messages. `called` holds the addresses that got
    a B call in the last second (see Fsk600Context). Fixed-length types are taken by their
    known tails; a remainder that is not a known type is reported as DATA when it
    is long enough. Parsing stops at the first invalid start (no '11' sync or
    failed parity).
    """
    out = []
    while True:
        bits = bits.lstrip("0")
        if len(bits) < 13:
            return out
        m, used = _parseOne(bits, called)
        if m is None:
            # no resynchronisation inside a burst: hunting for '11' in what is
            # left mostly finds false messages (checked against captures)
            return out
        out.append(m)
        bits = bits[used:]


def _parseOne(bits: str, called=frozenset()):
    if len(bits) < 13 or not bits.startswith("11"):
        return None, 0
    addr = int(bits[2:8], 2)
    if (bin(addr).count("1") + int(bits[8])) % 2 != 1:
        return None, 0
    tail = bits[9:]
    out = {"address": addr}
    # every observed data body starts with '0010' followed by 9..11 zeros before
    # its payload. A C message ('00100') followed by idle and the next burst
    # within the burst gap has 13 or more zeros there instead.
    # Bits alone cannot tell "C + idle + next burst" from a data body, so the
    # protocol context decides: C answers a B call to the same unit ~0.34 s
    # earlier (the caller passes the recently called addresses).
    if tail.startswith("0010") and "1" in tail[4:] and addr not in called:
        zeros = len(tail[4:]) - len(tail[4:].lstrip("0"))
        if 9 <= zeros <= 11:
            return _data(addr, bits)
    if len(tail) >= 11 and tail[4:11] == "0000100":
        out["type"], used = "A", 20
        out["status"] = int(tail[:4], 2)
        out["message"] = "status %s (%d)" % (tail[:4], out["status"])
    elif tail.startswith("000100"):
        out["type"], used, out["message"] = "B", 15, ""
    elif tail.startswith("00100"):
        out["type"], used, out["message"] = "C", 14, ""
    elif tail.startswith("0100") and (len(tail) == 4 or tail[4:6] == "00"):
        out["type"], used, out["message"] = "D", 13, ""
    elif len(tail) >= 12:
        return _data(addr, bits)
    else:
        return None, 0
    out["bits"] = bits[:used]
    out["role"] = ROLES[out["type"]]
    return out, used


def _data(addr: int, bits: str):
    out = {"address": addr, "type": "DATA", "bits": bits, "role": ROLES["DATA"], "message": _words9(bits[9:])}
    return out, len(bits)


def parseMessage(bits: str):
    """Parse a single message; returns the first valid one or None."""
    m = parseMessages(bits)
    return m[0] if m else None


def _words9(body: str) -> str:
    """Best-effort view of a data body as 9-bit words (marker bit + 8 data bits)."""
    best = None
    for k in range(9):
        n = (len(body) - k) // 9
        if n < 1:
            continue
        hit = sum(body[k + 9 * j] == "1" for j in range(n)) / n
        if best is None or hit > best[0]:
            best = (hit, k, n)
    if best is None or best[0] < 0.75:
        return "raw " + body
    hit, k, n = best
    return " ".join("%02X" % int(body[k + 9 * j + 1:k + 9 * j + 9], 2) for j in range(n)) + "  (raw %s)" % body


class Fsk600Context(object):
    """Tracks recent B calls so that C acknowledgements can be told from data."""

    def __init__(self, window: float = 1.0):
        self.window = window
        self.calls = {}

    def parse(self, bits: str, now: float) -> list:
        self.calls = {a: t for a, t in self.calls.items() if now - t <= self.window}
        msgs = parseMessages(bits, frozenset(self.calls))
        for m in msgs:
            if m["type"] == "B":
                self.calls[m["address"]] = now
        return msgs
