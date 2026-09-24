from unittest import TestCase
import math
import random

from owrx.fsk600 import Fsk600Decoder, Fsk600Context, parseMessages


def frame(addr, tail):
    a = format(addr, "06b")
    parity = "0" if a.count("1") % 2 else "1"   # odd parity over address + parity bit
    return "11" + a + parity + tail


def synth(bursts, fs=12000, baud=600, low=1300, high=1700, lead=0.08, gap=0.25, noise=0.0, seed=3):
    rnd = random.Random(seed)
    out, phase, t = [], 0.0, 0.0

    def emit(bit, duration):
        nonlocal phase, t
        start = len(out)
        t += duration
        f = high if bit == "1" else low
        for _ in range(int(round(t * fs)) - start):
            phase += 2 * math.pi * f / fs
            out.append(0.5 * math.sin(phase) + (rnd.gauss(0, noise) if noise else 0.0))

    for bits in bursts:
        emit("0", lead)
        for b in bits:
            emit(b, 1 / baud)
        emit("0", gap)
    return out


def decode(samples):
    dec = Fsk600Decoder(12000)
    ctx = Fsk600Context()
    msgs = []
    for i in range(0, len(samples), 1000):
        for t, bits, _ in dec.process(samples[i:i + 1000]):
            msgs += ctx.parse(bits, t / 12000)
    return msgs


A7 = frame(7, "0101" + "0000100")
B12 = frame(12, "000100")
C12 = frame(12, "00100")
D5 = frame(5, "0100")
DATA5 = frame(5, "0010000000000" + "10111001100")


class Fsk600Test(TestCase):
    def testAllTypes(self):
        msgs = decode(synth([A7, B12, C12, D5, DATA5]))
        self.assertEqual([(m["type"], m["address"]) for m in msgs],
                         [("A", 7), ("B", 12), ("C", 12), ("D", 5), ("DATA", 5)])
        self.assertEqual(msgs[0]["status"], 5)
        self.assertEqual(msgs[0]["role"], "status")
        self.assertEqual(msgs[1]["role"], "call")

    def testBackToBackInOneBurst(self):
        msgs = decode(synth([A7 + "00" + frame(63, "0010" + "0000100")]))
        self.assertEqual([(m["type"], m["address"], m.get("status")) for m in msgs], [("A", 7, 5), ("A", 63, 2)])

    def testNoise(self):
        msgs = decode(synth([A7, B12, C12], noise=0.1))
        self.assertEqual([(m["type"], m["address"]) for m in msgs], [("A", 7), ("B", 12), ("C", 12)])

    def testBadParityRejected(self):
        bad = "11" + format(7, "06b") + "1" + "0101" + "0000100"   # even parity
        self.assertEqual(decode(synth([bad])), [])

    def testWanderingToneAndNoiseGiveNothing(self):
        rnd = random.Random(5)
        fs, out, phase = 12000, [], 0.0
        f = 1000.0
        for _ in range(fs * 20):
            f = min(2400.0, max(300.0, f + rnd.gauss(0, 15)))
            phase += 2 * math.pi * f / fs
            out.append(0.4 * math.sin(phase) + rnd.gauss(0, 0.2))
        self.assertLessEqual(len(decode(out)), 1)

    def testAckFollowedByNextBurstInsideGap(self):
        # C, 11 idle bits, then more signal: after a B call to the unit it is a C
        tail = "00100" + "0" * 9 + "1011"
        ctx = Fsk600Context()
        ctx.parse(B12, 0.0)
        self.assertEqual([m["type"] for m in ctx.parse(frame(12, tail), 0.34)], ["C"])
        # without the call the same bits are a data message
        self.assertEqual([m["type"] for m in Fsk600Context().parse(frame(12, tail), 0.34)], ["DATA"])
