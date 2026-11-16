"""Synthesises the intro-theme score (no samples). usage: python3 music.py out.wav [--vivid] [--slow 1.5]

80 BPM, D lydian. Cues: 0.75 burst, 3.0 molecule, 6.0 sphere, 9.0 harmonics,
12.0 mandala (impact), 13.5 title.
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

K = float(sys.argv[sys.argv.index('--slow') + 1]) if '--slow' in sys.argv else 1.0   # time stretch of all cues
SR, DUR = 48000, 16.5 * K
n = int(SR * DUR)
t = np.arange(n) / SR
rng = np.random.default_rng(7)
mix = np.zeros((n, 2))      # dry bus
send = np.zeros((n, 2))     # reverb send

def hz(name):
    names = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) + 1) - 69) / 12)

def filt(x, kind, fc, order=2):
    return sosfilt(butter(order, fc, kind, fs=SR, output='sos'), x, axis=0)

def place(x, at, pan=0.0, gain=1.0, rev=0.3):
    i = int(at * K * SR)
    if i >= n: return
    x = x[: n - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    st = np.stack([x * l, x * r], 1) if x.ndim == 1 else x
    mix[i:i + len(x)] += st
    send[i:i + len(x)] += st * rev

def env(at, a, b, c, d):
    """0 before a, rises to 1 at b, holds to c, falls to 0 at d (smooth)."""
    a, b, c, d = a * K, b * K, c * K, d * K
    up = np.clip((at - a) / max(b - a, 1e-6), 0, 1); dn = np.clip((d - at) / max(d - c, 1e-6), 0, 1)
    s = lambda x: x * x * (3 - 2 * x)
    return s(up) * s(dn)

def piano(f, dur=4.0, bright=1.0):
    tt = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(tt)
    for k in range(1, 10):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > 9000: break
        y += np.sin(2 * np.pi * fk * tt + rng.uniform(0, 6.28)) * k ** (-1.9 / bright) * np.exp(-tt * (0.9 + 0.55 * k))
    y *= np.minimum(1, tt / 0.006)
    return y * np.minimum(1, (dur - tt) / 0.05)

def bowl(f, dur=7.0):
    tt = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(tt)
    for ratio, a, dec in [(1, 1, .45), (2.71, .55, .7), (5.15, .28, 1.1), (8.1, .12, 1.8)]:
        for det in (-0.6, 0.6):                                   # beating pairs
            y += a * np.sin(2 * np.pi * (f * ratio + det * ratio ** .5) * tt + rng.uniform(0, 6.28)) * np.exp(-tt * dec)
    return y * np.minimum(1, tt / 0.012)

def pad(notes, a, b, c, d, bright=(900, 2600)):
    """Detuned-saw string pad; filter opens from bright[0] to bright[1] over the note."""
    out = np.zeros((n, 2))
    for ch in range(2):
        y = np.zeros(n)
        for nm in notes:
            f = hz(nm)
            for det in rng.normal(0, 0.0035, 5):
                ph = rng.uniform(0, 1)
                vib = 1 + 0.0015 * np.sin(2 * np.pi * rng.uniform(4, 5.5) * t + rng.uniform(0, 6))
                y += 2 * ((np.cumsum(f * (1 + det) * vib) / SR + ph) % 1) - 1
        dark, lite = filt(y, 'low', bright[0], 4), filt(y, 'low', bright[1], 4)
        o = np.clip((t / K - a) / (d - a), 0, 1)
        out[:, ch] = (dark * (1 - o) + lite * o) * env(t, a, b, c, d) / len(notes)
    return out

def noise_swell(a, peak, end, lo, hi, gain):
    x = rng.normal(0, 1, (n, 2))
    x = filt(filt(x, 'high', lo), 'low', hi)
    a, peak, end = a * K, peak * K, end * K
    e = np.where(t < peak, np.clip((t - a) / (peak - a), 0, 1) ** 2.4, np.exp(-np.clip(t - peak, 0, None) / max((end - peak) / 4, 1e-3)))
    e[(t < a) | (t > end)] = 0
    return x * e[:, None] * gain

# --- drone + sub (roots: D, G, A, D) ---
for root, a, b, c, d, g in [('D1', 0.0, 2.5, 5.6, 6.6, .30), ('G1', 5.6, 6.4, 8.6, 9.6, .28), ('A1', 8.6, 9.6, 11.8, 12.1, .30), ('D1', 11.95, 12.05, 14.5, 16.5, .42)]:
    f = hz(root)
    y = (.7 * np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * 2 * f * t) + .12 * np.sin(2 * np.pi * 3 * f * t + 1)) * env(t, a, b, c, d) * g
    mix[:] += y[:, None]

# --- pads ---
mix_pad = (pad(['D2', 'A2', 'E3', 'F#3', 'A3'], 0.2, 3.0, 5.4, 6.6, (500, 1800)) * .50
           + pad(['G2', 'D3', 'F#3', 'A3', 'B3'], 5.5, 6.3, 8.5, 9.6, (900, 2400)) * .52
           + pad(['A2', 'E3', 'G3', 'B3', 'D4'], 8.5, 11.6, 11.9, 12.15, (900, 4200)) * .62
           + pad(['D2', 'A2', 'D3', 'F#3', 'A3', 'E4'], 11.95, 12.1, 14.3, 16.5, (3600, 1200)) * .80
           + pad(['A4', 'E5', 'F#5'], 11.95, 12.6, 14.0, 16.5, (5000, 2500)) * .16)
mix += mix_pad * .8; send += mix_pad * .5

# --- melody (one note per beat, rising to the downbeat at 12.0) ---
mel = [(.75, 'A4', .55), (1.5, 'D5', .5), (2.25, 'E5', .5), (3.0, 'F#5', .7), (3.75, 'A4', .45), (4.5, 'E5', .5), (5.25, 'D5', .45),
       (6.0, 'B4', .7), (6.75, 'D5', .5), (7.5, 'F#5', .55), (8.25, 'A5', .55), (9.0, 'E5', .7), (9.75, 'A5', .6), (10.5, 'B5', .65),
       (11.25, 'C#6', .7), (12.0, 'D6', .9), (12.0, 'A5', .6), (12.75, 'A5', .45), (13.5, 'F#5', .6), (13.5, 'D5', .45), (14.25, 'D5', .35), (15.0, 'A4', .28)]
dly = 0.5625
for k, (at, nm, g) in enumerate(mel):
    y = piano(hz(nm), 4.5)
    pan = .35 * (-1) ** k
    place(y, at, pan, g * .46, rev=.55)
    for r, fb in enumerate([.34, .16, .07]):                      # ping-pong dotted-eighth echoes
        place(filt(y, 'low', 2600 - 500 * r), at + dly * (r + 1), -pan if r % 2 == 0 else pan, g * .46 * fb, rev=.7)
for at, nm, g in [(0.75, 'D2', .5), (3.0, 'D3', .3), (6.0, 'G2', .42), (9.0, 'A2', .42), (12.0, 'D2', .8), (12.0, 'A2', .5), (12.0, 'D3', .5), (13.5, 'D3', .3)]:
    place(piano(hz(nm), 6.0, 1.3), at, 0, g * .5, rev=.4)

# --- singing bowl: the first point of light, and the mandala ---
place(bowl(hz('D4')), 0.3, -.1, .16, rev=.8)
place(bowl(hz('A4')), 0.75, .2, .07, rev=.8)
place(bowl(hz('D4')), 12.0, .1, .15, rev=.8)

# --- glass sparkle during message passing and geodesic flow ---
scale = ['D6', 'E6', 'F#6', 'A6', 'B6', 'D7', 'E7', 'F#7']
for a, b, dens in [(3.0, 5.1, 14), (6.2, 8.2, 20), (12.0, 14.5, 9)]:
    for at in np.sort(rng.uniform(a, b, int((b - a) * dens * K))):
        tt = np.arange(int(.5 * SR)) / SR
        f = hz(scale[rng.integers(len(scale))])
        place(np.sin(2 * np.pi * f * tt) * np.exp(-tt * 14) * np.minimum(1, tt / .003), at, rng.uniform(-.8, .8), rng.uniform(.012, .035), rev=1.2)

# --- transitions: soft air on each morph, riser into the downbeat ---
for a, pk, e, g in [(0.5, 0.78, 2.2, .028), (2.1, 2.95, 3.6, .02), (5.1, 5.95, 6.7, .022), (8.2, 8.95, 9.7, .022)]:
    w = noise_swell(a, pk, e, 500, 5000, g); mix += w; send += w * .8
r = noise_swell(10.4, 11.97, 12.0, 1500, 11000, .075) + noise_swell(9.8, 11.97, 12.0, 300, 1600, .05)
mix += r; send += r * .5
ts = t / K
gl = np.zeros(n); m = (ts > 10.5) & (ts < 12.0)                      # rising shimmer
for f0 in (hz('A4'), hz('E5'), hz('A5')):
    fr = f0 * np.ones(n)     # held pitch, swelling
    gl += np.sin(2 * np.pi * np.cumsum(fr) / SR)
gl *= m * np.clip((ts - 10.5) / 1.5, 0, 1) ** 3 * .035
mix += gl[:, None]; send += gl[:, None] * 1.0

# --- impact at 12.0 ---
tt = np.arange(int(3 * SR)) / SR
sub = np.sin(2 * np.pi * np.cumsum(36.7 + 50 * np.exp(-tt * 9)) / SR) * np.exp(-tt * 1.5) * np.minimum(1, tt / .004)
place(sub, 12.0, 0, .62, rev=.05)
th = filt(rng.normal(0, 1, len(tt)), 'low', 160, 4) * np.exp(-tt * 9)
place(th, 12.0, 0, .9, rev=.3)
air = filt(rng.normal(0, 1, (len(tt), 2)), 'high', 5000) * (np.exp(-tt * 1.6) * np.minimum(1, tt / .02))[:, None]
place(air, 12.0, 0, .022, rev=1.0)

# --- vivid variant: sixteenth-note arpeggio, soft kick, offbeat hats ---
if '--vivid' in sys.argv:
    def pluck(f, dur=.5, bright=1.0):
        tt = np.arange(int(dur * SR)) / SR; y = np.zeros_like(tt)
        for k in range(1, 9):
            if f * k > 8000: break
            y += np.sin(2 * np.pi * f * k * tt) / k * np.exp(-tt * (5 + 5 * k / bright))
        return y * np.minimum(1, tt / .003)
    arps = [(3.0, 6.0, ['D4', 'A4', 'E5', 'F#5', 'A5', 'F#5', 'E5', 'A4']),
            (6.0, 9.0, ['D4', 'G4', 'B4', 'F#5', 'A5', 'F#5', 'B4', 'G4']),
            (9.0, 11.625, ['E4', 'A4', 'B4', 'D5', 'G5', 'B5', 'G5', 'D5']),
            (12.0, 14.8, ['D4', 'A4', 'D5', 'F#5', 'A5', 'D6', 'A5', 'F#5'])]
    step = 0.1875
    for a, b, notes in arps:
        for k in range(int(round((b - a) / step))):
            at = a + k * step
            g = np.interp(at, [3.0, 6.0, 11.5, 12.0, 13.5, 14.8], [.25, .6, 1.0, 1.0, .7, .15])
            br = np.interp(at, [3.0, 11.5, 12.0, 14.8], [.6, 1.6, 1.8, .8])
            y = pluck(hz(notes[k % 8]), .5, br)
            pan = .55 * (-1) ** k
            place(y, at, pan, g * .085 * (1.25 if k % 4 == 0 else 1), rev=.6)
            place(y, at + step * 3, -pan, g * .03, rev=.9)
    tt = np.arange(int(.4 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(46 + 90 * np.exp(-tt * 38)) / SR) * np.exp(-tt * 11) * np.minimum(1, tt / .002)
    for at in list(np.arange(6.0, 11.3, .75)) + [12.75, 13.5]:
        place(kick, at, 0, .34 * np.interp(at, [6, 11.25], [.6, 1]), rev=.04)
    th = np.arange(int(.06 * SR)) / SR
    for at in list(np.arange(9.375, 11.6, .75)) + list(np.arange(12.375, 14.3, .75)):
        place(filt(rng.normal(0, 1, len(th)), 'high', 7000) * np.exp(-th * 90), at, .2, .045, rev=.5)

# --- reverb (synthetic hall) ---
L = int(4.2 * SR); ti = np.arange(L) / SR
ir = rng.normal(0, 1, (L, 2)) * np.exp(-ti * 1.75)[:, None]
ir = filt(filt(ir, 'low', 5200), 'high', 180); ir[: int(.018 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0))
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:n] for c in range(2)], 1)
out = mix + wet * .55

# --- master ---
out = filt(out, 'high', 24)
out *= env(t, 0, .02, 15.4, 16.45)[:, None]
out = np.tanh(out / np.abs(out).max() * 1.1)
out = out / np.abs(out).max() * 0.89
pcm = (out * 32767).astype('<i2')
import wave
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
rms = lambda a, b: 20 * np.log10(np.sqrt((out[int(a * SR):int(b * SR)] ** 2).mean()) + 1e-9)
print('rms dB by section:', [round(float(rms(a, a + 1.5 * K)), 1) for a in np.arange(0, DUR, 1.5 * K)])
