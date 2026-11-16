"""Original theme for the 24.75 s ("breathe") cut. usage: python3 score.py out.wav

Eleven bars of 4/4 in D major at 106.7 BPM (bar = 2.25 s), synthesised from
scratch. Picture cues fall on bar lines: molecule bar 3, sphere bar 5,
harmonics bar 7, mandala bar 9 (arrival), title bar 10.

  bars  1-2   Dsus2          bowl, drone, the motif stated alone (A D E F#)
  bars  3-4   Bm7 | Gmaj7    piano melody over a quiet arpeggio
  bars  5-6   D/F# | A       cello enters, pulse begins, melody an octave up
  bars  7-8   Em9 | A7sus4   climb to the leading note, riser
  bar   9     D              arrival
  bars 10-11  Dadd9 | G/D    the motif answered and resolved under the title
"""
import sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR, BAR, BEAT = 48000, 2.25, 0.5625
DUR = 11 * BAR
n = int(SR * DUR); t = np.arange(n) / SR
rng = np.random.default_rng(11)
mix = np.zeros((n, 2)); send = np.zeros((n, 2))

def hz(name):
    names = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) + 1) - 69) / 12)

def filt(x, kind, fc, order=2):
    return sosfilt(butter(order, fc, kind, fs=SR, output='sos'), x, axis=0)

def place(x, at, pan=0.0, gain=1.0, rev=0.3):
    i = int(at * SR)
    if i >= n or i < 0: return
    x = x[: n - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    st = np.stack([x * l, x * r], 1) if x.ndim == 1 else x
    mix[i:i + len(x)] += st; send[i:i + len(x)] += st * rev

def env(a, b, c, d):
    s = lambda x: x * x * (3 - 2 * x)
    return s(np.clip((t - a) / max(b - a, 1e-6), 0, 1)) * s(np.clip((d - t) / max(d - c, 1e-6), 0, 1))

def piano(f, dur=4.5, bright=1.0):
    tt = np.arange(int(dur * SR)) / SR; y = np.zeros_like(tt)
    for k in range(1, 10):
        fk = f * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > 9000: break
        y += np.sin(2 * np.pi * fk * tt + rng.uniform(0, 6.28)) * k ** (-1.9 / bright) * np.exp(-tt * (0.9 + 0.55 * k))
    return y * np.minimum(1, tt / 0.006) * np.minimum(1, (dur - tt) / 0.05)

def pluck(f, bright=1.0, dur=.5):
    tt = np.arange(int(dur * SR)) / SR; y = np.zeros_like(tt)
    for k in range(1, 9):
        if f * k > 8000: break
        y += np.sin(2 * np.pi * f * k * tt) / k * np.exp(-tt * (5 + 5 * k / bright))
    return y * np.minimum(1, tt / .003)

def bowl(f, dur=7.0):
    tt = np.arange(int(dur * SR)) / SR; y = np.zeros_like(tt)
    for ratio, a, dec in [(1, 1, .45), (2.71, .55, .7), (5.15, .28, 1.1), (8.1, .12, 1.8)]:
        for det in (-0.6, 0.6):
            y += a * np.sin(2 * np.pi * (f * ratio + det * ratio ** .5) * tt + rng.uniform(0, 6.28)) * np.exp(-tt * dec)
    return y * np.minimum(1, tt / 0.012)

def strings(notes, a, b, c, d, lo, hi, voices=5, vib=.0015):
    """Detuned-saw ensemble; the filter opens from lo to hi across the note."""
    out = np.zeros((n, 2)); e = env(a, b, c, d); o = np.clip((t - a) / (d - a), 0, 1)
    for ch in range(2):
        y = np.zeros(n)
        for nm in notes:
            f = hz(nm)
            for det in rng.normal(0, 0.0035, voices):
                v = 1 + vib * np.sin(2 * np.pi * rng.uniform(4.2, 5.6) * t + rng.uniform(0, 6))
                y += 2 * ((np.cumsum(f * (1 + det) * v) / SR + rng.uniform(0, 1)) % 1) - 1
        out[:, ch] = (filt(y, 'low', lo, 4) * (1 - o) + filt(y, 'low', hi, 4) * o) * e / len(notes)
    return out

def swell(a, peak, end, lo, hi, gain):
    x = filt(filt(rng.normal(0, 1, (n, 2)), 'high', lo), 'low', hi)
    e = np.where(t < peak, np.clip((t - a) / (peak - a), 0, 1) ** 2.4, np.exp(-np.clip(t - peak, 0, None) / max((end - peak) / 4, 1e-3)))
    e[(t < a) | (t > end)] = 0
    return x * e[:, None] * gain

B = lambda bar, beat=0.0: (bar - 1) * BAR + beat * BEAT           # bar/beat -> seconds

# --- harmony: (bar, pad voicing, bass, arpeggio, pad gain, filter lo, hi) ---
H = [(1,  ['D2', 'A2', 'E3', 'A3'],              'D1',  None, .34, 450, 1300),
     (2,  ['D2', 'A2', 'E3', 'A3', 'D4'],        'D1',  None, .40, 600, 1700),
     (3,  ['B2', 'F#3', 'A3', 'D4'],             'B1',  ['B3', 'F#4', 'A4', 'D5', 'F#5', 'D5', 'A4', 'F#4'], .46, 800, 2000),
     (4,  ['G2', 'D3', 'F#3', 'B3'],             'G1',  ['G3', 'D4', 'F#4', 'B4', 'D5', 'B4', 'F#4', 'D4'], .48, 800, 2200),
     (5,  ['F#2', 'D3', 'A3', 'E4'],             'F#1', ['D4', 'A4', 'D5', 'E5', 'F#5', 'E5', 'D5', 'A4'], .54, 1000, 2600),
     (6,  ['A2', 'E3', 'A3', 'C#4'],             'A1',  ['A3', 'E4', 'A4', 'C#5', 'E5', 'C#5', 'A4', 'E4'], .56, 1100, 2900),
     (7,  ['E2', 'B2', 'G3', 'D4', 'F#4'],       'E1',  ['E4', 'G4', 'B4', 'D5', 'F#5', 'D5', 'B4', 'G4'], .58, 1200, 3200),
     (8,  ['A2', 'E3', 'G3', 'D4'],              'A1',  ['A3', 'D4', 'E4', 'G4', 'A4', 'D5', 'E5', 'G5'], .68, 1400, 4600),
     (9,  ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4'], 'D1',  ['D4', 'F#4', 'A4', 'D5', 'F#5', 'A5', 'F#5', 'D5'], .86, 4200, 3000),
     (10, ['D2', 'A2', 'D3', 'F#3', 'A3', 'E4'], 'D1',  ['D4', 'A4', 'E5', 'F#5', 'A5', 'F#5', 'E5', 'A4'], .68, 2800, 1800),
     (11, ['D2', 'G2', 'B2', 'D3', 'G3'],        'D1',  ['D4', 'G4', 'B4', 'D5', 'G5', 'D5', 'B4', 'G4'], .52, 1800, 1200)]
for bar, notes, bass, arp, g, lo, hi in H:
    a = B(bar); last = bar == 11
    p = strings(notes, a - .25, a + (.9 if bar < 3 else .3), a + BAR - .2 if not last else a + BAR * .6, a + BAR + .35 if not last else DUR, lo, hi) * g
    mix[:] += p * .8; send[:] += p * .5
    f = hz(bass); e = env(a - .05, a + .12, a + BAR - .25 if not last else a + BAR * .5, a + BAR + .1 if not last else DUR)
    sub = (.6 * np.sin(2 * np.pi * f * t) + .55 * np.sin(2 * np.pi * 2 * f * t) + .14 * np.sin(2 * np.pi * 3 * f * t)) * e * (.2 + .22 * g)
    mix[:] += sub[:, None]
    if bar >= 2: place(piano(hz(notes[0]), 5, 1.2), a, 0, .16 + .2 * g, rev=.4)       # left hand
    if arp:
        mute = bar == 8
        for k in range(16):
            if mute and k >= 12: break                                               # breath before the arrival
            at = a + k * BEAT / 4
            lvl = {3: .35, 4: .45, 5: .6, 6: .7, 7: .8, 8: .95, 9: 1.0, 10: .7, 11: .32}[bar]
            y = pluck(hz(arp[k % 8]), .5 + lvl * 1.2); pan = .55 * (-1) ** k
            place(y, at, pan, lvl * .08 * (1.3 if k % 4 == 0 else 1), rev=.6)
            place(y, at + BEAT * .75, -pan, lvl * .028, rev=.9)

# --- melody (bar, beat, note, velocity); the motif is A D E F# ---
MEL = [(1, 2, 'A4', .5), (2, 0, 'D5', .5), (2, 1, 'E5', .5), (2, 2, 'F#5', .62),
       (3, 0, 'F#5', .7), (3, 1, 'E5', .5), (3, 1.5, 'D5', .48), (3, 2, 'B4', .55), (3, 3, 'D5', .5),
       (4, 0, 'B4', .62), (4, 1, 'D5', .5), (4, 2, 'F#5', .6), (4, 3, 'E5', .5),
       (5, 0, 'A5', .75), (5, 1, 'F#5', .52), (5, 1.5, 'E5', .5), (5, 2, 'D5', .55), (5, 3, 'F#5', .55),
       (6, 0, 'E5', .65), (6, 1, 'A5', .6), (6, 2, 'B5', .7), (6, 3.5, 'A5', .45),
       (7, 0, 'G5', .7), (7, 1, 'B5', .62), (7, 1.5, 'A5', .55), (7, 2, 'G5', .58), (7, 3, 'B5', .62),
       (8, 0, 'A5', .7), (8, 1, 'B5', .7), (8, 2, 'C#6', .8), (8, 3, 'E6', .85),
       (9, 0, 'D6', 1.0), (9, 0, 'F#5', .6), (9, 1, 'A5', .55), (9, 2, 'F#5', .55), (9, 3, 'A5', .5),
       (10, 0, 'F#5', .7), (10, 1, 'E5', .55), (10, 2, 'D5', .6),
       (11, 0, 'B4', .5), (11, 1, 'D5', .48), (11, 2, 'E5', .46), (11, 3.5, 'D5', .5), (11, 3.5, 'A4', .36)]
for k, (bar, beat, nm, v) in enumerate(MEL):
    y = piano(hz(nm), 4.5); pan = .3 * (-1) ** k; at = B(bar, beat)
    place(y, at, pan, v * .44, rev=.55)
    for r, fb in enumerate([.3, .13, .05]):
        place(filt(y, 'low', 2600 - 500 * r), at + BEAT * .75 * (r + 1), -pan if r % 2 == 0 else pan, v * .44 * fb, rev=.7)

# --- cello counter-line, bars 5-9 ---
for nm, a, d, g in [('F#3', B(5), B(6), .5), ('E3', B(6), B(7), .52), ('G3', B(7), B(7, 2), .55), ('B3', B(7, 2), B(8), .58), ('A3', B(8), B(8, 2), .62), ('C#4', B(8, 2), B(9), .7), ('D4', B(9), B(10, 2), .7)]:
    c = strings([nm], a - .05, a + .25, d - .15, d + .3, 700, 1900, voices=3, vib=.004) * g * .5
    mix[:] += c * .9; send[:] += c * .6

# --- pulse: half-time kick from bar 5, hats in bars 7-9 ---
tk = np.arange(int(.4 * SR)) / SR
kick = np.sin(2 * np.pi * np.cumsum(46 + 90 * np.exp(-tk * 38)) / SR) * np.exp(-tk * 11) * np.minimum(1, tk / .002)
for bar in range(5, 11):
    for beat in ((0, 2) if bar < 7 else (0, 1, 2, 3) if bar < 9 else (0, 2)):
        if bar == 8 and beat == 3: continue
        place(kick, B(bar, beat), 0, .2 + .03 * (bar - 5) if bar < 9 else .3, rev=.04)
th = np.arange(int(.06 * SR)) / SR
for bar in (7, 8, 9):
    for beat in (.5, 1.5, 2.5, 3.5):
        if bar == 8 and beat > 3: continue
        place(filt(rng.normal(0, 1, len(th)), 'high', 7000) * np.exp(-th * 90), B(bar, beat), .2, .04, rev=.5)

# --- colour: bowl, sparkle, air, riser, arrival ---
place(bowl(hz('D4')), 0.3, -.1, .16, rev=.8)
place(bowl(hz('A4')), B(1, 2), .2, .07, rev=.8)
place(bowl(hz('D4')), B(9), .1, .15, rev=.8)
scale = ['D6', 'E6', 'F#6', 'A6', 'B6', 'D7', 'E7', 'F#7']
for a, b, dens in [(B(3), B(5), 9), (B(5), B(7), 13), (B(9), B(11), 8)]:
    for at in np.sort(rng.uniform(a, b, int((b - a) * dens))):
        ts = np.arange(int(.5 * SR)) / SR
        place(np.sin(2 * np.pi * hz(scale[rng.integers(len(scale))]) * ts) * np.exp(-ts * 14) * np.minimum(1, ts / .003), at, rng.uniform(-.8, .8), rng.uniform(.012, .032), rev=1.2)
for bar, g in [(3, .02), (5, .022), (7, .022)]:
    w = swell(B(bar) - 1.3, B(bar) - .05, B(bar) + 1.2, 500, 5000, g); mix += w; send += w * .8
w = swell(B(1, 1.2), B(1, 2.05), B(2, 2), 500, 5000, .028); mix += w; send += w * .8
r = swell(B(8), B(9) - .03, B(9), 1500, 11000, .075) + swell(B(7, 2), B(9) - .03, B(9), 300, 1600, .05)
mix += r; send += r * .5
ti = np.arange(int(3 * SR)) / SR
place(np.sin(2 * np.pi * np.cumsum(36.7 + 50 * np.exp(-ti * 9)) / SR) * np.exp(-ti * 1.5) * np.minimum(1, ti / .004), B(9), 0, .6, rev=.05)
place(filt(rng.normal(0, 1, len(ti)), 'low', 160, 4) * np.exp(-ti * 9), B(9), 0, .9, rev=.3)
place(filt(rng.normal(0, 1, (len(ti), 2)), 'high', 5000) * (np.exp(-ti * 1.6) * np.minimum(1, ti / .02))[:, None], B(9), 0, .022, rev=1.0)
hi2 = strings(['A4', 'D5', 'F#5'], B(9) - .05, B(9) + .5, B(10, 2), DUR, 5000, 2500) * .15
mix += hi2; send += hi2 * .7

# --- hall reverb + master ---
L = int(4.2 * SR); tr = np.arange(L) / SR
ir = filt(filt(rng.normal(0, 1, (L, 2)) * np.exp(-tr * 1.75)[:, None], 'low', 5200), 'high', 180); ir[: int(.018 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0))
out = mix + np.stack([fftconvolve(send[:, c], ir[:, c])[:n] for c in range(2)], 1) * .55
out = filt(out, 'high', 24) * env(0, .02, DUR - 1.6, DUR - .05)[:, None]
out = np.tanh(out / np.abs(out).max() * 1.1); out = out / np.abs(out).max() * 0.89
with wave.open(sys.argv[1], 'wb') as wv:
    wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR); wv.writeframes((out * 32767).astype('<i2').tobytes())
rms = lambda a, b: 20 * np.log10(np.sqrt((out[int(a * SR):int(b * SR)] ** 2).mean()) + 1e-9)
print('rms dB per bar:', [round(float(rms(B(k), B(k + 1))), 1) for k in range(1, 12)])
