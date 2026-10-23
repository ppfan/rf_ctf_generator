#!/usr/bin/env python3
"""Simulated IQ recordings for the SDR++ spectrum-analysis CTF (10 tasks).

Each task becomes a 16-bit stereo (I/Q) WAV whose name carries the centre
frequency ("..._433500000Hz.wav"), so SDR++'s File source tunes itself.
An answer key (answers.md / answers.json) is written next to the files.

    python3 gen_ctf.py                      # all tasks, seed 1
    python3 gen_ctf.py --tasks 2,9 --seed 7 --flag "CTF{K0D}"
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from scipy import signal
from scipy.io import wavfile

NOISE_SIGMA = 0.01  # total receiver noise power = -40 dBFS
REF_FFT = 8192      # FFT size the "dB above floor" figures refer to
CHUNK = 1 << 22


# ---------------------------------------------------------------- helpers

def noise(rng, n, sigma=NOISE_SIGMA):
    re = rng.standard_normal(n, dtype=np.float32)
    im = rng.standard_normal(n, dtype=np.float32)
    return ((re + 1j * im) * np.float32(sigma / math.sqrt(2))).astype(np.complex64)


def tone_amp(snr_db, nfft=REF_FFT):
    """Amplitude of a CW tone sitting snr_db above the per-bin noise floor."""
    return math.sqrt(10 ** (snr_db / 10) * NOISE_SIGMA ** 2 / nfft)


def dbfs(amp):
    return 20 * math.log10(amp)


def floor_dbfs(nfft=REF_FFT):
    return 10 * math.log10(NOISE_SIGMA ** 2 / nfft)


def fm(finst, fs):
    """Phase-continuous oscillator following an instantaneous-frequency array (Hz)."""
    out = np.empty(len(finst), np.complex64)
    ph = 0.0
    for s in range(0, len(finst), CHUNK):
        p = ph + (2 * np.pi / fs) * np.cumsum(finst[s:s + CHUNK], dtype=np.float64)
        out[s:s + len(p)] = np.exp(1j * p)
        ph = p[-1] % (2 * np.pi)
    return out


def osc(f, fs, n):
    out = np.empty(n, np.complex64)
    for s in range(0, n, CHUNK):
        k = np.arange(s, min(n, s + CHUNK), dtype=np.float64)
        out[s:s + len(k)] = np.exp(2j * np.pi * ((f / fs * k) % 1.0))
    return out


def ramp_env(n, fs, on, off, edge=0.002):
    """0/1 envelope, on between sample indices [on, off), raised-cosine edges."""
    env = np.zeros(n, np.float32)
    env[max(on, 0):min(off, n)] = 1
    k = max(int(edge * fs), 1)
    win = np.hanning(2 * k + 1).astype(np.float32)
    return signal.oaconvolve(env, win / win.sum(), mode="same").astype(np.float32)


def psd_db(x, fs, nfft=REF_FFT, max_samples=4_000_000):
    f, p = signal.welch(x[:max_samples], fs, window="hann", nperseg=nfft,
                        return_onesided=False, scaling="spectrum")
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    return f, 10 * np.log10(p + 1e-30)


def smooth(y, k):
    return np.convolve(np.pad(y, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(y)]


def obw(f, p_db, frac=0.99):
    c = np.cumsum(10 ** (p_db / 10))
    c /= c[-1]
    return f[np.searchsorted(c, (1 - frac) / 2)], f[np.searchsorted(c, 1 - (1 - frac) / 2)]


def band_edges(f, p_db, level_db):
    idx = np.nonzero(p_db >= level_db)[0]
    return f[idx[0]], f[idx[-1]]


def write_iq(path, x, fs):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        for s in range(0, len(x), CHUNK):
            c = x[s:s + CHUNK]
            iq = np.empty(2 * len(c), np.float32)
            iq[0::2], iq[1::2] = c.real, c.imag
            w.writeframes(np.clip(np.round(iq * 32767), -32768, 32767).astype("<i2").tobytes())


def mhz(f):
    return f"{f / 1e6:.4f} МГц"


# ---------------------------------------------------------------- audio

def speech(text, wav_path):
    """Mono float audio + rate: user WAV, else espeak-ng (Ukrainian), else DTMF digits."""
    if wav_path:
        afs, a = wavfile.read(wav_path)
    elif shutil.which("espeak-ng"):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "v.wav")
            subprocess.run(["espeak-ng", "-v", "uk", "-s", "125", "-w", p, text], check=True)
            afs, a = wavfile.read(p)
    else:
        afs = 8000
        tones = {"0": (941, 1336), "1": (697, 1209), "2": (697, 1336), "3": (697, 1477),
                 "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "7": (852, 1209),
                 "8": (852, 1336), "9": (852, 1477)}
        t = np.arange(int(0.2 * afs)) / afs
        parts = []
        for ch in "".join(c for c in text if c.isdigit()) or "123":
            lo, hi = tones[ch]
            parts += [np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t), np.zeros(len(t))]
        a = np.concatenate(parts)
    a = a.astype(np.float64)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a / (np.max(np.abs(a)) + 1e-9), afs


def audio_to_fs(a, afs, fs):
    """Band-limit voice to 3 kHz and resample to the IQ rate (via 25 kHz)."""
    g = math.gcd(25000, int(afs))
    a = signal.resample_poly(a, 25000 // g, int(afs) // g)
    a = signal.sosfiltfilt(signal.butter(6, 3000, fs=25000, output="sos"), a)
    a = signal.resample_poly(a, int(fs) // 25000, 1)
    return (a / (np.max(np.abs(a)) + 1e-9)).astype(np.float32)


MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
    "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
    "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
    "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..", "0": "-----",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....",
    "7": "--...", "8": "---..", "9": "----.", "/": "-..-.", "-": "-....-", "=": "-...-",
    "?": "..--..", ".": ".-.-.-", "{": "-.--.", "}": "-.--.-", "_": "..--.-",
}


def morse_units(text):
    """List of (key_down, units) for a text."""
    seq = []
    for w, word in enumerate(text.upper().split()):
        if w:
            seq.append((0, 7))
        for c, ch in enumerate(word):
            if c:
                seq.append((0, 3))
            for e, el in enumerate(MORSE[ch]):
                if e:
                    seq.append((0, 1))
                seq.append((1, 1 if el == "." else 3))
    return seq


# ---------------------------------------------------------------- tasks

def t1_nfm(rng, a):
    fs, fc = 1_000_000, 433_500_000
    f_sig = 433_075_000 + 25_000 * int(rng.integers(0, 35))
    voice, afs = speech(a.voice_text, a.voice_wav)
    v = audio_to_fs(voice, afs, fs)
    lead, keyup, tail, gap = 1.0, 0.4, 0.5, 1.5
    n = int((lead + keyup + tail + gap) * fs) + len(v)
    mod = np.zeros(n, np.float32)
    s0 = int((lead + keyup) * fs)
    mod[s0:s0 + len(v)] = v
    on, off = int(lead * fs), s0 + len(v) + int(tail * fs)
    x = tone_amp(45) * ramp_env(n, fs, on, off) * fm((f_sig - fc) + 2500 * mod, fs)
    f_qrm = fc + rng.choice([-1, 1]) * rng.integers(150, 400) * 1000 + 6250
    x += tone_amp(18) * osc(f_qrm - fc, fs, n)
    x += noise(rng, n)
    return x, fs, fc, {
        "Центральна частота": mhz(f_sig),
        "Ширина смуги": "канал 12.5 кГц (девіація ±2.5 кГц, мова до 3 кГц, зайнята ≈11 кГц)",
        "Модуляція": "NFM (аналогова рація)",
        "Фраза": a.voice_text if not a.voice_wav else f"(з файлу {a.voice_wav})",
        "Відволікаюча несуча": f"{mhz(f_qrm)} — немодульована, НЕ відповідь",
    }


def lora_airtime(sf, bw, pl, cr=1, crc=1, ih=0, n_pre=8):
    de = 1 if (2 ** sf / bw) > 0.016 else 0
    tsym = 2 ** sf / bw
    n_pl = 8 + max(math.ceil((8 * pl - 4 * sf + 28 + 16 * crc - 20 * ih) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return tsym, n_pre + 4.25, n_pl, (n_pre + 4.25 + n_pl) * tsym


def t2_lora(rng, a):
    fs, fc = 1_000_000, 868_300_000
    sf, bw = a.lora_sf, a.lora_bw
    f_sig = float(rng.choice([868_100_000, 868_300_000, 868_500_000]))
    pl = int(rng.integers(8, 21))
    tsym, n_pre, n_pl, airtime = lora_airtime(sf, bw, pl)
    M, sps = 2 ** sf, int(round(fs * 2 ** sf / bw))
    t = np.arange(sps) / sps

    def up(k):
        return (((k / M + t) % 1.0) - 0.5) * bw

    def packet():
        syms = [up(0)] * 8 + [up(24 * M // 128), up(32 * M // 128)]
        down = -up(0)
        syms += [down, down, down[:sps // 4]]
        syms += [up(int(rng.integers(0, M))) for _ in range(n_pl)]
        return np.concatenate(syms).astype(np.float32) + np.float32(f_sig - fc)

    period, n_pk = 3.0, 2
    n = int(period * n_pk * fs)
    x = np.zeros(n, np.complex64)
    for i in range(n_pk):
        fi = packet()
        s0 = int((i * period + rng.uniform(0.3, period - airtime - 0.3)) * fs)
        env = ramp_env(len(fi), fs, 0, len(fi), 0.0002)
        x[s0:s0 + len(fi)] += tone_amp(35) * env * fm(fi, fs)
    x += noise(rng, n)
    return x, fs, fc, {
        "Тип сигналу": "LoRa (Chirp Spread Spectrum)",
        "Частота каналу": mhz(f_sig),
        "Ширина каналу": f"{bw / 1e3:.0f} кГц",
        "Параметри": f"SF{sf}, CR 4/5, явний заголовок, CRC, payload {pl} байт",
        "Тривалість символу": f"{tsym * 1e3:.3f} мс",
        "Airtime пакета": f"{airtime * 1e3:.1f} мс (преамбула {n_pre:g} + {n_pl} символів даних)",
        "Період": f"{n_pk} пакети за {period * n_pk:.0f} с файлу",
    }


def t3_fhss(rng, a):
    fs, fc, dur = 10_000_000, 915_000_000, 2.0
    sp = float(rng.choice([200e3, 250e3, 300e3]))
    nch = int(min(rng.integers(20, 33), 7.5e6 // sp + 1))
    f0 = -(nch - 1) * sp / 2 + 25e3 * int(rng.integers(-20, 21))
    chans = f0 + sp * np.arange(nch)
    dwell, burst, rb, dev = 0.004, 0.0016, 100e3, 50e3
    n, ns, nb = int(dur * fs), int(dwell * fs), int(burst * fs)
    hops = np.concatenate([rng.permutation(nch) for _ in range(int(n / ns / nch) + 2)])
    finst = np.zeros(n, np.float32)
    env = np.zeros(n, np.float32)
    spb = int(fs / rb)
    k = max(int(0.00002 * fs), 1)
    edge = np.hanning(2 * k)[:k].astype(np.float32)
    for i, s0 in enumerate(range(0, n - ns, ns)):
        bits = rng.integers(0, 2, nb // spb + 1) * 2 - 1
        b = np.repeat(bits, spb)[:nb].astype(np.float32)
        b = np.convolve(b, np.ones(spb // 2) / (spb // 2), "same")
        finst[s0:s0 + nb] = chans[hops[i]] + dev * b
        e = np.ones(nb, np.float32)
        e[:k], e[-k:] = edge, edge[::-1]
        env[s0:s0 + nb] = e
    x = tone_amp(32) * env * fm(finst, fs)
    x += noise(rng, n)
    return x, fs, fc, {
        "Тип": "FHSS (ППРЧ), 2-GFSK пакети",
        "Band Min (центр нижнього каналу)": mhz(fc + chans[0]),
        "Band Max (центр верхнього каналу)": mhz(fc + chans[-1]),
        "Межі по краях спектра": f"{mhz(fc + chans[0] - 75e3)} … {mhz(fc + chans[-1] + 75e3)}",
        "Крок сітки": f"{sp / 1e3:.0f} кГц, каналів: {nch}",
        "Темп стрибків": f"{1 / dwell:.0f} hop/с (dwell {dwell * 1e3:.0f} мс, пакет {burst * 1e3:.1f} мс)",
    }


def t4_ghosts(rng, a):
    fs, step_s = 2_000_000, 2.0
    fA = 433_920_000
    fc = fA - 150_000
    dA, dB = 150e3, 150e3 + float(rng.choice([180e3, 220e3, 260e3]))
    steps = [0, 5, 10, 15, 20, 25]
    ns = int(step_s * fs)
    n = ns * len(steps)
    tt = np.arange(n, dtype=np.float32) / fs
    clean = 0.5 * fm(dA + 2500 * np.sin(2 * np.pi * 1000 * tt), fs) + 0.15 * osc(dB, fs, n)
    del tt
    x = np.empty(n, np.complex64)
    c3, c5 = 0.3, 1.0
    for i, att in enumerate(steps):
        s = clean[i * ns:(i + 1) * ns] * np.float32(10 ** (-att / 20))
        m = np.abs(s) ** 2
        x[i * ns:(i + 1) * ns] = s * (1 - c3 * m - c5 * m * m)
    x += noise(rng, n)
    ghosts = {"IM3 2fA−fB": 2 * dA - dB, "IM3 2fB−fA": 2 * dB - dA,
              "IM5 3fA−2fB": 3 * dA - 2 * dB, "IM5 3fB−2fA": 3 * dB - 2 * dA}
    rows, clean_from = [], None
    for i, att in enumerate(steps):
        f, p = psd_db(x[i * ns:(i + 1) * ns], fs)
        fl = np.median(p)
        lv = {k: p[np.abs(f - g) < 15e3].max() - fl for k, g in ghosts.items()}
        rows.append(f"{att:>2} dB: " + ", ".join(f"{k} {v:+.0f} dB" for k, v in lv.items()))
        if clean_from is None and max(lv.values()) < 3:
            clean_from = att
    return x, fs, fc, {
        "Реальних джерел": f"2 — {mhz(fA)} (NFM, тон 1 кГц) та {mhz(fc + dB)} (CW)",
        "Фантоми (інтермодуляція)": ", ".join(f"{k} = {mhz(fc + g)}" for k, g in ghosts.items()),
        "Ознака фантома": "IM3 ширші (девіація ×2…×3) і при +5 dB атенюації падають на 15 dB, IM5 — на 25 dB",
        "Сценарій файлу": f"атенюатор 0→25 dB кроком 5 dB кожні {step_s:.0f} с (файл циклічний)",
        "Рівні фантомів над шумом (FFT 8192)": " | ".join(rows),
        "Достатнє загрублення": f"{clean_from} dB" if clean_from is not None else "> 25 dB",
    }


def t5_weak(rng, a):
    fs, fc, dur = 1_000_000, 162_000_000, 10.0
    n = int(dur * fs)
    off = float(rng.choice([-1, 1]) * rng.integers(40, 420) * 1000 + rng.integers(0, 1000))
    amp = tone_amp(-6)
    x = amp * osc(off, fs, n)
    for _ in range(3):
        fo = float(rng.uniform(-450e3, 450e3))
        if abs(fo - off) > 30e3:
            x += tone_amp(float(rng.uniform(20, 35))) * fm(
                fo + 2500 * np.sin(2 * np.pi * rng.uniform(400, 1500) * np.arange(n, dtype=np.float32) / fs), fs)
    x += noise(rng, n)
    return x, fs, fc, {
        "Частота маяка": f"{(fc + off) / 1e6:.6f} МГц",
        "Рівень маяка": f"≈ {dbfs(amp):.1f} dBFS (у dBm = dBFS + калібрувальна поправка приладу)",
        "Шумова полиця": " | ".join(f"FFT {k}: {floor_dbfs(k):.1f} dBFS (маяк {dbfs(amp) - floor_dbfs(k):+.1f} dB)"
                                    for k in (1024, 8192, 65536, 524288)),
        "Як знайти": "FFT ≥ 65536 (вузький RBW) + усереднення/Peak Hold; без цього маяк під шумом",
        "Відволікаючі": "3 сильні NFM-несучі — не відповідь",
    }


def t6_video(rng, a):
    fs, dur = a.wide_fs, 1.0
    race = dict(zip([f"R{i}" for i in range(1, 9)], [5658, 5695, 5732, 5769, 5806, 5843, 5880, 5917]))
    name = str(rng.choice(list(race)))
    ch = race[name] * 1e6
    off = 2.5e6
    fc = ch - off

    def us(t):
        return int(round(t * 1e-6 * fs))
    L, lines = us(64), 312
    a0, act = us(4.7 + 5.7), us(52)
    field = np.zeros((lines, L), np.float32)
    field[:, :us(4.7)] = -0.3
    bars = np.repeat(np.array([1, .89, .70, .59, .41, .30, .11, 0]) * 0.7, act // 8 + 1)[:act]
    field[23:160, a0:a0 + act] = bars
    scene = rng.random((10, 16)) * 0.5 + np.linspace(0.05, 0.3, 16)
    field[160:310, a0:a0 + act] = np.repeat(np.repeat(scene, 15, 0), act // 16 + 1, 1)[:150, :act]
    for ln in range(8):
        field[ln] = 0
        for h in (0, L // 2):
            field[ln, h:h + us(27.3)] = -0.3
    v = field.ravel()
    taps = signal.firwin(101, 5e6, fs=fs).astype(np.float32)
    v = np.convolve(np.concatenate([v[-100:], v]), taps, "valid").astype(np.float32)
    n = int(dur * fs)
    finst = np.tile(v, n // len(v) + 1)[:n] * np.float32(6e6) + np.float32(off)
    for s in range(0, n, CHUNK):
        t = np.arange(s, min(n, s + CHUNK), dtype=np.float64) / fs
        finst[s:s + len(t)] += (1.2e6 * np.sin(2 * np.pi * 6.5e6 * t + 50 * np.sin(2 * np.pi * 1e3 * t))).astype(np.float32)
    clean = 0.5 * fm(finst, fs)
    del finst
    f, p = psd_db(clean, fs, 4096, 3_000_000)
    lo99, hi99 = obw(f, p)
    p = smooth(p, int(1e6 / (f[1] - f[0])))
    lo, hi = band_edges(f, p, p.max() - 30)
    x = clean + noise(rng, n)
    return x, fs, fc, {
        "Тип": "аналогове FM-відео (PAL, 625/50) + аудіопіднесуча 6.5 МГц",
        "Канал": f"Raceband {name} = {race[name]} МГц",
        "Центр несучої": f"{mhz(ch)} (рівень гасіння = несуча; пік синхроімпульсів ≈ −1.8 МГц)",
        "Ширина по −30 dB (спектр згладжений 1 МГц)": f"{(hi - lo) / 1e6:.1f} МГц ({mhz(fc + lo)} … {mhz(fc + hi)})",
        "Зайнята смуга 99% потужності": f"{(hi99 - lo99) / 1e6:.1f} МГц",
        "Форма": "вузький пік несучої (рівень гасіння) + гребінь ліній яскравості, не рівна полиця",
        "Додатково": "бічні складові аудіопіднесучої на ±6.5 МГц від несучої",
    }


def t7_ofdm(rng, a):
    fs, dur = a.wide_fs, 1.0
    N = int(round(fs / (312.5e3 if a.ofdm_bw == 20 else 156.25e3)))
    cp, spacing = N // 4, fs / N
    wch = int(rng.integers(1, 12))
    ch = (2412 + 5 * (wch - 1)) * 1e6
    off = float(rng.choice([-3e6, -2e6, 2e6, 3e6]))
    fc = ch - off
    bins = np.r_[-26:0, 1:27] % N
    d = int(rng.integers(2, 5))
    h = np.zeros(d + 1, np.complex64)
    h[0], h[d] = 1, rng.uniform(0.2, 0.35) * np.exp(2j * np.pi * rng.random())
    n = int(dur * fs)
    x = np.zeros(n, np.complex64)
    pos = 0
    while pos < n:
        pos += int(rng.integers(int(16e-6 * fs), int(200e-6 * fs)))
        nsym = int(rng.integers(40, 400))
        X = np.zeros((nsym, N), np.complex64)
        X[:, bins] = ((rng.integers(0, 4, (nsym, 52)) * 2 - 3) + 1j * (rng.integers(0, 4, (nsym, 52)) * 2 - 3)) / math.sqrt(10)
        tsym = np.fft.ifft(X, axis=1) * (N / math.sqrt(52))
        pk = np.concatenate([tsym[:, -cp:], tsym], axis=1).ravel()
        pk = np.convolve(pk, h)[:len(pk)].astype(np.complex64)
        m = min(len(pk), n - pos)
        x[pos:pos + m] = pk[:m]
        pos += len(pk)
    x *= 0.3 * osc(off, fs, n)
    f, p = psd_db(x, fs, 4096, 3_000_000)
    flat = np.abs(f - off) < 25 * spacing
    flat &= np.abs(f - off) > 2.5 * spacing
    ps = smooth(p, int(spacing / (f[1] - f[0])))
    ripple = ps[flat].max() - ps[flat].min()
    lo, hi = band_edges(f, smooth(p, 3), np.median(p[flat]) - 20)
    x += noise(rng, n)
    return x, fs, fc, {
        "Тип": "OFDM (Wi-Fi-подібний, 52 піднесучі, 16-QAM, пакетний трафік)",
        "Канал": f"Wi-Fi 2.4 ГГц ch{wch} = {mhz(ch)}",
        "Номінальна ширина каналу": f"{a.ofdm_bw} МГц (зайнята {52 * spacing / 1e6:.2f} МГц, крок піднесучих {spacing / 1e3:.2f} кГц)",
        "Ширина по −20 dB від полиці": f"{(hi - lo) / 1e6:.1f} МГц",
        "Нерівномірність вершини (без DC-провалу)": f"≈ {ripple:.1f} dB (багатопроменевість: луна {d / fs * 1e9:.0f} нс, |h|={abs(h[d]):.2f})",
        "Відмінність від аналогового відео": "П-подібна полиця, немає піку несучої, круті краї",
    }


def t8_jammer(rng, a):
    fs, dur, t_on = a.wide_fs, 2.0, 0.4
    n, n0 = int(dur * fs), int(t_on * fs)
    W = float(rng.uniform(0.5, 0.72) * fs)
    fcen = float(rng.uniform(-(0.45 * fs - W / 2), 0.45 * fs - W / 2))
    lo, hi, tw = fcen - W / 2, fcen + W / 2, 1.5e6
    fpk = float(rng.uniform(lo + W / 4, hi - W / 4))
    fc = 1_250_000_000 + 500_000 * int(rng.integers(-10, 11))
    plateau, hump = 30.0, 6.0
    B = 1 << 20
    ff = np.fft.fftfreq(B, 1 / fs)
    mask = np.clip(np.minimum(ff - (lo - tw), (hi + tw) - ff) / tw, 0, 1)
    mask = 0.5 - 0.5 * np.cos(np.pi * mask)
    mask *= 1 + (10 ** (hump / 10) - 1) * np.exp(-0.5 * ((ff - fpk) / (W / 10)) ** 2)
    g = np.sqrt(mask * 10 ** (plateau / 10) * NOISE_SIGMA ** 2)
    x = np.zeros(n, np.complex64)
    if a.jam_mode == "sweep":
        k = np.arange(n - n0, dtype=np.float64)
        saw = ((k * 2000 / fs) % 1.0).astype(np.float32)
        x[n0:] = 0.3 * fm(lo + (hi - lo) * saw, fs)
    else:
        for s in range(n0, n, B):
            m = min(B, n - s)
            w = noise(rng, B, 1.0)
            x[s:s + m] = np.fft.ifft(np.fft.fft(w) * g)[:m]
    legit = rng.uniform(-0.45 * fs, 0.45 * fs, 6)
    for fo in legit:
        x += tone_amp(25) * osc(float(fo), fs, n)
    x += noise(rng, n)
    f, p_off = psd_db(x[:n0], fs, 4096)
    _, p_on = psd_db(x[n0:], fs, 4096)
    floor = np.median(p_off)
    ps = smooth(p_on, 15)
    inside = (f > lo) & (f < hi)
    plat = np.median(ps[inside])
    lo3, hi3 = band_edges(f, ps, plat - 3)
    lo10, hi10 = band_edges(f, ps, floor + 10)
    inband = sorted(round((fc + fo) / 1e6, 3) for fo in legit if lo < fo < hi)
    return x, fs, fc, {
        "Тип": "загороджувальна шумова перешкода" if a.jam_mode == "noise" else "загороджувальна перешкода зі свіпом CW (2 кГц)",
        "Межі (−3 dB від полиці)": f"{mhz(fc + lo3)} … {mhz(fc + hi3)} ({(hi3 - lo3) / 1e6:.1f} МГц)",
        "Межі (+10 dB над фоном)": f"{mhz(fc + lo10)} … {mhz(fc + hi10)}",
        "Максимум інтерференції": mhz(fc + f[np.argmax(np.where(inside, ps, -999))]),
        "Глибина подавлення": f"≈ {plat - floor:.0f} dB над фоном (у максимумі ≈ {ps[inside].max() - floor:.0f} dB)",
        "Сценарій": f"перші {t_on} с — чистий ефір (видно фон), далі перешкода",
        "Задушені корисні сигнали": ", ".join(f"{v} МГц" for v in inband) or "немає",
    }


def t9_morse(rng, a):
    fs, fc = 250_000, 145_000_000
    off = float(rng.choice([-1, 1]) * rng.integers(15, 90) * 1000 + rng.integers(0, 1000))
    unit = 1.2 / a.wpm
    msg = f"VVV {a.flag} {a.flag}"
    seq = morse_units(msg) + [(0, int(3 / unit))]
    keys = np.concatenate([np.full(int(u * unit * fs), k, np.float32) for k, u in seq])
    n = len(keys) + int(0.5 * fs)
    env = np.zeros(n, np.float32)
    env[int(0.5 * fs):] = keys
    kk = int(0.005 * fs)
    env = signal.oaconvolve(env, np.hanning(kk) / np.hanning(kk).sum(), "same").astype(np.float32)
    x = tone_amp(30) * env * osc(off, fs, n)
    x += tone_amp(15) * osc(float(rng.uniform(-110e3, 110e3)), fs, n)
    x += noise(rng, n)
    return x, fs, fc, {
        "Частота маяка": f"{(fc + off) / 1e6:.6f} МГц",
        "Модуляція": f"CW (OOK), {a.wpm} WPM, точка = {unit * 1e3:.0f} мс — слухати в режимі CW/USB",
        "Текст у циклі": msg,
        "ПРАПОР": a.flag,
    }


def t10_fox(rng, a):
    fs, fc, period = 250_000, 446_100_000, 36.0
    ch = int(rng.integers(1, 17))
    f_sig = 446_006_250 + 12_500 * (ch - 1)
    th0 = float(rng.integers(0, 360))
    n = int(period * fs)
    t = np.arange(n, dtype=np.float32) / fs
    theta = np.deg2rad(t * 360 / period)
    gain = ((1 + np.cos(theta - np.deg2rad(th0))) / 2) ** 6 + 0.02
    fade = np.interp(t, np.linspace(0, period, 200), rng.standard_normal(200)).astype(np.float32)
    amp = tone_amp(38) * (np.sqrt(gain) * 10 ** (0.075 * fade)).astype(np.float32)
    x = amp * fm((f_sig - fc) + 2500 * np.sin(2 * np.pi * 1000 * t), fs)
    ref = np.zeros(n, np.float32)
    for s in range(int(period)):
        dl = 0.6 if s == 0 else 0.3 if s % 9 == 0 else 0.06
        ref[int(s * fs):int((s + dl) * fs)] = 1
    f_ref = fc + (95_000 if f_sig < fc + 50_000 else -95_000)
    x += tone_amp(25) * ref * osc(f_ref - fc, fs, n)
    x += noise(rng, n)
    peak_db = 10 * math.log10(1.02) + 38
    min_db = 10 * math.log10(0.02) + 38
    return x, fs, fc, {
        "Ціль": f"PMR446 ch{ch} = {mhz(f_sig)} (NFM, тон 1 кГц)",
        "Компас-маяк": f"{mhz(f_ref)}: імпульс кожну 1 с = 10°; довгий (0.6 с) = 0° (Північ), середні (0.3 с) = 90°/180°/270°",
        "Обертання": f"360° за {period:.0f} с (10°/с), початок файлу = 0°",
        "Азимут максимуму": f"{th0:.0f}° (t = {th0 / 10:.1f} с від довгого імпульсу)",
        "Delta (максимум − шумова полиця, FFT 8192)": f"≈ {peak_db:.0f} dB",
        "Delta (максимум − мінімум діаграми / задня пелюстка)": f"≈ {peak_db - min_db:.0f} dB",
        "Примітка": "повільні завмирання ±1–2 dB накладені навмисно",
    }


TASKS = {
    1: ("Впізнай свій голос (NFM)", "nfm", t1_nfm),
    2: ("Пакети в ефірі (LoRa)", "lora", t2_lora),
    3: ("Мандрівна точка (FHSS)", "fhss", t3_fhss),
    4: ("Полювання на привидів (IMD)", "ghosts", t4_ghosts),
    5: ("Голка в купі сіна (слабкий CW)", "weak", t5_weak),
    6: ("Аналогове око (FM-відео)", "video", t6_video),
    7: ("Цифровий трафік (OFDM)", "ofdm", t7_ofdm),
    8: ("Смертоносна смуга (РЕБ)", "jammer", t8_jammer),
    9: ("Захована морзянка (CW)", "morse", t9_morse),
    10: ("Лисиця на рубежі (пеленг)", "fox", t10_fox),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--tasks", default="1-10", help="e.g. 1-10 or 2,5,9")
    ap.add_argument("--seed", type=int, default=1, help="change per group -> different answers")
    ap.add_argument("--flag", default="FLAG123", help="task 9 Morse text")
    ap.add_argument("--wpm", type=int, default=10)
    ap.add_argument("--voice-text", default="Беркут, Беркут, я Сокіл. Прапор: сім, два, дев'ять. Прийом.")
    ap.add_argument("--voice-wav", help="use your own recording for task 1 instead of TTS")
    ap.add_argument("--lora-sf", type=int, default=11, choices=range(7, 13))
    ap.add_argument("--lora-bw", type=float, default=125e3, choices=[125e3, 250e3, 500e3])
    ap.add_argument("--wide-fs", type=float, default=30e6, help="sample rate for tasks 6-8")
    ap.add_argument("--ofdm-bw", type=int, default=20, choices=[10, 20])
    ap.add_argument("--jam-mode", default="noise", choices=["noise", "sweep"])
    a = ap.parse_args()

    sel = set()
    for part in a.tasks.split(","):
        lo, _, hi = part.partition("-")
        sel.update(range(int(lo), int(hi or lo) + 1))
    os.makedirs(a.out, exist_ok=True)
    key_path = os.path.join(a.out, "answers.json")
    key = json.load(open(key_path, encoding="utf-8")) if os.path.exists(key_path) else {}

    for num in sorted(sel):
        title, slug, fn = TASKS[num]
        rng = np.random.default_rng([a.seed, num])
        print(f"[{num:2}] {title} ...", flush=True)
        x, fs, fc, ans = fn(rng, a)
        name = f"t{num:02d}_{slug}_{int(round(fc))}Hz.wav"
        for old in os.listdir(a.out):
            if old.startswith(f"t{num:02d}_") and old.endswith(".wav"):
                os.remove(os.path.join(a.out, old))
        write_iq(os.path.join(a.out, name), x, fs)
        key[str(num)] = {"title": title, "file": name, "center_hz": fc, "fs": fs,
                         "duration_s": round(len(x) / fs, 2), "seed": a.seed, "answers": ans}
        print(f"     {name}  fs={fs / 1e6:g} MS/s  {len(x) / fs:.1f} s")

    json.dump(key, open(key_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(a.out, "answers.md"), "w", encoding="utf-8") as md:
        md.write("# Ключ відповідей (тільки для інструктора)\n\n")
        md.write(f"Шумова полиця: {floor_dbfs():.1f} dBFS на бін при FFT {REF_FFT}. "
                 "Рівні в dBFS; для dBm додайте калібрувальну поправку приладу.\n\n")
        for k in sorted(key, key=int):
            e = key[k]
            md.write(f"## {k}. {e['title']}\n\n`{e['file']}` — центр {e['center_hz'] / 1e6:.4f} МГц, "
                     f"{e['fs'] / 1e6:g} MS/s, {e['duration_s']} с (цикл), seed {e['seed']}\n\n")
            for q, v in e["answers"].items():
                md.write(f"- **{q}:** {v}\n")
            md.write("\n")
    print(f"\nAnswer key: {os.path.join(a.out, 'answers.md')}")


if __name__ == "__main__":
    main()
