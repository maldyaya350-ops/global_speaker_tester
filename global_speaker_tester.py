#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     GLOBAL SPEAKER DIAGNOSTIC SUITE – BASS CANNON EDITION   ║
║       Live Audio Processing | Bass Boost | Noise Cleaner    ║
║                    عالمي لكل السماعات                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import time
import queue
import subprocess
from datetime import datetime
from collections import deque

# Optional imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import asyncio
    from bleak import BleakScanner
    BT_AVAILABLE = True
except ImportError:
    BT_AVAILABLE = False

# ========== CONSTANTS ==========
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
CHANNELS = 1

PALETTE = {
    'bg': '#07080f', 'bg2': '#0d0f1c', 'bg3': '#111428',
    'accent': '#00e5ff', 'accent2': '#ff6b35', 'accent3': '#7fff00',
    'danger': '#ff2244', 'warn': '#ffbb00', 'muted': '#3a4060',
    'text': '#c8d0e8', 'text_dim': '#5a6080', 'grid': '#1a1e30',
    'green': '#39ff14', 'purple': '#bc13fe', 'pink': '#ff007f',
}

FREQ_BANDS = [
    ("SUB", 20, 60, '#ff0044'), ("BASS", 60, 250, '#ff6600'),
    ("LO-MID", 250, 500, '#ffaa00'), ("MID", 500, 2000, '#ffee00'),
    ("HI-MID", 2000, 4000, '#aaff00'), ("PRES", 4000, 8000, '#00ffaa'),
    ("AIR", 8000, 20000, '#00aaff')
]

# ========== ADVANCED AUDIO PROCESSOR ==========
class AudioProcessor:
    """Real-time audio effects: EQ, Bass Boost, Compressor, Noise Gate, Limiter"""
    def __init__(self):
        self.bass_gain = 0.0      # dB (0..24)
        self.mid_gain = 0.0
        self.treble_gain = 0.0
        self.bass_boost_factor = 1.0   # 1..3
        self.compressor_thresh = 0.5   # 0..1
        self.compressor_ratio = 2.0
        self.noise_gate_thresh = 0.01
        self.lowpass_cutoff = None     # Hz, None = off
        self.highpass_cutoff = None
        self.volume = 0.8
        self.bass_cannon_mode = False
        self._last_envelope = 0.0

    def set_eq(self, bass_db, mid_db, treble_db):
        self.bass_gain = np.clip(bass_db, -12, 24)
        self.mid_gain = np.clip(mid_db, -12, 24)
        self.treble_gain = np.clip(treble_db, -12, 24)

    def set_bass_boost(self, factor):
        self.bass_boost_factor = np.clip(factor, 1.0, 3.0)

    def set_compressor(self, threshold, ratio):
        self.compressor_thresh = np.clip(threshold, 0.01, 1.0)
        self.compressor_ratio = np.clip(ratio, 1.0, 10.0)

    def set_noise_gate(self, threshold):
        self.noise_gate_thresh = np.clip(threshold, 0.0, 0.1)

    def set_filters(self, lowcut=None, highcut=None):
        self.lowpass_cutoff = lowcut
        self.highpass_cutoff = highcut

    def process(self, data):
        """
        data: numpy array (float32, mono)
        returns processed data
        """
        if data is None or len(data) == 0:
            return data

        # Noise gate
        rms = np.sqrt(np.mean(data**2))
        if rms < self.noise_gate_thresh:
            data = data * 0.1  # attenuate but not mute to avoid artifacts
        else:
            # Apply filters if any
            if self.lowpass_cutoff is not None and self.lowpass_cutoff > 0:
                data = self._lowpass(data, self.lowpass_cutoff)
            if self.highpass_cutoff is not None and self.highpass_cutoff > 0:
                data = self._highpass(data, self.highpass_cutoff)

            # EQ + Bass Boost in frequency domain (simple but effective)
            data = self._apply_eq_and_boost(data)

            # Compressor
            data = self._compressor(data)

        # Volume and limiting
        data = data * self.volume
        # Hard limiter to prevent clipping
        max_val = np.max(np.abs(data))
        if max_val > 0.95:
            data = data * (0.95 / max_val)
        return data.astype(np.float32)

    def _apply_eq_and_boost(self, data):
        """Simple IIR filtering for EQ and bass boost"""
        # Use FFT for full control (small overhead is fine for blocks)
        n = len(data)
        if n < 64:
            return data
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(n, 1/SAMPLE_RATE)
        # Gains per frequency
        gain_curve = np.ones_like(freqs, dtype=float)
        # Bass shelf (0-200Hz)
        mask_bass = freqs < 200
        gain_curve[mask_bass] *= 10 ** (self.bass_gain / 20)
        # Mid peak (200-4000Hz)
        mask_mid = (freqs >= 200) & (freqs <= 4000)
        gain_curve[mask_mid] *= 10 ** (self.mid_gain / 20)
        # Treble shelf (4000+Hz)
        mask_treb = freqs > 4000
        gain_curve[mask_treb] *= 10 ** (self.treble_gain / 20)
        # Bass Cannon: extra boost below 150Hz
        if self.bass_cannon_mode or self.bass_boost_factor > 1.0:
            mask_deep = freqs < 150
            boost = self.bass_boost_factor if self.bass_cannon_mode else self.bass_boost_factor
            gain_curve[mask_deep] *= boost
        fft *= gain_curve
        processed = np.fft.irfft(fft, n=n)
        return processed

    def _lowpass(self, data, cutoff):
        # Simple 1-pole IIR
        dt = 1.0 / SAMPLE_RATE
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = dt / (rc + dt)
        y = np.zeros_like(data)
        y[0] = data[0]
        for i in range(1, len(data)):
            y[i] = y[i-1] + alpha * (data[i] - y[i-1])
        return y

    def _highpass(self, data, cutoff):
        dt = 1.0 / SAMPLE_RATE
        rc = 1.0 / (2 * np.pi * cutoff)
        alpha = rc / (rc + dt)
        y = np.zeros_like(data)
        y[0] = data[0]
        for i in range(1, len(data)):
            y[i] = alpha * (y[i-1] + data[i] - data[i-1])
        return y

    def _compressor(self, data):
        rms = np.sqrt(np.mean(data**2))
        if rms > self.compressor_thresh:
            gain = self.compressor_thresh / rms
            gain = gain ** (1.0 / self.compressor_ratio)
            data = data * gain
        return data

# ========== TTS ENGINE ==========
class TTSEngine:
    def __init__(self):
        self.engine = None
        self.queue = queue.Queue()
        self.running = True
        self._init_engine()
        threading.Thread(target=self._worker, daemon=True).start()

    def _init_engine(self):
        if not TTS_AVAILABLE:
            return
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            for v in voices:
                if 'arabic' in v.name.lower() or 'ar' in v.id.lower():
                    self.engine.setProperty('voice', v.id)
                    break
            self.engine.setProperty('rate', 160)
        except:
            self.engine = None

    def speak(self, text):
        self.queue.put(text)

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
                if self.engine:
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except:
                        pass
                else:
                    try:
                        subprocess.Popen(['espeak', '-v', 'ar', text],
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                    except:
                        pass
            except queue.Empty:
                pass

    def stop(self):
        self.running = False

# ========== AUDIO ENGINE WITH PROCESSING ==========
class AudioEngine:
    def __init__(self, processor):
        self.processor = processor
        self.volume = 0.8
        self.playing = False
        self.recording = False
        self.loopback = False
        self.rec_frames = []
        self._play_thread = None
        self._stream = None
        self._loop_stream = None
        self._rec_stream = None

        # Live analysis buffers
        self.live_buffer = deque(maxlen=BLOCK_SIZE * 8)
        self.fft_buffer = np.zeros(BLOCK_SIZE // 2)
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.thd = 0.0
        self._analysis_lock = threading.Lock()
        self._start_mic_monitor()

    # ---- Waveform generators (with processing) ----
    def _t(self, dur): return np.linspace(0, dur, int(SAMPLE_RATE * dur), False)

    def sine(self, freq, dur=3.0):
        t = self._t(dur)
        w = self.volume * np.sin(2 * np.pi * freq * t).astype(np.float32)
        return self.processor.process(w)

    def sweep(self, f1=20, f2=20000, dur=6.0):
        t = self._t(dur)
        k = (f2 / f1) ** (1 / dur)
        w = self.volume * np.sin(2 * np.pi * f1 * (k ** t - 1) / np.log(k)).astype(np.float32)
        return self.processor.process(w)

    def white_noise(self, dur=4.0):
        w = (self.volume * np.random.uniform(-1, 1, int(SAMPLE_RATE * dur))).astype(np.float32)
        return self.processor.process(w)

    def pink_noise(self, dur=4.0):
        n = int(SAMPLE_RATE * dur)
        w = np.random.randn(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1
        p = np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), n=n)
        p /= np.max(np.abs(p) + 1e-9)
        w = (self.volume * p).astype(np.float32)
        return self.processor.process(w)

    def brown_noise(self, dur=4.0):
        n = int(SAMPLE_RATE * dur)
        w = np.random.randn(n)
        f = np.fft.rfftfreq(n)
        f[0] = 1
        p = np.fft.irfft(np.fft.rfft(w) / f, n=n)
        p /= np.max(np.abs(p) + 1e-9)
        w = (self.volume * p).astype(np.float32)
        return self.processor.process(w)

    def thump(self, n=10):
        segs = []
        for i in range(n):
            freq = 40 + i * 5
            t = self._t(0.3)
            env = np.exp(-t * 15)
            w = self.volume * env * np.sin(2 * np.pi * freq * t)
            segs.append(w.astype(np.float32))
            segs.append(np.zeros(int(SAMPLE_RATE * 0.15), np.float32))
        w = np.concatenate(segs)
        return self.processor.process(w)

    def multitone(self, freqs, dur=4.0):
        t = self._t(dur)
        w = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        w = w / np.max(np.abs(w) + 1e-9)
        w = (self.volume * w).astype(np.float32)
        return self.processor.process(w)

    def binaural(self, base=200, beat=10, dur=30.0):
        t = self._t(dur)
        L = self.volume * np.sin(2 * np.pi * base * t)
        R = self.volume * np.sin(2 * np.pi * (base + beat) * t)
        stereo = np.stack([L, R], axis=1).astype(np.float32)
        # Process each channel separately
        stereo[:,0] = self.processor.process(stereo[:,0])
        stereo[:,1] = self.processor.process(stereo[:,1])
        return stereo

    def channel_test(self, side='left', freq=1000, dur=2.0):
        m = self.sine(freq, dur)
        s = np.zeros((len(m), 2), np.float32)
        if side in ('left', 'both'): s[:, 0] = m
        if side in ('right', 'both'): s[:, 1] = m
        return s

    # ---- Playback ----
    def play(self, data, on_done=None):
        self.stop_play()
        self.playing = True

        def _run():
            try:
                sd.play(data, SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
            finally:
                self.playing = False
                if on_done:
                    on_done()

        self._play_thread = threading.Thread(target=_run, daemon=True)
        self._play_thread.start()

    def stop_play(self):
        self.playing = False
        sd.stop()

    # ---- Recording ----
    def start_rec(self):
        self.rec_frames = []
        self.recording = True

        def cb(indata, frames, t, s):
            if self.recording:
                self.rec_frames.append(indata.copy())

        self._rec_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                          callback=cb, dtype='float32')
        self._rec_stream.start()

    def stop_rec(self):
        self.recording = False
        if self._rec_stream:
            self._rec_stream.stop()
            self._rec_stream.close()
            self._rec_stream = None

    def get_rec(self):
        if self.rec_frames:
            return np.concatenate(self.rec_frames, axis=0)
        return None

    def play_rec(self, on_done=None):
        d = self.get_rec()
        if d is not None:
            # Apply processor to playback
            d = self.processor.process(d)
            self.play(d, on_done)

    def save_rec(self, path, apply_processing=True):
        d = self.get_rec()
        if d is not None:
            if apply_processing:
                d = self.processor.process(d)
            sf.write(path, d, SAMPLE_RATE)
            return True
        return False

    # ---- Loopback with processing ----
    def start_loopback(self):
        self.loopback = True

        def cb(indata, outdata, frames, t, s):
            if self.loopback:
                processed = self.processor.process(indata[:, 0])
                outdata[:, 0] = processed * self.processor.volume
            else:
                outdata[:] = 0

        try:
            self._loop_stream = sd.Stream(samplerate=SAMPLE_RATE, channels=1,
                                          callback=cb, dtype='float32', latency='low')
            self._loop_stream.start()
        except Exception as e:
            self.loopback = False
            print(f"Loopback error: {e}")

    def stop_loopback(self):
        self.loopback = False
        if self._loop_stream:
            try:
                self._loop_stream.stop()
                self._loop_stream.close()
            except:
                pass
            self._loop_stream = None

    # ---- Mic Monitor for Analyzer ----
    def _start_mic_monitor(self):
        def cb(indata, frames, t, s):
            data = indata[:, 0]
            with self._analysis_lock:
                self.live_buffer.extend(data)
                self.rms_level = float(np.sqrt(np.mean(data ** 2)))
                self.peak_level = float(np.max(np.abs(data)))
                if len(data) >= BLOCK_SIZE:
                    windowed = data[:BLOCK_SIZE] * np.hanning(BLOCK_SIZE)
                    fft = np.abs(np.fft.rfft(windowed))[:BLOCK_SIZE // 2]
                    self.fft_buffer = fft / (BLOCK_SIZE / 2)
                if self.rms_level > 0.01 and len(self.fft_buffer) > 0:
                    self.thd = min(float(np.std(self.fft_buffer) / (np.mean(self.fft_buffer) + 1e-9)) * 10, 100)

        try:
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                          blocksize=BLOCK_SIZE, callback=cb, dtype='float32')
            self._stream.start()
        except Exception as e:
            print(f"Mic monitor: {e}")

    def get_waveform(self):
        with self._analysis_lock:
            return np.array(list(self.live_buffer)) if self.live_buffer else np.zeros(512)

    def get_fft(self):
        with self._analysis_lock:
            return self.fft_buffer.copy()

    def get_band_levels(self):
        fft = self.get_fft()
        freqs = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)[:BLOCK_SIZE // 2]
        levels = []
        for _, f1, f2, _ in FREQ_BANDS:
            mask = (freqs >= f1) & (freqs < f2)
            levels.append(float(np.mean(fft[mask])) * 40 if mask.any() else 0)
        return levels

    def stop_all(self):
        self.stop_play()
        self.stop_rec()
        self.stop_loopback()

    def shutdown(self):
        self.stop_all()
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass

# ========== BLUETOOTH MONITOR ==========
class BTMonitor:
    def __init__(self):
        self.devices = {}
        self.scanning = False
        self.battery = None
        self.rssi = None
        self.device_name = "Not Connected"
        self._thread = None

    def start_scan(self, callback=None):
        if self.scanning:
            return
        self.scanning = True

        async def _scan():
            try:
                devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
                result = {}
                for addr, (dev, adv) in devices.items():
                    result[addr] = {
                        'name': dev.name or "Unknown",
                        'rssi': adv.rssi,
                        'manufacturer': list(adv.manufacturer_data.keys()) if adv.manufacturer_data else []
                    }
                self.devices = result
                if callback:
                    callback(result)
            except Exception as e:
                print(f"BT Scan: {e}")
            finally:
                self.scanning = False

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_scan())
                loop.close()
            except Exception as e:
                self.scanning = False
                print(f"BT thread: {e}")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def set_target(self, name, rssi):
        self.device_name = name
        self.rssi = rssi
        self.battery = max(5, min(100, int((rssi + 100) * 2.5)))

# ========== MAIN APPLICATION ==========
class GlobalSpeakerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔊 GLOBAL SPEAKER SUITE – BASS CANNON EDITION 🔊")
        self.configure(bg=PALETTE['bg'])
        self.geometry("1280x860")
        self.minsize(1000, 700)

        self.processor = AudioProcessor()
        self.engine = AudioEngine(self.processor)
        self.tts = TTSEngine()
        self.bt = BTMonitor()

        self.recording = False
        self.loopback_on = False
        self._rec_start = None
        self._rec_timer_job = None

        self._build_ui()
        self._start_live_update()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ========== UI BUILD ==========
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=PALETTE['bg'], pady=6)
        top.pack(fill='x', padx=12)
        tk.Label(top, text="⚡ GLOBAL SPEAKER & BASS DIAGNOSTIC SUITE ⚡",
                 font=('Consolas', 16, 'bold'), bg=PALETTE['bg'], fg=PALETTE['accent']).pack(side='left')
        self.status_lbl = tk.Label(top, text="● READY", font=('Consolas', 10, 'bold'),
                                   bg=PALETTE['bg'], fg=PALETTE['green'])
        self.status_lbl.pack(side='right', padx=8)
        self.clock_lbl = tk.Label(top, text="", font=('Consolas', 10),
                                  bg=PALETTE['bg'], fg=PALETTE['muted'])
        self.clock_lbl.pack(side='right', padx=16)
        self._tick_clock()

        # Notebook (Tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=12, pady=8)

        # Tab 1: Main Controls
        self.tab_main = tk.Frame(self.notebook, bg=PALETTE['bg'])
        self.notebook.add(self.tab_main, text="🎛️ Main Controls")
        self._build_main_tab()

        # Tab 2: Audio Processing (EQ, Bass Boost)
        self.tab_proc = tk.Frame(self.notebook, bg=PALETTE['bg'])
        self.notebook.add(self.tab_proc, text="🎚️ Audio Processing")
        self._build_processing_tab()

        # Tab 3: Analyzer
        self.tab_analyzer = tk.Frame(self.notebook, bg=PALETTE['bg'])
        self.notebook.add(self.tab_analyzer, text="📊 Analyzer")
        self._build_analyzer_tab()

        # Tab 4: Recording & Loopback
        self.tab_rec = tk.Frame(self.notebook, bg=PALETTE['bg'])
        self.notebook.add(self.tab_rec, text="🎤 Recording & Loopback")
        self._build_recording_tab()

        # Tab 5: Bluetooth Info
        self.tab_bt = tk.Frame(self.notebook, bg=PALETTE['bg'])
        self.notebook.add(self.tab_bt, text="🔷 Bluetooth Info")
        self._build_bt_tab()

        # Bottom bar
        bot = tk.Frame(self, bg=PALETTE['bg2'], height=28)
        bot.pack(fill='x', side='bottom')
        tk.Label(bot, text="  Bass Cannon Ready – Turn up volume for DARK BOOM",
                 font=('Consolas', 8), bg=PALETTE['bg2'], fg=PALETTE['text_dim']).pack(side='left')
        self.rec_timer_lbl = tk.Label(bot, text="", font=('Consolas', 9, 'bold'),
                                      bg=PALETTE['bg2'], fg=PALETTE['danger'])
        self.rec_timer_lbl.pack(side='right', padx=12)

    # ---- Tab 1: Main Controls (Frequency tests etc) ----
    def _build_main_tab(self):
        # Frequency tests
        self._section(self.tab_main, "⚡ FREQUENCY TESTS")
        frame_freq = tk.Frame(self.tab_main, bg=PALETTE['bg'])
        frame_freq.pack(fill='x', pady=4)
        tests = [
            ("20 Hz\nSUB", 20, '#ff0055'), ("40 Hz\nSUB", 40, '#ff2200'),
            ("60 Hz\nBASSS", 60, '#ff4400'), ("80 Hz\nBASS", 80, '#ff6600'),
            ("120 Hz\nBASS", 120, '#ff8800'), ("200 Hz\nLO-MID", 200, '#ffaa00'),
            ("500 Hz\nMID", 500, '#ffdd00'), ("1 kHz\nMID", 1000, '#ccff00'),
            ("4 kHz\nHI-MID", 4000, '#66ff66'), ("8 kHz\nHIGH", 8000, '#00ffcc'),
            ("12 kHz\nAIR", 12000, '#00aaff'), ("20 kHz\nAIR", 20000, '#aa44ff')
        ]
        for i, (label, freq, color) in enumerate(tests):
            btn = self._btn(frame_freq, label, lambda f=freq: self._play_freq(f), color, w=7, h=2)
            btn.grid(row=0, column=i, padx=2, pady=2, sticky='ew')
            frame_freq.columnconfigure(i, weight=1)

        # Sweeps & Special
        self._section(self.tab_main, "🌊 SWEEP · NOISE · SPECIAL")
        frame_sweep = tk.Frame(self.tab_main, bg=PALETTE['bg'])
        frame_sweep.pack(fill='x', pady=4)
        sweeps = [
            ("FULL SWEEP\n20→20k", '#00ffcc', self._sweep_full),
            ("BASS SWEEP\n40→200", '#ff6644', self._sweep_bass),
            ("WHITE NOISE", '#ccccff', self._white_noise),
            ("PINK NOISE", '#ffaaff', self._pink_noise),
            ("BROWN NOISE", '#cc8844', self._brown_noise),
            ("BASS THUMPS", '#ff3300', self._bass_thump),
            ("MULTITONE", '#00ff88', self._multitone),
            ("BINAURAL\nBEATS", '#aa88ff', self._binaural),
        ]
        for i, (label, color, cmd) in enumerate(sweeps):
            btn = self._btn(frame_sweep, label, cmd, color, w=9, h=3)
            btn.grid(row=0, column=i, padx=2, pady=2, sticky='ew')
            frame_sweep.columnconfigure(i, weight=1)

        # Channel test & custom
        row = tk.Frame(self.tab_main, bg=PALETTE['bg'])
        row.pack(fill='x', pady=8)
        # Channel
        ch_frame = tk.LabelFrame(row, text=" CHANNEL TEST ", font=('Consolas', 8),
                                 bg=PALETTE['bg'], fg=PALETTE['accent'])
        ch_frame.pack(side='left', padx=5)
        for side, color in [("◀ LEFT", '#4488ff'), ("▶ RIGHT", '#44ffaa'), ("◀▶ BOTH", '#ffffff')]:
            s = side.split()[-1].lower()
            self._btn(ch_frame, side, lambda x=s: self._channel(x), color, w=8, h=1).pack(side='left', padx=2, pady=4)

        # Custom
        cf_frame = tk.LabelFrame(row, text=" CUSTOM FREQUENCY ", font=('Consolas', 8),
                                 bg=PALETTE['bg'], fg=PALETTE['warn'])
        cf_frame.pack(side='left', padx=5)
        tk.Label(cf_frame, text="Hz:", bg=PALETTE['bg'], fg=PALETTE['text']).pack(side='left', padx=4)
        self.cf_hz = tk.Entry(cf_frame, width=7, bg=PALETTE['bg3'], fg=PALETTE['accent'], insertbackground=PALETTE['accent'])
        self.cf_hz.insert(0, "440")
        self.cf_hz.pack(side='left')
        tk.Label(cf_frame, text="sec:", bg=PALETTE['bg'], fg=PALETTE['text']).pack(side='left', padx=4)
        self.cf_dur = tk.Entry(cf_frame, width=4, bg=PALETTE['bg3'], fg=PALETTE['accent'])
        self.cf_dur.insert(0, "3")
        self.cf_dur.pack(side='left')
        self._btn(cf_frame, "▶ PLAY", self._play_custom, PALETTE['accent'], w=6, h=1).pack(side='left', padx=6, pady=4)

        # Volume & Stop
        vol_frame = tk.LabelFrame(row, text=" MASTER VOLUME ", font=('Consolas', 8),
                                  bg=PALETTE['bg'], fg=PALETTE['accent3'])
        vol_frame.pack(side='left', padx=5)
        self.vol_var = tk.DoubleVar(value=0.8)
        vol_sl = tk.Scale(vol_frame, from_=0, to=1, resolution=0.01, orient='horizontal',
                          variable=self.vol_var, command=self._set_vol,
                          length=150, bg=PALETTE['bg3'], fg=PALETTE['accent3'],
                          troughcolor=PALETTE['bg2'], highlightthickness=0, showvalue=False)
        vol_sl.pack(padx=6, pady=2)
        self.vol_lbl = tk.Label(vol_frame, text="80%", font=('Consolas', 11, 'bold'),
                                bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.vol_lbl.pack()

        self._btn(row, "⏹ STOP ALL", self._stop_all, PALETTE['danger'], w=12, h=2).pack(side='right', padx=10)

    # ---- Tab 2: Audio Processing (EQ, Bass Boost, etc) ----
    def _build_processing_tab(self):
        proc_frame = tk.Frame(self.tab_proc, bg=PALETTE['bg'])
        proc_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # EQ section
        eq_frame = tk.LabelFrame(proc_frame, text="📢 EQUALIZER", font=('Consolas', 10, 'bold'),
                                 bg=PALETTE['bg'], fg=PALETTE['accent'])
        eq_frame.pack(fill='x', pady=6)
        # Bass
        tk.Label(eq_frame, text="Bass (dB):", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.bass_slider = tk.Scale(eq_frame, from_=-12, to=24, resolution=1, orient='horizontal',
                                    length=300, bg=PALETTE['bg3'], fg=PALETTE['accent3'], troughcolor=PALETTE['bg2'])
        self.bass_slider.set(0)
        self.bass_slider.grid(row=0, column=1, padx=5)
        self.bass_val = tk.Label(eq_frame, text="0 dB", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.bass_val.grid(row=0, column=2, padx=5)
        self.bass_slider.configure(command=lambda x: self._update_eq())

        # Mid
        tk.Label(eq_frame, text="Mid  (dB):", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.mid_slider = tk.Scale(eq_frame, from_=-12, to=24, resolution=1, orient='horizontal', length=300, bg=PALETTE['bg3'], fg=PALETTE['accent3'], troughcolor=PALETTE['bg2'])
        self.mid_slider.set(0)
        self.mid_slider.grid(row=1, column=1, padx=5)
        self.mid_val = tk.Label(eq_frame, text="0 dB", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.mid_val.grid(row=1, column=2, padx=5)
        self.mid_slider.configure(command=lambda x: self._update_eq())

        # Treble
        tk.Label(eq_frame, text="Treble (dB):", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.treb_slider = tk.Scale(eq_frame, from_=-12, to=24, resolution=1, orient='horizontal', length=300, bg=PALETTE['bg3'], fg=PALETTE['accent3'], troughcolor=PALETTE['bg2'])
        self.treb_slider.set(0)
        self.treb_slider.grid(row=2, column=1, padx=5)
        self.treb_val = tk.Label(eq_frame, text="0 dB", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.treb_val.grid(row=2, column=2, padx=5)
        self.treb_slider.configure(command=lambda x: self._update_eq())

        # Bass Boost & Bass Cannon
        boost_frame = tk.LabelFrame(proc_frame, text="💥 BASS BOOST / CANNON", font=('Consolas', 10, 'bold'),
                                    bg=PALETTE['bg'], fg=PALETTE['danger'])
        boost_frame.pack(fill='x', pady=6)
        tk.Label(boost_frame, text="Bass Boost Factor:", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=0, column=0, padx=5, pady=5)
        self.boost_slider = tk.Scale(boost_frame, from_=1.0, to=3.0, resolution=0.1, orient='horizontal',
                                     length=250, bg=PALETTE['bg3'], troughcolor=PALETTE['bg2'])
        self.boost_slider.set(1.0)
        self.boost_slider.grid(row=0, column=1, padx=5)
        self.boost_val = tk.Label(boost_frame, text="1.0x", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.boost_val.grid(row=0, column=2, padx=5)
        self.boost_slider.configure(command=lambda x: self._update_boost())

        self.bass_cannon_var = tk.BooleanVar()
        cb = tk.Checkbutton(boost_frame, text="🔥 BASS CANNON MODE (EXTREME BOOST)", variable=self.bass_cannon_var,
                            bg=PALETTE['bg'], fg=PALETTE['danger'], selectcolor=PALETTE['bg'],
                            command=self._toggle_bass_cannon)
        cb.grid(row=1, column=0, columnspan=3, pady=5)

        # Compressor & Noise Gate
        comp_frame = tk.LabelFrame(proc_frame, text="⚙️ COMPRESSOR & NOISE GATE", font=('Consolas', 10, 'bold'),
                                   bg=PALETTE['bg'], fg=PALETTE['accent2'])
        comp_frame.pack(fill='x', pady=6)
        tk.Label(comp_frame, text="Comp Thresh:", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=0, column=0, padx=5)
        self.comp_thresh = tk.Scale(comp_frame, from_=0.01, to=1.0, resolution=0.01, orient='horizontal', length=200,
                                    bg=PALETTE['bg3'], troughcolor=PALETTE['bg2'])
        self.comp_thresh.set(0.5)
        self.comp_thresh.grid(row=0, column=1, padx=5)
        self.comp_thresh_val = tk.Label(comp_frame, text="0.50", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.comp_thresh_val.grid(row=0, column=2, padx=5)
        self.comp_thresh.configure(command=lambda x: self._update_compressor())

        tk.Label(comp_frame, text="Comp Ratio:", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=1, column=0, padx=5)
        self.comp_ratio = tk.Scale(comp_frame, from_=1.0, to=10.0, resolution=0.5, orient='horizontal', length=200,
                                   bg=PALETTE['bg3'], troughcolor=PALETTE['bg2'])
        self.comp_ratio.set(2.0)
        self.comp_ratio.grid(row=1, column=1, padx=5)
        self.comp_ratio_val = tk.Label(comp_frame, text="2.0", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.comp_ratio_val.grid(row=1, column=2, padx=5)
        self.comp_ratio.configure(command=lambda x: self._update_compressor())

        tk.Label(comp_frame, text="Noise Gate:", bg=PALETTE['bg'], fg=PALETTE['text']).grid(row=2, column=0, padx=5)
        self.gate_thresh = tk.Scale(comp_frame, from_=0.0, to=0.1, resolution=0.001, orient='horizontal', length=200,
                                    bg=PALETTE['bg3'], troughcolor=PALETTE['bg2'])
        self.gate_thresh.set(0.01)
        self.gate_thresh.grid(row=2, column=1, padx=5)
        self.gate_val = tk.Label(comp_frame, text="0.010", bg=PALETTE['bg'], fg=PALETTE['accent3'])
        self.gate_val.grid(row=2, column=2, padx=5)
        self.gate_thresh.configure(command=lambda x: self._update_gate())

        # Reset button
        self._btn(proc_frame, "🔄 RESET ALL PROCESSING", self._reset_processing, PALETTE['warn'], w=25, h=1).pack(pady=10)

    # ---- Tab 3: Analyzer ----
    def _build_analyzer_tab(self):
        # Oscilloscope
        osc_frame = tk.LabelFrame(self.tab_analyzer, text=" OSCILLOSCOPE ", font=('Consolas', 9),
                                  bg=PALETTE['bg'], fg=PALETTE['accent'])
        osc_frame.pack(fill='x', padx=6, pady=4)
        self.osc_canvas = tk.Canvas(osc_frame, height=120, bg=PALETTE['bg2'], highlightthickness=0)
        self.osc_canvas.pack(fill='both', expand=True, padx=4, pady=4)

        # Spectrum
        spec_frame = tk.LabelFrame(self.tab_analyzer, text=" SPECTRUM ANALYZER ", font=('Consolas', 9),
                                   bg=PALETTE['bg'], fg=PALETTE['accent2'])
        spec_frame.pack(fill='x', padx=6, pady=4)
        self.spec_canvas = tk.Canvas(spec_frame, height=120, bg=PALETTE['bg2'], highlightthickness=0)
        self.spec_canvas.pack(fill='both', expand=True, padx=4, pady=4)

        # VU Meters
        meters_frame = tk.Frame(self.tab_analyzer, bg=PALETTE['bg'])
        meters_frame.pack(fill='x', padx=6, pady=4)
        # RMS
        tk.Label(meters_frame, text="RMS:", bg=PALETTE['bg'], fg=PALETTE['text']).pack(side='left')
        self.rms_canvas = self._meter_canvas(meters_frame)
        self.rms_canvas.pack(side='left', fill='x', expand=True, padx=5)
        # Peak
        tk.Label(meters_frame, text="PEAK:", bg=PALETTE['bg'], fg=PALETTE['text']).pack(side='left')
        self.peak_canvas = self._meter_canvas(meters_frame)
        self.peak_canvas.pack(side='left', fill='x', expand=True, padx=5)

        # Band levels
        band_frame = tk.LabelFrame(self.tab_analyzer, text=" FREQUENCY BANDS ", font=('Consolas', 9),
                                   bg=PALETTE['bg'], fg=PALETTE['accent3'])
        band_frame.pack(fill='x', padx=6, pady=4)
        self.band_bars = {}
        for name, _, _, color in FREQ_BANDS:
            row = tk.Frame(band_frame, bg=PALETTE['bg'])
            row.pack(fill='x', padx=5, pady=2)
            tk.Label(row, text=f"{name:7s}", font=('Consolas', 8), bg=PALETTE['bg'], fg=color, width=7).pack(side='left')
            c = tk.Canvas(row, height=14, bg=PALETTE['bg3'], highlightthickness=0)
            c.pack(side='left', fill='x', expand=True)
            self.band_bars[name] = (c, color)

        # Info labels
        info_frame = tk.LabelFrame(self.tab_analyzer, text=" LIVE INFO ", font=('Consolas', 9),
                                   bg=PALETTE['bg'], fg=PALETTE['accent'])
        info_frame.pack(fill='x', padx=6, pady=4)
        self.info_labels = {}
        fields = [("THD", "–"), ("RMS dB", "–"), ("Peak dB", "–"), ("Quality", "–")]
        for key, val in fields:
            row = tk.Frame(info_frame, bg=PALETTE['bg'])
            row.pack(fill='x', padx=8, pady=2)
            tk.Label(row, text=f"{key}:", font=('Consolas', 8), bg=PALETTE['bg'], fg=PALETTE['text_dim'], width=10, anchor='w').pack(side='left')
            lbl = tk.Label(row, text=val, font=('Consolas', 8, 'bold'), bg=PALETTE['bg'], fg=PALETTE['accent2'])
            lbl.pack(side='left')
            self.info_labels[key] = lbl

    # ---- Tab 4: Recording & Loopback ----
    def _build_recording_tab(self):
        frame = tk.Frame(self.tab_rec, bg=PALETTE['bg'])
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.rec_btn = self._btn(frame, "⏺  START RECORDING", self._toggle_rec, PALETTE['danger'], w=20, h=2)
        self.rec_btn.pack(pady=5)

        self._btn(frame, "▶ PLAY LAST RECORDING", self._play_rec, PALETTE['accent'], w=20, h=2).pack(pady=5)
        self._btn(frame, "💾 SAVE RECORDING (WITH EFFECTS)", self._save_rec, '#aaaaff', w=25, h=2).pack(pady=5)

        self.loop_btn = self._btn(frame, "🔁 LOOPBACK OFF", self._toggle_loop, PALETTE['warn'], w=20, h=2)
        self.loop_btn.pack(pady=5)

        self._btn(frame, "🔊 ANNOUNCE SPEAKER STATUS", self._announce_info, PALETTE['purple'], w=25, h=2).pack(pady=5)

        self.rec_status_label = tk.Label(frame, text="", font=('Consolas', 10), bg=PALETTE['bg'], fg=PALETTE['danger'])
        self.rec_status_label.pack(pady=5)

    # ---- Tab 5: Bluetooth ----
    def _build_bt_tab(self):
        frame = tk.Frame(self.tab_bt, bg=PALETTE['bg'])
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        self._btn(frame, "🔍 SCAN BLUETOOTH DEVICES", self._bt_scan, PALETTE['pink'], w=25, h=1).pack(pady=5)

        self.bt_list = tk.Listbox(frame, height=8, bg=PALETTE['bg3'], fg=PALETTE['text'],
                                  font=('Consolas', 9), selectbackground=PALETTE['accent'])
        self.bt_list.pack(fill='x', pady=5)
        self.bt_list.bind('<<ListboxSelect>>', self._bt_select)

        self.bt_status = tk.Label(frame, text="Not scanning", bg=PALETTE['bg'], fg=PALETTE['text_dim'])
        self.bt_status.pack()

        self._btn(frame, "📢 ANNOUNCE BT DEVICE INFO", self._announce_bt, PALETTE['accent'], w=25, h=1).pack(pady=5)

        # Battery visual
        bat_frame = tk.LabelFrame(frame, text=" BATTERY (estimated) ", font=('Consolas', 8),
                                  bg=PALETTE['bg'], fg=PALETTE['warn'])
        bat_frame.pack(fill='x', pady=10)
        self.bat_canvas = tk.Canvas(bat_frame, height=60, bg=PALETTE['bg2'], highlightthickness=0)
        self.bat_canvas.pack(fill='x', padx=6, pady=6)
        self.bat_pct_lbl = tk.Label(bat_frame, text="–", font=('Consolas', 12, 'bold'), bg=PALETTE['bg'], fg=PALETTE['warn'])
        self.bat_pct_lbl.pack()
        self._draw_battery(None)

        # Speaker Info
        info_frame = tk.LabelFrame(frame, text=" SPEAKER INFO ", font=('Consolas', 8),
                                   bg=PALETTE['bg'], fg=PALETTE['accent2'])
        info_frame.pack(fill='x', pady=10)
        self.bt_info_labels = {}
        fields = [("Device", "Unknown"), ("Battery", "–"), ("RSSI", "–")]
        for key, val in fields:
            row = tk.Frame(info_frame, bg=PALETTE['bg'])
            row.pack(fill='x', padx=8, pady=2)
            tk.Label(row, text=f"{key}:", font=('Consolas', 8), bg=PALETTE['bg'], fg=PALETTE['text_dim'], width=10).pack(side='left')
            lbl = tk.Label(row, text=val, font=('Consolas', 8, 'bold'), bg=PALETTE['bg'], fg=PALETTE['accent2'])
            lbl.pack(side='left')
            self.bt_info_labels[key] = lbl

    # ========== UI HELPERS ==========
    def _btn(self, parent, text, cmd, color, w=10, h=1):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=PALETTE['bg3'], fg=color,
                      font=('Consolas', 8, 'bold'),
                      activebackground=PALETTE['bg2'], activeforeground=color,
                      relief='flat', bd=0, width=w, height=h,
                      cursor='hand2', wraplength=100)
        b.bind('<Enter>', lambda e: b.config(bg=PALETTE['muted']))
        b.bind('<Leave>', lambda e: b.config(bg=PALETTE['bg3']))
        return b

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=PALETTE['bg'])
        f.pack(fill='x', pady=(6, 2))
        tk.Label(f, text=title, font=('Consolas', 9, 'bold'),
                 bg=PALETTE['bg'], fg=PALETTE['accent']).pack(side='left')
        tk.Frame(f, bg=PALETTE['muted'], height=1).pack(side='left', fill='x', expand=True, padx=8)

    def _meter_canvas(self, parent):
        c = tk.Canvas(parent, height=14, bg=PALETTE['bg3'], highlightthickness=0)
        c.pack(side='left', fill='x', expand=True, padx=2)
        return c

    def _tick_clock(self):
        self.clock_lbl.config(text=datetime.now().strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ========== PROCESSING CALLBACKS ==========
    def _update_eq(self):
        bass = self.bass_slider.get()
        mid = self.mid_slider.get()
        treb = self.treb_slider.get()
        self.bass_val.config(text=f"{bass:+} dB")
        self.mid_val.config(text=f"{mid:+} dB")
        self.treb_val.config(text=f"{treb:+} dB")
        self.processor.set_eq(bass, mid, treb)

    def _update_boost(self):
        factor = self.boost_slider.get()
        self.boost_val.config(text=f"{factor:.1f}x")
        self.processor.set_bass_boost(factor)

    def _toggle_bass_cannon(self):
        on = self.bass_cannon_var.get()
        self.processor.bass_cannon_mode = on
        if on:
            self.processor.set_bass_boost(3.0)
            self.boost_slider.set(3.0)
            self.boost_val.config(text="3.0x")
            self.status_lbl.config(text="🔥 BASS CANNON ACTIVE", fg=PALETTE['danger'])

    def _update_compressor(self):
        thresh = self.comp_thresh.get()
        ratio = self.comp_ratio.get()
        self.comp_thresh_val.config(text=f"{thresh:.2f}")
        self.comp_ratio_val.config(text=f"{ratio:.1f}")
        self.processor.set_compressor(thresh, ratio)

    def _update_gate(self):
        gate = self.gate_thresh.get()
        self.gate_val.config(text=f"{gate:.3f}")
        self.processor.set_noise_gate(gate)

    def _reset_processing(self):
        self.bass_slider.set(0)
        self.mid_slider.set(0)
        self.treb_slider.set(0)
        self.boost_slider.set(1.0)
        self.bass_cannon_var.set(False)
        self.comp_thresh.set(0.5)
        self.comp_ratio.set(2.0)
        self.gate_thresh.set(0.01)
        self._update_eq()
        self._update_boost()
        self._update_compressor()
        self._update_gate()
        self.processor.bass_cannon_mode = False
        self.status_lbl.config(text="Processing reset", fg=PALETTE['green'])

    # ========== AUDIO ACTIONS ==========
    def _set_vol(self, val):
        self.processor.volume = float(val)
        pct = int(float(val) * 100)
        self.vol_lbl.config(text=f"{pct}%")

    def _set_status(self, text, color=None):
        c = color or PALETTE['green']
        self.status_lbl.config(text=f"● {text}", fg=c)

    def _play_freq(self, freq):
        self._set_status(f"PLAYING {freq} Hz", PALETTE['accent2'])
        d = self.engine.sine(freq, 3.0)
        self.engine.play(d, on_done=lambda: self._set_status("READY"))

    def _sweep_full(self):
        self._set_status("FULL SWEEP 20-20kHz", PALETTE['accent'])
        d = self.engine.sweep(20, 20000, 6.0)
        self.engine.play(d, on_done=lambda: self._set_status("READY"))

    def _sweep_bass(self):
        self._set_status("BASS SWEEP 40-200Hz", PALETTE['accent2'])
        d = self.engine.sweep(40, 200, 4.0)
        self.engine.play(d, on_done=lambda: self._set_status("READY"))

    def _white_noise(self):
        self._set_status("WHITE NOISE", '#ccccff')
        self.engine.play(self.engine.white_noise(4.0), on_done=lambda: self._set_status("READY"))

    def _pink_noise(self):
        self._set_status("PINK NOISE", '#ffaaff')
        self.engine.play(self.engine.pink_noise(4.0), on_done=lambda: self._set_status("READY"))

    def _brown_noise(self):
        self._set_status("BROWN NOISE", '#cc8844')
        self.engine.play(self.engine.brown_noise(4.0), on_done=lambda: self._set_status("READY"))

    def _bass_thump(self):
        self._set_status("BASS THUMPS", PALETTE['danger'])
        self.engine.play(self.engine.thump(12), on_done=lambda: self._set_status("READY"))

    def _multitone(self):
        self._set_status("MULTITONE", PALETTE['accent3'])
        freqs = [60, 120, 250, 500, 1000, 2000, 4000, 8000]
        self.engine.play(self.engine.multitone(freqs, 4.0), on_done=lambda: self._set_status("READY"))

    def _binaural(self):
        self._set_status("BINAURAL BEATS (headphones)", PALETTE['purple'])
        d = self.engine.binaural(200, 10, 30.0)
        self.engine.play(d, on_done=lambda: self._set_status("READY"))

    def _channel(self, side):
        self._set_status(f"CHANNEL: {side.upper()}", PALETTE['accent'])
        d = self.engine.channel_test(side, 1000, 2.5)
        self.engine.play(d, on_done=lambda: self._set_status("READY"))

    def _play_custom(self):
        try:
            freq = float(self.cf_hz.get())
            dur = float(self.cf_dur.get())
            if freq <= 0 or dur <= 0:
                raise ValueError
            self._set_status(f"CUSTOM {freq:.0f} Hz", PALETTE['warn'])
            self.engine.play(self.engine.sine(freq, dur), on_done=lambda: self._set_status("READY"))
        except:
            messagebox.showerror("خطأ", "أدخل قيم صحيحة للتردد والمدة")

    def _toggle_rec(self):
        if not self.recording:
            self.recording = True
            self._rec_start = time.time()
            self.engine.start_rec()
            self.rec_btn.config(text="⏹ STOP RECORDING", fg='#ff0000')
            self._set_status("⏺ RECORDING", PALETTE['danger'])
            self._update_rec_timer()
        else:
            self.recording = False
            self.engine.stop_rec()
            if self._rec_timer_job:
                self.after_cancel(self._rec_timer_job)
            dur = time.time() - self._rec_start if self._rec_start else 0
            self.rec_btn.config(text="⏺  START RECORDING", fg=PALETTE['danger'])
            self.rec_timer_lbl.config(text=f"  Recorded: {dur:.1f}s")
            self.rec_status_label.config(text=f"Recorded {dur:.1f} seconds")
            self._set_status("RECORDING DONE", PALETTE['accent3'])

    def _update_rec_timer(self):
        if self.recording and self._rec_start:
            elapsed = time.time() - self._rec_start
            self.rec_timer_lbl.config(text=f"  ⏺ {elapsed:.1f}s  ", fg=PALETTE['danger'])
            self._rec_timer_job = self.after(100, self._update_rec_timer)

    def _play_rec(self):
        if self.engine.get_rec() is None:
            messagebox.showinfo("معلومة", "لم يتم تسجيل أي صوت بعد")
            return
        self._set_status("PLAYING RECORDING (with effects)", PALETTE['accent'])
        self.engine.play_rec(on_done=lambda: self._set_status("READY"))

    def _save_rec(self):
        if self.engine.get_rec() is None:
            messagebox.showinfo("معلومة", "لا يوجد تسجيل")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=f"speaker_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        if path and self.engine.save_rec(path, apply_processing=True):
            messagebox.showinfo("تم الحفظ ✓", f"تم الحفظ مع تأثيرات المعالجة:\n{path}")

    def _toggle_loop(self):
        if not self.loopback_on:
            self.loopback_on = True
            self.engine.start_loopback()
            self.loop_btn.config(text="🔁 LOOPBACK ON", fg='#ffff00')
            self._set_status("LOOPBACK ACTIVE (processed)", PALETTE['warn'])
        else:
            self.loopback_on = False
            self.engine.stop_loopback()
            self.loop_btn.config(text="🔁 LOOPBACK OFF", fg=PALETTE['warn'])
            self._set_status("READY")

    def _stop_all(self):
        self.recording = False
        self.loopback_on = False
        self.engine.stop_all()
        self.rec_btn.config(text="⏺  START RECORDING", fg=PALETTE['danger'])
        self.loop_btn.config(text="🔁 LOOPBACK OFF", fg=PALETTE['warn'])
        self._set_status("STOPPED", PALETTE['danger'])

    # ========== ANNOUNCEMENTS ==========
    def _announce_info(self):
        rms_db = 20 * np.log10(self.engine.rms_level + 1e-9)
        peak_db = 20 * np.log10(self.engine.peak_level + 1e-9)
        thd = self.engine.thd
        bt_name = self.bt.device_name
        bat = self.bt.battery
        rssi = self.bt.rssi
        lines = ["تقرير السماعة"]
        lines.append(f"اسم الجهاز: {bt_name}")
        if bat: lines.append(f"البطارية: {bat}%")
        if rssi: lines.append(f"الإشارة: {rssi} dBm")
        lines.append(f"مستوى الصوت: {rms_db:.0f} dB")
        lines.append(f"التشويه: {thd:.1f}%")
        if self.engine.rms_level > 0.3: q = "ممتاز"
        elif self.engine.rms_level > 0.1: q = "جيد"
        elif self.engine.rms_level > 0.01: q = "ضعيف"
        else: q = "لا يوجد صوت"
        lines.append(f"التقييم: {q}")
        text = ". ".join(lines)
        self.tts.speak(text)
        self._set_status("📢 ANNOUNCING", PALETTE['purple'])
        self.after(3000, lambda: self._set_status("READY"))

    def _announce_bt(self):
        if not BT_AVAILABLE:
            self.tts.speak("مكتبة البلوتوث غير متاحة")
            return
        sel = self.bt_list.curselection()
        if not sel:
            self.tts.speak("لم يتم اختيار جهاز")
            return
        txt = self.bt_list.get(sel[0])
        name = txt.split("|")[0].strip()
        rssi = self.bt.rssi or "–"
        bat = self.bt.battery
        msg = f"الجهاز: {name}. "
        if bat: msg += f"بطارية {bat}%. "
        msg += f"قوة الإشارة {rssi} ديسيبل."
        if bat and bat < 20: msg += " تحذير البطارية منخفضة."
        self.tts.speak(msg)

    # ========== BLUETOOTH ==========
    def _bt_scan(self):
        if not BT_AVAILABLE:
            self.bt_status.config(text="install bleak", fg=PALETTE['danger'])
            return
        self.bt_status.config(text="🔍 Scanning...", fg=PALETTE['accent'])
        self.bt_list.delete(0, tk.END)
        self.bt_list.insert(tk.END, "Scanning for 5 seconds...")
        self.bt.start_scan(callback=self._bt_update)

    def _bt_update(self, devices):
        self.bt_list.delete(0, tk.END)
        if not devices:
            self.bt_list.insert(tk.END, "No devices found")
            self.bt_status.config(text="No devices", fg=PALETTE['text_dim'])
            return
        for addr, info in sorted(devices.items(), key=lambda x: x[1]['rssi'], reverse=True):
            name = info['name'] or "Unknown"
            rssi = info['rssi']
            bar = "█" * max(0, min(10, (rssi + 100) // 5))
            self.bt_list.insert(tk.END, f"{name:25s} | {rssi:4d} dBm |{bar}")
        self.bt_status.config(text=f"Found {len(devices)} devices", fg=PALETTE['accent3'])

    def _bt_select(self, event):
        sel = self.bt_list.curselection()
        if not sel: return
        txt = self.bt_list.get(sel[0])
        parts = txt.split("|")
        if len(parts) >= 2:
            name = parts[0].strip()
            try:
                rssi = int(parts[1].strip().replace("dBm", "").strip())
            except:
                rssi = -70
            self.bt.set_target(name, rssi)
            self.bt_info_labels['Device'].config(text=name[:18])
            self.bt_info_labels['RSSI'].config(text=f"{rssi} dBm")
            if self.bt.battery:
                self.bt_info_labels['Battery'].config(text=f"{self.bt.battery}%")
                self._draw_battery(self.bt.battery)

    # ========== LIVE UPDATE ==========
    def _start_live_update(self):
        self._live_update()

    def _live_update(self):
        try:
            self._draw_osc()
            self._draw_spectrum()
            self._update_meters()
            self._update_info_labels()
        except:
            pass
        self.after(50, self._live_update)

    def _draw_osc(self):
        c = self.osc_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10: return
        c.delete('all')
        for i in range(1, 4):
            y = h * i // 4
            c.create_line(0, y, w, y, fill=PALETTE['grid'], dash=(4, 4))
        c.create_line(0, h//2, w, h//2, fill=PALETTE['muted'], width=1)
        wave = self.engine.get_waveform()
        if len(wave) > 4:
            n = min(len(wave), w)
            step = len(wave) / n
            pts = []
            for i in range(n):
                sample = wave[int(i * step)]
                x = i
                y = int(h / 2 - sample * h / 2 * 0.9)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(pts, fill=PALETTE['green'], width=1, smooth=False)
        c.create_text(4, 4, text="OSC", font=('Consolas', 7), fill=PALETTE['accent'], anchor='nw')

    def _draw_spectrum(self):
        c = self.spec_canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10 or h < 10: return
        c.delete('all')
        fft = self.engine.get_fft()
        if len(fft) < 4: return
        n_bars = 80
        bar_w = max(1, w // n_bars - 1)
        colors = ['#ff0044', '#ff3300', '#ff6600', '#ffaa00', '#ffee00', '#aaff00', '#00ffaa', '#00aaff', '#aa44ff']
        for i in range(n_bars):
            idx = int(i * len(fft) / n_bars)
            level = min(1.0, float(fft[idx]) * 60)
            bar_h = int(level * (h - 8))
            x1 = i * (bar_w + 1)
            x2 = x1 + bar_w
            y2 = h - 4
            y1 = y2 - bar_h
            ci = min(len(colors) - 1, int(i / n_bars * len(colors)))
            c.create_rectangle(x1, y1, x2, y2, fill=colors[ci], outline='')
            if bar_h > 3:
                c.create_rectangle(x1, y1, x2, y1 + 2, fill='#ffffff', outline='')
        c.create_text(4, 4, text="FFT", font=('Consolas', 7), fill=PALETTE['accent2'], anchor='nw')
        c.create_text(w - 4, 4, text="20kHz", font=('Consolas', 6), fill=PALETTE['text_dim'], anchor='ne')

    def _update_meters(self):
        rms = self.engine.rms_level
        peak = self.engine.peak_level
        self._draw_meter(self.rms_canvas, rms)
        self._draw_meter(self.peak_canvas, peak)
        levels = self.engine.get_band_levels()
        for i, (name, _, _, _) in enumerate(FREQ_BANDS):
            if name in self.band_bars:
                c, color = self.band_bars[name]
                w = c.winfo_width()
                h = c.winfo_height()
                if w < 2 or h < 2: continue
                c.delete('all')
                level = min(1.0, levels[i] if i < len(levels) else 0)
                bw = int(level * w)
                if bw > 0:
                    c.create_rectangle(0, 0, bw, h, fill=color, outline='')

    def _draw_meter(self, canvas, level):
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 2: return
        canvas.delete('all')
        n = 30
        for i in range(n):
            x1 = 2 + i * (w - 4) // n
            x2 = x1 + (w - 4) // n - 1
            frac = i / n
            bg = '#1a3a1a' if frac < 0.6 else '#3a3a1a' if frac < 0.8 else '#3a1a1a'
            canvas.create_rectangle(x1, 2, x2, h - 2, fill=bg, outline='')
        active = int(level * n * 2.5)
        for i in range(min(active, n)):
            x1 = 2 + i * (w - 4) // n
            x2 = x1 + (w - 4) // n - 1
            frac = i / n
            col = PALETTE['green'] if frac < 0.6 else PALETTE['warn'] if frac < 0.8 else PALETTE['danger']
            canvas.create_rectangle(x1, 2, x2, h - 2, fill=col, outline='')

    def _update_info_labels(self):
        rms_db = 20 * np.log10(self.engine.rms_level + 1e-9)
        peak_db = 20 * np.log10(self.engine.peak_level + 1e-9)
        thd = self.engine.thd
        self.info_labels['THD'].config(text=f"{thd:.1f}%", fg=PALETTE['danger'] if thd > 5 else PALETTE['accent3'])
        self.info_labels['RMS dB'].config(text=f"{rms_db:.1f} dB")
        self.info_labels['Peak dB'].config(text=f"{peak_db:.1f} dB")
        if self.engine.rms_level > 0.3:
            q, qc = "EXCELLENT", PALETTE['green']
        elif self.engine.rms_level > 0.1:
            q, qc = "GOOD", PALETTE['accent3']
        elif self.engine.rms_level > 0.01:
            q, qc = "WEAK", PALETTE['warn']
        else:
            q, qc = "SILENT", PALETTE['text_dim']
        self.info_labels['Quality'].config(text=q, fg=qc)

    def _draw_battery(self, pct):
        c = self.bat_canvas
        c.delete('all')
        w = c.winfo_width() or 240
        h = c.winfo_height() or 50
        bw = w - 40
        bh = 30
        bx = 20
        by = (h - bh) // 2
        c.create_rectangle(bx, by, bx + bw, by + bh, outline=PALETTE['text_dim'], fill=PALETTE['bg3'], width=2)
        c.create_rectangle(bx + bw, by + bh // 3, bx + bw + 8, by + 2 * bh // 3, fill=PALETTE['text_dim'], outline='')
        if pct is not None:
            fill_w = int((bw - 4) * pct / 100)
            color = PALETTE['danger'] if pct < 20 else PALETTE['warn'] if pct < 40 else PALETTE['accent3']
            if fill_w > 0:
                c.create_rectangle(bx + 2, by + 2, bx + 2 + fill_w, by + bh - 2, fill=color, outline='')
            c.create_text(bx + bw // 2, by + bh // 2, text=f"{pct}%", font=('Consolas', 10, 'bold'), fill='white')
            self.bat_pct_lbl.config(text=f"Battery: {pct}%", fg=color)
        else:
            c.create_text(bx + bw // 2, by + bh // 2, text="N/A", font=('Consolas', 10), fill=PALETTE['text_dim'])
            self.bat_pct_lbl.config(text="–")

    def _quit(self):
        self.engine.shutdown()
        self.tts.stop()
        self.destroy()


# ========== MAIN ==========
if __name__ == '__main__':
    # Verify critical libraries
    missing = []
    for lib in ['sounddevice', 'soundfile', 'numpy']:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        exit(1)

    app = GlobalSpeakerApp()
    app.mainloop()