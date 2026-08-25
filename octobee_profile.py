#!/usr/bin/env python3
"""
octobee_profile.py -- find out what is actually slow, on the machine it is slow on.

"The app goes sluggish once data arrives" has several quite different causes,
and they need different fixes:

  * the arithmetic (decode, VCM subtraction, tesla conversion, decimation)
  * the drawing we ask for (setData on curves, colours on 16 meshes)
  * the drawing Qt then does (the OpenGL paint, which is where a machine
    without GPU acceleration loses seconds at a time)
  * something else entirely blocking the event loop

Measuring only the first two -- which is all a Python profiler sees -- points
the finger in the wrong place when the answer is the third. So this module
times named spans, times the GL paint separately by hooking paintGL, and
watches the Qt event loop for lag. The lag number is the honest answer to "is
it frozen": it is how late a timer that asked for 100 ms actually fired.

Overhead when disabled is a boolean test per span.
"""

import time
from collections import deque

_now = time.perf_counter


class _Span:
    """One reusable timing context. Not reentrant, which it never needs to be."""

    __slots__ = ("name", "prof", "t0")

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = _now()
        return self

    def __exit__(self, *exc):
        self.prof.add(self.name, _now() - self.t0)
        return False


class _NullSpan:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL = _NullSpan()


class Profiler:
    """Named spans with call counts, mean and worst case, over a rolling window."""

    def __init__(self, enabled=False, window_s=5.0):
        self.enabled = bool(enabled)
        self.window_s = float(window_s)
        self._spans = {}
        self._samples = {}          # name -> deque of (t, seconds)
        self._notes = {}
        self.t0 = _now()

    # ---- recording -------------------------------------------------------
    def time(self, name):
        if not self.enabled:
            return _NULL
        s = self._spans.get(name)
        if s is None:
            s = self._spans[name] = _Span(self, name)
        return s

    def add(self, name, seconds):
        if not self.enabled:
            return
        d = self._samples.get(name)
        if d is None:
            d = self._samples[name] = deque()
        t = _now()
        d.append((t, seconds))
        cutoff = t - self.window_s
        while d and d[0][0] < cutoff:
            d.popleft()

    def note(self, key, value):
        """Record a one-off fact worth reporting alongside the timings."""
        self._notes[key] = value

    @property
    def notes(self):
        return dict(self._notes)

    def reset(self):
        self._samples.clear()
        self.t0 = _now()

    # ---- reporting -------------------------------------------------------
    def rows(self):
        """
        One row per span: (name, calls/s, mean ms, worst ms, % of wall clock).

        The percentage is the share of real time spent inside that span. Spans
        that contain others double-count on purpose -- seeing that 'view tick'
        is 80% and 'GL paint' is 75% of it is the whole point.
        """
        out = []
        now = _now()
        span = min(self.window_s, max(now - self.t0, 1e-6))
        for name, d in self._samples.items():
            if not d:
                continue
            vals = [s for _, s in d]
            total = sum(vals)
            out.append({
                "name": name,
                "per_s": len(vals) / span,
                "mean_ms": total / len(vals) * 1e3,
                "max_ms": max(vals) * 1e3,
                "pct": total / span * 100.0,
            })
        out.sort(key=lambda r: -r["pct"])
        return out

    def text(self, width=78):
        if not self.enabled:
            return ("Profiling is off.\n\n"
                    "Turn it on to find out where the time goes -- it costs "
                    "almost nothing to leave running.")
        rows = self.rows()
        lines = [f"rolling {self.window_s:g} s window",
                 "",
                 f"{'what':<30} {'calls/s':>8} {'mean ms':>9} "
                 f"{'worst ms':>9} {'% time':>8}"]
        lines.append("-" * width)
        for r in rows:
            lines.append(f"{r['name']:<30} {r['per_s']:8.1f} {r['mean_ms']:9.2f} "
                         f"{r['max_ms']:9.1f} {r['pct']:7.1f}%")
        if not rows:
            lines.append("(nothing recorded yet)")
        if self._notes:
            lines += ["", "environment:"]
            for k, v in self._notes.items():
                lines.append(f"  {k:<26} {v}")
        return "\n".join(lines)


class LagMonitor:
    """
    How late a fixed-interval timer actually fires.

    This is the number that says whether the application is blocked, as opposed
    to merely busy. A timer asking for 100 ms that fires at 900 ms means
    something held the event loop for 800 ms, and no amount of making the maths
    faster will help until that something is found.
    """

    def __init__(self, interval_ms=100, window=100):
        self.interval = interval_ms / 1000.0
        self.samples = deque(maxlen=window)
        self._last = None

    def tick(self):
        t = _now()
        if self._last is not None:
            self.samples.append(max(0.0, (t - self._last) - self.interval))
        self._last = t

    def reset(self):
        self.samples.clear()
        self._last = None

    @property
    def mean_ms(self):
        return (sum(self.samples) / len(self.samples) * 1e3) if self.samples else 0.0

    @property
    def max_ms(self):
        return (max(self.samples) * 1e3) if self.samples else 0.0

    def verdict(self):
        m = self.max_ms
        if not self.samples:
            return "no data yet"
        if m < 50:
            return f"responsive (worst {m:.0f} ms late)"
        if m < 250:
            return f"occasional stalls (worst {m:.0f} ms late)"
        return (f"BLOCKED -- the event loop was held for {m:.0f} ms. "
                f"Whatever is at the top of the table above is doing it.")


def gl_info(widget):
    """
    Vendor/renderer/version of the live OpenGL context, or why we could not ask.

    Worth reporting prominently: a renderer of 'GDI Generic', 'Microsoft Basic
    Render Driver' or 'llvmpipe' means there is no GPU acceleration and every
    3D repaint is being done on the CPU. That turns the probe head from a
    millisecond into a second, which is exactly what a freeze feels like.
    """
    try:
        from OpenGL import GL  # noqa: PLC0415  (optional dependency)
        widget.makeCurrent()
        try:
            def s(enum):
                v = GL.glGetString(enum)
                return v.decode(errors="replace") if isinstance(v, bytes) else str(v)
            return {"GL vendor": s(GL.GL_VENDOR),
                    "GL renderer": s(GL.GL_RENDERER),
                    "GL version": s(GL.GL_VERSION)}
        finally:
            widget.doneCurrent()
    except Exception as e:
        return {"GL renderer": f"could not query ({type(e).__name__}: {e})"}


SOFTWARE_RENDERERS = ("gdi generic", "microsoft basic render", "llvmpipe",
                      "softpipe", "swiftshader", "software rasterizer")


def is_software_renderer(info):
    r = str(info.get("GL renderer", "")).lower()
    return any(k in r for k in SOFTWARE_RENDERERS)
