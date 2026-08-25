"""Numbers the window and its tabs share: rates, windows, plot budgets."""



from octobee.calib import geometry as pgeom

N_SENSORS = pgeom.N_SENSORS


AXES = ("Bx", "By", "Bz")


OUT_RATES = (100.0, 200.0, 500.0, 1000.0, 2000.0)


# Reducing the ADC clock is NOT free. The SENM3Dx analog low-pass sits at
# 100 kHz (PWM_CTRL bits 5:4, already at its narrowest setting), so sampling
# below 200 kSPS folds noise from 0-100 kHz into 0-fs/2. The density penalty is
# sqrt(100kHz / (fs/2)) -- measured 3.1x at 20 kSPS, matching the predicted 3.16x.
# You trade noise for stream bandwidth; the label states the cost.
STREAM_RATES = {"leave the box alone (recommended)": 0.0,
                "200 kSPS (no aliasing, best noise)": 200000.0,
                "50 kSPS (2.0x noise, aliased)": 50000.0,
                "20 kSPS (3.2x noise, aliased)": 20000.0}


# The per-channel health scan is the most expensive thing in the refresh loop
# and its answer changes only when a connector does, so it runs on its own
# slower clock over a short window. Left on the display clock over the full
# history it starves the reader threads and the carriers' queues overflow.
# Redraw rate for the live view. The acquisition tick is independent of this,
# so turning it down costs you smoothness and nothing else -- no samples, no
# recorded data. 10 Hz is already far beyond what a hand-passed magnet needs.
VIEW_RATES = (2.0, 5.0, 10.0, 20.0)


DEFAULT_VIEW_HZ = 10.0


MAX_VIEW_INTERVAL_MS = 2000        # slowest the automatic backoff will go


HEALTH_PERIOD_S = 2.0


HEALTH_WINDOW_S = 1.0


RAW_HISTORY_S = 5.0


# Antialiasing off, and every plot pen exactly 1 pixel wide. This is not a
# cosmetic preference, it is the difference between a usable application and an
# unusable one. Qt strokes a cosmetic pen of non-integer width through a
# completely different and vastly slower path: measured on a 20 s window of 15
# traces, a single repaint took 39 SECONDS at width 1.6 with antialiasing, and
# 74 ms at width 1. Turning antialiasing off as well brings it to 45 ms. The
# symptom is a live plot that appears to hang the moment real data arrives,
# with the cost invisible to any timing of our own code because it happens
# inside Qt's paint.
PLOT_PEN_WIDTH = 1


# Points handed to each curve, as a multiple of the plot's width in pixels.
# 0.5 gives one min/max pair per ~4 pixels, which still renders a hand-passed
# magnet spike over many bins while costing a third of what 1.0 does. Raise it
# if you need finer structure on screen and can afford the repaint.
PLOT_TARGET_MULT = 0.5
