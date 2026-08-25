# Long runs and the data rate

**You do not need to reduce anything.** Both carriers stream full 200 kSPS
continuously, indefinitely, with zero sample loss. Long continuous measurement at
full bandwidth and full resolution is exactly what this system does.

### Measured, both boxes together

| test | duration | combined rate | gaps | lost |
|---|---|---|---|---|
| 100 kSPS | 90 s | 19.3 MB/s | 0 | 0 |
| 150 kSPS | 60 s | 28.8 MB/s | 0 | 0 |
| 200 kSPS | 180 s | 39.3 MB/s | 0 | 0 |
| 200 kSPS + host decimation to disk | 150 s | 39.1 MB/s | 0 | 0 |

In the 180 s run the lag held flat at 2.78 s (694) and 3.06 s (695) from the
20 s mark onward. Flat lag means keeping up; a growing lag would mean the box's
512 MB buffer pool was filling.

### The shape to use for long logs

Receive at 200 kSPS (the ADC does your anti-aliasing), decimate on the host,
write the decimated stream. Measured over 150 s with both boxes:

```
200 kSPS in  ->  host boxcar average  ->  1 kHz float32 out
   0 gaps, 0 lost, lag flat at 2.6 s / 3.1 s
   453 MB/hour per box  =  10.9 GB/day per box, 21.7 GB/day for both
```

Output size scales with the rate you choose, and the noise improves as √decim:

| output rate | per box | both boxes | noise floor |
|---|---|---|---|
| 1 kHz | 10.9 GB/day | 21.7 GB/day | ~4.9 µT |
| 100 Hz | 1.1 GB/day | 2.2 GB/day | ~1.5 µT |
| 10 Hz | 0.11 GB/day | 0.22 GB/day | ~0.7 µT |
| 1 Hz | 0.011 GB/day | 0.022 GB/day | ~1 µT (drift floor, §7) |

So a month of continuous 10 Hz logging from all 16 sensors is about 6.6 GB. The
binding constraint is host disk, not the instrument.

### If you *do* lower the clock, it costs you

The SENM3Dx analog low-pass is fixed at 100 kHz (`PWM_CTRL` bits 5:4, already at
its narrowest), so sampling below 200 kSPS folds 0–100 kHz into 0–fs/2 and noise
density rises by `sqrt(100kHz/(fs/2))`:

| rate | stream/box | penalty | measured |
|---|---|---|---|
| 200 kSPS | 19.2 MB/s | 1.00× | baseline |
| 50 kSPS | 4.8 MB/s | 2.00× | — |
| 20 kSPS | 1.9 MB/s | 3.16× | **3.1×** |

Note the penalty largely **washes out below ~1 Hz**, where offset drift dominates
instead of white noise: at 1 Hz, 20 kSPS measured 0.75 µT against ~0.7 µT at full
rate. So for very slow logging the aliasing barely matters — but there is no
reason to accept it, since full rate streams fine.

```bash
octobee probe rate                    # report rate, load, aliasing cost
octobee probe rate --fs 50000         # set (rarely needed)
octobee probe restore                 # back to 200 kSPS
```

On the carrier: `/usr/local/CARE/set-sample-rate.sh [hz]`. In the GUI, the
**stream rate** dropdown now defaults to "leave the box alone".

### The FPGA oversampling filter is still missing

D-TACQ's `nacc` accumulate/decimate (manual §13.1) would decimate in the FPGA
with no aliasing and √N less noise. It is inert here: `NACC:DISA` was `TRUE`, and
with it `FALSE` and `nacc=8`/`nacc=32` set, the knob echoes `8,3,0,1`/`32,5,0,1`
while the output rate stays 200 kSPS and the noise stays 63 µT. The FPGA is a
custom `ACQ1001_TOP_09_74_32B-OCTOBEE` bitstream, so the block is probably not in
this build. Worth asking D-TACQ for — but it is now a nice-to-have, not a
blocker, since full-rate streaming works.

### Onboard capture as an alternative

`nbuffers=512 × bufferlen=1 MB` = **512 MB** of dedicated capture RAM, i.e. ~26 s
gapless at 200 kSPS (96 B/sample) with no network involved. Useful for
triggered bursts. Not needed for continuous work. Not yet wired into these tools.

---

---

*Part of the [OCTO-BEE documentation](../README.md).*
