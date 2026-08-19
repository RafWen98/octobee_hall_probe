#!/bin/sh
# onbox_set_sample_rate.sh -- RUNS ON THE ACQ1001 CARRIER.
#
# Set (or just report) the ACQ423 sample rate, with the aliasing cost stated.
#
#   set-sample-rate.sh              # report current rate and stream load
#   set-sample-rate.sh 200000       # full rate, no aliasing
#   set-sample-rate.sh 50000        # quarter rate, 2.0x noise density
#   set-sample-rate.sh 20000        # 20 kSPS, 3.2x noise density
#
# WHY LOWERING THE RATE COSTS YOU NOISE
#   The SENM3Dx analog low-pass is fixed at 100 kHz (PWM_CTRL bits 5:4, already
#   at its narrowest). Nothing between the sensor and the ADC filters further.
#   So sampling below 200 kSPS folds all noise from 0-100 kHz into 0-fs/2 and
#   the noise density rises by sqrt(100000 / (fs/2)).
#
#   200 kSPS -> 1.0x (critically sampled, no penalty)
#    50 kSPS -> 2.0x
#    20 kSPS -> 3.2x   (measured 3.1x on this hardware)
#
#   Averaging on the host AFTER sampling at 200 kSPS is strictly better than
#   sampling slower, because the ADC does the anti-aliasing for you. Only lower
#   the clock when the stream bandwidth is the binding constraint.
#
# NOTE: the FPGA oversampling filter (nacc) would do this properly -- decimate
# in the FPGA with no aliasing and sqrt(N) less noise -- but it is inert in this
# custom OCTO-BEE bitstream (ACQ1001_TOP_09_74_32B-OCTOBEE). The knob accepts
# and echoes values while the output rate and noise stay unchanged. Worth asking
# D-TACQ to include the oversampling block in the OCTO-BEE build.

# /usr/local/bin is not on PATH for non-login ssh shells, so be explicit.
PATH=/usr/local/bin:$PATH
export PATH

SITE=1
SSB=$(cat /etc/acq400/0/ssb 2>/dev/null)
[ -z "$SSB" ] && SSB=96

report() {
    FS=$(/usr/local/bin/get.site 0 SIG:CLK_S${SITE}:FREQ 2>/dev/null | awk '{print $NF}')
    DIV=$(cat /etc/acq400/$SITE/clkdiv 2>/dev/null)
    if [ -z "$FS" ]; then
        echo "could not read SIG:CLK_S${SITE}:FREQ -- is the EPICS IOC up?" >&2
        return 1
    fi
    echo "sample rate : ${FS} Hz  (clkdiv ${DIV}, ssb ${SSB} B)"
    awk -v fs="$FS" -v ssb="$SSB" 'BEGIN{
        mb = fs*ssb/1e6
        pen = (fs/2 >= 100000) ? 1.0 : sqrt(100000/(fs/2))
        printf "stream load : %.2f MB/s   (carrier delivers ~10-15 MB/s)\n", mb
        printf "aliasing    : %.2fx noise density vs 200 kSPS\n", pen
        if (pen > 1.05)
            printf "              (sensor LPF is 100 kHz; below 200 kSPS you fold it in)\n"
    }'
}

if [ -z "$1" ]; then
    report
    exit 0
fi

case "$1" in
    ''|*[!0-9]*) echo "usage: $0 [sample_rate_hz]" >&2; exit 2 ;;
esac

CLK_MB=$(/usr/local/bin/get.site 0 SIG:CLK_MB:FREQ 2>/dev/null | awk '{print $NF}')
DIV=$(awk -v c="$CLK_MB" -v f="$1" 'BEGIN{d=int(c/f + 0.5); if (d<1) d=1; print d}')

echo "motherboard clock ${CLK_MB} Hz / clkdiv ${DIV}"
/usr/local/bin/set.site $SITE clkdiv $DIV
sleep 3
report
