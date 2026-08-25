#!/bin/sh
# Repair ~/.ssh/authorized_keys on an OCTO-BEE carrier and install the
# calibration workstation's public key.
#
# Why this exists: pasting a long `echo '<key>' >> authorized_keys` into a
# terminal that soft-wraps puts real newlines inside the quoted string, so the
# key lands as three fragments instead of one line. Copying the .pub file up and
# appending it here cannot suffer that, because the key never touches a command
# line.
#
# Run as:  scp id_ed25519.pub fix_carrier_keys.sh root@<carrier>:/tmp/
#          ssh root@<carrier> "sh /tmp/fix_carrier_keys.sh"

# Stop on any failure. Every step below feeds the next, and a half-run of this
# script rewrites the file that lets you back in.
set -e

AK="$HOME/.ssh/authorized_keys"
PUB=/tmp/id_ed25519.pub

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$AK"

# Back up before touching anything. This script rewrites the only file standing
# between you and a headless box you can no longer log into, and it is designed
# to be re-runnable -- so the case where the filter keeps nothing and the .pub
# is absent (it deletes the .pub at the end) leaves an empty authorized_keys.
# One copy makes that recoverable over the serial console instead of fatal.
cp "$AK" "$AK.bak"

# Keep only well-formed public keys. Every fragment a wrapped paste leaves
# behind -- a bare "ssh-ed25519", a naked base64 blob, a trailing comment --
# is a single field, whereas a real key is always "<type> <payload> [comment]".
# So requiring both a key type in field 1 and a payload in field 2 drops the
# wreckage and keeps anything genuine that was already there.
awk 'NF > 1 && $1 ~ /^(ssh-|ecdsa-|sk-)/' "$AK" > /tmp/ak.clean

before=$(wc -l < "$AK")
kept=$(wc -l < /tmp/ak.clean)

if [ -f "$PUB" ]; then
    cat "$PUB" >> /tmp/ak.clean
fi

# Refuse to install an empty file. Locking yourself out is the one outcome this
# script must never produce, and it is exactly what "no genuine keys kept and
# no .pub uploaded" would do.
if [ ! -s /tmp/ak.clean ]; then
    echo "REFUSING to write an empty authorized_keys." >&2
    echo "  $before line(s) were present, none parsed as a public key, and" >&2
    echo "  $PUB is not there to add one. Copy your .pub up first:" >&2
    echo "      scp id_ed25519.pub root@<carrier>:/tmp/" >&2
    echo "  $AK is unchanged; the backup is at $AK.bak" >&2
    rm -f /tmp/ak.clean
    exit 1
fi

# De-duplicate while preserving order, so re-running this is harmless.
awk '!seen[$0]++' /tmp/ak.clean > "$AK"
chmod 600 "$AK"
rm -f /tmp/ak.clean "$PUB"

echo "authorized_keys: $before line(s) before, $kept genuine kept, $(wc -l < "$AK") now"
echo "previous contents kept at $AK.bak"
echo "keys present:"
cut -d' ' -f1,3 "$AK" | sed 's/^/  /'

# D-TACQ carriers keep persistent state under /mnt/local; if $HOME is on a
# tmpfs the key evaporates at the next power cycle, which is worth knowing
# before anyone relies on it.
echo "--- persistence of $HOME/.ssh ---"
df -P "$HOME/.ssh" 2>/dev/null | tail -1
