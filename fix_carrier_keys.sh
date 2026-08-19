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

AK="$HOME/.ssh/authorized_keys"
PUB=/tmp/id_ed25519.pub

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
touch "$AK"

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

# De-duplicate while preserving order, so re-running this is harmless.
awk '!seen[$0]++' /tmp/ak.clean > "$AK"
chmod 600 "$AK"
rm -f /tmp/ak.clean "$PUB"

echo "authorized_keys: $before line(s) before, $kept genuine kept, $(wc -l < "$AK") now"
echo "keys present:"
cut -d' ' -f1,3 "$AK" | sed 's/^/  /'

# D-TACQ carriers keep persistent state under /mnt/local; if $HOME is on a
# tmpfs the key evaporates at the next power cycle, which is worth knowing
# before anyone relies on it.
echo "--- persistence of $HOME/.ssh ---"
df -P "$HOME/.ssh" 2>/dev/null | tail -1
