
# ThreeDHALLInterface lives here; set-device-gain.py does not add it itself
export PYTHONPATH=$PYTHONPATH:/usr/local/senm3dx

# Set sensor gains (belt-and-braces: the EEPROM EGain_sel byte already selects
# gain 3000 at power-up, this re-asserts it in case the EEPROM is ever invalid)
for sensor in 0 1 2 3 4 5 6 7; do
    /usr/local/CARE/set-device-gain.py $sensor 3000
done
