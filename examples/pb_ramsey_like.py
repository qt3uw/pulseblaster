# example: Pulse blaster does a pulse, waits a long while and does another pulse

from pulseblaster.PBInd import PBInd
import pulseblaster.spinapi
import time

board_idx = 0
pulse_duration = 100 * pulseblaster.spinapi.us
ramsey_time = 300000 * pulseblaster.spinapi.us
hardware_pins = [16, 17]  # 17 is the trigger, 16 is the mw pulser
delays = []          # delays for each individual channel

pb=PBInd(pins=hardware_pins,
		 on_time=pulse_duration,
		 DEBUG_MODE=0,
		 auto_stop=0,
		 res=10) # default resolution is 50 ns


pb.spinapi.pb_select_board(board_idx)

# initialize board
if pb.spinapi.pb_init() != 0:
	print("Error initializing board: %s" % pb.spinapi.pb_get_error())
	input("Please press a key to continue.")
	exit(-1)

# Configure the core clock
pb.spinapi.pb_core_clock(100 * pulseblaster.spinapi.MHz) # MHz
pb.spinapi.pb_reset()
# initialize PBInd object to individually program pulse blaster pins

# program hardware_pins to be on from t0=0 to tend=cycle_length
pb.spinapi.pb_start_programming(0)
# Program the first pi/2 pulse
pb.on(hardware_pins[0], 0, pulse_duration)
pb.on(hardware_pins[1], 0, pulse_duration)
pb.program(delays, 1)
# Program free precession
pb.off(hardware_pins[0], 0, pulse_duration)
pb.program(delays, int(round(ramsey_time / pulse_duration)))
# Program second pi/2 pulse
pb.on(hardware_pins[0], 0, pulse_duration)
pb.program(delays, 1)

# Turn channels off
[pb.off(pin, 0, pulse_duration) for pin in hardware_pins]
pb.program(delays, 1)
pb.spinapi.pb_stop_programming()


# trigger the board
pb.spinapi.pb_reset()
pb.spinapi.pb_start()
# to stop, use pb_stop()
# pb.spinapi.pb_stop()
# pb.spinapi.pb_reset()


# to close connection to pulseblaster
pb.spinapi.pb_close()
