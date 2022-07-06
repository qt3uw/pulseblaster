# example: pulse blaster pin held on indefinitely
from PBInd import PBInd
from spinapi import *

cycle_length = 1e3   # ns
hardware_pins = [23] # using pin 23 (AOM modulation) 
delays = []          # delays for each individual channel
N=float('inf')       # number of loops (N = float('inf') to repeat indefinitely

# select board 1
pb_select_board(1)

# initialize board
if pb_init() != 0:
	print("Error initializing board: %s" % pb_get_error())
	input("Please press a key to continue.")
	exit(-1)

# Configure the core clock
pb_core_clock(100) # MHz
pb_reset()
# initialize PBInd object to individually program pulse blaster pins
pb=PBInd(pins=hardware_pins,on_time=cycle_length,DEBUG_MODE=0,auto_stop=0) # default resolution is 50 ns

#program hardware_pins to be on from t0=0 to tend=cycle_length
pb_start_programming(0)
pb.on(hardware_pins[0],0,cycle_length)
pb.program(delays,N)
pb_stop_programming()


#trigger the board
pb_reset()
pb_start()

#to stop, use pb_stop()
pb_stop()

#to close connection to pulseblaster
pb_close()