from typing import Sequence
import pulseblaster.spinapi as spinapi
import numpy as np

def initialize(board_idx=0, f_clock=100.E6):
    """
    Sets clock, and initializes board at board index, throws exception if not initialized successfully
    """
    spinapi.pb_select_board(board_idx)
    if spinapi.pb_init() != 0:
        raise(ConnectionError(f'Board {board_idx} could not be initialized.'))
    spinapi.pb_core_clock(f_clock * spinapi.Hz)
    spinapi.pb_reset()


def _array_to_bool(arr: Sequence):
    bstr = '0b'
    for el in arr:
        bstr += str(int(el))
    return bstr

def _array_to_hex(arr: Sequence):
    bstr = _array_to_bool(arr)
    hexstr = hex(int(bstr, 2))
    return hexstr


def close():
    spinapi.pb_close()


def reset():
    spinapi.pb_reset()

def write_sequence(durations, states, loops: int=1):
    for i, duration in enumerate(durations):
        if i == 0:
            loop = spinapi.pb_inst_pbonly(_array_to_hex(states[i]), spinapi.Inst.LOOP, loops,
                                          durations[i] * 1000 * spinapi.ms)
            print(loop)
        elif i == len(durations) - 1:
            spinapi.pb_inst_pbonly(_array_to_hex(states[i]), spinapi.Inst.END_LOOP, loop,
                                   durations[i] * 1000 * spinapi.ms)
        else:
            spinapi.pb_inst_pbonly(_array_to_hex(states[i]), spinapi.Inst.CONTINUE, loops,
                                   durations[i] * 1000 * spinapi.ms)



def break_to(name):
    raise(NotImplementedError)


def check_sequence():
    durations = [1, 1, 1]
    state0 = np.zeros(24)
    state1 = np.ones(24)
    #state1[16] = 1
    state2 = np.zeros(24)
    states = [state0, state1, state2]
    initialize()
    spinapi.pb_start_programming(0)
    write_sequence(durations, states, loops=1)
    spinapi.pb_stop_programming()
    spinapi.pb_start()
    spinapi.pb_stop()
    spinapi.pb_reset()


if __name__ == "__main__":
    check_sequence()