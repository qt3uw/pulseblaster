import pulseblaster.spinapi as spincore_spinapi
import numpy as np

class PBInd:
    '''
    PBInd is a client of the SpinCore Pulseblaster API that allows the user
    to program the pins of the Pulseblaster independently of one another in python. 
    
    Notes: 
        All times are inputted in nanoseconds (as ints).
        PulseBlaster specification:
            - Minimum Pulse Width: 50ns
            - Pulse resolution: 10ns (increments)
            - Pulses can not be less than 50ns apart (across all channels)
            - Puses can not overlap by less than 50ns
    '''
    def __init__(self,
                pins,
                DEBUG_MODE = 0,
                on_time = 4000,
                res = 10,
                minimum_pulse = 50,
                auto_stop = 0):
        self._pins = pins   # array of valid pin numbers (0-23) from Pulseblaster
        self._DEBUG_MODE = DEBUG_MODE   # print instructions to screen after programming them, no physical pb connection
        if (on_time%10 != 0):
            raise Exception('total instruction time is not a multiple of 10ns: ' + str(on_time) + 'ns')
        else:
            self._on_time = on_time   # total time of instruction (in nanoseconds)
        self._res  = res # length of grid unit in nanoseconds
        self._minimum_pulse = minimum_pulse
        self._smps = round(self._on_time/self._res) # total length of instruction in samples (clock periods) (before looping)
        self._output_chs = ['0'*self._smps]*(len(self._pins))# 'bitset' representation of output
        #self._output_length = 0   # number of units to downsample sequence to
        self.instructions = "" # string representation of commands issued to Pulseblaster
        self._auto_stop = auto_stop  # boolean flag that programs the stop and does 'pb_stop_programming()'
                    # if turned off, this allows the client to program after a call to PBInd.program()
        self.spinapi = spincore_spinapi

    def on(self, pin, start, length):
        '''Sets 'pin' to high voltage for 'length' (ns) starting at 'start'.'''
        self.set(pin, start, length, 1)

    def off(self, pin, start, length):
        '''Sets 'pin' to low voltage for 'length' (ns) starting at 'start'.'''
        self.set(pin, start, length, 0)

    def make_clock(self, pin, period):
        '''
        Configures 'pin' as a clock output with 50% duty cycle.
        Period should always be >2*minimum pulse width.
        Typically period should be 4*minimum pulse width or more if using 10ns pulse resolution.
        '''
        if period/2 < self._minimum_pulse:
            raise Exception('requested clock is impossible: less than resolution')
        ticks = 0  # count the number of actual ticks that are programmed
        cursor = 0
        # sweep across chs and add ticks
        while cursor + period <= self._on_time:
            self.on(pin, cursor, period/2)
            cursor = cursor + period/2
            self.off(pin, cursor, period/2)
            cursor = cursor + period/2
            ticks = ticks + 1

    def program(self, offsets, loops):
        '''
        Programs pulseblaster with the current channel states (chs matrix).
        Validates the pulse sequence, raising an error if timing constraints are violated.
        
        Note: The offset capability for adjusting channels is not fully compatible with validation
        '''
        if loops < 1:
            raise Exception('number of loops must be positive integer')

        if not self._DEBUG_MODE & self._auto_stop:
            self.spinapi.pb_start_programming('PULSE_PROGRAM')

        if len(offsets)==0:
            # the client did not request delays, so the matrix is unchanged
            self.validate_pulse_instructions(self._output_chs)
            self.write_instruction(self._output_chs, loops)
        elif len(offsets) == len(self._pins):
            # shift individual channels according to requested delays
            actual_chs, front_ind, back_ind = self.get_offset_chs[offsets]

            # validate and program front end of pulse loop
            self.validate_pulse_instructions(actual_chs[:, 1:front_ind])
            self.write_instruction(actual_chs[:, 1:front_ind], 1)

            # program middle of loop, accounting for delays
            front_chs = actual_chs[:, 1:front_ind]
            front_chs[:, len(self._output_chs)] = 0 # extend this array to make it same dimension as mid_chs
            shift_amount = self._smps - front_ind # front must be shifted right, then bitor'ed
            front_chs = np.roll(front_chs, -shift_amount)
            back_chs = actual_chs[:, back_ind:-1]
            back_chs[:, self._smps] = 0  # extend this array to make it same dimension as mid_chs

            # combine front, middle, and end loops, then program it
            mid_chs = actual_chs[:, (front_ind+1):back_ind]
            mid_chs = mid_chs|front_chs
            mid_chs = mid_chs|back_chs
            # TODO: should this be loops - 1?
            self.validate_pulse_instructions(mid_chs)
            self.write_instruction(mid_chs, loops)

            # program back end of pulse loop
            self.validate_pulse_instructions(actual_chs[:, (back_ind+1):-1])
            self.write_instruction(actual_chs[:, (back_ind+1):-1], 1)
        else:
            raise Exception('vector of offsets is incorrect len - must be len of pins vector')

        if self._DEBUG_MODE:
            self.instructions = self.instructions + "pb_inst_pbonly(0, 'STOP', 0, " + str(2*self._res) + ")\n"

        if (not self._DEBUG_MODE) & self._auto_stop:
            self.spinapi.pb_inst_pbonly(0, 'STOP', 0, 2 * self._res)
            self.spinapi.pb_stop_programming()

        self.print_instructions()

    def print_instructions(self):
        '''
        Prints out the string representation of the instructions
        issued to the Pulseblaster
        '''
        if self._DEBUG_MODE:
            print(self.instructions)

    ## PRIVATE
    def write_instruction(self, actual_chs, loops):
        '''Produces the instructions used to program the PulseBlaster from the CHS matrix'''
        if len(actual_chs)==0:
            return
        # Given a chs matrix, program a sequence of Pulseblaster instructions. 
        # The 'command' can be self.spinapi.Inst.CONTINUE or self.spinapi.Inst.LOOP
        prev_state_d = 0  # index of 'prev_state'
        prev_state =  self.get_state(prev_state_d, actual_chs, len(self._pins)) # vertical slice of states as bitset

        actual_smps = len(actual_chs[0])

        cur_command = self.spinapi.Inst.CONTINUE  # the next instruction
        last_command = self.spinapi.Inst.CONTINUE
        if loops == float('inf'):
            # in this case we want an infinite loop
            last_command = self.spinapi.Inst.BRANCH
            loops=0
        elif loops > 1:
            # if loops > 1, then the first and last commands will be loop commands
            cur_command = self.spinapi.Inst.LOOP
            last_command = self.spinapi.Inst.END_LOOP

        first_inst = float('inf')  # this will eventually change to the first instruction ID

        for d in range(1,actual_smps):
            current_state = self.get_state(d, actual_chs, len(self._pins))
            if self._DEBUG_MODE:
                print(current_state)
            compare = current_state == prev_state
            if not compare:
                # at least one channel changed state, issue new instruction to PB
                hex_flag = self.hex_flag(prev_state)
                duration = (d - prev_state_d) * self._res
                if not self._DEBUG_MODE:
                    first_inst = min(self.spinapi.pb_inst_pbonly(hex_flag, cur_command, int(loops), duration * self.spinapi.ns), first_inst)  # we want inst to be the lowest instruction ID
                else:
                    first_inst = 0

                if self._DEBUG_MODE:
                    self.instructions = self.instructions + "pb_inst_pbonly(" + str(hex_flag) + "," + str(cur_command) + "," + str(loops) + "," + str(duration * self.spinapi.ns) + ")\n"

                cur_command = self.spinapi.Inst.CONTINUE  # even if loops > 1, the next commands will all be CONTINUE commands (until last END_LOOP command)
                prev_state = current_state
                prev_state_d = d

        # we have reached the end of the chs matrix. Now issue the last instruction
        hex_flag = self.hex_flag(prev_state)
        duration = (actual_smps - prev_state_d) * self._res  # the plus one is needed otherwise there is an off by one error
        if (first_inst == float('inf')) & (loops > 1):
            # in this case, the matrix was homogeneous and no
            # instructions were issued in the loop. Therefore the last
            # command CANNOT be an END_LOOP (there was no 'begin loop').
            # Simply change it to a CONTINUE command
            last_command = self.spinapi.Inst.CONTINUE
            duration = duration * loops

        if not self._DEBUG_MODE:
            if first_inst == float('inf'):
                first_inst = 0
            self.spinapi.pb_inst_pbonly(hex_flag, last_command, first_inst, duration * self.spinapi.ns)  # this instruction could be an END_LOOP command

        if self._DEBUG_MODE:
            self.instructions = self.instructions + "pb_inst_pbonly(" + str(hex_flag)+ "," + str(last_command) +","+ str(first_inst) +","+str(duration) +")\n"

    def get_offset_chs(self, offsets):
        '''
        Returns a new chs matrix that takes into account the offset
        value for each channel. Also returns the front and back
        index of the main body of the chs matrix.
        '''
        offsets_smps = round(offsets / self._res)

        if max(abs(offsets_smps)) >= len(self._output_chs):
            raise Exception('this special case is not supported as of 07/02/2017')

        front_ind = max(-offsets_smps)  # largest negative offset
        back_ind = self._smps + front_ind  # largest positive offset
        adj_smps = back_ind + max(offsets_smps)  # length of actual chs
        #offset_chs = zeros(len(self._pins), adj_smps)
        for ch in range(self._pins):
            cpy_start = offsets_smps(ch) + 1 + front_ind  # start copying into this index
            cpy_end = cpy_start + self._smps - 1

            # copy over actual bits from unshifted array, leave other
            # bits at zero state
            offset_chs[ch, cpy_start:cpy_end] = self._output_chs[ch, :]

        return offset_chs, front_ind, back_ind

    def set(self, pin, start, len, val):
        '''Sets 'pin' to specific 'val' (high/low) for 'length' (ns) starting at 'start'.'''
        if (start%10 != 0):
            raise Exception('start is not a multiple of 10ns: ' + str(start) + 'ns')
        if (len%10 != 0):
            raise Exception('len is not a multiple of 10ns: ' + str(len) + 'ns')
        start_smp = round(start / self._res)
        stop_smp = start_smp + round(len / self._res)-1

        if start_smp < 1 | start_smp > self._smps:
            raise Exception('invalid start: ' + str(start_smp))
        elif stop_smp > self._smps:
            raise Exception('invalid stop time: ' + str(stop_smp))

        if stop_smp >= start_smp:
            ch = self.get_ch(pin)
            self._output_chs[ch] = self._output_chs[ch][min(start_smp,0):start_smp] + str(val)*(stop_smp-start_smp+1) + self._output_chs[ch][stop_smp:-1]
            # TODO: possibly add rounding feature to improve downsample

    def get_ch(self, pin):
        index = -1
        for d in range(len(self._pins)):
            if self._pins[d] == pin:
                index = d
                break

        if index == -1:
            raise Exception('invalid pin requested: ' + str(pin))

        return index

    def hex_flag(self, state):
        hex_flag = 0
        for d in range(len(state)):
            if state[d] == '1':
                # this pin is on, add to flag
                # TODO: somewhere, clarify self._pins(d) vs 2^self._pins(d)
                hex_flag = hex_flag|2**(self._pins[d])
        return hex_flag

    def get_state(self, state_index, state_array, num_elements):
            stateslice=""
            for i in range(num_elements):
                stateslice = stateslice + state_array[i][state_index]

            return stateslice

    def validate_pulse_instructions(self, chs_matrix):
        """
        Validates that all instructions respect PulseBlaster timing constraints.
            - Minimum Pulse Width: 50ns
            - Pulse resolution: 10ns (increments)
            - Pulses can not be less than 50ns apart (across all channels)
            - Puses can not overlap by less than 50ns

        Raises:
            Exception: If an instruction change faster than the minimum allowed duration
        """
        min_instruction_span = int(self._minimum_pulse / self._res)  # Required minimum span in columns

        chs_matrix = np.array([list(row) for row in chs_matrix])
    
        current_streak = 1
        streak_start = 0
        change_indices = []
        min_streak = float('inf')
        min_streak_index = 0
        problem_channels_start = []  # Will store which channels started the shortest instruction
        problem_channels_end = []  # Will store which channels ended the shortest instruction
    
        # Compare entire columns
        for i in range(1, chs_matrix.shape[1]):
            if np.array_equal(chs_matrix[:, i], chs_matrix[:, i-1]):
                current_streak += 1
            else:
                # Column changed, check if previous streak was long enough
                if current_streak < min_streak:
                    min_streak = current_streak
                    min_streak_index = streak_start
                    # Find which channels changing at this change interface and prior interface
                    if (streak_start > 0):
                        problem_channels_start = np.where(chs_matrix[:, streak_start] != chs_matrix[:, streak_start-1])[0]
                    else:
                        problem_channels_start = np.array([]) # All channels change at start
                    problem_channels_end = np.where(chs_matrix[:, i] != chs_matrix[:, i-1])[0]
            
                change_indices.append(i)
                streak_start = i
                current_streak = 1
    
        # Check final streak
        if current_streak < min_streak:
            min_streak = current_streak
            min_streak_index = streak_start
            if streak_start > 0:
                problem_channels_start = np.where(chs_matrix[:, streak_start] != chs_matrix[:, streak_start-1])[0]
            else:
                problem_channels_start = np.array([])
            problem_channels_end = np.array([])
    
        # If minimum streak is too short, raise exception
        if min_streak < min_instruction_span:
            start_channels_str = "initial state" if len(problem_channels_start) == 0 else problem_channels_start.tolist()
            end_channels_str = "final state" if len(problem_channels_end) == 0 else problem_channels_end.tolist()
        
            error_msg = (f'Instruction duration {min_streak * self._res}ns shorter than required '
                        f'{self._minimum_pulse}ns starting at time {min_streak_index * self._res}ns (index {min_streak_index}). '
                        f'Instructions changed at indices: {change_indices}. '
                        f'Channels that changed at instruction\'s start: {start_channels_str}. '
                        f'Channels that changed at instruction\'s end: {end_channels_str}.')
            
            raise Exception(error_msg)
        return True
