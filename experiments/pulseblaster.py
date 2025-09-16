"""
Defines and determines PB instructions for basic pulse sequences: CWODMR, PODMR/Rabi, Ramsey/Hahn Echo/DD.
The Soundcore PulseBlaster can only handle changes spaced >=50ns apart, with 10ns resolution.
The PulseBlasterAuto class has helper functions to automate finding of the minimum padding for pulse sequences.
"""
# Imports
import numpy as np
from qt3utils.pulsers.interface import ExperimentPulser
from qt3utils.errors import PulseBlasterInitError, PulseBlasterError, PulseTrainWidthError

# Attempt to import PulseBlaster modules with error handling for development purposes
try:
    from pulseblaster.PBInd import PBInd
    import pulseblaster.spinapi
except NameError as e:
    # Handling this error allows for development of pulse sequences without needing hardware interaction
    print(e)
    print('Pulse Blaster software has not been properly installed.')

class PulseBlaster(ExperimentPulser):
  def start(self):
    """
    Starts the PulseBlaster device by initializing it, starting the pulse sequence,
    and closing the connection. Raises a PulseBlasterError if the start operation fails.

    :raises PulseBlasterError: If the start operation fails or an error occurs during the process.
    """
    self.open()
    ret = pulseblaster.spinapi.pb_start()
    if ret != 0:
      raise PulseBlasterError(f'{ret}: {pulseblaster.spinapi.pb_get_error()}')
    self.close()

  def stop(self):
    """
    Stops the PulseBlaster device by stopping the pulse sequence and closing the connection.
    Raises a PulseBlasterError if the stop operation fails.

    :raises PulseBlasterError: If the stop operation fails or an error occurs during the process.
    """
    self.open()
    ret = pulseblaster.spinapi.pb_stop()
    if ret != 0:
      raise PulseBlasterError(f'{ret}: {pulseblaster.spinapi.pb_get_error()}')
    self.close()

  def reset(self):
    """
    Resets the PulseBlaster device by reinitializing it and closing the connection.
    Raises a PulseBlasterError if the reset operation fails.

    :raises PulseBlasterError: If the reset operation fails or an error occurs during the process.
    """
    self.open()
    ret = pulseblaster.spinapi.pb_reset()
    if ret != 0:
      raise PulseBlasterError(f'{ret}: {pulseblaster.spinapi.pb_get_error()}')
    self.close()

  def open(self):
    """
    Opens the PulseBlaster device by selecting the board and initializing the hardware.
    Configures the core clock frequency. Raises a PulseBlasterInitError if the initialization fails.

    :raises PulseBlasterInitError: If the PulseBlaster initialization fails.
    """
    pulseblaster.spinapi.pb_select_board(self.pb_board_number)
    ret = pulseblaster.spinapi.pb_init()
    if ret != 0:
      self.close()  # Attempt to close before raising error
      raise PulseBlasterInitError(f'{ret}: {pulseblaster.spinapi.pb_get_error()}')
    pulseblaster.spinapi.pb_core_clock(100 * pulseblaster.spinapi.MHz)

  def close(self):
    """
    Closes the PulseBlaster device connection. Raises a PulseBlasterError if closing fails.

    :raises PulseBlasterError: If the close operation fails.
    """
    ret = pulseblaster.spinapi.pb_close()
    if ret != 0:
      raise PulseBlasterError(f'{ret}: {pulseblaster.spinapi.pb_get_error()}')

  def stop_programming(self):
    """
    Stops the programming of the PulseBlaster device. Raises a PulseBlasterError if the operation fails.

    :raises PulseBlasterError: If the stop programming operation fails.
    """
    if pulseblaster.spinapi.pb_stop_programming() != 0:
      raise PulseBlasterError(pulseblaster.spinapi.pb_get_error())

  def start_programming(self):
    """
    Starts programming the PulseBlaster device for a new pulse sequence. Raises a PulseBlasterError
    if the operation fails.

    :raises PulseBlasterError: If the start programming operation fails.
    """
    if pulseblaster.spinapi.pb_start_programming(0) != 0:
      raise PulseBlasterError(pulseblaster.spinapi.pb_get_error())

class PulseBlasterAuto(PulseBlaster):
  """
  This class generalizes and automates the process of finding the minimum padding 
  for programming an RF pulse sequence on the pulseblaster, to avoid timing violations.
  It also features the 'validate_params` function to automate timescale conversions
  (s to ns) and the validation of timing parameters.
  """
  def get_clock_edges(self, lower_bound, upper_bound, clock_period):
    """
    Produces an array of all the clock edges that falls within (/near) a time window
    for a 50% duty cycle clock, that starts at time 0.
    
    :param lower_bound: time window lower bound
    :param upper_bound: time window upper bound
    :param clock_period: Clock period in ns
    :returns: array of clock edges
    """
    clock_half_period = clock_period/2
    first_edge = lower_bound - (lower_bound % clock_half_period)
    return np.arange(first_edge, upper_bound + clock_half_period, clock_half_period)
  
  def minimum_padding_finder(self, rf_edges, start_time: int, end_time: int, clock_period: int, other_edges):
    """
    Finds the minimum padding needed to satisfy timing constraints, given a minimum instruction duration of 50ns.
    
    :param rf_edges: List of RF pulse edge time relative to nominal start time
    :param start_time: The start of the RF window (when the first RF edge can be programmed)
    :param end_time: The end of the RF window (when the last RF edge must be progammed by)
    :param clock_period: The clock period for the clock signal output by the PulseBlaster (for the NiDaq)
    :param other_edges: List of times at which non clock edges are fixed
    :returns: Required pre-RF padding time in ns
    :raises ValueError: Valid padding that would fit all the RF edges within the RF window, not found 
    """
    min_separation = 50 # minimum seperation time

    # Get all possible conflict edges
    clock_edges = self.get_clock_edges(start_time, end_time, clock_period)
    conflict_edges = np.concatenate([clock_edges, other_edges])
    
    pre_rf_padding = 0
    sequence_invalid = True
    
    # Calculate maximum allowed padding
    maximum_rf_padding = end_time - start_time - max(rf_edges)
    
    trials = 0
    while sequence_invalid and pre_rf_padding < maximum_rf_padding:
      sequence_invalid = False
      trials += 1
      for rf_edge in rf_edges:
        adjusted_edge = start_time + pre_rf_padding + rf_edge
        
        for conflict_edge in conflict_edges:
          edge_difference = conflict_edge - adjusted_edge
          
          # Valid cases:
          # 1. Edges exactly align (difference = 0)
          # 2. Edges are separated by >= min_separation
          if (edge_difference == 0) or (abs(edge_difference) >= min_separation):
            continue
          else:
            sequence_invalid = True
            # Calculate required padding adjustment
            pre_rf_padding += edge_difference if (adjusted_edge < conflict_edge) else min_separation + edge_difference
            break
        if sequence_invalid:
          break
    
    # Check if we found a valid solution
    if pre_rf_padding >= maximum_rf_padding:
      raise ValueError(f"Could not find valid padding within maximum allowed time, with {trials} attempts")
    else:
      print(f'RF padding = {int(pre_rf_padding)} ns') # broadcast as logging.info instead
    return pre_rf_padding

  def validate_params(self):
    """
    Validate and update pulse or clock parameters based on their names.    
    Raises:
        ValueError: For invalid pulse parameters.
        PulseTrainWidthError: For invalid clock parameters.
    """    
    for param_name in self.PARAMS_NS.keys():
      value = getattr(self, param_name, None)
      if value is None:
        print(f"{param_name} is not defined in the class instance.")
        continue
        
      else:
        if (int(value*1e9) != int(np.round(value*1e9, -1))):
          print(f"{param_name} was rounded: {value}s --> {int(np.round(value*1e9, -1))}ns, pulse resolution is limited to 10ns")
        value = int(np.round(value*1e9, -1))

        # Validate parameter as a clock if it is named as such
        if 'clock' in param_name.lower():
          # Clock period validation
          if (value < 300) or (value % 20 != 0):
            raise PulseTrainWidthError(
              f"{param_name} value of {value}ns is invalid. "
              f"Periodic pulses must be >= 300ns, and a multiple of 20ns"
            )
        else: # Validate parameter as a regular pulse
          if (0 < value < 50) or (value % 10 != 0) or (value < 0):
            raise ValueError(
              f"{param_name} value of {value}ns is invalid. "
              f"Pulses must be 0 or >= 50 ns, and a 10ns multiple."
            )
      self.PARAMS_NS[param_name] = value
    return self.PARAMS_NS

  def experimental_conditions(self):
    """
    Returns a dictionary of parameters that are pertinent for the relevant experiment.
    """
    if hasattr(self, 'PARAM_NAMES') and isinstance(self.PARAM_NAMES, (list, tuple)):
        return {param_name: getattr(self, param_name, None) for param_name in self.PARAM_NAMES}
    else:
        raise AttributeError("The 'param_names' attribute must be defined as a list or tuple in the class instance.")

class PulseBlasterCWODMR(PulseBlasterAuto):
  '''
  Programs the pulse sequences needed for CWODMR.

  Provides an
    * always ON channel for an AOM.
    * 50% duty cycle pulse for RF switch
    * clock signal for use with a data acquisition card
    * trigger signal for use with a data acquisition card
  '''
  def __init__(self, pb_board_number = 0,
                aom_channel = 0,
                rf_channel = 1,
                clock_channel = 2,
                trigger_channel = 3,
                rf_pulse_duration = 5e-6,
                clock_period = 200e-9,
                trigger_width = 500e-9):
      """
      pb_board_number - the board number (0, 1, ...)
      aom_channel output controls the AOM by holding a positive voltage
      rf_channel output controls a RF switch
      clock_channel output provides a clock input to the NI DAQ card
      trigger_channel output provides a rising edge trigger for the NI DAQ card
      """
      for param, value in locals().items():
        if param != 'self':
            setattr(self, param, value)

      # Pulse parameters that need validation, these are used by the code
      self.PARAMS_NS = {'rf_pulse_duration': None, 'clock_period': None, 'trigger_width': None}
      # Parameters of interest for USER
      self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'rf_pulse_duration', 'clock_period', 'trigger_width']

  def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
      if rf_pulse_duration:
        self.rf_pulse_duration = rf_pulse_duration
      self.validate_params()

      cycle_length_ns = 2*self.PARAMS_NS['rf_pulse_duration']

      hardware_pins = [self.aom_channel, self.rf_channel,
                        self.clock_channel, self.trigger_channel]

      self.open()
      pb = PBInd(pins = hardware_pins, on_time = cycle_length_ns)
      self.start_programming()

      pb.on(self.trigger_channel, 0, self.PARAMS_NS['trigger_width'])
      pb.make_clock(self.clock_channel, self.PARAMS_NS['clock_period'])
      pb.on(self.aom_channel, 0, cycle_length_ns)
      pb.on(self.rf_channel, 0, self.PARAMS_NS['rf_pulse_duration'])
      pb.program(float('inf'))

      self.stop_programming()
      self.close()
      return np.round(cycle_length_ns / self.PARAMS_NS['clock_period']).astype(int)

class PulseBlasterPulsedODMR(PulseBlasterAuto):
  # add functionality to make sure full_cycle_width is a multiple of NiDaq Clock
  '''
  Programs the pulse sequences needed for pulsed ODMR.

  AOM on / RF off , AOM off / RF on , AOM on / RF off , AOM off / RF off

  Provides
    * AOM channel with user-specified width
    * RF channel with user-specified width
    * RF pulse left, center, or right justified pulse
    * padding between the AOM and RF pulses
    * support for specifying AOM/RF hardware response times in order to fine-tune position of pulses
    * control of the full cycle width
    * clock signal for use with a data acquisition card
    * trigger signal for use with a data acquisition card
  '''
  def __init__(self, pb_board_number = 0,
                aom_channel = 0,
                rf_channel = 1,
                clock_channel = 2,
                trigger_channel = 3,
                clock_period = 500e-9,
                trigger_width = 500e-9,
                rf_pulse_duration = 5e-6,
                aom_width = 500e-9,
                aom_response_time = 200e-9,
                rf_response_time = 200e-9,
                pre_rf_pad = 100e-9,
                post_rf_pad = 100e-9,
                full_cycle_width = int(30e-6)):
    """
    pb_board_number - the board number (0, 1, ...)
    rf_channel output controls a RF switch
    clock_channel output provides a clock input to the NI DAQ card
    trigger_channel output provides a rising edge trigger for the NI DAQ card
    """
    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)
    
    # Pulse parameters that need validation, these are used by the code
    self.PARAMS_NS = {'rf_pulse_duration': None, 'aom_width': None, 'aom_response_time': None, 
                      'post_rf_pad': None, 'pre_rf_pad': None, 'full_cycle_width': None, 
                      'clock_period': None, 'rf_response_time': None, 'trigger_width': None}
    # Parameters of interest for USER
    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel',
                        'trigger_channel', 'clock_period', 'trigger_width', 'rf_pulse_duration',
                        'aom_width', 'aom_response_time', 'rf_response_time', 'pre_rf_pad',
                        'post_rf_pad', 'full_cycle_width']

  def _compute_rf_pulse_sequence(self, rf_pulse_duration, min_start_time, max_end_time, clock_period):
    additional_rf_delay = self.minimum_padding_finder([0, rf_pulse_duration], min_start_time, max_end_time, 
                                                      clock_period, [0, self.PARAMS_NS['trigger_width'], self.PARAMS_NS['aom_width']])
    return min_start_time + additional_rf_delay

  def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
    if rf_pulse_duration:
      self.raise_for_pulse_width(rf_pulse_duration)
      self.rf_pulse_duration = rf_pulse_duration
    else:
      self.raise_for_pulse_width(self.rf_pulse_duration)

    self.validate_params()

    half_cycle_width = int(self.PARAMS_NS['full_cycle_width'] / 2)

    # Observed that not setting the Full Cycle Width to an integer multiple of 2*clock_period, produces poor results
    # This is likely due to how the Nidaq processing function operates
    if (half_cycle_width % self.PARAMS_NS['clock_period'] != 0):
       raise PulseTrainWidthError(f"Set Full Cycle Width to an integer multiple of 2*clock_period. Try .")

    delay_rf_channel = self.PARAMS_NS['aom_width'] + self.PARAMS_NS['aom_response_time'] \
                        + self.PARAMS_NS['pre_rf_pad'] - self.PARAMS_NS['rf_response_time']

    rf_start_time = self._compute_rf_pulse_sequence(self.PARAMS_NS['rf_pulse_duration'], delay_rf_channel,
                                                    half_cycle_width, self.PARAMS_NS['clock_period'])

    hardware_pins = [self.aom_channel, self.rf_channel,
                      self.clock_channel, self.trigger_channel]
    self.open()

    pb = PBInd(pins = hardware_pins, on_time = self.PARAMS_NS['full_cycle_width'])
    self.start_programming()

    pb.on(self.trigger_channel, 0, self.PARAMS_NS['trigger_width'])
    pb.make_clock(self.clock_channel, self.PARAMS_NS['clock_period'])
    pb.on(self.aom_channel, 0, self.PARAMS_NS['aom_width'])
    pb.on(self.rf_channel, rf_start_time, self.PARAMS_NS['rf_pulse_duration'])
    pb.on(self.aom_channel, half_cycle_width, self.PARAMS_NS['aom_width'])
    pb.program(float('inf'))

    self.stop_programming()
    self.close()
    return np.round(self.PARAMS_NS['full_cycle_width'] / self.PARAMS_NS['clock_period']).astype(int)

  def raise_for_pulse_width(self, rf_pulse_duration):
    #the following enforces that the full cycle width is large enough
    requested_total_width = self.aom_width
    requested_total_width += self.aom_response_time
    requested_total_width += self.pre_rf_pad
    requested_total_width += rf_pulse_duration
    requested_total_width += self.post_rf_pad

    if requested_total_width >= self.full_cycle_width / 2:
        raise PulseTrainWidthError(f"half cycle width, {self.full_cycle_width / 2}, is not large enough to support requested pulse sequence, {requested_total_width}.")

class PulseBlasterRamHahnDD(PulseBlasterAuto):
  '''
  Programs pulse sequences for Ramsey, Hahn Echo, and Dynamical Decoupling experiments.
  
  Features:
    - User-specified AOM and RF pulse widths
    - Flexible padding and response time adjustments
    - Supports refocusing pulses during free precession
    - Clock and trigger signals for data acquisition
  '''
  def __init__(self, pb_board_number = 0,
                      aom_channel=0,
                      rf_channel=1,
                      clock_channel=2,
                      trigger_channel=3,
                      clock_period=400e-9,
                      trigger_width=500e-9,
                      rf_half_pi_pulse_width=1e-6,
                      aom_width=500e-9,
                      aom_response_time=200e-9,
                      rf_response_time=200e-9,
                      pre_rf_pad=100e-9,
                      post_rf_pad=100e-9,
                      free_precession_time=5e-6,
                      n_refocussing_pi_pulses=0):
    '''
    Initializes hardware and experiment parameters.
    
    Fixed Parameters:
      - rf_pi_pulse_width: Duration of a pi RF pulse
      - pre_rf_pad/post_rf_pad: Padding before/after RF pulses
      - aom_width: Laser pulse duration
    
    Variable Parameters:
      - free_precession_time: Time between RF pulses
      - n_refocussing_pi_pulses: Number of refocusing pulses
    '''
    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)
    
    self.PARAMS_NS = {'rf_half_pi_pulse_width': None, 'aom_width': None, 'aom_response_time': None, 
                    'post_rf_pad': None, 'pre_rf_pad': None, 'clock_period': None, 
                    'rf_response_time': None, 'trigger_width': None, 'free_precession_time': None}
    
    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 
                        'trigger_channel', 'clock_period', 'trigger_width', 'rf_half_pi_pulse_width', 
                        'aom_width', 'aom_response_time', 'rf_response_time', 'pre_rf_pad', 
                        'post_rf_pad', 'free_precession_time', 'n_refocussing_pi_pulses']
    
    self.validate_params()
      
  def program_pulser_state(self, free_precession_time=None, n_refocussing_pi_pulses=None):
    '''
    Update pulse sequence with new free precession time or refocusing pulses.
    '''
    if free_precession_time is not None:
        self.free_precession_time = free_precession_time
    elif n_refocussing_pi_pulses is not None:
        self.n_refocussing_pi_pulses = n_refocussing_pi_pulses
    
    self.validate_params()
    self.raise_for_pulse_width()

    # Get rf_edges and half_cycle_width
    rf_edges, half_cycle_width = self._compute_rf_pulse_sequence()
    full_cycle_width = half_cycle_width*2

    # Program hardware
    hardware_pins = [self.aom_channel, self.rf_channel,
                      self.clock_channel, self.trigger_channel]
    self.open()
    pb = PBInd(pins=hardware_pins, on_time=int(full_cycle_width))
    self.start_programming()

    # Program NiDaq clock and trigger signals
    pb.make_clock(self.clock_channel, self.PARAMS_NS['clock_period'])
    pb.on(self.trigger_channel, 0, self.PARAMS_NS['trigger_width'])

    # Program first half cycle's AOM pulses
    pb.on(self.aom_channel, 0, self.PARAMS_NS['aom_width'])

    # Program RF pulses
    for rf_pulse in range(len(rf_edges)//2):
        rf_pulse_duration = rf_edges[rf_pulse*2+1]-rf_edges[rf_pulse*2]
        pb.on(self.rf_channel, rf_edges[rf_pulse*2], rf_pulse_duration)
    
    # Program second half cycle's AOM pulses
    pb.on(self.aom_channel, half_cycle_width, self.PARAMS_NS['aom_width'])
    pb.program(float('inf'))

    self.stop_programming()
    self.close()
    
    # Used for DAC calculations and programming 
    return (full_cycle_width // self.PARAMS_NS['clock_period'])

  def _compute_rf_pulse_sequence(self):
    '''
    Calculate RF pulse timings.
    '''
    pi_pulse, pi_over_2 = 2*self.PARAMS_NS['rf_half_pi_pulse_width'], self.PARAMS_NS['rf_half_pi_pulse_width']
    aom_width, aom_rt = self.PARAMS_NS['aom_width'], self.PARAMS_NS['aom_response_time']
    free_precession_time = self.PARAMS_NS['free_precession_time']
    pre_rf_pad, post_rf_pad = self.PARAMS_NS['pre_rf_pad'], self.PARAMS_NS['post_rf_pad']
    clock_period = self.PARAMS_NS['clock_period']
    rf_rt = self.PARAMS_NS['rf_response_time']
    
    # Determine half_cycle_width based on free precession time
    fixed_times = aom_width + aom_rt + pi_pulse + free_precession_time + pre_rf_pad + post_rf_pad
    half_cycle_width = (fixed_times // clock_period + 2) * clock_period

    # Determine the bounds
    lower_rf_bound = aom_width + aom_rt + pre_rf_pad - rf_rt
    upper_rf_bound = half_cycle_width - rf_rt - post_rf_pad

    # RF edges for starting pi/2 pulse
    rf_edges = np.zeros(4+2*self.n_refocussing_pi_pulses)
    rf_edges[:2] = [0, self.PARAMS_NS['rf_half_pi_pulse_width']]

    # RF edges for ending pi/2 pulse
    ending_rf_pulse = pi_over_2 + free_precession_time
    rf_edges[-2:] = [ending_rf_pulse, ending_rf_pulse + pi_over_2]

    # RF edges for pi pulses
    remaining_void_increments = int((free_precession_time - self.n_refocussing_pi_pulses * pi_pulse)//10)
    for pulse_n in range(self.n_refocussing_pi_pulses):
        pulse_buffer = int(remaining_void_increments // (self.n_refocussing_pi_pulses - pulse_n + 1))
        remaining_void_increments -= pulse_buffer
        rf_edges[pulse_n*2+2] = pulse_n[pulse_n*2+1] + pulse_buffer*10
        rf_edges[pulse_n*2+3] = pulse_n[pulse_n*2+2] + pi_pulse

    # Calculate additional padding needed for PulseBlaster
    additional_rf_delay = self.minimum_padding_finder(rf_edges, lower_rf_bound, upper_rf_bound, clock_period, [aom_width, half_cycle_width])
    rf_edges = rf_edges + (additional_rf_delay + lower_rf_bound)
    return rf_edges, half_cycle_width

  def raise_for_pulse_width(self, tau=None, n_refocussing_pi_pulses=None):
    '''
    Validate that free precession time supports the requested number of pulses.
    '''
    if tau is None:
      tau = self.PARAMS_NS['free_precession_time']
    else:
      tau = np.round(tau*1e9)
    if n_refocussing_pi_pulses is None:
      n_refocussing_pi_pulses = self.n_refocussing_pi_pulses

    required_time = 50 + n_refocussing_pi_pulses * (2 * self.PARAMS_NS['rf_half_pi_pulse_width'] + 50)
    
    if tau < required_time:
      print("errored")
      raise PulseTrainWidthError(
        f"Free precession time too short for the # of requested pulses.\n"
        f"For {n_refocussing_pi_pulses} pulses, must be >{required_time}"
      )

# Arbitrary Pulses enable easier creation of less precise pulses
class PulseBlasterArb(PulseBlaster):
    """
    Class allows for easy creation of arbitrary pulses, but with a reduced resolution of only 50ns.
    Makes it easier to make pulse sequences without having to account for delays to satisfy the minimum pulse time constraint.
    Timing Parameters should be in ns.
    """
    def __init__(self, pb_board_number = 0):
        self.pb_board_number = pb_board_number
        self.reset()

    def reset(self):
        self.clock_channels = []
        self.clock_period = None
        self.channel_settings = []
        self.full_cycle_width = 0

    def set_clock_channels(self, pulse_blaster_channels, clock_period):
        '''
        pulse_blaster_channel can be an int or a list of ints
        '''
        if type(pulse_blaster_channels) == int:
            pulse_blaster_channels = [pulse_blaster_channels]

        if (clock_period%100e-9 != 0):  
              print(f"Clock period will be adjusted to: {int(clock_period/100e-9)*100e-9}ns. \n" +
                    f"Must be a multiple of 100ns")
          
        self.clock_channels = pulse_blaster_channels
        self.clock_period = clock_period

    def add_channels(self, pulse_blaster_channels, start_time, pulse_width):
        '''
        pulse_blaster_channel can be an int or a list of ints
        start_time and pulse_width are in seconds (and will be rounded
        to the nearest 50ns)

        One side-effect of this function is that it will increase the object's
        self.full_cycle_width_ns value if a requested channel's start_time + pulse_width
        exceeds the current full_cycle_width_ns value.

        Otherwise, one may also set the full cycle width manually with set_full_cycle_length.
        '''
        if type(pulse_blaster_channels) == int:
            pulse_blaster_channels = [pulse_blaster_channels]

        for pulse_blaster_channel in pulse_blaster_channels:
            self.channel_settings.append({
                'channel':pulse_blaster_channel,
                'start': start_time,
                'width': pulse_width
                })
            if start_time + pulse_width > self.full_cycle_width:
                self.full_cycle_width = start_time + pulse_width

    def set_full_cycle_length(self, cycle_width):
        '''
        cycle_width is in units of seconds and will be rounded to the nearest 50ns.

        Be sure to set this to a value greater than or equal to the time you need
        for your individual channels.
        '''
        if (cycle_width%100 != 0):
                self.full_cycle_width = np.round(cycle_width/100e-9)*100e-9
                print(f"Cycle width is being rounded to {self.full_cycle_width} s")

    def program_pulser_state(self, *args, **kwargs):
        '''
        Programs the pulser based on the state of the object, as instructed
        by calls to set_clock_channels, set_channels and set_full_cycle_length

        If a clock channel has been specified, will return full_cycle_length / clock_period,
        which is the number of clock "ticks" for each full pulse sequence cycle.
        This useful for a data acquisition device that utilizes the clock signal.

        If no clock channel has been specified, will return 0. 
        '''
        hardware_pins = self.clock_channels + [s['channel'] for s in self.channel_settings]
        
        self.open()
        pb = PBInd(pins = hardware_pins, on_time = int(self.full_cycle_width*1e7)*100)
        self.start_programming()

        for clock_channel in self.clock_channels:
            pb.make_clock(clock_channel, int(self.clock_period*1e7)*100)

        for a_chan_setting in self.channel_settings:
            pb.on(a_chan_setting['channel'], int(a_chan_setting['start']*1e9/50)*50, int(a_chan_setting['width']*1e9/50)*50)

        pb.program(float('inf'))
        self.stop_programming()
        self.close()

        if self.clock_period:
            return np.round(int(self.full_cycle_width*1e7)*100 / int(self.clock_period*1e7)*100).astype(int)
        else:
            return 0

    def experimental_conditions(self):
        '''
        Returns a dictionary of paramters that are pertinent for the relevant experiment
        '''
        return {
            'full_cycle_width':self.full_cycle_width,
            'clock_channels':self.clock_channels,
            'clock_period':self.clock_period,
            'channel_settings':self.channel_settings
        }

class PulseBlasterHoldAOM(PulseBlasterArb):
  '''
  Holds the AOM channel open indefinitely by programming it
  to output a constant positive voltage.
  Useful for confocal scanning.
  '''
  def __init__(self, pb_board_number = 0,
                    aom_channel = 0,
                    cycle_width = 10e-6):
    """
    pb_board_number - the board number (0, 1, ...)
    aom_channel output controls the AOM by holding a positive voltage
    cycle_width - the length of the programmed pulse. Since aom channel is held on, this value is arbitrary
    """
    super().__init__(pb_board_number)
    self.add_channels(aom_channel, 0, cycle_width)

  def turn_on(self):
    self.program_pulser_state()
    self.start()
    self.stop()
      