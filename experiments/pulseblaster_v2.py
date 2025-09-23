"""
Defines and determines PB instructions for basic pulse sequences: CWODMR, PODMR/Rabi, Ramsey/Hahn Echo/DD.
The Soundcore PulseBlaster can only handle changes spaced >=50ns apart, with 10ns resolution.
The PulseBlasterAuto class has helper functions to automate finding of the minimum padding for pulse sequences.
"""
# Imports
import numpy as np
from qt3utils.pulsers.interface import ExperimentPulser
from qt3utils.errors import PulseBlasterInitError, PulseBlasterError, PulseTrainWidthError
import sys
sys.path.append(r'C:\Users\QT3\Documents\Victor\pulseblaster')

# Attempt to import PulseBlaster modules with error handling for development purposes
try:
  #  from pulseblaster.PBInd import PBInd
    from pulseblaster.pb_instruct import PB_Instruct, PB_Channel
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

class PulseBlasterArbClock(PulseBlaster):
  """
  Use to finely select which portion of of the clock signal to integrate over.
  Clock signal is not even duty cycle, but rather a short (50ns) pulse at the start of each clock period.
  Compatible with clock periods >=100ns, with 10ns resolution.
  """
  def __init__(self, pb_board_number = 0,
                aom_channel = 0,
                rf_channel = 1,
                clock_channel = 2,
                trigger_channel = 3):
    self.pb_board_number = pb_board_number # Necessary for the open() function of the parent class
    self.aom = PB_Channel(pin = aom_channel, channel_name = 'aom')
    self.rf = PB_Channel(pin = rf_channel, channel_name = 'rf')
    self.clock = PB_Channel(pin = clock_channel, channel_name = 'clock')
    self.trigger = PB_Channel(pin = trigger_channel, channel_name = 'trigger')

    # Initialize start_stop lists
    for channel in [self.aom, self.rf, self.clock, self.trigger]:
        channel.start_stop = []

  def add_clock_tick(self, channel, start_time):
    """
    Adds a clock tick to the pulse sequence at the specified start time.

    :param clock_channel: The channel number for the clock signal.
    :param start_time: The start time for the clock tick in seconds.
    """    
    channel.start_stop.append((start_time, start_time + 100e-9))

  def experimental_conditions(self):
    """
    Returns a dictionary of parameters that are pertinent for the relevant experiment.
    """
    if hasattr(self, 'PARAM_NAMES') and isinstance(self.PARAM_NAMES, (list, tuple)):
        return {param_name: getattr(self, param_name, None) for param_name in self.PARAM_NAMES}
    else:
        raise AttributeError("The 'param_names' attribute must be defined as a list or tuple in the class instance.")

class CWODMR(PulseBlasterArbClock):
  def __init__(self, pb_board_number = 0,
                aom_channel = 0,
                rf_channel = 1,
                clock_channel = 2,
                trigger_channel = 3,

                trigger_width = 100e-9,
                integration_time = 1e-6):
    
    super().__init__(pb_board_number, aom_channel, rf_channel, clock_channel, trigger_channel)

    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)

    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'trigger_width', 'integration_time']

  def program_pulser_state(self, *args, **kwargs):
    self.cycle_period = 2 * (self.integration_time)

    for channel in [self.aom, self.rf, self.clock, self.trigger]:
      channel.start_stop = []

    # Add validation after defining timing
    if self.integration_time < 200e-9:
        raise ValueError(f"Integration time {self.integration_time} should be >200ns")

    # TODO: add more param validation later 

    self.pb_instruct = PB_Instruct(active_channels = [self.aom, self.rf, self.clock, self.trigger],
                                    cycle_period = self.cycle_period,
                                    clock_pin = self.clock_channel,
                                    clock_type = 'arb_gate_free_falling_edge',
                                    instruction_conflict_resolution_method = 'abort'
                                    )

    # Trigger, AOM, RO Clock
    self.trigger.start_stop.append((0, self.trigger_width))
    self.aom.start_stop.append((0, self.cycle_period))
    self.add_clock_tick(self.clock, 0)
    self.rf.start_stop.append((self.integration_time, self.cycle_period))
    self.add_clock_tick(self.clock, self.integration_time)

    self.pb_instruct.generate_instructions()
    self.pb_instruct.program_pb_loop_with_alloffs_and_run(
            check_visualization = True,
            number_of_loop_rpts = np.inf,
            all_off_duration_ns = 0, # Possible to add delay
            pb_board_number = self.pb_board_number)
    
    return self.cycle_period, 2

class PulseBlasterPulsedODMR(PulseBlasterArbClock):
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

                trigger_width = 100e-9,
                integration_time = 300e-9,

                aom_width = 500e-9,

                rf_pulse_duration = 100e-9,
                ro_delay = 700e-9,

                padding = 100e-9):
    
    super().__init__(pb_board_number, aom_channel, rf_channel, clock_channel, trigger_channel)

    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)

    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'trigger_width', 'integration_time', 'aom_width', 'rf_pulse_duration', 
                           'ro_delay', 'padding']

  def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
    if rf_pulse_duration:
      self.rf_pulse_duration = rf_pulse_duration
  
    self.cycle_period = 2 * (self.aom_width + self.ro_delay + self.padding + self.rf_pulse_duration + self.padding)

    for channel in [self.aom, self.rf, self.clock, self.trigger]:
      channel.start_stop = []

    # Add validation after defining timing
    # if self.integration_time > self.aom_width - 50e-9:
    #     raise ValueError(f"Integration time {self.integration_time} should be >50ns shorter than AOM width {self.aom_width}")

    # TODO: add more param validation later 

    self.pb_instruct = PB_Instruct(active_channels = [self.aom, self.rf, self.clock, self.trigger],
                                    cycle_period = self.cycle_period,
                                    clock_pin = self.clock_channel,
                                    clock_type = 'arb_gate_free_falling_edge',
                                    instruction_conflict_resolution_method = 'abort'
                                    )
    
    first_aom = self.padding
    rf_start = first_aom + self.aom_width + self.ro_delay + self.padding
    second_aom = rf_start + self.rf_pulse_duration + self.padding

    # Trigger, AOM, RO Clock
    self.trigger.start_stop.append((0, self.padding))
    self.aom.start_stop.append((first_aom, first_aom + self.aom_width))
    self.add_clock_tick(self.clock, first_aom + self.ro_delay)
    self.add_clock_tick(self.clock, first_aom + self.ro_delay + self.integration_time)

    # RF
    self.rf.start_stop.append((rf_start, rf_start + self.rf_pulse_duration))

    # AOM, RO Clock 2
    self.aom.start_stop.append((second_aom, second_aom + self.aom_width))
    self.add_clock_tick(self.clock, second_aom + self.ro_delay)
    self.add_clock_tick(self.clock, second_aom + self.ro_delay + self.integration_time)

    # No RF

    self.pb_instruct.generate_instructions()
    self.pb_instruct.program_pb_loop_with_alloffs_and_run(
            check_visualization = False,
            number_of_loop_rpts = np.inf,
            all_off_duration_ns = 0, # Possible to add delay
            pb_board_number = self.pb_board_number)

    return self.cycle_period, 4

class PulseBlasterRamHahnDD(PulseBlasterArbClock):
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

                rf_half_pi_duration = 100e-9,

                trigger_width = 100e-9,
                integration_time = 300e-9,
                aom_width = 300e-9,
                rf_pulse_duration = 100e-9,
                padding = 100e-9):
    
    super().__init__(pb_board_number, aom_channel, rf_channel, clock_channel, trigger_channel)

    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)

    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'rf_half_pi_duration', 'trigger_width', 'integration_time', 'aom_width', 'rf_pulse_duration', 'padding']

  def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
    if rf_pulse_duration:
      self.rf_pulse_duration = rf_pulse_duration
  
    self.cycle_period = 2 * (self.aom_width + self.padding + self.rf_pulse_duration + self.padding)

    for channel in [self.aom, self.rf, self.clock, self.trigger]:
      channel.start_stop = []

    # Add validation after defining timing
    if self.integration_time > self.aom_width - 50e-9:
        raise ValueError(f"Integration time {self.integration_time} should be >50ns shorter than AOM width {self.aom_width}")

    # TODO: add more param validation later 

    self.pb_instruct = PB_Instruct(active_channels = [self.aom, self.rf, self.clock, self.trigger],
                                    cycle_period = self.cycle_period,
                                    clock_pin = self.clock_channel,
                                    clock_type = 'arb_gate_free_falling_edge',
                                    instruction_conflict_resolution_method = 'abort'
                                    )
    
    first_aom = 0
    rf_start = self.aom_width + self.padding
    second_aom = rf_start + self.rf_pulse_duration + self.padding

    # Trigger, AOM, RO Clock
    self.trigger.start_stop.append((0, self.trigger_width))
    self.aom.start_stop.append((first_aom, first_aom + self.aom_width))
    self.add_clock_tick(self.clock, first_aom)
    self.add_clock_tick(self.clock, first_aom + self.integration_time)

    # RF
    self.rf.start_stop.append((rf_start, rf_start + self.rf_pulse_duration))

    # AOM, RO Clock 2
    self.aom.start_stop.append((second_aom, second_aom + self.aom_width))
    self.add_clock_tick(self.clock, second_aom)
    self.add_clock_tick(self.clock, second_aom + self.integration_time)

    # No RF

    self.pb_instruct.generate_instructions()
    self.pb_instruct.program_pb_loop_with_alloffs_and_run(
            check_visualization = False,
            number_of_loop_rpts = np.inf,
            all_off_duration_ns = 0, # Possible to add delay
            pb_board_number = self.pb_board_number)

    return self.cycle_period, 4

class PulseBlasterRamHahnDD(PulseBlasterArbClock):
  '''
  Programs pulse sequences for Ramsey, Hahn Echo, and Dynamical Decoupling experiments.

  Trigger, Integration Clock, AOM On
  TAU delay
  Loop:
    RF On
    TAU delay
  Integratopm Clock, AOM on
  TAU delay
  Loop:
    RF Off
    TAU delay
  
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

                      trigger_width=500e-9,
                      rf_half_pi_width=1e-6,
                      aom_width=500e-9,
                      aom_response_time=200e-9,
                      rf_response_time=200e-9,
                      pre_rf_pad=100e-9,
                      post_rf_pad=100e-9,
                      free_precession_time=5e-6,
                      n_refocussing_pi_pulses=0):


    
    super().__init__(pb_board_number, aom_channel, rf_channel, clock_channel, trigger_channel)

    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)

    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'trigger_width', 'integration_time', 'aom_width', 'rf_pulse_duration', 'padding']

  def program_pulser_state(self, rf_pulse_duration = None, *args, **kwargs):
    if rf_pulse_duration:
      self.rf_pulse_duration = rf_pulse_duration
  
    self.cycle_period = 2 * (self.aom_width + self.padding + self.rf_pulse_duration + self.padding)

    for channel in [self.aom, self.rf, self.clock, self.trigger]:
      channel.start_stop = []

    # Add validation after defining timing
    if self.integration_time > self.aom_width - 50e-9:
        raise ValueError(f"Integration time {self.integration_time} should be >50ns shorter than AOM width {self.aom_width}")

    # TODO: add more param validation later 

    self.pb_instruct = PB_Instruct(active_channels = [self.aom, self.rf, self.clock, self.trigger],
                                    cycle_period = self.cycle_period,
                                    clock_pin = self.clock_channel,
                                    clock_type = 'arb_gate_free_falling_edge',
                                    instruction_conflict_resolution_method = 'abort'
                                    )
    
    first_aom = 0
    rf_start = self.aom_width + self.padding
    second_aom = rf_start + self.rf_pulse_duration + self.padding

    # Trigger, AOM, RO Clock
    self.trigger.start_stop.append((0, self.trigger_width))
    self.aom.start_stop.append((first_aom, first_aom + self.aom_width))
    self.add_clock_tick(self.clock, first_aom)
    self.add_clock_tick(self.clock, first_aom + self.integration_time)

    # RF
    self.rf.start_stop.append((rf_start, rf_start + self.rf_pulse_duration))

    # AOM, RO Clock 2
    self.aom.start_stop.append((second_aom, second_aom + self.aom_width))
    self.add_clock_tick(self.clock, second_aom)
    self.add_clock_tick(self.clock, second_aom + self.integration_time)

    # No RF

    self.pb_instruct.generate_instructions()
    self.pb_instruct.program_pb_loop_with_alloffs_and_run(
            check_visualization = False,
            number_of_loop_rpts = np.inf,
            all_off_duration_ns = 0, # Possible to add delay
            pb_board_number = self.pb_board_number)

    return self.cycle_period, 4
  


class TestDelayPulse(PulseBlasterArbClock):
  def __init__(self, pb_board_number = 0,
                      aom_channel=0,
                      rf_channel=1,
                      clock_channel=2,
                      trigger_channel=3,

                      trigger_width=100e-9,
                      aom_off_width=1000e-9,
                      aom_width=1000e-9,
                      integration_delay=50e-6):

    super().__init__(pb_board_number, aom_channel, rf_channel, clock_channel, trigger_channel)

    for param, value in locals().items():
      if param != 'self':
        setattr(self, param, value)

    self.PARAM_NAMES = ['pb_board_number', 'aom_channel', 'rf_channel', 'clock_channel', 'trigger_channel',
                          'trigger_width', 'aom_width', 'integration_delay']

  def program_pulser_state(self, integration_delay = None, *args, **kwargs):
    if integration_delay:
      self.integration_delay = integration_delay
  
    self.cycle_period = self.aom_off_width + self.aom_width

    for channel in [self.aom, self.rf, self.clock, self.trigger]:
      channel.start_stop = []

    # TODO: add more param validation later 

    self.pb_instruct = PB_Instruct(active_channels = [self.aom, self.rf, self.clock, self.trigger],
                                    cycle_period = self.cycle_period,
                                    clock_pin = self.clock_channel,
                                    clock_type = 'arb_gate_free_falling_edge',
                                    instruction_conflict_resolution_method = 'abort'
                                    )
    
    # 0: trigger, AOM off, Throw-Away Clock
    self.trigger.start_stop.append((0, self.trigger_width))
    self.add_clock_tick(self.clock, 0)

    # aom_off_width - 500e-9 + integration_delay: Dark Counts Clock
    self.add_clock_tick(self.clock, self.aom_off_width - 500e-9 + self.integration_delay)

    # aom_off_width: AOM on
    self.aom.start_stop.append((self.aom_off_width, self.cycle_period))

    # aom_off_width + integration_delay: Light Counts Clock
    self.add_clock_tick(self.clock, self.aom_off_width + self.integration_delay)

    # aom_off_width + integration_delay + 500e-9: Throw-Away Clock 2
    self.add_clock_tick(self.clock, self.aom_off_width + self.integration_delay + 500e-9)

    self.pb_instruct.generate_instructions()
    self.pb_instruct.program_pb_loop_with_alloffs_and_run(
            check_visualization = True,
            number_of_loop_rpts = np.inf,
            all_off_duration_ns = 0, # Possible to add delay
            pb_board_number = self.pb_board_number)

    return self.cycle_period, 4