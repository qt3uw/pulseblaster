import logging
import time
import numpy as np
import nidaqmx.errors
import nidaqmx.constants
import qt3utils.experiments.common
import qt3utils.experiments.podmr

from qt3utils.errors import PulseTrainWidthError

logger = logging.getLogger(__name__)

def signal_to_background(trace, pre_trigger, aom_width, rf_pulse_duration, verbose=False,
                        aom_width_duty = 1.0):
    '''
    Assumes trace produced by qt3utils.experiments.rabi.Rabi class and
    is the aggregated data for a particular RF width.

    The inputs `pre_trigger`, `aom_width` and `rf_pulse_duration` are all in units of index of the trace.
    That is, they are in units of clock ticks.

    Assumes that trace is of shape
        * pre_trigger
        * aom_width: aom on / rf off (background)
        * rf_pulse_duration:  aom off / rf on
        * aom_width: aom on/ rf off  (signal)

    returns sum(signal) / sum(background)

    '''
    background_end = pre_trigger + int(aom_width*aom_width_duty)
    signal_start = pre_trigger + aom_width + rf_pulse_duration
    signal_end = signal_start + int(aom_width*aom_width_duty)

    background = np.sum(trace[pre_trigger:background_end])
    signal = np.sum(trace[signal_start:signal_end])

    if verbose:
        print('background')
        print(trace[pre_trigger:background_end])
        print('signal')
        print(trace[signal_start:signal_end])

    return signal / background


class Rabi(qt3utils.experiments.common.Experiment):

    def __init__(self, podmr_pulser, rfsynth, edge_counter_config,
                       photon_counter_nidaq_terminal = 'PFI0',
                       clock_nidaq_terminal = 'PFI12',
                       trigger_nidaq_terminal = 'PFI1',
                       rf_pulse_duration_low = 100e-9,
                       rf_pulse_duration_high = 10e-6,
                       rf_pulse_duration_step = 50e-9,
                       rf_power = -20,
                       rf_frequency = 2870e6,
                       rfsynth_channel = 0):
        '''
        The input parameters to this object specify the conditions
        of an experiment and the hardware system setup.

        Hardware Settings
            podmr_pulser - a qt3utils.pulsers.interface.ODMRPulser object (such as qt3utils.pulsers.qcsapphire.QCSapphPulsedODMRPulser)
            rfsynth - a qt3rfsynthcontrol.Pulser object
            edge_counter_config - a qt3utils.nidaq.config.EdgeCounter object

            NI DAQ Connections
            * photon_counter_nidaq_terminal - terminal connected to TTL pulses that indicate a photon
            * clock_nidaq_terminal - terminal connected to the clock_pulser_channel
            * trigger_nidaq_terminal - terminal connected to the trigger_pulser_channel

        Experimental parameters

            The rf width parameters define the range and step size of the scan.
                The scan is inclusive of rf_pulse_duration_low and rf_pulse_duration_high.
            The rf_power specifices the power of the MW source in units of dB mWatt.

        The user is responsible for analyzing the data. However, during acquisition,
        a callback function can be supplied in order to perform an analysis
        during the scan. The default callback function is defined in this module,
        qt3utils.experiments.cwodmr.aggregate_data.

        Without a callback function the raw data will be stored and could require
        prohibitive amounts of memory.

        '''

        ## TODO: assert conditions on rf width low, high and step sizes
        # to be compatible with pulser.

        self.rf_pulse_duration_low = np.round(rf_pulse_duration_low, 9)
        self.rf_pulse_duration_high = np.round(rf_pulse_duration_high, 9)
        self.rf_pulse_duration_step = np.round(rf_pulse_duration_step, 9)
        self.rf_power = rf_power
        self.rf_frequency = rf_frequency

        self.pulser = podmr_pulser
        #assert (type(self.pulser) = qcsapphire.Pulser) or (type(self.pulser) = pulseblaster.Pulser)
        self.rfsynth = rfsynth
        self.rfsynth_channel = rfsynth_channel
        #assert(type(self.rfsynth) = qt3rfsynthcontrol.QT3SynthHD)

        self.photon_counter_nidaq_terminal = photon_counter_nidaq_terminal
        self.clock_nidaq_terminal = clock_nidaq_terminal
        self.trigger_nidaq_terminal  = trigger_nidaq_terminal

        self.edge_counter_config = edge_counter_config
        
        # Track the counter task reference
        self.counter_task = None

    def experimental_conditions(self):
        '''
        Returns a dictionary that captures the essential experimental conditions.
        '''
        return {
            'rf_pulse_duration_low':self.rf_pulse_duration_low,
            'rf_pulse_duration_high':self.rf_pulse_duration_high,
            'rf_pulse_duration_step':self.rf_pulse_duration_step,
            'rf_power':self.rf_power,
            'rf_frequency':self.rf_frequency,
            'pulser':self.pulser.experimental_conditions()
        }

    def _stop_and_close_daq_tasks(self, trim_errors=True):
        """
        Safely stops and closes DAQ tasks, handling any exceptions that occur.
        Only attempts to stop/close if the task exists and is not already closed.
        
        Parameters:
        trim_errors (bool): If True, only log the error type without details
        """
        # First check if the counter task exists and has a valid handle
        has_valid_task = (hasattr(self.edge_counter_config, 'counter_task') and 
                        self.edge_counter_config.counter_task is not None)
        
        # Don't attempt operations on invalid tasks
        if not has_valid_task:
            return
            
        # Try to stop the task first
        try:
            # Check if the task has a valid handle
            if not getattr(self.edge_counter_config.counter_task, '_handle', None) is None:
                self.edge_counter_config.counter_task.stop()
                logger.debug("Counter task stopped successfully")
        except Exception as e:
            # Only log non-trivial errors with trimmed message
            if not "invalid or does not exist" in str(e):
                error_msg = f"{type(e).__name__}" if trim_errors else f"{type(e).__name__}: {str(e)}"
                logger.error(f'Error stopping counter task: {error_msg}')
        
        # Wait a moment for the device to register the stop command
        time.sleep(0.05)
            
        # Then try to close the task
        try:
            if not getattr(self.edge_counter_config.counter_task, '_handle', None) is None:
                self.edge_counter_config.counter_task.close()
                logger.debug("Counter task closed successfully")
        except Exception as e:
            if not "invalid or does not exist" in str(e):
                error_msg = f"{type(e).__name__}" if trim_errors else f"{type(e).__name__}: {str(e)}"
                logger.error(f'Error closing counter task: {error_msg}')
        finally:
            # Always set to None after trying to close, even if it failed
            self.edge_counter_config.counter_task = None
            self.edge_counter_config.counter_reader = None
            
        # Give the hardware a moment to fully release resources
        time.sleep(0.1)

    def _acquire_data_at_parameter(self, current_rf_pulse_duration, N_cycles = 10000,
                              post_process_function = qt3utils.experiments.podmr.simple_measure_contrast, trim_error=True):
        """
        Acquires data for a specific RF pulse duration with retry logic.
        Will attempt to acquire data up to 5 times if needed.
        """
        self.N_cycles = int(N_cycles)
        
        # Program the pulser for this RF pulse duration
        self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state(current_rf_pulse_duration)
        
        # Calculate acquisition parameters
        self.N_clock_ticks_per_frequency = int(self.N_clock_ticks_per_cycle * self.N_cycles)
        self.daq_time = self.N_clock_ticks_per_frequency * self.pulser.clock_period
        
        logger.debug(f'Acquiring {self.N_clock_ticks_per_frequency} total samples')
        logger.debug(f'  sample period of {self.pulser.clock_period} seconds')
        logger.debug(f'  acquisition time of {self.daq_time} seconds')

        # Retry mechanism
        max_retries = 5
        retry_count = 0
        acquisition_success = False
        data_buffer = None
        samples_read = 0

        while retry_count < max_retries and not acquisition_success:
            if retry_count > 0:
                logger.info(f'Retry {retry_count} for RF duration: {current_rf_pulse_duration} seconds')
            
            # Ensure any previous tasks are properly cleaned up and resources released
            self._stop_and_close_daq_tasks()
            
            try:
                # Set up with appropriate sampling mode
                self.edge_counter_config.configure_counter_period_measure(
                    source_terminal = self.photon_counter_nidaq_terminal,
                    N_samples_to_acquire_or_buffer_size = int(self.N_clock_ticks_per_frequency),
                    clock_terminal = self.clock_nidaq_terminal,
                    trigger_terminal = self.trigger_nidaq_terminal)
                
                self.edge_counter_config.create_counter_reader()
                
                # Prepare data buffer
                data_buffer = np.zeros(self.N_clock_ticks_per_frequency, dtype=np.float64)
                
                # Start the pulser first
                self.pulser.start()
                
                # Wait a small amount to make sure the pulser is running
                time.sleep(0.01)
                
                self.edge_counter_config.counter_task.start()
                
                # Use a timeout that's reasonably longer than the expected acquisition time
                timeout = max(10, self.daq_time * 1.2)  # At least 10 seconds or 20% longer than expected
                
                # Wait for task to complete
                self.edge_counter_config.counter_task.wait_until_done(timeout=timeout)
                
                # Read available samples (up to our buffer size)
                samples_read = self.edge_counter_config.counter_reader.read_many_sample_double(
                    data_buffer,
                    number_of_samples_per_channel=self.N_clock_ticks_per_frequency,
                    timeout=timeout)
                
                logger.debug(f"Read {samples_read} samples at once")
                
                # If we read all expected samples, consider it a success
                if samples_read == self.N_clock_ticks_per_frequency:
                    acquisition_success = True
                    logger.debug(f"Acquisition successful with {samples_read}/{self.N_clock_ticks_per_frequency} samples")
                else:
                    logger.warning(f"Insufficient samples: {samples_read}/{self.N_clock_ticks_per_frequency}")
                    retry_count += 1
            
            except Exception as e:
                # Use simplified error message format
                error_msg = f'{type(e).__name__}' if trim_error else f'{type(e).__name__}: {str(e)}'
                logger.error(f'Error during acquisition: {error_msg}')
                retry_count += 1
            
            finally:
                # Always stop the pulser regardless of read success
                try:
                    self.pulser.stop()
                except Exception as e:
                    error_msg = f'{type(e).__name__}' if trim_error else f'{type(e).__name__}: {str(e)}'
                    logger.error(f'Error stopping pulser: {error_msg}')
                
                # Clean up DAQ tasks with same error verbosity setting
                self._stop_and_close_daq_tasks(trim_errors=trim_error)

        if not acquisition_success:
            logger.error(f"Acquisition failed after {retry_count} retries for RF duration={current_rf_pulse_duration}")
            return [current_rf_pulse_duration, np.nan]
        elif retry_count > 0:
            logger.info(f"Acquisition succeeded after {retry_count} retries for RF duration={current_rf_pulse_duration}")

        if post_process_function is not None:
            data_buffer = post_process_function(data_buffer, self)
        
        return [current_rf_pulse_duration, data_buffer]

    def run(self, N_cycles = 50000,
                  post_process_function = qt3utils.experiments.podmr.simple_measure_contrast, trim_error=True):
        """
        Performs the scan over the specificed range of RF widths.

        For each RF width, some number of cycles of data are acquired. A cycle
        is one full sequence of the pulse train used in the experiment.
        For Rabi, a cycle is {AOM on, AOM off/RF on, AOM on, AOM off/RF off}.

        The N_cycles specifies the total number of these cycles to
        acquire. Your choice depends on your desired resolution or signal-to-noise
        ratio, your post-data acquisition processing choices, and the amount of memory
        available on your computer.

        For each width, the number of data read from the NI DAQ will be
        N_clock_ticks_per_cycle * N_cycles, where N_clock_ticks_per_cycle
        is the value returned by self.set_pulser_state(rf_pulse_duration).

        Given the way our pulser is configured, N_clock_ticks_per_cycle will
        grow linearly by rf_pulse_duration.

        The acquired data are stored in a data_buffer within this method. They
        may be analyzed with a function passed to post_process_function,
        which is useful to reduce the required memory to hold the raw data.

        After data acquisition for each width in the scan,
        the post_process_function is called and takes two arguments:
            1) data_buffer: the full trace of data acquired
            2) self: a reference to an instance of this object

        The output of post_process_function is recorded in the data
        returned by this function.

        If post_process_function = None, the full raw data trace will be kept.

        The return from this function is a numpy array. Each element of the array
        is a list of the following values
            RF width (float),
            data_post_processing_output, or raw data trace (typically of type numpy array(dtype = float))

        Because of the mixed types in this array, the numpy array data type returned
        here is an 'object'.

        The remaining (fixed) values for analysis can be obtained from the
        self.experimental_conditions function.
        """

        # First check that the pulser width is large enough
        try:
            self.pulser.raise_for_pulse_width(self.rf_pulse_duration_high)
            # Question: should we automatically increase the pulser width or force the user to do it?
        except PulseTrainWidthError as e:
            logger.error(f'The largest requested RF width pulse, self.rf_pulse_duration_high = {self.rf_pulse_duration_high}, is too large.')
            raise e

        # Configure RF synthesizer once at the beginning
        self.rfsynth.stop_sweep()
        self.rfsynth.trigger_mode('disabled')
        self.rfsynth.set_power(self.rfsynth_channel, self.rf_power)
        self.rfsynth.set_frequency(self.rfsynth_channel, self.rf_frequency)
        self.rfsynth.rf_on(self.rfsynth_channel)
        
        # Short wait for RF box to stabilize
        time.sleep(0.5)

        data = []
        rf_pulse_duration_list = np.arange(self.rf_pulse_duration_low, self.rf_pulse_duration_high + self.rf_pulse_duration_step, self.rf_pulse_duration_step)

        try:
            for pulse_index, rf_pulse_duration in enumerate(rf_pulse_duration_list):
                current_rf_pulse_duration = np.round(rf_pulse_duration, 9)
                logger.info(f'RF Width: {current_rf_pulse_duration} seconds -- ({pulse_index+1}/{len(rf_pulse_duration_list)})')
                data.append(self._acquire_data_at_parameter(current_rf_pulse_duration, N_cycles, post_process_function, trim_error))
                
                # Allow a short delay between measurements for system stability
                time.sleep(0.2)  # 200ms between measurements
                
        except Exception as e:
            error_msg = f'{type(e).__name__}' if trim_error else f'{type(e).__name__}: {str(e)}'
            logger.error(f'Experiment error: {error_msg}')
            raise e

        finally:
            # Final cleanup
            self._stop_and_close_daq_tasks(trim_errors=trim_error)
            
            # Turn off RF synthesizer
            try:
                self.rfsynth.rf_off(self.rfsynth_channel)
            except Exception as e:
                error_msg = f'{type(e).__name__}' if trim_error else f'{type(e).__name__}: {str(e)}'
                logger.error(f'Error turning off RF synthesizer: {error_msg}')
            
            # Convert data to numpy array and return
            data = np.array(data, dtype=object)
            return data