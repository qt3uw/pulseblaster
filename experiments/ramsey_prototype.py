import logging
import time
import numpy as np
import nidaqmx.errors

import qt3utils.experiments.podmr
from qt3utils.errors import PulseTrainWidthError

logger = logging.getLogger(__name__)


class Ramsey:

    def __init__(self, ramsey_pulser, rfsynth, edge_counter_config,
                       photon_counter_nidaq_terminal = 'PFI0',
                       clock_nidaq_terminal = 'PFI12',
                       trigger_nidaq_terminal = 'PFI1',
                       tau_low = 1e-6,
                       tau_high = 10e-6,
                       tau_step = 0.2e-6,
                       rf_power = -25,
                       rf_frequency = 2870e6,
                       rfsynth_channel = 0):
        '''
        The input parameters to this object specify the conditions
        of an experiment and the hardware system setup.

        Hardware Settings
            ramsey_pulser - a qt3utils.pulsers.pulseblaster.PulseBlasterRamHahnDD object
            rfsynth - a qt3rfsynthcontrol.Pulser object
            edge_counter_config - a qt3utils.nidaq.config.EdgeCounter object

            NI DAQ Connections
            * photon_counter_nidaq_terminal - terminal connected to TTL pulses that indicate a photon
            * clock_nidaq_terminal - terminal connected to the clock_pulser_channel
            * trigger_nidaq_terminal - terminal connected to the trigger_pulser_channel

        Experimental parameters

            The free precession time is scanned from tau_low, to tau_high,
            in step sizes of tau_step.
            The scan is inclusive of tau_low and tau_high.
            The rf_power specifices the power of the MW source in units of dB mWatt.

            In order to control the width of the pi/2 pulses, use the ramsey_pulser
            object to set the rf_pi_pulse_width value.

        The user is responsible for analyzing the data. However, during acquisition,
        a callback function can be supplied in order to perform an analysis
        during the scan. The default callback function is defined in this module,
        qt3utils.experiments.cwodmr.aggregate_data.

        Without a callback function the raw data will be stored and could require
        prohibitive amounts of memory.

        '''

        ## TODO: assert conditions on rf width low, high and step sizes
        # to be compatible with pulser.

        self.tau_low = np.round(tau_low, 9)
        self.tau_high = np.round(tau_high, 9)
        self.tau_step = np.round(tau_step, 9)
        self.rf_power = rf_power
        self.rf_frequency = rf_frequency

        self.pulser = ramsey_pulser
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
            'tau_low':self.tau_low,
            'tau_high':self.tau_high,
            'tau_step':self.tau_step,
            'rf_power':self.rf_power,
            'rf_frequency':self.rf_frequency,
            'pulser':self.pulser.experimental_conditions()
        }

    def _stop_and_close_daq_tasks(self):
        """
        Safely stops and closes DAQ tasks, handling any exceptions that occur.
        Only attempts to stop/close if the task exists and is not already closed.
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
            # Only log non-trivial errors (i.e., not "task doesn't exist")
            if not "invalid or does not exist" in str(e):
                logger.error(f'Error stopping counter task: {type(e)}: {e}')
        
        # Wait a moment for the device to register the stop command
        time.sleep(0.05)
            
        # Then try to close the task
        try:
            if not getattr(self.edge_counter_config.counter_task, '_handle', None) is None:
                self.edge_counter_config.counter_task.close()
                logger.debug("Counter task closed successfully")
        except Exception as e:
            if not "invalid or does not exist" in str(e):
                logger.error(f'Error closing counter task: {type(e)}: {e}')
        finally:
            # Always set to None after trying to close, even if it failed
            self.edge_counter_config.counter_task = None
            self.edge_counter_config.counter_reader = None
            
        # Give the hardware a moment to fully release resources
        time.sleep(0.1)

    def run(self, N_cycles = 50000, post_process_function = qt3utils.experiments.podmr.simple_measure_contrast, trim_error=True):
        """
        Performs the scan over the specificed range of free precession times.

        For each RF pulse delay, tau, some number of cycles of data are acquired. A cycle
        is one full sequence of the pulse train used in the experiment as specified
        by the supplied ramsey_pulser object.

        The N_cycles specifies the total number of these cycles to
        acquire. Your choice depends on your desired resolution or signal-to-noise
        ratio, your post-data acquisition processing choices, and the amount of memory
        available on your computer.

        For each free precession time, tau, the number of data read from the NI DAQ will be
        N_clock_ticks_per_cycle * N_cycles, where N_clock_ticks_per_cycle
        is the value returned by self.set_pulser_state(tau).

        The acquired data are stored in a data_buffer within this method. They
        may be processed with a function passed to post_process_function,
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
            RF Frequency (float),
            data_post_processing_output, or raw data trace (typically of type numpy array(dtype = float))

        Because of the mixed types in this array, the numpy array data type returned
        here is an 'object'.

        The remaining (fixed) values for analysis can be obtained from the
        self.experimental_conditions function.
        """
        
        # First check that the pulser width is large enough
        try:
            self.pulser.raise_for_pulse_width(self.tau_low, 0)
            # This should NEVER raise an exception. But we keep it here
            # because this kind of check should be performed for all experiments
            # before a run is started. HahnEcho and Dynamic Decoupling classes
            # should follow this pattern.
        except PulseTrainWidthError as e:
            logger.error(f'The smallest requested free precession time, self.tau_low = {self.tau_low}, is too small.')
            raise e
        
        self.N_cycles = int(N_cycles)

        # Configure RF synthesizer once at the beginning
        self.rfsynth.stop_sweep()
        self.rfsynth.trigger_mode('disabled')
        self.rfsynth.set_power(self.rfsynth_channel, self.rf_power)
        self.rfsynth.set_frequency(self.rfsynth_channel, self.rf_frequency)
        self.rfsynth.rf_on(self.rfsynth_channel)
        
        # Short wait for RF box to stabilize
        time.sleep(0.1)

        data = []
        tau_list = np.arange(self.tau_low, self.tau_high + self.tau_step, self.tau_step)

        try:
            for tau_index, self.current_tau in enumerate(tau_list):
                logger.info(f'Free Precession Time, tau: {self.current_tau} seconds -- ({tau_index+1}/{len(tau_list)})')
                
                # Retry mechanism
                max_retries = 5
                retry_count = 0
                acquisition_success = False
                data_buffer = None
                samples_read = 0

                while retry_count < max_retries and not acquisition_success:
                    if retry_count > 0:
                        logger.info(f'Retry {retry_count} for tau: {self.current_tau} seconds')
                    
                    # Ensure any previous tasks are properly cleaned up and resources released
                    self._stop_and_close_daq_tasks()
                    
                    try:
                        # Program the pulser for this tau value
                        self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state(self.current_tau)
                        
                        # Calculate acquisition parameters
                        self.N_clock_ticks_at_tau = int(self.N_clock_ticks_per_cycle * self.N_cycles)
                        self.daq_time = self.N_clock_ticks_at_tau * self.pulser.clock_period
                        
                        logger.debug(f'Acquiring {self.N_clock_ticks_at_tau} total samples')
                        logger.debug(f'  sample period of {self.pulser.clock_period} seconds')
                        logger.debug(f'  acquisition time of {self.daq_time} seconds')

                        # Set up with a continuous sampling mode instead of finite
                        # This is done by passing the sampling_mode parameter
                        self.edge_counter_config.configure_counter_period_measure(
                            source_terminal = self.photon_counter_nidaq_terminal,
                            N_samples_to_acquire_or_buffer_size = int(self.N_clock_ticks_at_tau),
                            clock_terminal = self.clock_nidaq_terminal,
                            trigger_terminal = self.trigger_nidaq_terminal)
                            # sampling_mode = nidaqmx.constants.AcquisitionType.CONTINUOUS)
                        
                        self.edge_counter_config.create_counter_reader()
                        
                        # Prepare data buffer
                        data_buffer = np.zeros(self.N_clock_ticks_at_tau, dtype=np.float64)
                        
                        # Start the pulser first
                        self.pulser.start()
                        
                        # Wait a small amount to make sure the pulser is running
                        time.sleep(0.01)
                        
                        self.edge_counter_config.counter_task.start()
                        
                        # Use a timeout that's reasonably longer than the expected acquisition time
                        timeout = max(10, self.daq_time * 1.2)  # At least 10 seconds or 20% longer than expected
                        self.edge_counter_config.counter_task.wait_until_done(timeout=timeout)
                        
                        # # Wait for task to complete or reach a specific state before reading
                        # try:
                        #     # Using a shorter timeout to check if task is done or has data available
                        #     wait_timeout = min(5.0, self.daq_time * 0.5)
                        #     self.edge_counter_config.counter_task.wait_until_done(timeout=wait_timeout)
                        # except Exception as e:
                        #     # The task might not be done, but could still have data to read
                        #     # Just log and continue with reading
                        #     logger.debug(f"Wait until done: {type(e)}: {e}")
                        
                        # Read available samples (up to our buffer size)
                        samples_read = self.edge_counter_config.counter_reader.read_many_sample_double(
                            data_buffer,
                            number_of_samples_per_channel=self.N_clock_ticks_at_tau,
                            timeout=timeout)
                        
                        logger.debug(f"Read {samples_read} samples at once")
                        
                        # If we read all expected samples, consider it a success
                        if samples_read == self.N_clock_ticks_at_tau:
                            acquisition_success = True
                            logger.debug(f"Acquisition successful with {samples_read}/{self.N_clock_ticks_at_tau} samples")
                        else:
                            logger.warning(f"Insufficient samples: {samples_read}/{self.N_clock_ticks_at_tau}")
                            retry_count += 1
                    
                    except Exception as e:
                        if trim_error:
                            logger.error(f'Error during acquisition: {type(e)}')
                        else:
                            logger.error(f'Error during acquisition: {type(e)}: {e}')
                        retry_count += 1
                    
                    finally:
                        # Always stop the pulser regardless of read success
                        try:
                            self.pulser.stop()
                        except:
                            pass

                if not acquisition_success:
                    logger.error(f"Acquisition failed after {retry_count} retries for tau={self.current_tau}")
                    data.append([self.current_tau, np.nan])
                    continue
                elif retry_count > 0:
                    logger.info(f"Acquisition succeeded after {retry_count} retries for tau={self.current_tau}")
                processed_buffer = post_process_function(data_buffer[:samples_read], self)
                data.append([self.current_tau, processed_buffer])
                
                # Allow a short delay between measurements for system stability
                time.sleep(0.2)  # 200ms between measurements

        except Exception as e:
            logger.error(f'Experiment error: {type(e)}: {e}')
            raise e

        finally:
            # Final cleanup
            self._stop_and_close_daq_tasks()
            # Turn off RF synthesizer
            try:
                self.rfsynth.rf_off(self.rfsynth_channel)
            except Exception as e:
                logger.error(f'Error turning off RF synthesizer: {type(e)}: {e}')
            
            # Convert data to numpy array and return
            data = np.array(data, dtype=object)
            return data