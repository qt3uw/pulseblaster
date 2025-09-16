import logging
import time
import numpy as np

import qt3utils.experiments.common
from qt3utils.errors import PulseTrainWidthError

class RABI(qt3utils.experiments.common.Experiment):
    def __init__(self, pulser, rfsynth, edge_counter_config,
                 rf_pulse_duration_low=100e-9,
                 rf_pulse_duration_high=10e-6,
                 rf_pulse_duration_step=50e-9,
                 rf_frequency=2870e6,
                 rf_power=-20, **kwargs):
        
        # Call parent constructor
        super().__init__(pulser, rfsynth, edge_counter_config, **kwargs)
        
        # Set RABI-specific attributes
        self.rf_pulse_duration_low = rf_pulse_duration_low
        self.rf_pulse_duration_high = rf_pulse_duration_high
        self.rf_pulse_duration_step = rf_pulse_duration_step
        self.rf_frequency = rf_frequency
        self.rf_power = rf_power

    def experimental_conditions(self):
        '''
        Returns a dictionary that captures the essential experimental conditions.
        '''
        return {
            'rf_pulse_duration_low': self.rf_pulse_duration_low,
            'rf_pulse_duration_high': self.rf_pulse_duration_high,
            'rf_pulse_duration_step': self.rf_pulse_duration_step,
            'rf_frequency': self.rf_frequency,
            'rf_power': self.rf_power,
            'pulser': self.pulser.experimental_conditions()
        }
    
    def run(self, N_cycles=500000,
            post_process_function=qt3utils.experiments.common.measure_readout_contrast,
            random_order=False):
        """
        Performs a RABI experiment scanning over RF pulse durations.

        For each pulse duration, some number of cycles of data are acquired. A cycle
        is one full sequence of the pulse train used in the experiment. For RABI,
        a cycle typically includes {initialization, RF pulse (variable duration), readout}.

        The N_cycles specifies the total number of these cycles to acquire.

        The return from this function is a numpy array. Each element contains:
            RF Pulse Duration (float),
            data_post_processing_output, or raw data trace
        """
        self.N_cycles = int(N_cycles)
        if self.N_cycles <= 0:
            raise ValueError("N_cycles must be positive")

        # Validate pulse width before starting
        try:
            self.pulser.raise_for_pulse_width(self.rf_pulse_duration_high)
        except PulseTrainWidthError as e:
            self.logger.error(f'The largest requested RF width pulse, {self.rf_pulse_duration_high}, is too large.')
            raise e

        # Configure RF synthesizer
        self.rfsynth.stop_sweep()
        self.rfsynth.trigger_mode('disabled')
        self.rfsynth.set_power(self.rfsynth_channel, self.rf_power)
        self.rfsynth.set_frequency(self.rfsynth_channel, self.rf_frequency)
        self.rfsynth.rf_on(self.rfsynth_channel)
        time.sleep(1)  # wait for RF box to fully turn on

        data = []
        rf_pulse_duration_list = np.arange(
            self.rf_pulse_duration_low, 
            self.rf_pulse_duration_high + self.rf_pulse_duration_step, 
            self.rf_pulse_duration_step)

        if random_order:
            np.random.shuffle(rf_pulse_duration_list)

        try:
            for pulse_index, pulse_duration in enumerate(rf_pulse_duration_list):
                self.current_pulse_duration = np.round(pulse_duration, 9)

                self.logger.info(f'RF Width: {self.current_pulse_duration} seconds -- ({pulse_index+1}/{len(rf_pulse_duration_list)})')

                data_buffer = self._run_and_acquire_step(self.current_pulse_duration)

                if post_process_function:
                    data_buffer = post_process_function(data_buffer, self)

                data.append([self.current_pulse_duration, data_buffer])

        except Exception as e:
            self.logger.error(f'{type(e).__name__}: {e}')
            raise e

        finally:
            self._stop_and_close_daq_tasks()
            # Turn off RF synthesizer
            try:
                self.rfsynth.rf_off(self.rfsynth_channel)
            except Exception as e:
                self.logger.error(f'Error turning off RF synthesizer: {type(e).__name__}')
            
            data = np.array(data, dtype=object)
            data = data[data[:,0].argsort()]
            return data
        
    def _run_and_acquire_step(self, pulse_duration):
        """
        Acquires data for a specific RF pulse duration.
        """
        try:
            # Program the pulser with the new pulse duration
            self.cycle_period, self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state(pulse_duration)
            
            # Compute timing for this pulse sequence
            self.daq_time = self.cycle_period * self.N_cycles
            self.total_clock_ticks = self.N_clock_ticks_per_cycle * self.N_cycles

            self.logger.debug(f'Acquiring {self.total_clock_ticks} samples')
            self.logger.debug(f'   acquisition time of {self.daq_time} seconds')

            # Configure DAQ for this acquisition
            self.edge_counter_config.configure_counter_period_measure(
                source_terminal=self.photon_counter_nidaq_terminal,
                N_samples_to_acquire_or_buffer_size=self.total_clock_ticks,
                clock_terminal=self.clock_nidaq_terminal,
                trigger_terminal=self.trigger_nidaq_terminal)

            self.edge_counter_config.create_counter_reader()
            
            # Prepare data buffer
            data_buffer = np.zeros(self.total_clock_ticks, dtype=np.float64)

            # Start the pulser
            self.pulser.start()
            time.sleep(0.01)  # Brief wait for pulser to start

            # Start DAQ and acquire data
            self.edge_counter_config.counter_task.wait_until_done()
            self.edge_counter_config.counter_task.start()
            
            # Use timeout that's longer than expected acquisition time
            timeout = max(10, self.daq_time * 1.2)
            time.sleep(self.daq_time * 1.1)

            samples_read = self.edge_counter_config.counter_reader.read_many_sample_double(
                data_buffer,
                number_of_samples_per_channel=self.total_clock_ticks,
                timeout=timeout)

            self.edge_counter_config.counter_task.stop()

            # Validate sample count
            if samples_read != self.total_clock_ticks:
                self.logger.warning(f"Unexpected sample count: {samples_read}/{self.total_clock_ticks}")

            return data_buffer

        except Exception as e:
            self.logger.error(f'Acquisition error: {type(e).__name__}: {e}')
            raise e

        finally:
            # Always stop the pulser and clean up DAQ
            try:
                self.pulser.stop()
            except Exception as e:
                self.logger.error(f'Error stopping pulser: {type(e).__name__}')
            
            # Clean up DAQ tasks
            self._stop_and_close_daq_tasks()