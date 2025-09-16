import logging
import time
import numpy as np

import qt3utils.experiments.common
from qt3utils.errors import PulseTrainWidthError

class RAMSEY(qt3utils.experiments.common.Experiment):
    def __init__(self, pulser, rfsynth, edge_counter_config,
                 tau_low=100e-9,
                 tau_high=10e-6,
                 tau_step=50e-9,
                 rf_frequency=2870e6,
                 rf_power=-20, **kwargs):
        
        # Call parent constructor
        super().__init__(pulser, rfsynth, edge_counter_config, **kwargs)
        
        # Set RAMSEY-specific attributes
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.tau_step = tau_step
        self.rf_frequency = rf_frequency
        self.rf_power = rf_power

    def experimental_conditions(self):
        '''
        Returns a dictionary that captures the essential experimental conditions.
        '''
        return {
            'tau_low': self.tau_low,
            'tau_high': self.tau_high,
            'tau_step': self.tau_step,
            'rf_frequency': self.rf_frequency,
            'rf_power': self.rf_power,
            'pulser': self.pulser.experimental_conditions()
        }
    
    def run(self, N_cycles=500000,
            post_process_function=qt3utils.experiments.common.measure_readout_contrast,
            random_order=False):
        """
        Performs a RAMSEY experiment scanning over tau (free evolution time) values.

        For each tau value, some number of cycles of data are acquired. A cycle
        is one full sequence of the pulse train used in the experiment. For RAMSEY,
        a cycle typically includes {initialization, π/2 pulse, free evolution (tau), π/2 pulse, readout}.

        The N_cycles specifies the total number of these cycles to acquire.

        The return from this function is a numpy array. Each element contains:
            Tau (float),
            data_post_processing_output, or raw data trace
        """
        self.N_cycles = int(N_cycles)
        if self.N_cycles <= 0:
            raise ValueError("N_cycles must be positive")

        # Validate pulse width before starting
        try:
            self.pulser.raise_for_pulse_width(self.tau_high)
        except PulseTrainWidthError as e:
            self.logger.error(f'The largest requested tau value, {self.tau_high}, is too large.')
            raise e

        # Configure RF synthesizer
        self.rfsynth.stop_sweep()
        self.rfsynth.trigger_mode('disabled')
        self.rfsynth.set_power(self.rfsynth_channel, self.rf_power)
        self.rfsynth.set_frequency(self.rfsynth_channel, self.rf_frequency)
        self.rfsynth.rf_on(self.rfsynth_channel)
        time.sleep(1)  # wait for RF box to fully turn on

        data = []
        tau_list = np.arange(
            self.tau_low, 
            self.tau_high + self.tau_step, 
            self.tau_step)

        if random_order:
            np.random.shuffle(tau_list)

        try:
            for tau_index, tau in enumerate(tau_list):
                self.current_tau = np.round(tau, 9)

                self.logger.info(f'Tau: {self.current_tau} seconds -- ({tau_index+1}/{len(tau_list)})')

                data_buffer = self._run_and_acquire_step(self.current_tau)

                if post_process_function:
                    data_buffer = post_process_function(data_buffer, self)

                data.append([self.current_tau, data_buffer])

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
        
    def _run_and_acquire_step(self, tau):
        """
        Acquires data for a specific tau (free evolution time).
        """
        try:
            # Program the pulser with the new tau value
            self.cycle_period, self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state(tau)
            
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