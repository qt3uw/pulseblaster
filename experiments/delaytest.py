import logging
import time
import numpy as np

import experiments.common

logger = logging.getLogger(__name__)

class DELAYTEST(experiments.common.Experiment):
    def __init__(self, pulser, rfsynth, edge_counter_config,
                 integration_delay_low=100e-9,
                 integration_delay_high=500e-6,
                 integration_delay_step=10e-9,
                 **kwargs):
        
        # Call parent constructor
        super().__init__(pulser, rfsynth, edge_counter_config, **kwargs)
        
        # Set specific attributes
        self.integration_delay_low = integration_delay_low
        self.integration_delay_high = integration_delay_high
        self.integration_delay_step = integration_delay_step


    def experimental_conditions(self):
        '''
        Returns a dictionary that captures the essential experimental conditions.
        '''
        return {
            'integration_delay_low': self.integration_delay_low,
            'integration_delay_high': self.integration_delay_high,
            'integration_delay_step': self.integration_delay_step,
            'pulser': self.pulser.experimental_conditions()
        }
    
    def run(self, N_cycles=500000,
            post_process_function=experiments.common.aggregate_sum,
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

        data = []
        integration_delay_list = np.arange(
            self.integration_delay_low, 
            self.integration_delay_high + self.integration_delay_step, 
            self.integration_delay_step)

        if random_order:
            np.random.shuffle(integration_delay_list)

        try:
            for delay_index, integration_delay in enumerate(integration_delay_list):
                self.current_integration_delay = np.round(integration_delay, 8)

                self.logger.info(f'RF Width: {self.current_integration_delay} seconds -- ({delay_index+1}/{len(integration_delay_list)})')

                data_buffer = self._run_and_acquire_step(self.current_integration_delay)

                if post_process_function:
                    data_buffer = post_process_function(data_buffer, self)
                data.append([self.current_integration_delay,
                              data_buffer])

        except Exception as e:
            self.logger.error(f'{type(e).__name__}: {e}')
            raise e

        finally:
            self._stop_and_close_daq_tasks()
            # rfsynth.rf_off(self.rfsynth_channel) # uncomment if needed
            data = np.array(data, dtype=object)
            data = data[data[:,0].argsort()]
            return data
        
    def _run_and_acquire_step(self, integration_delay):
        """
        Acquires data for a specific RF pulse duration.
        """
        try:
            # Program the pulser with the new pulse duration
            self.cycle_period, self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state(integration_delay)
            # Compute timing for this pulse sequence
            self.daq_time = self.cycle_period * self.N_cycles
            self.total_clock_ticks = int(self.N_clock_ticks_per_cycle * self.N_cycles)

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
            data_buffer = np.zeros(self.total_clock_ticks)

            # Start the pulser
            self.pulser.start()
            time.sleep(0.01)  # Brief wait for pulser to start

            # Start DAQ and acquire data
            self.edge_counter_config.counter_task.wait_until_done()
            self.edge_counter_config.counter_task.start()
            time.sleep(self.daq_time * 1.1)
            timeout = max(5, self.daq_time * 0.1)

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
            self.logger.error(f'{type(e)}: {e}')
            raise e

        finally:
            # Always stop the pulser and clean up DAQ
            try:
                self.pulser.stop()
            except Exception as e:
                self.logger.error(f'Error stopping pulser: {type(e).__name__}')
            
            # Clean up DAQ tasks
            self._stop_and_close_daq_tasks()