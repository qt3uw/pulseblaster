import logging
import time
import numpy as np

import experiments.common

class PODMR(experiments.common.Experiment):
    def __init__(self, pulser, rfsynth, edge_counter_config,
                 freq_low=2820e6, freq_high=2920e6,
                 freq_step=1e6, rf_power=-20, **kwargs):
        
        # Call parent constructor
        super().__init__(pulser, rfsynth, edge_counter_config, **kwargs)
        
        # Set PODMR-specific attributes
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.freq_step = freq_step
        self.rf_power = rf_power

    def experimental_conditions(self):
        '''
        Returns a dictionary that captures the essential experimental conditions.
        '''
        return {
            'freq_low':self.freq_low,
            'freq_high':self.freq_high,
            'freq_step':self.freq_step,
            'rf_power':self.rf_power,
            'pulser':self.pulser.experimental_conditions()
        }
    
    def run(self, N_cycles = 500000,
                  post_process_function = experiments.common.measure_readout_contrast,
                  random_order = False):
        """
        Performs the PulsedODMR scan over the specificed range of frequencies.

        For each frequency, some number of cycles of data are acquired. A cycle
        is one full sequence of the pulse train used in the experiment. For PulsedODMR,
        a cycle is {AOM on, AOM off/RF on, AOM on, AOM off/RF off}.

        The N_cycles specifies the total number of these cycles to
        acquire. Your choice depends on your desired resolution or signal-to-noise
        ratio, your post-data acquisition processing choices, and the amount of memory
        available on your computer.

        For each frequency, the number of data read from the NI DAQ will be
        N_cycles * N_clock_ticks_per_cycle (usually 4 when using arbitrary clock, 2 for signal, 2 for background)..

        These data are found in a data_buffer within this method. They
        may be analyzed with a function passed to the argument `post_process_function`,
        which is useful to reduce the required memory to hold the raw data.

        After data acquisition for each frequency in the scan,
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
        self.N_cycles = int(N_cycles)
        if self.N_cycles <= 0:
            raise ValueError("N_cycles must be positive")

        self.rfsynth.stop_sweep()
        self.rfsynth.trigger_mode('disabled')
        self.rfsynth.set_power(self.rfsynth_channel, self.rf_power)
        self.rfsynth.rf_on(self.rfsynth_channel)
        time.sleep(1) #wait for RF box to fully turn on

        self.cycle_period, self.N_clock_ticks_per_cycle = self.pulser.program_pulser_state()
        self.pulser.start() #start the pulser

        # compute the total number of samples to be acquired and the DAQ time
        # these will be the same for each RF frequency through the scan
        self.daq_time = self.cycle_period * self.N_cycles
        self.total_clock_ticks = self.N_clock_ticks_per_cycle * self.N_cycles

        self.edge_counter_config.configure_counter_period_measure(
            source_terminal = self.photon_counter_nidaq_terminal,
            N_samples_to_acquire_or_buffer_size = self.total_clock_ticks,
            clock_terminal = self.clock_nidaq_terminal,
            trigger_terminal = self.trigger_nidaq_terminal)

        self.edge_counter_config.create_counter_reader()

        data = []
        rf_frequency_list = np.arange(self.freq_low, self.freq_high + self.freq_step, self.freq_step)
        if random_order:
            np.random.shuffle(rf_frequency_list)
        try:
            for rf_freq in rf_frequency_list:

                self.current_rf_freq = np.round(rf_freq, 9)
                self.rfsynth.set_frequency(self.rfsynth_channel, self.current_rf_freq)

                self.logger.info(f'RF frequency: {self.current_rf_freq*1e-9} GHz')
                self.logger.debug(f'Acquiring {self.total_clock_ticks} samples')
                self.logger.debug(f'   acquisition time of {self.daq_time} seconds')

                data_buffer = np.zeros(self.total_clock_ticks)

                self.edge_counter_config.counter_task.wait_until_done()
                self.edge_counter_config.counter_task.start()
                time.sleep(self.daq_time*1.1)

                read_samples = self.edge_counter_config.counter_reader.read_many_sample_double(
                                        data_buffer,
                                        number_of_samples_per_channel=self.total_clock_ticks,
                                        timeout=5)

                self.edge_counter_config.counter_task.stop()
                if post_process_function:
                    data_buffer = post_process_function(data_buffer, self)

                data.append([self.current_rf_freq,
                             data_buffer])
        except Exception as e:
            self.logger.error(f'{type(e)}: {e}')
            raise e

        finally:
            self._stop_and_close_daq_tasks()
            # rfsynth.rf_off(self.rfsynth_channel) # uncomment if needed
            data = np.array(data, dtype=object)
            data = data[data[:,0].argsort()]
            return data