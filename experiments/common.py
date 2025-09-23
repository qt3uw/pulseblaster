import logging
import time
import numpy as np

def aggregate_sum(data_buffer, experiment):
    return data_buffer.reshape(int(experiment.N_cycles), int(experiment.N_clock_ticks_per_cycle)).sum(axis=0)

def measure_total_contrast(data_buffer, experiment):
    trace = aggregate_sum(data_buffer, experiment)
    background = trace[:len(trace)//2]
    signal = trace[len(trace)//2:]
    return np.sum(signal)/np.sum(background)

def measure_readout_contrast(data_buffer, experiment):
    trace = aggregate_sum(data_buffer, experiment)
    background = trace[0]
    signal = trace[2]
    return np.sum(signal)/np.sum(background)

class Experiment:
    def __init__(self, pulser, rfsynth, edge_counter_config,
                       photon_counter_nidaq_terminal='PFI0',
                       clock_nidaq_terminal='PFI12',
                       trigger_nidaq_terminal='PFI1',
                       rfsynth_channel=0,
                       name=None, 
                       log_level=logging.INFO):
        """
        Base class for ODMR, RABI, RAMSEY, HAHN ECHO, CPMG experiments.

        Hardware Settings
            pulser - a qt3utils.pulsers.interface.ODMRPulser object (such as qt3utils.pulsers.qcsapphire.QCSapphPulsedODMRPulser)
            rfsynth - a qt3rfsynthcontrol.Pulser object
            The rfsynth_channel specifies which output channel from the Windfreak RF SynthHD is used to provde the RF signal (either 0 or 1)
            edge_counter_config - a qt3utils.nidaq.config.EdgeCounter object

            
            NI DAQ Connections
            * photon_counter_nidaq_terminal - terminal connected to TTL pulses that indicate a photon
            * clock_nidaq_terminal - terminal connected to the clock_pulser_channel
            * trigger_nidaq_terminal - terminal connected to the trigger_pulser_channel
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"experiments.{self.name}")
        self.logger.setLevel(log_level)

        self.pulser = pulser

        self.rfsynth = rfsynth
        self.rfsynth_channel = rfsynth_channel

        self.photon_counter_nidaq_terminal = photon_counter_nidaq_terminal
        self.clock_nidaq_terminal = clock_nidaq_terminal
        self.trigger_nidaq_terminal  = trigger_nidaq_terminal

        self.edge_counter_config = edge_counter_config

    def run(self, N_cycles, post_process_function, *args, **kwargs):
        raise NotImplementedError()

    def experimental_conditions(self):
        raise NotImplementedError()
    
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
                self.logger.debug("Counter task stopped successfully")
        except Exception as e:
            # Only log non-trivial errors with trimmed message
            if not "invalid or does not exist" in str(e):
                error_msg = f"{type(e).__name__}" if trim_errors else f"{type(e).__name__}: {str(e)}"
                self.logger.error(f'Error stopping counter task: {error_msg}')
        
        # Wait a moment for the device to register the stop command
        time.sleep(0.05)
            
        # Then try to close the task
        try:
            if not getattr(self.edge_counter_config.counter_task, '_handle', None) is None:
                self.edge_counter_config.counter_task.close()
                self.logger.debug("Counter task closed successfully")
        except Exception as e:
            if not "invalid or does not exist" in str(e):
                error_msg = f"{type(e).__name__}" if trim_errors else f"{type(e).__name__}: {str(e)}"
                self.logger.error(f'Error closing counter task: {error_msg}')
        finally:
            # Always set to None after trying to close, even if it failed
            self.edge_counter_config.counter_task = None
            self.edge_counter_config.counter_reader = None
            
        # Give the hardware a moment to fully release resources
        time.sleep(0.1)