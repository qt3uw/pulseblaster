# PulseBlaster Control

This repository contains Python control code for **PulseBlaster** pulse generation hardware.

This code creates an object **PBInd** that relies on the [SpinCore spinapi.py wrapper](http://www.spincore.com/support/SpinAPI_Python_Wrapper/spinapi.py) to **program individual pulseblaster pins independently**.

## Features
- **Independent Pin Control**: Program PulseBlaster pins **individually** rather than setting all at once.
- **NEW - Flexible Timing**:  
  - Supports **pulse increments as small as 10 ns** (previously limited to 50 ns increments).  
  - Ensures **minimum pulse width of 50 ns** to meet hardware constraints.  
- **NEW - Built-in Validation**:  
  - Ensures **pulse sequences are programmable** by verifying that changes across all channels meet the minimum required pulse duration.  
  - **Triggers exceptions** detailing conflicting channels and the times of conflicts.  
- **NEW - Strict Error Handling**:  
  - Instead of automatic rounding, an **exception is raised** if an invalid pulse duration or start time is provided.
- **Dropped Support for Offsets**:  
  - Functions handling **channel offsetting** have been removed.

## Installation

### Prerequisites

Spin Core Driver: http://www.spincore.com/support/spinapi/

### Python Install

```
> pip install pulseblaster
```

## Usage

### Examples

[pb_infinite_on_example.py](pb_infinite_on_example.py)
[pb_infinite_square_wave_example.py](pb_infinite_square_wave_example.py)
[pb_clock_and_short_square_wave_example.py](pb_clock_and_short_square_wave_example.py)

## PBInd - Additional Important Info

### Timing Considerations
PulseBlaster hardware **issues channel values in one large instruction** with a **minimum duration of 50 ns**.  
Each change in channel value **triggers a new instruction**, meaning:
- When using increments **that are not multiples of 50 ns**, changes across multiple channels must be carefully managed to ensure the following:
  - Channel values **must remain constant for at least 50 ns** after any change.
  - **Multiple channels can change simultaneously**, but individual changes **cannot overlap at intervals shorter than 50 ns**.

### Configuring PulseBlaster Specifications
Ensure the following parameters are correctly set to match your PulseBlaster model:
- **Resolution (`resolution`)**: Defines the pulse increment granularity (**default: 10 ns** for most models).
- **Minimum Pulse (`minimum_pulse`)**: Defines the shortest valid pulse/instruction duration (**default: 50 ns**).

### Updating Legacy Code
- `self.program(loops)` **no longer supports the `offset` argument**—remove any references to it.
- **Pulse durations and start times** must be **multiples of the specified resolution** and **≥ minimum_pulse (or 0)**.
- If using increments **that are not multiples of 50 ns**, ensure **channel changes do not overlap at intervals shorter than the minimum pulse width** (unless occurring simultaneously).  
  - If they do, an **exception will be triggered**, as such sequences **cannot be programmed on the PulseBlaster**.