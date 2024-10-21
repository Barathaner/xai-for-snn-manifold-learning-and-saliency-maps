from brian2 import *
import matplotlib.pyplot as plt  # Importiere matplotlib zum Plotten


# Define parameters
N = 100  # Number of neurons
tau = 10*ms  # Time constant
R = 1*ohm  # Widerstand, um den Strom in Spannung umzuwandeln

# Input current (in Ampere)
I_values = TimedArray([0, 1, 0, 0, 1]*nA, dt=1*ms)

# Create neuron group with the current converted to voltage using the resistance R
neurons = NeuronGroup(N, '''
    dv/dt = (R*I_values(t) - v) / tau : volt
    ''', threshold='v > 1*mV', reset='v = 0*mV')

# Monitor the output
monitor = StateMonitor(neurons, 'v', record=True)

# Run the simulation
run(100*ms)

# Plot the results
plot(monitor.t/ms, monitor.v[0], label='Neuron 0')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
legend()
show()
