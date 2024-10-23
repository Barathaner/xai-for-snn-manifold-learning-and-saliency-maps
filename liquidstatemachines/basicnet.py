from brian2 import *

taum = 20*ms
taue = 5*ms
taui = 10*ms
Vt = -50*mV
Vr = -60*mV
El = -49*mV

# Modell der Neuronen
eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

# Externe Eingangsgruppe für variierenden Input
input_group = NeuronGroup(1, 'rates : Hz', threshold='rand() < rates*dt', method='exact')

# Neuronengruppe für das Hauptnetzwerk
P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0*mV
P.gi = 0*mV

# Ausgabeneuronen für Klassifikation (2 Neuronen: Klasse 0 und Klasse 1)
output_group = NeuronGroup(2, 'v : volt', threshold='v > Vt', reset='v = Vr', method='euler')

# Synapsenverbindungen im Netzwerk
we = (60*0.27/10)*mV  # Erregende synaptische Gewichte
wi = (-20*4.5/10)*mV  # Hemmende synaptische Gewichte
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.02)
Ci.connect('i>=3200', p=0.02)

# Verbindungen von Input zu den Neuronen
Cext = Synapses(input_group, P[:3200], on_pre='ge += we')
Cext.connect(p=0.1)

# Verbindungen zu den Ausgabeneuronen
C = Synapses(P, output_group, on_pre='v += 1*mV')
C.connect(p=0.1)

# STDP-Regel für synaptisches Lernen (synaptische Gewichte anpassen)
stdp = Synapses(P, output_group,
                model='''w : volt
                         dpre/dt = -pre/taupre : 1 (event-driven)
                         dpost/dt = -post/taupost : 1 (event-driven)''',
                on_pre='''v_post += w
                          pre = 1
                          w = clip(w + post * eta, 0*mV, wmax)''',  # 0 in Millivolt geändert
                on_post='''post = 1
                           w = clip(w + pre * eta, 0*mV, wmax)''')  # 0 in Millivolt geändert

# Parameter für STDP
taupre = 20*ms
taupost = 20*ms
eta = 0.05*mV  # Lernrate verringert
wmax = 5*mV  # Maximales Gewicht reduziert

# Verbinde STDP mit den bestehenden Verbindungen
stdp.connect(p=0.1)  # Verbinde 10% der Neuronen zufällig
stdp.w = 'rand() * wmax'  # Zufällige Anfangsgewichte

# Monitore zur Überwachung der Aktivitäten
spike_monitor_output = SpikeMonitor(output_group)
s_mon = SpikeMonitor(P)
v_mon_output = StateMonitor(output_group, 'v', record=True)  # Überwache Membranpotenziale der Output-Neuronen

v_mon = StateMonitor(P, 'v', record=[0, 100])  # Neuron 0 und 100 überwachen
ge_mon = StateMonitor(P, 'ge', record=[0, 100])  # Erregender Strom überwachen
gi_mon = StateMonitor(P, 'gi', record=[0, 100])  # Hemmender Strom überwachen
w_mon = StateMonitor(stdp, 'w', record=True)  # Überwachung der synaptischen Gewichte

# Schleife zur Simulation mit wechselndem Input
for cycle in range(20):  # Mehr Durchläufe (20 anstelle von 5)
    # Klasse 1 (Hohe Frequenz)
    input_group.rates = 50*Hz  # Hohe Frequenz für Klasse 1
    run(1000 * ms)  # Simuliere für 1000 ms (längere Simulationszeit)
    print(f"Durchlauf {cycle+1}, Input: Hohe Frequenz")

    # Ruhephase zwischen den Frequenzwechseln (Nullaktivität)
    input_group.rates = 0*Hz  # Setze den Input auf 0 Hz für Ruhe
    run(500 * ms)  # Ruhephase von 500 ms

    # Klasse 0 (Niedrige Frequenz)
    input_group.rates = 5*Hz  # Niedrige Frequenz für Klasse 0
    run(1000 * ms)  # Simuliere für 1000 ms (längere Simulationszeit)
    print(f"Durchlauf {cycle+1}, Input: Niedrige Frequenz")

    # Ruhephase zwischen den Frequenzwechseln (Nullaktivität)
    input_group.rates = 0*Hz  # Setze den Input auf 0 Hz für Ruhe
    run(500 * ms)  # Ruhephase von 500 ms

    # Auswertung: Welches Ausgabeneuron hat öfter gefeuert?
    if spike_monitor_output.count[0] > spike_monitor_output.count[1]:
        print("Klasse 0 erkannt")
    else:
        print("Klasse 1 erkannt")

# Plot der Ergebnisse
plot(s_mon.t/ms, s_mon.i, ',k')
xlabel('Time (ms)')
ylabel('Neuron index')
show()

plot(v_mon.t/ms, v_mon.v[0]/mV)
xlabel('Time (ms)')
ylabel('Membrane potential (mV)')
show()

plot(ge_mon.t/ms, ge_mon.ge[0]/mV, label='ge (Neuron 0)')
plot(gi_mon.t/ms, gi_mon.gi[0]/mV, label='gi (Neuron 0)')
xlabel('Time (ms)')
ylabel('Synaptic current (mV)')
legend()
show()

# Plot der synaptischen Gewichte über die Zeit
plot(w_mon.t/ms, w_mon.w[0]/mV, label='Synapse 0')
xlabel('Time (ms)')
ylabel('Synaptic weight (mV)')
legend()
show()

# Plot der Membranpotenziale der Output-Neuronen
plot(v_mon_output.t/ms, v_mon_output.v[0]/mV, label='Output Neuron 0')
plot(v_mon_output.t/ms, v_mon_output.v[1]/mV, label='Output Neuron 1')
xlabel('Time (ms)')
ylabel('Membrane potential (mV)')
legend()
show()
