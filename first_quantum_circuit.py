
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create a 2-qubit circuit
qc = QuantumCircuit(2)

# Create superposition
qc.h(0)

# Entangle the two qubits
qc.cx(0, 1)

# Measure both qubits
qc.measure_all()

# Simulate locally on your CPU
simulator = AerSimulator()
compiled = transpile(qc, simulator)

result = simulator.run(
    compiled,
    shots=1024
).result()

print("Measurement results:")
print(result.get_counts())

print("\nCircuit diagram:")
print(qc.draw())
