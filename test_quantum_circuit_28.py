
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "quantum_data_28" / "quantum_features_28.npz"
)

N_QUBITS = 8
N_LAYERS = 2
SHOTS = 1024
SEED = 42


# ============================================================
# 2. Load one example from the quantum dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Quantum dataset not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH)

features = data["train_angles"][0]
label = int(data["y_train"][0])

if features.shape != (N_QUBITS,):
    raise ValueError(
        f"Expected 8 input angles, received {features.shape}"
    )

if not np.isfinite(features).all():
    raise ValueError("Input features contain invalid values.")

print("=" * 55)
print("QUANTUM CIRCUIT TEST")
print("=" * 55)
print(f"Input features: {features.shape}")
print(f"Example label:  {label}")
print(f"Input angles:   {np.round(features, 4)}")


# ============================================================
# 3. Define the trainable quantum circuit
# ============================================================

input_parameters = ParameterVector("x", N_QUBITS)

# Each layer has two trainable parameters per qubit:
# one RY angle and one RZ angle.
weights = ParameterVector(
    "theta",
    N_LAYERS * N_QUBITS * 2,
)

circuit = QuantumCircuit(N_QUBITS)

# Encode the eight classical PCA features.
for qubit in range(N_QUBITS):
    circuit.ry(input_parameters[qubit], qubit)

circuit.barrier()

# Trainable rotation and entanglement layers.
weight_index = 0

for layer in range(N_LAYERS):

    # Trainable single-qubit rotations.
    for qubit in range(N_QUBITS):
        circuit.ry(weights[weight_index], qubit)
        weight_index += 1

        circuit.rz(weights[weight_index], qubit)
        weight_index += 1

    # Entangle neighboring qubits in a ring.
    for qubit in range(N_QUBITS - 1):
        circuit.cx(qubit, qubit + 1)

    circuit.cx(N_QUBITS - 1, 0)

    circuit.barrier()

circuit.measure_all()


# ============================================================
# 4. Initialize example trainable parameters
# ============================================================

rng = np.random.default_rng(SEED)

initial_weights = rng.uniform(
    low=-0.1,
    high=0.1,
    size=len(weights),
)

parameter_values = {}

for parameter, value in zip(input_parameters, features):
    parameter_values[parameter] = float(value)

for parameter, value in zip(weights, initial_weights):
    parameter_values[parameter] = float(value)

bound_circuit = circuit.assign_parameters(
    parameter_values,
    inplace=False,
)

print(f"\nNumber of qubits: {N_QUBITS}")
print(f"Trainable parameters: {len(weights)}")
print(f"Circuit depth before transpilation: {circuit.depth()}")
print(f"Total circuit operations: {circuit.size()}")


# ============================================================
# 5. Run the circuit on the Aer simulator
# ============================================================

print("\nRunning circuit on Qiskit Aer...")

simulator = AerSimulator()

compiled_circuit = transpile(
    bound_circuit,
    simulator,
    seed_transpiler=SEED,
)

job = simulator.run(
    compiled_circuit,
    shots=SHOTS,
    seed_simulator=SEED,
)

result = job.result()
counts = result.get_counts()


# ============================================================
# 6. Inspect measurement results
# ============================================================

total_shots = sum(counts.values())

if total_shots != SHOTS:
    raise RuntimeError(
        f"Expected {SHOTS} shots, received {total_shots}"
    )

print("\nMeasurement results:")
for bitstring, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True,
)[:10]:
    print(f"{bitstring}: {count} shots")

print(f"\nUnique measured states: {len(counts)}")
print(f"Total measurement shots: {total_shots}")


# ============================================================
# 7. Calculate single-qubit expectation values
# ============================================================

# Qiskit displays classical bitstrings with the highest-index
# classical bit on the left. We reverse the string to associate
# each bit with its corresponding qubit index.

expectations = np.zeros(N_QUBITS, dtype=np.float64)

for bitstring, count in counts.items():
    probability = count / total_shots
    bits_by_qubit = bitstring[::-1]

    for qubit, bit in enumerate(bits_by_qubit):
        # Z expectation: +1 for |0>, -1 for |1>.
        expectations[qubit] += probability * (
            1.0 if bit == "0" else -1.0
        )

print("\nMeasured Z expectation values:")
for qubit, value in enumerate(expectations):
    print(f"Qubit {qubit}: {value:.4f}")


# ============================================================
# 8. Save circuit diagram and test results
# ============================================================

OUTPUT_DIR = ROOT / "quantum_circuit_test_28"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

diagram_path = OUTPUT_DIR / "quantum_circuit_28.txt"
results_path = OUTPUT_DIR / "circuit_test_results_28.npz"

with open(diagram_path, "w", encoding="utf-8") as file:
    file.write(str(circuit.draw(output="text")))

np.savez_compressed(
    results_path,
    input_features=features,
    example_label=np.int64(label),
    initial_weights=initial_weights,
    counts_bitstrings=np.array(list(counts.keys())),
    counts_values=np.array(list(counts.values())),
    z_expectations=expectations,
    shots=np.int64(SHOTS),
    n_qubits=np.int64(N_QUBITS),
    n_layers=np.int64(N_LAYERS),
)

print("\n" + "=" * 55)
print("QUANTUM CIRCUIT TEST PASSED")
print("=" * 55)
print(f"Circuit diagram: {diagram_path}")
print(f"Test results:    {results_path}")
print("\nThe circuit encoded one example and produced measurements.")
print("The trainable parameters have NOT been optimized yet.")
