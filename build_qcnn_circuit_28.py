
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "quantum_data_28" / "quantum_features_28.npz"
OUTPUT_DIR = ROOT / "qcnn_circuit_test_28"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD ONE PREPARED DIGIT
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run prepare_quantum_data_28.py first."
    )

data = np.load(DATA_PATH)

angles = np.asarray(data["train_angles"][0], dtype=float)
label = int(data["y_train"][0])

if angles.shape != (8,):
    raise ValueError(
        f"Expected 8 input angles, got shape {angles.shape}"
    )

print("=" * 60)
print("QCNN CIRCUIT BUILD AND TEST")
print("=" * 60)
print("Input feature shape:", angles.shape)
print("Example digit label:", label)


# ============================================================
# 3. CREATE QCNN PARAMETERS
#
# Convolution stage 1: 4 pairs x 6 parameters = 24
# Pooling stage 1:     4 parameters
# Convolution stage 2: 2 pairs x 6 parameters = 12
# Pooling stage 2:     2 parameters
# Convolution stage 3: 1 pair  x 6 parameters = 6
# Pooling stage 3:     1 parameter
#
# Total = 49 trainable parameters
# ============================================================

theta = ParameterVector("theta", 49)
parameter_index = 0


# ============================================================
# 4. BUILD THE CIRCUIT
# ============================================================

# Eight quantum bits and eight classical bits.
qc = QuantumCircuit(8, 8)

# Encode the 8 PCA features using RY rotations.
for qubit in range(8):
    qc.ry(float(angles[qubit]), qubit)

qc.barrier()


# ------------------------------------------------------------
# CONVOLUTION BLOCK
# Each pair gets trainable rotations and entanglement.
# ------------------------------------------------------------

def convolution_block(circuit, q1, q2):
    global parameter_index

    circuit.ry(theta[parameter_index], q1)
    parameter_index += 1

    circuit.rz(theta[parameter_index], q1)
    parameter_index += 1

    circuit.ry(theta[parameter_index], q2)
    parameter_index += 1

    circuit.rz(theta[parameter_index], q2)
    parameter_index += 1

    circuit.cx(q1, q2)

    circuit.ry(theta[parameter_index], q2)
    parameter_index += 1

    circuit.rz(theta[parameter_index], q2)
    parameter_index += 1


# ------------------------------------------------------------
# POOLING BLOCK
# Apply a controlled rotation, then measure the qubit
# that is discarded from the next stage.
# ------------------------------------------------------------

def pooling_block(circuit, survivor, discarded, classical_bit):
    global parameter_index

    circuit.cry(
        theta[parameter_index],
        discarded,
        survivor
    )
    parameter_index += 1

    # Measurement implements the pooling/discard step.
    # The discarded qubit is not used in later stages.
    circuit.measure(discarded, classical_bit)


# ============================================================
# 5. QCNN STAGE 1: 8 QUBITS -> 4 ACTIVE QUBITS
# ============================================================

for q1, q2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
    convolution_block(qc, q1, q2)

qc.barrier()

# Keep qubits 0, 2, 4, 6.
# Measure and discard qubits 1, 3, 5, 7.
for survivor, discarded, classical_bit in [
    (0, 1, 0),
    (2, 3, 1),
    (4, 5, 2),
    (6, 7, 3),
]:
    pooling_block(qc, survivor, discarded, classical_bit)

qc.barrier()


# ============================================================
# 6. QCNN STAGE 2: 4 ACTIVE QUBITS -> 2 ACTIVE QUBITS
# ============================================================

for q1, q2 in [(0, 2), (4, 6)]:
    convolution_block(qc, q1, q2)

qc.barrier()

# Keep qubits 0 and 4.
# Measure and discard qubits 2 and 6.
pooling_block(qc, survivor=0, discarded=2, classical_bit=4)
pooling_block(qc, survivor=4, discarded=6, classical_bit=5)

qc.barrier()


# ============================================================
# 7. QCNN STAGE 3: 2 ACTIVE QUBITS -> 1 ACTIVE QUBIT
# ============================================================

convolution_block(qc, 0, 4)

qc.barrier()

# Keep qubit 0 and discard qubit 4.
pooling_block(qc, survivor=0, discarded=4, classical_bit=6)

qc.barrier()


# ============================================================
# 8. FINAL OUTPUT MEASUREMENT
# ============================================================

qc.measure(0, 7)

assert parameter_index == len(theta), (
    f"Used {parameter_index} parameters, "
    f"but created {len(theta)}."
)

print("\nCircuit construction successful.")
print("Number of qubits:", qc.num_qubits)
print("Trainable parameters:", len(theta))
print("Circuit depth:", qc.depth())
print("Circuit operations:", qc.count_ops())


# ============================================================
# 9. SAVE THE PARAMETERIZED CIRCUIT DIAGRAM
# ============================================================

diagram_path = OUTPUT_DIR / "qcnn_circuit_28.txt"

diagram_path.write_text(
    str(qc.draw(output="text", fold=120)),
    encoding="utf-8"
)

print("\nCircuit diagram saved to:")
print(diagram_path)


# ============================================================
# 10. TEST WITH RANDOM PARAMETERS ON AER SIMULATOR
# ============================================================

rng = np.random.default_rng(42)

random_values = rng.uniform(
    low=-0.1,
    high=0.1,
    size=len(theta)
)

parameter_bindings = {
    theta[i]: float(random_values[i])
    for i in range(len(theta))
}

bound_circuit = qc.assign_parameters(
    parameter_bindings,
    inplace=False
)

simulator = AerSimulator()

compiled_circuit = transpile(
    bound_circuit,
    simulator,
    seed_transpiler=42
)

job = simulator.run(
    compiled_circuit,
    shots=1024,
    seed_simulator=42
)

result = job.result()
counts = result.get_counts()

print("\n" + "=" * 60)
print("QCNN CIRCUIT SIMULATION RESULTS")
print("=" * 60)

print("Total shots:", sum(counts.values()))
print("Distinct measured bitstrings:", len(counts))

print("\nMost frequent measurement results:")
for bitstring, count in sorted(
    counts.items(),
    key=lambda item: item[1],
    reverse=True
)[:10]:
    print(f"{bitstring}: {count} shots")


# Save results for later inspection.
np.savez(
    OUTPUT_DIR / "qcnn_circuit_results_28.npz",
    counts_bitstrings=np.array(list(counts.keys())),
    counts_values=np.array(list(counts.values())),
    input_angles=angles,
    example_label=label,
    random_parameters=random_values,
)

print("\n" + "=" * 60)
print("QCNN CIRCUIT TEST COMPLETED")
print("=" * 60)
print("Circuit diagram:", diagram_path)
print("Results folder:", OUTPUT_DIR)
print("Note: parameters are random and have NOT been trained.")
