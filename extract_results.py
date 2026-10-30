import numpy as np
try:
    data = np.load('pure_qcnn_8qubit_final_output/test_results.npz')
    with open('current_results.txt', 'w') as f:
        f.write(f"Accuracy: {data['test_accuracy']}\n")
        f.write(f"Hinge: {data['test_hinge_accuracy']}\n")
except Exception as e:
    with open('current_results.txt', 'w') as f:
        f.write(f"Error: {str(e)}")
