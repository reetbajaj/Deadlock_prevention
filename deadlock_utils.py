import numpy as np
from collections import deque

def generate_snapshot(num_processes, num_resources):
    allocated = np.random.randint(0, 5, size=(num_processes, num_resources))
    requested = np.random.randint(0, 5, size=(num_processes, num_resources))
    priorities = np.random.randint(1, 4, size=(num_processes,))
    total_units = np.random.randint(5, 15, size=(num_resources,))
    available = total_units - allocated.sum(axis=0)
    available = np.clip(available, 0, None)

    blocked = 0
    high_priority_blocked = 0
    dependency_depth = 0
    for i in range(num_processes):
        waiting_for = sum(requested[i][j] > available[j] for j in range(num_resources))
        if waiting_for > 0:
            dependency_depth += waiting_for
            if priorities[i] == 1:
                blocked += 1
                high_priority_blocked += 1

    deadlock_risk = 1 if blocked > 2 else 0
    utilization = (allocated / np.maximum(total_units, 1)).flatten().tolist()

    features = (
        allocated.flatten().tolist()
        + requested.flatten().tolist()
        + available.tolist()
        + priorities.tolist()
        + utilization
        + [dependency_depth]
    )

    return features, allocated, dependency_depth, deadlock_risk, high_priority_blocked

# Generate sequences for LSTM
def generate_sequence_dataset(n_samples, seq_len, num_processes, num_resources):
    X_seq, y_seq = [], []
    buffer = deque(maxlen=seq_len+1)
    for _ in range(n_samples + seq_len):
        features, _, _, risk, _ = generate_snapshot(num_processes, num_resources)
        # Normalize features
        features = np.array(features) / 20.0  # scale to ~0-1
        buffer.append(features)
        if len(buffer) == seq_len + 1:
            X_seq.append(list(buffer)[:-1])
            y_seq.append(list(buffer)[-1][-1])  # last feature: dependency depth / risk
    return np.array(X_seq), np.array(y_seq)
