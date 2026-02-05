# Real-Time Intrusion Detection Using XGBoost and CICFlowMeter

## Overview

This project implements a **real-time Network Intrusion Detection System (NIDS)** using machine learning. The system captures live network traffic, extracts flow-based features using CICFlowMeter, and classifies each flow with a trained **XGBoost** model to detect malicious activity.

The final system is designed to operate in near real time and focuses on **reproducible, high-volume network attacks**, particularly UDP-based denial-of-service traffic.

---

## Abstract

This project presents a real-time Network Intrusion Detection System leveraging an XGBoost classifier trained on labeled network flow data. Network flows are extracted live using CICFlowMeter, aligned to the training schema, scaled, feature-selected, and classified using a persisted model pipeline.

While early experiments used only CICIDS2017, the final system combines **benign traffic from CICIDS2017** with **attack traffic from CICDDoS2019**, creating a more realistic and targeted threat model. Live testing and simulated attacks demonstrate strong detection performance, particularly for volumetric UDP floods.

---

## Motivation

Traditional signature-based IDS systems struggle with:
- Zero-day attacks
- Obfuscated payloads
- High-volume, low-variance traffic

Machine-learning–based IDS approaches address these limitations by learning statistical traffic patterns rather than relying on known signatures. This project explores whether a **flow-based ML pipeline** can operate reliably in real time while maintaining high accuracy.

---

## Background

### XGBoost
XGBoost is a gradient boosting framework widely used for structured data classification due to:
- Built-in regularization
- Robust handling of missing values
- High performance and scalability

### Datasets
- **CICIDS2017**: Used primarily for benign traffic, providing realistic background network behavior.
- **CICDDoS2019**: Used for attack traffic, focusing on high-volume and reproducible DDoS patterns such as:
  - DrDoS_UDP
  - DrDoS_NTP
  - SYN Floods

### CICFlowMeter
CICFlowMeter extracts over 80 statistical features from packet captures, including:
- Packet counts
- Byte rates
- Inter-arrival times
- Flow duration metrics

These features form the basis of both offline training and live inference.

---

## System Architecture

The system follows a **streaming ML pipeline**:

1. **Packet Capture**
   - Live traffic captured using `tcpdump`

2. **Flow Extraction**
   - CICFlowMeter converts packet streams into flow records

3. **Feature Alignment**
   - Live features are renamed and reindexed to match the training schema

4. **Preprocessing**
   - StandardScaler applied (fit on raw training data)
   - SelectKBest feature selector applied

5. **Inference**
   - XGBoost classifier predicts attack class probabilities

6. **Output**
   - Predicted labels decoded
   - Class frequency logging for live monitoring

---

## Dataset Construction

### Benign Traffic
- Extracted from CICIDS2017
- Represents normal enterprise network behavior

### Attack Traffic
- Extracted from CICDDoS2019
- Focused on attacks that can be realistically reproduced in a lab environment

### Dataset Characteristics
- Over **1.9 million rows** merged
- Balanced class distribution
- Multi-class labels preserved (not binary-only)

---

## Preprocessing Pipeline

- Dropped non-numeric columns
- Removed constant and zero-variance features
- Encoded attack categories using `LabelEncoder`
- Fitted `StandardScaler` on **raw, unscaled training data**
- Persisted both scaler and feature selector for live inference
- Selected top **50 most informative features** using SelectKBest

Maintaining **identical preprocessing steps** between training and inference proved critical to system correctness.

---

## Model Training

- Algorithm: XGBoost
- Objective: `multi:softprob`
- Train/Test Split: 80/20
- Feature Selection: SelectKBest (k = 50)
- Evaluation:
  - Training and validation accuracy
  - Training and validation loss
  - Early stopping behavior analysis

Evaluation curves were generated using XGBoost’s `evals_result` logs.

---

## Results

### Offline Performance
- Test-set accuracy exceeded **99.9%**
- Stable validation loss
- Clean class separation

### Live Testing Challenges
Several issues emerged during real-time deployment:

#### Feature Misalignment
- Live flows used different column names (e.g., `flow_byts_s` vs `Flow Bytes/s`)
- Required explicit renaming and reindexing

#### Missing Values
- Reindexing introduced NaNs
- Resolved by dropping rows with missing values after alignment

#### Scaler Drift
- Early live tests incorrectly used a scaler fit on already-normalized data
- Re-fitting and persisting the scaler on raw features significantly improved results

---

## Live Detection Behavior

### Corrected Pipeline Results
- Benign traffic correctly classified
- High-volume UDP floods detected reliably

### Verified Attack Detection
- `iperf -u -b 100M` consistently triggered **DrDoS_UDP** detections
- Confirmed end-to-end pipeline correctness

### Attack Simulation Observations
- `hping3` and `nping` often failed to generate sustained flow patterns
- These tools produced traffic that was either:
  - Too short-lived
  - Too low-volume
  - Aggregated as benign flows

This highlighted the importance of **flow-level realism**, not just packet generation.

---

## Performance and Optimization

### Bottlenecks
- CICFlowMeter performance degraded under heavy packet floods

### Mitigations
- Reduced capture window duration
- Minimized per-flow processing overhead

### Future Improvements
- Asynchronous flow processing
- Batch inference
- Flow window aggregation
- Containerized deployment for scalable edge monitoring

---

## Visualization and Analysis

- Line charts for training/validation accuracy and loss
- Pie charts verifying balanced class distributions
- Frequency logs for live classification behavior

These visualizations were used to validate both offline training stability and live inference behavior.

---

## Conclusion

This project demonstrates a functional **real-time ML-based NIDS** capable of detecting high-volume network attacks using flow-based features. Transitioning to a hybrid dataset—benign traffic from CICIDS2017 and attack traffic from CICDDoS2019—significantly improved real-world relevance.

Accurate live detection depended on:
- Precise feature alignment
- Correct scaler usage
- Consistent preprocessing across training and inference

With these components properly synchronized, the system successfully detected simulated UDP flood attacks in real time, validating the feasibility of flow-based ML intrusion detection.

---

## Future Work

- Expand detection to additional attack classes
- Introduce time-windowed detection logic
- Improve performance under burst traffic
- Deploy as a containerized edge-monitoring service

---

## License

Academic project. Not licensed for production or commercial deployment.
