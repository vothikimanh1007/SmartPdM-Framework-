# A Data-Driven Framework for Predictive Maintenance: From Knowledge Topology to WDCNN-BiLSTM-Attention Architecture

**Lead Researcher & Author:** Vo Thi Kim Anh (ORCID: 0009-0001-5691-7665)

**Affiliations:** 1. Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam

2\. Faculty of Electrical Engineering and Computer Science, VSB - Technical University of Ostrava, Ostrava, Czech Republic

## 📖 Abstract

Predictive Maintenance (PdM) is a critical component of Industry 4.0. However, bridging the gap between theoretical consensus and practical Deep Learning architecture design remains a challenge. This paper proposes a novel "Model-Centric" pipeline.

First, we conduct a bibliometric analysis and keyword topology mapping using Web of Science (WoS) data to identify research hotspots. Second, Structural Equation Modeling (SEM) is applied to quantitatively validate the necessity of hybrid AI architectures. Driven by the SEM findings, we propose a Wide-Kernel Convolutional Neural Network combined with Bidirectional LSTM and Dual Attention Mechanism (**WDCNN-BiLSTM-DualAttn**). Finally, a Dataset Discovery approach is employed to benchmark the proposed framework.

Experimental results demonstrate that our framework achieves a state-of-the-art accuracy of **97.50%** on the CWRU dataset and robustly generalizes with **89.22%** on the highly noisy Paderborn dataset, utilizing Local Z-Score Normalization.

## 🎯 Core Contributions

This research moves beyond standard "trial-and-error" deep learning by establishing a mathematically and theoretically grounded pipeline:

- **Knowledge Topology:** Extraction of research topological networks from Web of Science (WoS) metadata to objectively identify the convergence of Deep Learning and Fault Diagnosis.
- **SEM Validation:** Mathematical justification (Path Coefficient ![](data:image/png;base64,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)) proving the necessity of integrating spatial (CNN), temporal (LSTM), and focus (Attention) layers for diagnostic robustness.
- **Advanced Neural Architecture:** Proposal of the WDCNN-BiLSTM-DualAttn architecture, utilizing a Wide-Kernel (64) CNN as an initial morphological low-pass filter to suppress mechanical background noise.
- **Generalization Benchmark:** Implementation of a "Dataset Discovery" mechanism testing the framework on both Low-Noise (CWRU) and High-Noise (Paderborn) datasets using Local Z-Score Normalization to ensure load/speed invariance.

## 🧠 Framework Architecture: WDCNN-BiLSTM-DualAttn

The proposed model accepts a ![](data:image/png;base64,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) raw vibration acceleration window and processes it through four distinct blocks:

- **Block 1: Wide-Kernel CNN (WDCNN):** Uses a large kernel (size=64, stride=8) to capture long-period structural anomalies and reject high-frequency noise.
- **Block 2: Deep Feature Extraction:** Standard convolutional layers (size=16) to learn fine-grained mechanical features.
- **Block 3: Temporal Recurrency:** A Bidirectional LSTM layer to capture the time-series dependencies and periodic impact signatures of rolling element bearings.
- **Block 4: Dual Attention Mechanism:** A novel combination of GlobalMaxPooling1D (to catch sharp fracture peaks) and GlobalAveragePooling1D (to retain underlying signal energy trends), weighted by a custom attention matrix.

## 📊 Experimental Results

The framework was benchmarked automatically via the pipeline.py Dataset Discovery module.

| **Industrial Context**      | **Dataset Used**     | **Sampling Rate** | **Noise Profile** | **Test Accuracy** |
| --------------------------- | -------------------- | ----------------- | ----------------- | ----------------- |
| **Context A (Lab Setting)** | CWRU Bearing Data    | 12 kHz            | Clean / Low       | **97.50%**        |
| **Context B (Shop Floor)**  | Paderborn University | 64 kHz            | Heavy / High      | **89.22%**        |

_Note: The high accuracy on the Paderborn dataset without specific fine-tuning demonstrates the exceptional noise-resilience of the Wide-Kernel and Dual Attention combination._

## 💻 Repository Structure & Reproducibility

This repository contains the complete, replicable codebase for the CSoNet 2026 submission.

SmartPdM-Framework/  
├── src/  
│ ├── pipeline.py # Full Python research pipeline (Generates Fig 1-12 & Models)  
│ └── index.html # Industrial Web Demo (TF.js Frontend)  
├── models/  
│ ├── context*a_cwru/ # Exported TF.js weights (97.50% Acc)  
│ └── context_b_paderborn/ # Exported TF.js weights (89.22% Acc)  
├── Research_Project_AI_PDU/ # Output directory for generated artifacts  
│ ├── Table_1_SEM_Results.csv # SEM Structural Path validation  
│ ├── Fig*\*.png # All generated manuscript figures  
│ └── Model\_\*.keras # Raw trained Keras models  
└── README.md

### How to Run the Pipeline

The entire research methodology-from generating bibliometric charts and SEM diagrams to training the dual-context models-is consolidated into a single script.

\# 1. Install required scientific libraries  
pip install kagglehub wordcloud networkx seaborn scikit-learn tensorflow scipy semopy graphviz  
<br/>\# 2. Execute the comprehensive pipeline  
python src/pipeline.py

_(Artifacts, models, and figures will be exported automatically to the Research_Project_AI_PDU folder)._

## 🌐 Live Industrial Demo

To prove the deployability of this framework on Edge-AI environments, we have included a client-side Web Application powered by **TensorFlow.js**.

The demo features:

- **Dynamic Model Routing:** Instantly switch between Lab (CWRU) and Shop (PAD) neural weights based on your environment.
- **100% Client-Side Privacy:** Real-time signal inference in the browser. No sensor data is sent to external servers.
- **Live Pre-processing:** Implements the exact Local Z-Score normalization used in training.

**\[View Live Interactive Demo Here\] (Insert Vercel/Render Link)**

## ✉️ Contact & Citation

If you find this framework useful for your research, please consider citing our CSoNet 2026 paper (Citation details pending publication).

For academic inquiries, collaboration, or data access, please contact:

- **Vo Thi Kim Anh** - [vothikimanh@tdtu.edu.vn](mailto:vothikimanh@tdtu.edu.vn) | [thi.kim.anh.vo.st@vsb.cz](mailto:thi.kim.anh.vo.st@vsb.cz)
