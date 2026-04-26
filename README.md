# A Data-Driven Framework for Predictive Maintenance: From Knowledge Topology to WDCNN-BiLSTM-Attention Architecture

**Lead Researcher & Author:** Vo Thi Kim Anh (ORCID: 0009-0001-5691-7665)

**Affiliations:** 1. Faculty of Information Technology, Ton Duc Thang University, Ho Chi Minh City, Vietnam

2\. Faculty of Electrical Engineering and Computer Science, VSB - Technical University of Ostrava, Ostrava, Czech Republic

## 📖 Abstract

Predictive Maintenance (PdM) is a cornerstone of Industry 4.0. This research proposes a novel "Model-Centric" pipeline that bridges the gap between theoretical consensus and practical Deep Learning deployment. We utilize bibliometric topology on Web of Science (WoS) data to identify research hotspots, followed by Structural Equation Modeling (SEM) to quantitatively validate the necessity of hybrid AI architectures. Driven by these findings, we propose the **WDCNN-BiLSTM-DualAttn** framework. Experimental results demonstrate a state-of-the-art accuracy of **97.50%** on the CWRU dataset and a robust **89.44%** on the highly noisy Paderborn dataset, achieved through Local Z-Score Normalization.

## 📊 Empirical Benchmarking Results

Our framework utilizes a **Dataset Discovery** approach to ensure real-world reliability across different industrial noise profiles.

| **Industrial Context**    | **Dataset Source**   | **Signal Complexity**          | **Accuracy** |
| ------------------------- | -------------------- | ------------------------------ | ------------ |
| **Context A: Laboratory** | CWRU Bearing Data    | Low-Noise                      | **97.50%**   |
| ---                       | ---                  | ---                            | ---          |
| **Context B: Shop Floor** | Paderborn University | **High-Noise / Variable Load** | **89.44%**   |
| ---                       | ---                  | ---                            | ---          |

**Scientific Note:** Achieving ~89.44% accuracy on the Paderborn dataset is highly significant for industrial deployment. It demonstrates the framework's ability to generalize across variable motor speeds and torque levels without specific fine-tuning.

## 🧠 Core Methodology

- **Knowledge Topology:** Identification of spatial-temporal convergence in global PdM research.
- **SEM Validation:** Mathematical proof (![](data:image/png;base64,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)) justifying the integration of CNN, LSTM, and Attention layers.
- **WDCNN-BiLSTM-DualAttn:** A wide-kernel architecture designed as a morphological filter to suppress mechanical background noise.
- **Local Z-Score Normalization:** Ensures load and speed invariance during real-time inference.

## 💻 Repository Structure

SmartPdM-Framework/  
├── src/  
│ ├── pipeline.py # Full Python research pipeline (WoS + SEM + Training)  
│ └── index.html # Industrial Web Demo (TF.js Frontend)  
├── models/  
│ ├── context_a_cwru/ # Exported TF.js weights (97.50% Acc)  
│ └── context_b_paderborn/ # Exported TF.js weights (89.44% Acc)  
└── README.md

## 🌐 Live Industrial Demo

The included index.html provides a 100% client-side inference engine using **TensorFlow.js**.

- **Live Link:** https://smart-pd-m-framework.vercel.app
- **Dynamic Routing:** Instantly switch between Lab and Shop-optimized weights.
- **Privacy First:** All vibration data is processed locally in the browser.

## ✉️ Contact

For academic inquiries or collaboration:

- **Vo Thi Kim Anh** - [vothikimanh@tdtu.edu.vn](mailto:vothikimanh@tdtu.edu.vn) | [thi.kim.anh.vo.st@vsb.cz](mailto:thi.kim.anh.vo.st@vsb.cz)
