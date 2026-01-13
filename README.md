# UIDAI-HACKATHON
# 👁️ Project Drishti: Predictive Intelligence for Aadhaar
### *Data-Driven Resource Allocation & Anomaly Detection Framework*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Library](https://img.shields.io/badge/Library-Pandas%20%7C%20Sklearn-orange) ![Status](https://img.shields.io/badge/Status-Hackathon%20Submission-green)

---

## 📌 Overview
**Project Drishti** is an AI-powered analytics framework designed to optimize the operations of Aadhaar Seva Kendras. By analyzing over **1 million enrollment records**, this tool transitions the ecosystem from a **Reactive** model (managing crowds) to a **Predictive** model (deploying resources *before* demand peaks).

It solves two critical problems:
1.  **Overcrowding:** Predicting seasonal surges (e.g., School Admissions) to prevent long queues.
2.  **Fraud:** Detecting hyper-local operational anomalies using Unsupervised Machine Learning.

---

## 🚀 Key Features
* **📅 Temporal Forecasting:** Identifies the "September Surge" (600% demand spike) for proactive staffing.
* **🚌 Target Segmentation (Clustering):** Automatically segments districts into "School Hubs" vs. "Birthing Centers" to guide Mobile Van deployment.
* **🚨 Automated Fraud Detection:** Uses **Isolation Forest** algorithms to flag pincodes with statistically impossible activity levels (Top 0.1% outliers).

---

## 📂 Project Structure

```text
📦 UIDAI-HACKATHON
 ┣ 📜 project_drishti_analysis.py    # Main AI & Analysis Script
 ┣ 📜 dashboard.png                  # Output Visualizations (The Graph)
 ┣ 📜 api_data_aadhar_enrolment_0_500000.csv
 ┣ 📜 api_data_aadhar_enrolment_500000_1000000.csv
 ┣ 📜 api_data_aadhar_enrolment_1000000_1006029.csv
 ┗ 📜 README.md                      # Project Documentation
```
---
## 📊 Methodology & AI Models

### 1. Unsupervised Clustering (K-Means)
Instead of manual thresholds, we used **K-Means Clustering** to segment districts based on their demographic ratio (New Births vs. Student Updates).
* **Result:** Identified **178 High-Priority Districts** where student workload > 40%, making them ideal targets for "School Camp" interventions.

### 2. Anomaly Detection (Isolation Forest)
We implemented the **Isolation Forest** algorithm to detect non-linear outliers in pincode activity.
* **Result:** Filtered millions of transactions to flag exactly **5 Critical Pincodes** for immediate vigilance audit.

---

## 📉 Visual Insights
The code generates a 4-panel dashboard highlighting:
1.  **The "September Surge":** Clear visualization of the Q3 demand spike.
2.  **Demographic Clusters:** Visual proof of "Student-Heavy" vs. "Birth-Heavy" districts.
3.  **School Targets:** Top 10 districts requiring Mobile Vans.
4.  **Fraud Hotspots:** Pincodes with Z-Scores > 25.

*(See `dashboard.png` in the repository for the output graph)*

---

## 💻 How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Davish-code/UIDAI-HACKATHON.git](https://github.com/Davish-code/UIDAI-HACKATHON.git)
    cd UIDAI-HACKATHON
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Analysis**
    ```bash
    python project_drishti_analysis.py
    ```

---


## 🔮 Future Scope
* **Real-Time Integration:** Deploying the Z-Score API to the live UIDAI dashboard.
* **Biometric Failure Analysis:** Extending the Anomaly model to detect faulty hardware via "failure rate" patterns.

---

*Submitted for **UIDAI Data Hackathon 2026** by [Your Team Name]*
