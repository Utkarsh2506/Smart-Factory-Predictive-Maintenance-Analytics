# Smart Factory Predictive Maintenance Analytics

---

### # Problem Statement

In manufacturing, unplanned machine downtime is a primary source of significant financial loss. A single critical failure can halt production lines, leading to missed deadlines, high costs for emergency repairs, and inefficient resource allocation. Traditional run-to-failure or time-based preventive maintenance schedules are often inefficient, either allowing preventable failures to occur or resulting in unnecessary maintenance on healthy machines.

This project addresses this challenge by developing an end-to-end Business Intelligence solution to forecast machine failures before they happen and quantify the financial impact of a proactive maintenance strategy.

---

### # Solution Overview

To solve this problem, I developed a complete analytics pipeline that integrates machine learning, financial modelling, and business intelligence visualization

1.  **Data Ingestion & Warehousing:** Processed raw sensor data and stored it in a MySQL database to simulate a real-world data environment.
2.  **Predictive Modelling (Machine Learning):** Trained an XGBoost classification model on historical sensor data (temperature, vibration, torque) to accurately predict the probability of machine failure within a specific time window.
3.  **Financial Analysis (ROI Modelling): Built a detailed cost-benefit analysis model in Excel to compare the high cost of unplanned failures against the lower cost of proactive interventions. This model calculated potential ROI and tested assumptions using What-If analysis.
4.  **Root Cause Analysis (RCA):** Employed a Fishbone (Ishikawa) diagram to identify and categorize potential root causes of failures, aligning with Total Quality Management (TQM) principles.
5.  **BI Dashboard Integration:** Consolidated all operational and financial insights into an interactive Power BI dashboard designed for plant managers and business leaders.

---

### # Key Results & Dashboard Features

The analysis demonstrated significant value in shifting to a predictive maintenance strategy, projecting annual savings of approximately ¥577 Million with an ROI of exactly 1308%.

Key Features of the Power BI Dashboard

 * **Page 1: Predictive Maintenance**
     High-level financial KPIs Total Annual Savings, ROI, and Program Cost.
     Real-time machine health status monitoring.
     Alerts for machines with a high probability of near-term failure.
 * **Page 2: Quality Control & Failure Analysis**
     Pareto charts identifying the most frequent failure types.
     Integration of the Root Cause Analysis (Fishbone diagram) explaining why failures occur.
     Historical trends of sensor readings leading up to failure events.

---

### # Technology Stack

 * **Data Analysis:** Python (Pandas, NumPy)
 * **Machine Learning:** Scikit-learn, XGBoost
 * **Database:** MySQL
 * **Financial Modelling & RCA:** Advanced Excel (What-If Analysis, Fishbone Diagram)
 * **Visualization:** Power BI, MatplotlibSeaborn

---

**For a detailed breakdown of the methodology, model evaluation, and financial analysis, please see the [PROJECT REPORT.pdf](https://github.com/user-attachments/files/22232328/PROJECT.REPORT.pdf).**

---

### # Project Structure

```plaintext
Smart Factory Predictive Maintenance Analytics
│
├── Data/                                                                # Contains the original raw dataset
│   └── Factory Dataset.csv
│
├── SQL/                                                                 # Database schema
│   └── Schema_SQL
│
├── Notebooks/                                                           # Python notebooks for data import and ML modelling
│   ├── Data Import.ipynb
│   └── Machine Failure Predictor.ipynb
│
├── Analysis Files/                                                      # Final Power BI (.pbix) and Excel (.xlsx) files
│   ├── Smart Factory Dashboard.pbix
│   └── Smart Factory Model.xlsx
│
├── README.md                                                            # Project summary
│
└── Project Report                                                       # Detailed project report
