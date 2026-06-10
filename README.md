# Validating Non-Fatal Intentional Poisoning Incidents as a National Sentinel Indicator for Suicide Mortality Trends in Malaysia (2006–2021)

This repository contains the official reproducible data analysis pipeline, advanced time-series modeling architectures, and publication-grade data visualization code for the manuscript submitted to *BMC Public Health*.

---

## 📋 Project Architecture Overview

Relying exclusively on official suicide vital statistics often introduces severe 1- to 2-year reporting delays due to administrative verification, forensic review, and legal investigations. This lag creates a significant operational gap, forcing public health interventions to remain reactive. 

This project mathematically validates near real-time toxicovigilance tracking from the National Poison Centre (NPC-USM) database as a high-sensitivity sentinel indicator for national distress. By integrating 16 years of continuous, population-level intentional exposure tallies with official mortality registries from the Department of Statistics Malaysia (DOSM), this pipeline evaluates historical baseline alignments, models long-term temporal momentum using Autoregressive Integrated Moving Average with Exogenous Predictors (ARIMAX), and executes a clinicoeconomic simulation based on the Value of Statistical Life (VSL) framework.

---

## 🔒 Data Availability Statement & Access Protocols

To protect personal privacy and ensure compliance with medical confidentiality guidelines, the underlying clinical dataset—containing raw biological, chemical, and pharmaceutical exposure incidents—cannot be hosted publicly in this open-access repository. 

### Clinical Data Access Policy
* **Source Dataset:** `Kes Poisoning 0624 Mini.csv`
* **Data Steward:** National Poison Centre, Universiti Sains Malaysia (USM).
* **Access Conditions:** The non-fatal intentional exposure records used in this ecological correlation study are available from the corresponding author upon reasonable request, subject to formal institutional review and permission from the Director of the National Poison Centre, USM. 

### Formal Access Requests
To request data access for research validation or academic replication, please contact the corresponding author:

**Ts. Dr. Syazwan Aizat Ismail** National Poison Centre, Universiti Sains Malaysia  
Gelugor, 11800, Pulau Pinang, Malaysia  
📧 **Email:** drsai@usm.my  

---

## 🛠️ Pipeline Setup & Environment Configuration

The pipeline script is designed for zero-dependency cross-platform reproducibility using standard scientific computing architectures in Python 3.10+.

### Prerequisites
Ensure the following core Python libraries are installed before running the analysis:

```bash
pip install pandas numpy scipy statsmodels matplotlib seabed
