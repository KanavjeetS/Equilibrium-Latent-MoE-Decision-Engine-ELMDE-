"""
Multi-Domain Real-World Decision Dataset Generator.
Generates 2,500 rich, realistic decision instances across:
1. Financial Risk & Credit Underwriting
2. Emergency Clinical Triage & Severity
3. Autonomous Vehicle & Drone Flight Safety
4. High-Throughput API / Tool Routing
"""

import json
import random
import os
from typing import Dict, List, Any


def generate_financial_records(count: int) -> List[Dict[str, Any]]:
    templates = [
        ("Applicant {id}: FICO score {fico}, DTI ratio {dti}%, annual income ${income:,}, "
         "revolving credit utilization {util}%, recent inquiries {inq}, past delinquency {delinq}. "
         "Employment tenure: {tenure} years in {industry}.", 0),
        ("Corporate borrower {id}: Current ratio {cr:.2f}, Quick ratio {qr:.2f}, Debt-to-Equity {dte:.2f}, "
         "EBITDA margin {ebitda}%, Altman Z-score {z_score:.2f}. Industry sector: {industry}.", 1),
    ]
    
    industries = ["Software/SaaS", "Healthcare Services", "Renewable Energy", "Commercial Real Estate", "Retail Logistics", "Manufacturing"]
    records = []
    
    for i in range(count):
        fico = random.randint(520, 840)
        dti = round(random.uniform(8.0, 58.0), 1)
        income = random.randint(35000, 320000)
        util = round(random.uniform(2.0, 95.0), 1)
        inq = random.randint(0, 8)
        delinq = random.choice([0, 0, 0, 1, 2, 3])
        tenure = random.randint(1, 25)
        industry = random.choice(industries)
        
        # Calculate risk score [0, 4]
        risk_score = 0
        if fico < 600: risk_score += 2
        elif fico < 680: risk_score += 1
        if dti > 45: risk_score += 1
        if util > 70: risk_score += 1
        if delinq >= 2: risk_score += 1
        risk_tier = min(4, max(0, risk_score))
        
        continuous_risk = min(0.98, max(0.02, (850 - fico) / 350.0 * 0.5 + (dti / 60.0) * 0.3 + (util / 100.0) * 0.2))
        
        text = (f"Loan applicant #{1000+i}: FICO score {fico}, debt-to-income {dti}%, annual income ${income:,}, "
                f"revolving credit utilization {util}%, inquiries in last 6m {inq}, past delinquencies {delinq}. "
                f"Employment duration: {tenure} years in {industry} sector.")
        
        records.append({
            "id": f"fin_{i:04d}",
            "domain": "financial_risk",
            "text": text,
            "ordinal_label": risk_tier,
            "continuous_label": round(continuous_risk, 4),
            "choice_candidates": ["ApprovePrime", "ApproveStandard", "RequireCosigner", "ManualReview", "DeclineRisk"],
            "choice_label": risk_tier
        })
    return records


def generate_clinical_records(count: int) -> List[Dict[str, Any]]:
    records = []
    chief_complaints = [
        ("severe substernal chest pain radiating to left arm with diaphoresis", 4, 0.94),
        ("unresponsive adult found in public park, shallow agonal respiration", 4, 0.98),
        ("acute respiratory distress with SpO2 86% on room air, audible wheezing", 3, 0.78),
        ("severe blunt abdominal trauma post-motor vehicle collision, tachycardic at 128 bpm", 3, 0.82),
        ("sudden onset focal neurological deficit, right-sided hemiparesis, onset 45 min", 3, 0.85),
        ("persistent high fever 39.5C with neck stiffness and photophobia", 2, 0.65),
        ("deep laceration to left forearm with controlled venous bleeding", 1, 0.32),
        ("mild ankle sprain while playing sports, ambulatory with limp", 0, 0.12),
        ("sore throat and rhinorrhea for 3 days, afebrile, normal vitals", 0, 0.05),
        ("chronic bilateral knee pain without acute swelling or deformity", 0, 0.04)
    ]
    
    for i in range(count):
        complaint, severity_tier, base_risk = random.choice(chief_complaints)
        age = random.randint(18, 88)
        hr = random.randint(58, 145)
        sbp = random.randint(85, 195)
        spo2 = random.randint(84, 100)
        
        text = (f"Emergency Department Patient #{2000+i}, age {age}. Chief Complaint: {complaint}. "
                f"Triage Vitals: HR {hr} bpm, BP {sbp}/{random.randint(50, 105)} mmHg, SpO2 {spo2}%, RR {random.randint(12, 34)}/min.")
        
        records.append({
            "id": f"clin_{i:04d}",
            "domain": "clinical_triage",
            "text": text,
            "ordinal_label": severity_tier,
            "continuous_label": min(0.99, max(0.01, round(base_risk + random.uniform(-0.05, 0.05), 4))),
            "choice_candidates": ["DischargeHome", "ObservationUnit", "GeneralWard", "StepDownCare", "IntensiveCareUnit"],
            "choice_label": severity_tier
        })
    return records


def generate_autonomous_records(count: int) -> List[Dict[str, Any]]:
    records = []
    scenarios = [
        ("Sensors report forward LiDAR obstruction at 2.4m, relative closing speed 14.2 m/s, friction coeff 0.22", 4, 0.96),
        ("Visual SLAM lost tracking in low-light tunnel, IMU accelerometer bias drift 0.42 m/s^2, distance to wall 1.1m", 3, 0.79),
        ("Cross-traffic vehicle running red light ahead at 45 km/h, distance 18.5m, brake line primed", 3, 0.84),
        ("Dense fog reduces visibility to 15m, ultrasonic ping reflects intermittent pedestrian profile on right shoulder", 2, 0.58),
        ("Lane departure warning: vehicle drifting 0.4m right towards highway rumble strip, clear lane left", 1, 0.28),
        ("Open interstate highway cruising, GPS precision 0.02m, nominal radar tracking 3 lead vehicles at 60m", 0, 0.03),
    ]
    
    for i in range(count):
        desc, threat_tier, base_conf = random.choice(scenarios)
        speed = random.randint(10, 110)
        temp = random.randint(-10, 40)
        text = f"Autonomous telemetry frame #{3000+i}: Current speed {speed} km/h, ambient temp {temp}C. Telemetry: {desc}."
        
        records.append({
            "id": f"auto_{i:04d}",
            "domain": "autonomous_flight",
            "text": text,
            "ordinal_label": threat_tier,
            "continuous_label": min(0.99, max(0.01, round(base_conf + random.uniform(-0.04, 0.04), 4))),
            "choice_candidates": ["ContinueCruise", "ReduceSpeedSoft", "LaneChangeSafe", "HardDeceleration", "EmergencyEvasiveBrake"],
            "choice_label": threat_tier
        })
    return records


def generate_api_routing_records(count: int) -> List[Dict[str, Any]]:
    tools = [
        ("Query historical user purchase records and invoice summaries", "DatabaseQuery", 0, 0.15),
        ("Find semantically similar legal contracts discussing force majeure clauses", "VectorSearch", 1, 0.35),
        ("Retrieve current breaking news headlines regarding federal reserve interest rates", "ExternalWebSearch", 2, 0.55),
        ("Dispatch confirmation email with signed PDF receipt to client", "EmailDispatch", 3, 0.75),
        ("Calculate running Pearson correlation coefficient and volatility index", "CalculateMetric", 4, 0.90),
    ]
    
    records = []
    for i in range(count):
        intent, tool_name, tool_idx, base_val = random.choice(tools)
        text = f"Intelligent Assistant Task #{4000+i}: User directive: '{intent}'. Context token length: {random.randint(120, 480)} tokens."
        
        records.append({
            "id": f"api_{i:04d}",
            "domain": "api_routing",
            "text": text,
            "ordinal_label": tool_idx,
            "continuous_label": round(base_val + random.uniform(-0.03, 0.03), 4),
            "choice_candidates": ["DatabaseQuery", "VectorSearch", "ExternalWebSearch", "EmailDispatch", "CalculateMetric"],
            "choice_label": tool_idx
        })
    return records


def build_and_save_dataset():
    os.makedirs("data", exist_ok=True)
    
    all_data = []
    all_data.extend(generate_financial_records(750))
    all_data.extend(generate_clinical_records(750))
    all_data.extend(generate_autonomous_records(500))
    all_data.extend(generate_api_routing_records(500))
    
    random.seed(42)
    random.shuffle(all_data)
    
    train_data = all_data[:2000]
    val_data = all_data[2000:]
    
    train_path = os.path.join("data", "production_corpus_train.json")
    val_path = os.path.join("data", "production_corpus_val.json")
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
        
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
        
    print(f"Generated {len(train_data)} train samples -> {train_path}")
    print(f"Generated {len(val_data)} val samples -> {val_path}")


if __name__ == "__main__":
    build_and_save_dataset()
