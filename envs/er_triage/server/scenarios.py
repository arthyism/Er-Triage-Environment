"""
Patient scenario bank for the ER Triage environment.

Each ground_truth entry now carries per-patient key_signals so the
reasoning scorer checks only signals relevant to that specific patient.

ESI Quick Reference:
  1 → Immediate    | resuscitation_bay  | life threat right now
  2 → Emergency    | monitored_bed      | high risk, needs rapid MD
  3 → Urgent       | exam_room          | stable, needs 2+ resources
  4 → Less urgent  | exam_room          | stable, needs 1 resource
  5 → Non-urgent   | waiting_room       | no resources needed
"""

import os
import sys
import random

try:
    from ..models import Patient, Vitals
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(__file__))
    from models import Patient, Vitals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p(pid, age, sex, complaint, bp_s, bp_d, hr, rr, spo2, temp, pain,
       mental, arrival, history=None):
    return Patient(
        patient_id=pid, age=age, sex=sex,
        chief_complaint=complaint,
        vitals=Vitals(
            bp_systolic=bp_s, bp_diastolic=bp_d,
            heart_rate=hr, respiratory_rate=rr,
            spo2=spo2, temperature=temp, pain_score=pain
        ),
        mental_status=mental,
        arrival_mode=arrival,
        history=history
    )


# ---------------------------------------------------------------------------
# EASY — single patient, textbook ESI, no ambiguity
# ---------------------------------------------------------------------------

EASY_SCENARIOS = [

    {
        "scenario_id": "E001",
        "patients": [
            _p("P001", 45, "M",
               "unresponsive, found collapsed at home",
               60, 40, 140, 32, 78.0, 36.8, 10,
               "unresponsive", "ambulance", history="unknown")
        ],
        "ground_truth": [
            {"patient_id": "P001", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["unresponsive", "spo2", "critical", "immediate",
                             "airway", "blood pressure", "resuscitation"]}
        ],
    },

    {
        "scenario_id": "E002",
        "patients": [
            _p("P002", 62, "F",
               "sudden severe chest pain, radiating to left arm, sweating",
               92, 64, 108, 22, 94.0, 37.0, 8,
               "alert", "ambulance", history="hypertension, smoker")
        ],
        "ground_truth": [
            {"patient_id": "P002", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["chest pain", "radiation", "diaphoresis",
                             "cardiac", "high risk", "ecg", "troponin"]}
        ],
    },

    {
        "scenario_id": "E003",
        "patients": [
            _p("P003", 34, "M",
               "right wrist pain after fall, mild swelling, no deformity",
               122, 78, 82, 16, 98.0, 36.9, 4,
               "alert", "walk-in")
        ],
        "ground_truth": [
            {"patient_id": "P003", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["wrist", "swelling", "stable", "x-ray", "one resource"]}
        ],
    },

    {
        "scenario_id": "E004",
        "patients": [
            _p("P004", 8, "F",
               "fever and ear pain, pulling at right ear, eating normally",
               102, 66, 94, 20, 99.0, 38.4, 3,
               "alert", "walk-in", history="no significant history")
        ],
        "ground_truth": [
            {"patient_id": "P004", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["fever", "ear", "stable", "one resource", "otitis"]}
        ],
    },

    {
        "scenario_id": "E005",
        "patients": [
            _p("P005", 28, "M",
               "laceration on left forearm, 3cm clean cut, bleeding controlled",
               118, 76, 78, 16, 99.0, 36.7, 3,
               "alert", "walk-in")
        ],
        "ground_truth": [
            {"patient_id": "P005", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["laceration", "controlled", "stable", "suture", "one resource"]}
        ],
    },

    {
        "scenario_id": "E006",
        "patients": [
            _p("P006", 71, "M",
               "sudden onset worst headache of life, neck stiffness, photophobia",
               148, 92, 96, 18, 97.0, 37.2, 9,
               "confused", "ambulance", history="no prior headaches")
        ],
        "ground_truth": [
            {"patient_id": "P006", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["worst headache", "thunderclap", "meningismus",
                             "subarachnoid", "confused", "urgent", "ct"]}
        ],
    },

    {
        "scenario_id": "E007",
        "patients": [
            _p("P007", 19, "F",
               "prescription refill request, no acute complaints",
               116, 74, 72, 14, 99.0, 36.6, 0,
               "alert", "walk-in")
        ],
        "ground_truth": [
            {"patient_id": "P007", "esi_level": 5, "resource": "waiting_room",
             "key_signals": ["no acute", "stable", "no resources", "non-urgent", "prescription"]}
        ],
    },

    {
        "scenario_id": "E008",
        "patients": [
            _p("P008", 55, "F",
               "difficulty breathing, wheezing, unable to speak full sentences",
               130, 82, 118, 28, 89.0, 37.1, 6,
               "alert", "walk-in", history="severe persistent asthma")
        ],
        "ground_truth": [
            {"patient_id": "P008", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["wheezing", "spo2", "respiratory distress",
                             "asthma", "high risk", "nebuliser"]}
        ],
    },

    {
        "scenario_id": "E009",
        "patients": [
            _p("P009", 42, "M",
               "abdominal pain, nausea, mild tenderness right lower quadrant",
               124, 80, 88, 18, 98.0, 37.8, 5,
               "alert", "walk-in")
        ],
        "ground_truth": [
            {"patient_id": "P009", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["abdominal pain", "tenderness", "labs", "imaging",
                             "appendicitis", "multiple resources"]}
        ],
    },

    {
        "scenario_id": "E010",
        "patients": [
            _p("P010", 3, "M",
               "high fever, inconsolable crying, bulging fontanelle",
               88, 58, 168, 44, 96.0, 40.1, 8,
               "alert", "ambulance")
        ],
        "ground_truth": [
            {"patient_id": "P010", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["bulging fontanelle", "meningitis", "infant",
                             "fever", "immediate", "critical", "pediatric"]}
        ],
    },

]


# ---------------------------------------------------------------------------
# MEDIUM — 3 simultaneous patients, agent must triage all
# ---------------------------------------------------------------------------

MEDIUM_SCENARIOS = [

    {
        "scenario_id": "M001",
        "patients": [
            _p("PA01", 67, "M",
               "crushing chest pain radiating to jaw, diaphoretic",
               86, 58, 114, 24, 91.0, 36.9, 9,
               "confused", "ambulance", history="diabetes, hypertension"),
            _p("PA02", 23, "F",
               "twisted ankle playing football, walking with limp",
               118, 76, 80, 16, 99.0, 36.7, 4,
               "alert", "walk-in"),
            _p("PA03", 50, "M",
               "sudden facial droop left side, slurred speech, arm weakness",
               162, 94, 88, 18, 97.0, 37.0, 2,
               "confused", "ambulance", history="atrial fibrillation"),
        ],
        "ground_truth": [
            {"patient_id": "PA01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["chest pain", "shock", "low bp", "diaphoretic",
                             "cardiac", "immediate", "confused"]},
            {"patient_id": "PA02", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["ankle", "stable", "x-ray", "one resource", "minor"]},
            {"patient_id": "PA03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["stroke", "facial droop", "slurred speech",
                             "neurological", "FAST", "atrial fibrillation"]},
        ],
    },

    {
        "scenario_id": "M002",
        "patients": [
            _p("PB01", 78, "F",
               "fall from standing, hip pain, unable to bear weight",
               134, 82, 92, 18, 97.0, 36.8, 7,
               "alert", "ambulance", history="osteoporosis, warfarin"),
            _p("PB02", 31, "M",
               "severe allergic reaction, throat tightening, hives spreading",
               98, 64, 118, 28, 93.0, 36.9, 6,
               "alert", "walk-in", history="known peanut allergy"),
            _p("PB03", 14, "F",
               "mild sore throat, low grade fever, eating and drinking fine",
               112, 72, 84, 16, 99.0, 37.9, 2,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PB01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["hip fracture", "anticoagulant", "warfarin",
                             "elderly", "osteoporosis", "fall"]},
            {"patient_id": "PB02", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["anaphylaxis", "airway", "epinephrine",
                             "throat tightening", "allergy", "immediate"]},
            {"patient_id": "PB03", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["sore throat", "stable", "mild", "one resource", "strep"]},
        ],
    },

    {
        "scenario_id": "M003",
        "patients": [
            _p("PC01", 45, "F",
               "right flank pain radiating to groin, nausea, vomiting",
               126, 80, 96, 20, 98.0, 37.4, 8,
               "alert", "walk-in", history="previous kidney stones"),
            _p("PC02", 60, "M",
               "confusion and fever, urinary catheter in place",
               96, 62, 104, 22, 95.0, 39.2, 4,
               "confused", "ambulance", history="prostate cancer"),
            _p("PC03", 25, "M",
               "cut finger, minor bleeding, band-aid applied at home",
               120, 78, 76, 14, 99.0, 36.6, 2,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PC01", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["renal colic", "flank pain", "imaging",
                             "pain management", "ureter", "multiple resources"]},
            {"patient_id": "PC02", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["urosepsis", "fever", "confusion",
                             "immunocompromised", "cancer", "sepsis", "urgent"]},
            {"patient_id": "PC03", "esi_level": 5, "resource": "waiting_room",
             "key_signals": ["minor", "controlled", "no resources", "non-urgent", "stable"]},
        ],
    },

    {
        "scenario_id": "M004",
        "patients": [
            _p("PD01", 38, "F",
               "12 weeks pregnant, heavy vaginal bleeding, severe cramping",
               100, 66, 112, 22, 97.0, 36.8, 8,
               "alert", "ambulance"),
            _p("PD02", 70, "M",
               "blood glucose 28 mg/dL, diaphoretic, barely rousable",
               104, 68, 108, 18, 97.0, 36.7, 2,
               "confused", "ambulance", history="type 1 diabetes, insulin"),
            _p("PD03", 29, "M",
               "low back pain after lifting, mild spasm, ambulating independently",
               122, 80, 78, 16, 99.0, 36.9, 5,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PD01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["miscarriage", "hemorrhage", "obstetric",
                             "pregnant", "bleeding", "high risk"]},
            {"patient_id": "PD02", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["hypoglycemia", "dextrose", "altered mental status",
                             "insulin", "glucose", "immediate", "critical"]},
            {"patient_id": "PD03", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["musculoskeletal", "stable", "one resource",
                             "back pain", "ambulating"]},
        ],
    },

    {
        "scenario_id": "M005",
        "patients": [
            _p("PE01", 55, "M",
               "coughing up blood, 2 cups, known lung cancer",
               102, 66, 110, 26, 88.0, 37.3, 7,
               "alert", "ambulance", history="stage 3 lung cancer, chemotherapy"),
            _p("PE02", 44, "F",
               "palpitations, heart racing for 2 hours, no chest pain",
               124, 80, 164, 18, 98.0, 36.8, 3,
               "alert", "walk-in", history="previous SVT"),
            _p("PE03", 10, "M",
               "mild rash on forearm, no fever, no respiratory symptoms",
               110, 70, 88, 18, 99.0, 36.7, 1,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PE01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["hemoptysis", "spo2", "airway", "cancer",
                             "massive bleeding", "immediate"]},
            {"patient_id": "PE02", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["tachycardia", "SVT", "cardiac", "ecg",
                             "palpitations", "high risk"]},
            {"patient_id": "PE03", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["rash", "mild", "stable", "one resource", "non-urgent"]},
        ],
    },

    {
        "scenario_id": "M006",
        "patients": [
            _p("PF01", 82, "F",
               "altered mental status, not recognising family, no fever",
               148, 90, 88, 18, 95.0, 36.7, 0,
               "confused", "ambulance", history="dementia, blood thinners"),
            _p("PF02", 36, "M",
               "eye injury from metal fragment at work, vision blurry",
               128, 82, 84, 16, 99.0, 36.8, 6,
               "alert", "walk-in"),
            _p("PF03", 50, "F",
               "requesting paperwork for insurance, no medical complaint",
               118, 76, 74, 14, 99.0, 36.6, 0,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PF01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["altered mental status", "anticoagulant",
                             "acute change", "elderly", "dementia", "urgent"]},
            {"patient_id": "PF02", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["eye injury", "foreign body", "imaging",
                             "ophthalmology", "vision", "multiple resources"]},
            {"patient_id": "PF03", "esi_level": 5, "resource": "waiting_room",
             "key_signals": ["no complaint", "non-urgent", "no resources",
                             "paperwork", "stable"]},
        ],
    },

    {
        "scenario_id": "M007",
        "patients": [
            _p("PG01", 66, "M",
               "not feeling well, blood sugar 480 mg/dL, excessive thirst",
               112, 72, 104, 24, 97.0, 37.2, 3,
               "alert", "walk-in", history="type 2 diabetes"),
            _p("PG02", 22, "F",
               "severe abdominal pain, rigid abdomen, rebound tenderness",
               96, 62, 116, 26, 97.0, 38.4, 9,
               "alert", "ambulance"),
            _p("PG03", 48, "M",
               "headache for 3 days, mild, relieved partially by ibuprofen",
               128, 82, 78, 16, 99.0, 36.8, 4,
               "alert", "walk-in"),
        ],
        "ground_truth": [
            {"patient_id": "PG01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["hyperglycemia", "DKA", "ketoacidosis",
                             "insulin", "glucose", "labs", "urgent"]},
            {"patient_id": "PG02", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["peritonitis", "surgical", "perforation",
                             "rigid abdomen", "rebound", "high risk"]},
            {"patient_id": "PG03", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["headache", "stable", "imaging", "multiple resources",
                             "chronic", "neurological workup"]},
        ],
    },

    {
        "scenario_id": "M008",
        "patients": [
            _p("PH01", 5, "F",
               "ingested unknown tablets, found beside open bottle",
               104, 68, 128, 28, 95.0, 36.8, 6,
               "confused", "ambulance"),
            _p("PH02", 39, "M",
               "sprained thumb, minor swelling, full grip strength",
               120, 78, 76, 16, 99.0, 36.7, 3,
               "alert", "walk-in"),
            _p("PH03", 74, "F",
               "3-day increasing shortness of breath, bilateral leg oedema",
               140, 88, 100, 26, 90.0, 36.9, 4,
               "alert", "walk-in", history="heart failure, furosemide"),
        ],
        "ground_truth": [
            {"patient_id": "PH01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["ingestion", "poison", "child", "toxicology",
                             "overdose", "immediate", "confused"]},
            {"patient_id": "PH02", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["sprain", "stable", "x-ray", "minor", "one resource"]},
            {"patient_id": "PH03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["heart failure", "pulmonary oedema", "spo2",
                             "decompensated", "diuretic", "urgent"]},
        ],
    },

]


# ---------------------------------------------------------------------------
# HARD — 6-8 patients, resource constraints active, some deteriorate
# ---------------------------------------------------------------------------

HARD_SCENARIOS = [

    {
        "scenario_id": "H001",
        "description": "Multi-vehicle accident. 7 patients. 2 resus bays, 3 monitored beds, 4 exam rooms.",
        "patients": [
            _p("Q01", 30, "M", "unresponsive, GCS 3, head trauma",
               70, 40, 140, 32, 82.0, 36.5, 10, "unresponsive", "ambulance"),
            _p("Q02", 45, "F", "open femur fracture, significant blood loss",
               88, 56, 124, 24, 93.0, 36.3, 10, "confused", "ambulance"),
            _p("Q03", 28, "M", "chest wall bruising, painful breathing, SpO2 dropping",
               110, 72, 118, 30, 91.0, 37.0, 8, "alert", "ambulance"),
            _p("Q04", 60, "F", "cervical spine pain, ambulatory, no neuro deficit",
               128, 82, 84, 16, 98.0, 36.9, 6, "alert", "ambulance"),
            _p("Q05", 19, "M", "abrasions both arms, minor lacerations, vitals stable",
               120, 78, 78, 16, 99.0, 36.8, 3, "alert", "walk-in"),
            _p("Q06", 55, "M", "pelvic pain, suspected pelvic fracture, cannot stand",
               100, 64, 108, 22, 95.0, 37.0, 9, "alert", "ambulance"),
            _p("Q07", 10, "F", "frightened, minor bruising, no loss of consciousness",
               110, 70, 96, 18, 99.0, 36.7, 2, "alert", "ambulance"),
        ],
        "available_resources": {
            "resuscitation_bay": 2, "monitored_bed": 3,
            "exam_room": 4, "waiting_room": 999
        },
        "ground_truth": [
            {"patient_id": "Q01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["GCS", "head trauma", "unresponsive", "immediate",
                             "airway", "critical", "resuscitation"]},
            {"patient_id": "Q02", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["hemorrhage", "femur fracture", "blood loss",
                             "shock", "immediate", "transfusion"]},
            {"patient_id": "Q03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["pneumothorax", "chest", "spo2", "rib",
                             "breathing", "urgent", "dropping"]},
            {"patient_id": "Q04", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["cervical spine", "c-spine", "imaging",
                             "ct", "stable", "multiple resources"]},
            {"patient_id": "Q05", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["minor", "abrasions", "stable",
                             "one resource", "wound care"]},
            {"patient_id": "Q06", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["pelvic fracture", "hemorrhage risk",
                             "cannot stand", "urgent", "imaging"]},
            {"patient_id": "Q07", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["pediatric", "bruising", "observation",
                             "exam", "stable", "multiple resources"]},
        ],
        "deteriorating_patients": {
            "Q03": {"trigger_steps": 2, "new_spo2": 84.0,
                    "new_mental_status": "confused", "escalate_to_esi": 1}
        }
    },

    {
        "scenario_id": "H002",
        "description": "Chemical spill. 6 patients. 1 resus bay, 2 monitored beds, 3 exam rooms.",
        "patients": [
            _p("R01", 40, "M", "chemical splash to eyes and face, severe pain",
               128, 82, 92, 20, 97.0, 36.8, 9, "alert", "ambulance"),
            _p("R02", 35, "F", "inhaled fumes, coughing, slight wheeze",
               122, 78, 96, 22, 95.0, 37.0, 5, "alert", "ambulance"),
            _p("R03", 50, "M", "large chemical burn to arm and torso, 20% BSA",
               100, 64, 116, 26, 94.0, 37.2, 10, "alert", "ambulance"),
            _p("R04", 28, "F", "minor skin irritation, no inhalation",
               118, 76, 78, 16, 99.0, 36.7, 2, "alert", "walk-in"),
            _p("R05", 60, "M", "severe inhalation, stridor, drooling",
               96, 62, 124, 34, 87.0, 37.1, 8, "confused", "ambulance",
               history="COPD"),
            _p("R06", 22, "F", "anxiety, hyperventilating, no chemical contact",
               130, 84, 108, 32, 99.0, 36.8, 4, "alert", "walk-in"),
        ],
        "available_resources": {
            "resuscitation_bay": 1, "monitored_bed": 2,
            "exam_room": 3, "waiting_room": 999
        },
        "ground_truth": [
            {"patient_id": "R01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["eye injury", "chemical", "irrigation",
                             "ophthalmology", "urgent", "vision"]},
            {"patient_id": "R02", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["inhalation", "wheeze", "fumes",
                             "monitoring", "multiple resources", "respiratory"]},
            {"patient_id": "R03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["burns", "BSA", "fluid resuscitation",
                             "parkland", "chemical burn", "urgent"]},
            {"patient_id": "R04", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["minor", "skin irritation", "stable",
                             "one resource", "decontamination"]},
            {"patient_id": "R05", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["stridor", "airway compromise", "COPD",
                             "immediate", "intubation", "drooling"]},
            {"patient_id": "R06", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["panic attack", "hyperventilation", "no exposure",
                             "stable", "anxiety", "one resource"]},
        ],
        "deteriorating_patients": {
            "R02": {"trigger_steps": 2, "new_spo2": 90.0,
                    "new_mental_status": "confused", "escalate_to_esi": 2}
        }
    },

    {
        "scenario_id": "H003",
        "description": "Night shift surge. 8 patients. 2 resus, 3 monitored, 4 exam rooms.",
        "patients": [
            _p("S01", 58, "M", "crushing substernal pain, diaphoretic, vomiting",
               84, 54, 120, 26, 92.0, 36.8, 10, "confused", "ambulance",
               history="CABG 3 years ago"),
            _p("S02", 33, "F", "seizure in waiting room, post-ictal now",
               118, 76, 98, 18, 96.0, 36.9, 3, "confused", "walk-in",
               history="epilepsy"),
            _p("S03", 72, "M", "slurred speech, right arm drift, onset 45 mins ago",
               162, 96, 86, 18, 97.0, 37.0, 2, "confused", "ambulance",
               history="atrial fibrillation, warfarin"),
            _p("S04", 25, "F", "UTI symptoms, burning urination, no fever",
               116, 74, 76, 14, 99.0, 36.6, 3, "alert", "walk-in"),
            _p("S05", 80, "F", "found on floor, unable to get up, hip pain",
               132, 84, 92, 18, 97.0, 36.8, 7, "confused", "ambulance",
               history="dementia, warfarin"),
            _p("S06", 15, "M", "asthma attack, not improving with own inhaler",
               128, 82, 118, 30, 90.0, 37.0, 6, "alert", "walk-in",
               history="moderate persistent asthma"),
            _p("S07", 42, "M", "dental pain, swollen jaw, difficulty opening mouth",
               126, 80, 88, 18, 98.0, 37.8, 7, "alert", "walk-in"),
            _p("S08", 65, "F", "nausea and vomiting 2 days, mild dehydration",
               118, 76, 92, 18, 98.0, 37.3, 4, "alert", "walk-in",
               history="hypertension"),
        ],
        "available_resources": {
            "resuscitation_bay": 2, "monitored_bed": 3,
            "exam_room": 4, "waiting_room": 999
        },
        "ground_truth": [
            {"patient_id": "S01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["STEMI", "cardiac", "chest pain", "shock",
                             "immediate", "diaphoretic", "cabg"]},
            {"patient_id": "S02", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["seizure", "post-ictal", "confused",
                             "epilepsy", "monitoring", "urgent"]},
            {"patient_id": "S03", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["stroke", "tPA", "arm drift", "slurred speech",
                             "atrial fibrillation", "time critical", "immediate"]},
            {"patient_id": "S04", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["UTI", "stable", "urine", "one resource",
                             "antibiotic", "urinalysis"]},
            {"patient_id": "S05", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["hip fracture", "anticoagulant", "warfarin",
                             "elderly", "confused", "fall risk"]},
            {"patient_id": "S06", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["asthma", "spo2", "nebuliser", "not improving",
                             "respiratory", "high risk", "pediatric"]},
            {"patient_id": "S07", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["dental abscess", "Ludwig", "airway risk",
                             "trismus", "imaging", "multiple resources"]},
            {"patient_id": "S08", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["dehydration", "vomiting", "iv fluids",
                             "labs", "stable", "multiple resources"]},
        ],
        "deteriorating_patients": {
            "S06": {"trigger_steps": 2, "new_spo2": 85.0,
                    "new_mental_status": "confused", "escalate_to_esi": 1}
        }
    },

    {
        "scenario_id": "H004",
        "description": "Paediatric surge. 6 patients. 1 resus bay, 2 monitored, 3 exam rooms.",
        "patients": [
            _p("T01", 2, "M", "fever 41.2°C, stiff neck, purple rash spreading",
               80, 52, 160, 46, 94.0, 41.2, 9, "confused", "ambulance"),
            _p("T02", 7, "F", "broken arm after fall from bike, deformity visible",
               112, 72, 96, 20, 99.0, 36.8, 7, "alert", "walk-in"),
            _p("T03", 4, "M", "choking episode, object partially cleared, mild stridor",
               108, 68, 124, 32, 93.0, 37.0, 6, "alert", "ambulance"),
            _p("T04", 12, "F", "suspected appendicitis, RLQ pain worsening 6 hours",
               124, 80, 94, 20, 98.0, 38.1, 7, "alert", "walk-in"),
            _p("T05", 9, "M", "mild asthma, 2 puffs salbutamol, now comfortable",
               114, 72, 86, 18, 97.0, 36.9, 2, "alert", "walk-in",
               history="mild asthma"),
            _p("T06", 6, "F", "rash over torso after antibiotic, no swelling, breathing fine",
               110, 70, 82, 16, 99.0, 36.7, 1, "alert", "walk-in"),
        ],
        "available_resources": {
            "resuscitation_bay": 1, "monitored_bed": 2,
            "exam_room": 3, "waiting_room": 999
        },
        "ground_truth": [
            {"patient_id": "T01", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["meningococcal", "purpura", "meningitis",
                             "immediate", "infant", "fever", "rash"]},
            {"patient_id": "T02", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["fracture", "x-ray", "deformity",
                             "stable", "multiple resources", "pediatric"]},
            {"patient_id": "T03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["foreign body", "airway", "stridor",
                             "choking", "urgent", "partial obstruction"]},
            {"patient_id": "T04", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["appendicitis", "surgical", "RLQ",
                             "worsening", "labs", "imaging", "urgent"]},
            {"patient_id": "T05", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["asthma", "controlled", "salbutamol",
                             "monitoring", "stable", "multiple resources"]},
            {"patient_id": "T06", "esi_level": 4, "resource": "exam_room",
             "key_signals": ["drug rash", "mild", "antibiotic",
                             "stable", "one resource", "no anaphylaxis"]},
        ],
        "deteriorating_patients": {}
    },

    {
        "scenario_id": "H005",
        "description": "Psychiatric + medical mix. 6 patients. 2 resus, 2 monitored, 3 exam rooms.",
        "patients": [
            _p("U01", 27, "F", "overdose on paracetamol, 20 tablets 3 hours ago",
               118, 76, 96, 18, 98.0, 36.8, 4, "alert", "walk-in",
               history="depression"),
            _p("U02", 68, "M", "acute confusion, fever 39.8°C, productive cough",
               96, 62, 108, 28, 88.0, 39.8, 5, "confused", "ambulance",
               history="COPD, hypertension"),
            _p("U03", 35, "M", "agitated, threatening staff, unknown substance use",
               148, 92, 132, 22, 99.0, 37.4, 3, "confused", "police"),
            _p("U04", 55, "F", "severe bone pain, unable to walk",
               124, 80, 88, 18, 96.0, 37.1, 9, "alert", "ambulance",
               history="metastatic breast cancer"),
            _p("U05", 22, "M", "suicidal ideation, calm, no immediate plan",
               118, 76, 74, 14, 99.0, 36.6, 0, "alert", "walk-in"),
            _p("U06", 80, "M", "septic shock, temp 40.1, BP critically low",
               74, 44, 136, 32, 90.0, 40.1, 6, "confused", "ambulance",
               history="recent UTI, diabetes"),
        ],
        "available_resources": {
            "resuscitation_bay": 2, "monitored_bed": 2,
            "exam_room": 3, "waiting_room": 999
        },
        "ground_truth": [
            {"patient_id": "U01", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["paracetamol overdose", "N-acetylcysteine",
                             "toxicology", "liver", "time critical", "NAC"]},
            {"patient_id": "U02", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["pneumonia", "COPD", "spo2", "sepsis",
                             "confusion", "fever", "urgent"]},
            {"patient_id": "U03", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["agitation", "substance", "intoxication",
                             "safety", "monitoring", "psychiatric"]},
            {"patient_id": "U04", "esi_level": 3, "resource": "exam_room",
             "key_signals": ["cancer", "bone pain", "palliative",
                             "pain management", "stable", "multiple resources"]},
            {"patient_id": "U05", "esi_level": 2, "resource": "monitored_bed",
             "key_signals": ["suicidal", "psychiatric", "safety",
                             "monitoring", "risk assessment", "mental health"]},
            {"patient_id": "U06", "esi_level": 1, "resource": "resuscitation_bay",
             "key_signals": ["septic shock", "vasopressor", "critical",
                             "low bp", "immediate", "fluids", "resuscitation"]},
        ],
        "deteriorating_patients": {
            "U02": {"trigger_steps": 2, "new_spo2": 82.0,
                    "new_mental_status": "unresponsive", "escalate_to_esi": 1}
        }
    },

]


# ---------------------------------------------------------------------------
# Public API — used by server/environment.py
# ---------------------------------------------------------------------------

SCENARIO_POOLS = {
    "easy":   EASY_SCENARIOS,
    "medium": MEDIUM_SCENARIOS,
    "hard":   HARD_SCENARIOS,
}


def sample_scenario(task_level: str, exclude_id: str = None) -> dict:
    pool = SCENARIO_POOLS[task_level]
    available = [s for s in pool if s["scenario_id"] != exclude_id]
    if not available:
        available = pool
    return random.choice(available)


def get_scenario_by_id(scenario_id: str) -> dict:
    for pool in SCENARIO_POOLS.values():
        for scenario in pool:
            if scenario["scenario_id"] == scenario_id:
                return scenario
    raise ValueError(f"Scenario '{scenario_id}' not found")


def get_default_resources(task_level: str, scenario: dict) -> dict:
    if "available_resources" in scenario:
        return scenario["available_resources"]
    return {
        "resuscitation_bay": 2,
        "monitored_bed":     3,
        "exam_room":         4,
        "waiting_room":      999,
    }