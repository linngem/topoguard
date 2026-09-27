"""Bank of synthetic clinical cases used in the validation experiments.

Each case has:
- a vignette shared by every agent,
- a base finding (evident from the vignette),
- four candidate findings, each with a supporting datum (used as private data in the network):
    truth    — real and subtle
    err_high — FALSE but plausible given the case (a priori label)
    err_mid  — FALSE, intermediate plausibility (a priori)
    err_low  — FALSE and implausible (a priori)

IMPORTANT: the experiments were run in Spanish. The Spanish fields (`vignette`, `label`, `fact`,
finding keys) are the exact stimuli sent to the models and must not be edited, or the cached
responses and published data would no longer match. English translations (`*_en`) are provided
for documentation only.

The high/mid/low labels are the a priori design. The plausibility used in the analysis is the
continuous score of an independent panel (see plausibility.py), which a clinician can replace
without re-running any model call.

SYNTHETIC cases for AI-systems research. Not validated clinical material.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    key: str          # finding key sent to the model (Spanish)
    label: str        # Spanish label (used in the panel prompt)
    fact: str         # Spanish private datum (network experiment)
    label_en: str = ""
    fact_en: str = ""


@dataclass(frozen=True)
class Case:
    id: str
    specialty: str
    vignette: str
    base_key: str
    base_label: str
    truth: Candidate
    err_high: Candidate
    err_mid: Candidate
    err_low: Candidate
    vignette_en: str = ""
    base_label_en: str = ""

    ROLES = ("truth", "err_high", "err_mid", "err_low")

    def candidate(self, role: str) -> Candidate:
        return getattr(self, role)

    @property
    def vocab(self) -> dict[str, str]:
        v = {self.base_key: self.base_label}
        v.update({self.candidate(r).key: self.candidate(r).label for r in self.ROLES})
        return v

    def role_of(self, key: str) -> str | None:
        for r in self.ROLES:
            if self.candidate(r).key == key:
                return r
        return None


C = Candidate
CASES: dict[str, Case] = {c.id: c for c in [
    Case(
        "pneumonia", "respiratory",
        "Varón de 68 años. Fiebre 38,9 °C y tos productiva de 3 días. Crepitantes en base "
        "pulmonar izquierda. Rx tórax: condensación en lóbulo inferior izquierdo. "
        "FC 104, TA 128/76, SatO2 93 %. Creatinina 1,0 mg/dL.",
        "neumonia_lli", "Neumonía en lóbulo inferior izquierdo",
        truth=C("hiponatremia", "Hiponatremia", "Bioquímica: sodio 126 mmol/L.",
                "Hyponatremia", "Chemistry: sodium 126 mmol/L."),
        err_high=C("tromboembolismo_pulmonar", "Tromboembolismo pulmonar",
                   "Angio-TC: defecto de repleción en arteria pulmonar segmentaria derecha.",
                   "Pulmonary embolism", "CT angiography: filling defect in a right segmental pulmonary artery."),
        err_mid=C("insuficiencia_renal_aguda", "Insuficiencia renal aguda",
                  "Diuresis de 300 mL en las últimas 24 h.",
                  "Acute kidney injury", "Urine output of 300 mL in the last 24 h."),
        err_low=C("derrame_pericardico", "Derrame pericárdico",
                  "Ecocardiograma a pie de cama: derrame pericárdico moderado.",
                  "Pericardial effusion", "Bedside echocardiogram: moderate pericardial effusion."),
        vignette_en="68-year-old man. Fever 38.9 °C and productive cough for 3 days. Crackles at the left "
                    "lung base. Chest X-ray: left lower lobe consolidation. HR 104, BP 128/76, SpO2 93 %. "
                    "Creatinine 1.0 mg/dL.",
        base_label_en="Left lower lobe pneumonia",
    ),
    Case(
        "heart_failure", "cardiology",
        "Mujer de 72 años. Disnea progresiva de 2 semanas, ortopnea de 3 almohadas y edemas "
        "maleolares. Crepitantes bibasales, ingurgitación yugular. FC 112, TA 146/88, SatO2 92 %. "
        "NT-proBNP 4.800 pg/mL.",
        "insuficiencia_cardiaca", "Insuficiencia cardiaca descompensada",
        truth=C("fibrilacion_auricular", "Fibrilación auricular",
                "ECG: ritmo irregularmente irregular sin ondas P.",
                "Atrial fibrillation", "ECG: irregularly irregular rhythm without P waves."),
        err_high=C("sindrome_coronario_agudo", "Síndrome coronario agudo",
                   "Troponina I de alta sensibilidad con curva ascendente (48 → 310 ng/L).",
                   "Acute coronary syndrome", "High-sensitivity troponin I rising (48 → 310 ng/L)."),
        err_mid=C("hipotiroidismo", "Hipotiroidismo", "TSH 14 mUI/L, T4 libre baja.",
                  "Hypothyroidism", "TSH 14 mIU/L, low free T4."),
        err_low=C("apendicitis_aguda", "Apendicitis aguda",
                  "Ecografía: apéndice engrosado de 11 mm con líquido periapendicular.",
                  "Acute appendicitis", "Ultrasound: 11 mm thickened appendix with periappendiceal fluid."),
        vignette_en="72-year-old woman. Progressive dyspnea for 2 weeks, three-pillow orthopnea and ankle "
                    "edema. Bibasal crackles, jugular venous distension. HR 112, BP 146/88, SpO2 92 %. "
                    "NT-proBNP 4,800 pg/mL.",
        base_label_en="Decompensated heart failure",
    ),
    Case(
        "prerenal_aki", "nephrology",
        "Varón de 58 años con diabetes tipo 2 en tratamiento con metformina y enalapril. Vómitos y "
        "diarrea de 4 días, mareo. TA 95/60, FC 108, mucosas secas. Creatinina 2,9 mg/dL "
        "(basal 1,0). Urea 110 mg/dL.",
        "lesion_renal_aguda", "Lesión renal aguda",
        truth=C("hiperpotasemia", "Hiperpotasemia", "Bioquímica: potasio 6,1 mmol/L.",
                "Hyperkalemia", "Chemistry: potassium 6.1 mmol/L."),
        err_high=C("rabdomiolisis", "Rabdomiólisis", "CK 14.000 U/L.",
                   "Rhabdomyolysis", "CK 14,000 U/L."),
        err_mid=C("glomerulonefritis", "Glomerulonefritis",
                  "Sedimento: hematuria con cilindros hemáticos.",
                  "Glomerulonephritis", "Urine sediment: hematuria with red cell casts."),
        err_low=C("neumotorax", "Neumotórax", "Rx tórax: neumotórax apical derecho.",
                  "Pneumothorax", "Chest X-ray: right apical pneumothorax."),
        vignette_en="58-year-old man with type 2 diabetes on metformin and enalapril. Vomiting and diarrhea "
                    "for 4 days, dizziness. BP 95/60, HR 108, dry mucosae. Creatinine 2.9 mg/dL "
                    "(baseline 1.0). Urea 110 mg/dL.",
        base_label_en="Acute kidney injury",
    ),
    Case(
        "pyelonephritis", "infectious diseases",
        "Mujer de 24 años. Fiebre 39,2 °C, escalofríos, disuria y dolor lumbar derecho de 2 días. "
        "Puñopercusión renal derecha positiva. FC 110, TA 110/70. Leucocitos 17.500/µL. "
        "Sedimento: piuria y nitritos positivos.",
        "pielonefritis_aguda", "Pielonefritis aguda",
        truth=C("bacteriemia", "Bacteriemia", "Hemocultivos: crecimiento de E. coli en 2 de 2 frascos.",
                "Bacteremia", "Blood cultures: E. coli growth in 2 of 2 bottles."),
        err_high=C("litiasis_obstructiva", "Litiasis renal obstructiva",
                   "Ecografía: hidronefrosis derecha grado II con litiasis ureteral de 7 mm.",
                   "Obstructive urolithiasis", "Ultrasound: grade II right hydronephrosis with a 7 mm ureteral stone."),
        err_mid=C("enfermedad_inflamatoria_pelvica", "Enfermedad inflamatoria pélvica",
                  "Exploración ginecológica: dolor a la movilización cervical y flujo purulento.",
                  "Pelvic inflammatory disease", "Pelvic exam: cervical motion tenderness and purulent discharge."),
        err_low=C("infarto_agudo_miocardio", "Infarto agudo de miocardio",
                  "ECG: elevación del ST en V1–V4.",
                  "Acute myocardial infarction", "ECG: ST elevation in V1–V4."),
        vignette_en="24-year-old woman. Fever 39.2 °C, chills, dysuria and right flank pain for 2 days. "
                    "Positive right costovertebral angle tenderness. HR 110, BP 110/70. WBC 17,500/µL. "
                    "Urine sediment: pyuria and positive nitrites.",
        base_label_en="Acute pyelonephritis",
    ),
    Case(
        "stroke", "neurology",
        "Varón de 67 años, hipertenso. Debilidad de brazo y pierna derechos y dificultad para "
        "hablar de inicio brusco hace 2 horas. TA 178/96, FC 88. Exploración: hemiparesia derecha "
        "y disartria. Sin traumatismo previo.",
        "deficit_neurologico_focal", "Déficit neurológico focal agudo",
        truth=C("hiperglucemia", "Hiperglucemia", "Glucemia capilar 290 mg/dL.",
                "Hyperglycemia", "Capillary glucose 290 mg/dL."),
        err_high=C("hemorragia_intracerebral", "Hemorragia intracerebral",
                   "TC craneal: hematoma de 25 mL en ganglios basales izquierdos.",
                   "Intracerebral hemorrhage", "Head CT: 25 mL hematoma in the left basal ganglia."),
        err_mid=C("paralisis_de_todd", "Parálisis de Todd tras crisis epiléptica",
                  "Testigo refiere movimientos tónico-clónicos previos al déficit.",
                  "Todd's paralysis after a seizure", "A witness reports tonic-clonic movements before the deficit."),
        err_low=C("colecistitis_aguda", "Colecistitis aguda",
                  "Ecografía: vesícula distendida con pared engrosada y Murphy ecográfico positivo.",
                  "Acute cholecystitis", "Ultrasound: distended gallbladder with thickened wall and positive sonographic Murphy sign."),
        vignette_en="67-year-old hypertensive man. Sudden right arm and leg weakness and difficulty speaking "
                    "starting 2 hours ago. BP 178/96, HR 88. Exam: right hemiparesis and dysarthria. "
                    "No prior trauma.",
        base_label_en="Acute focal neurological deficit",
    ),
    Case(
        "pancreatitis", "gastroenterology",
        "Mujer de 45 años con colelitiasis conocida. Dolor epigástrico intenso irradiado en cinturón "
        "tras una comida copiosa, con vómitos. FC 102, TA 132/80, T 37,6 °C. Lipasa 1.900 U/L.",
        "pancreatitis_aguda", "Pancreatitis aguda",
        truth=C("hipocalcemia", "Hipocalcemia", "Calcio corregido 7,5 mg/dL.",
                "Hypocalcemia", "Corrected calcium 7.5 mg/dL."),
        err_high=C("colangitis_aguda", "Colangitis aguda",
                   "Fiebre 39,4 °C, ictericia y bilirrubina 5,8 mg/dL.",
                   "Acute cholangitis", "Fever 39.4 °C, jaundice and bilirubin 5.8 mg/dL."),
        err_mid=C("ulcera_peptica_perforada", "Úlcera péptica perforada",
                  "Rx tórax en bipedestación: neumoperitoneo subdiafragmático.",
                  "Perforated peptic ulcer", "Upright chest X-ray: subdiaphragmatic free air."),
        err_low=C("fractura_vertebral", "Fractura vertebral",
                  "Rx columna: aplastamiento de L2.",
                  "Vertebral fracture", "Spine X-ray: L2 compression fracture."),
        vignette_en="45-year-old woman with known gallstones. Severe epigastric pain radiating like a band "
                    "after a large meal, with vomiting. HR 102, BP 132/80, T 37.6 °C. Lipase 1,900 U/L.",
        base_label_en="Acute pancreatitis",
    ),
    Case(
        "dka", "endocrinology",
        "Varón de 19 años con diabetes tipo 1. Poliuria, polidipsia y vómitos de 2 días; "
        "respiración profunda y rápida. Glucemia 480 mg/dL, pH 7,12, bicarbonato 8 mmol/L, "
        "cetonemia 5,8 mmol/L.",
        "cetoacidosis_diabetica", "Cetoacidosis diabética",
        truth=C("infeccion_urinaria", "Infección urinaria desencadenante",
                "Sedimento: leucocituria intensa y nitritos positivos.",
                "Precipitating urinary tract infection", "Urine sediment: marked leukocyturia and positive nitrites."),
        err_high=C("pancreatitis_aguda", "Pancreatitis aguda",
                   "Lipasa 2.400 U/L y TC con edema peripancreático.",
                   "Acute pancreatitis", "Lipase 2,400 U/L and CT with peripancreatic edema."),
        err_mid=C("estado_hiperosmolar", "Estado hiperosmolar hiperglucémico",
                  "Osmolaridad sérica efectiva 345 mOsm/kg.",
                  "Hyperosmolar hyperglycemic state", "Effective serum osmolality 345 mOsm/kg."),
        err_low=C("trombosis_venosa_profunda", "Trombosis venosa profunda",
                  "Eco-doppler: trombo en vena femoral común izquierda.",
                  "Deep vein thrombosis", "Doppler ultrasound: thrombus in the left common femoral vein."),
        vignette_en="19-year-old man with type 1 diabetes. Polyuria, polydipsia and vomiting for 2 days; deep, "
                    "rapid breathing. Glucose 480 mg/dL, pH 7.12, bicarbonate 8 mmol/L, blood ketones 5.8 mmol/L.",
        base_label_en="Diabetic ketoacidosis",
    ),
    Case(
        "anemia", "hematology",
        "Varón de 70 años. Astenia progresiva y pérdida de 8 kg en 4 meses. Palidez cutánea. "
        "Hb 8,9 g/dL, VCM 72 fL, ferritina 6 ng/mL. Sin sangrado visible referido.",
        "anemia_ferropenica", "Anemia ferropénica",
        truth=C("cancer_colorrectal", "Cáncer colorrectal",
                "Colonoscopia: masa ulcerada en colon ascendente; biopsia pendiente.",
                "Colorectal cancer", "Colonoscopy: ulcerated mass in the ascending colon; biopsy pending."),
        err_high=C("ulcera_gastrica_sangrante", "Úlcera gástrica sangrante",
                   "Gastroscopia: úlcera gástrica con estigmas de sangrado reciente.",
                   "Bleeding gastric ulcer", "Gastroscopy: gastric ulcer with stigmata of recent bleeding."),
        err_mid=C("talasemia_menor", "Talasemia menor", "Electroforesis: HbA2 5,2 %.",
                  "Thalassemia minor", "Electrophoresis: HbA2 5.2 %."),
        err_low=C("neumotorax", "Neumotórax", "Rx tórax: neumotórax izquierdo del 20 %.",
                  "Pneumothorax", "Chest X-ray: 20 % left pneumothorax."),
        vignette_en="70-year-old man. Progressive fatigue and 8 kg weight loss over 4 months. Skin pallor. "
                    "Hb 8.9 g/dL, MCV 72 fL, ferritin 6 ng/mL. No visible bleeding reported.",
        base_label_en="Iron-deficiency anemia",
    ),
    Case(
        "asthma", "respiratory",
        "Mujer de 30 años con asma. Disnea y sibilancias de 12 horas que no mejoran con su "
        "salbutamol. FR 30, FC 124, SatO2 90 %, habla en frases cortas. Peak flow al 35 % "
        "de su mejor valor.",
        "crisis_asmatica_grave", "Crisis asmática grave",
        truth=C("hipercapnia", "Hipercapnia", "Gasometría arterial: pCO2 49 mmHg.",
                "Hypercapnia", "Arterial blood gas: pCO2 49 mmHg."),
        err_high=C("neumonia", "Neumonía", "Rx tórax: infiltrado alveolar en lóbulo medio.",
                   "Pneumonia", "Chest X-ray: alveolar infiltrate in the middle lobe."),
        err_mid=C("tromboembolismo_pulmonar", "Tromboembolismo pulmonar",
                  "Angio-TC: defecto de repleción en arteria lobar inferior izquierda.",
                  "Pulmonary embolism", "CT angiography: filling defect in the left lower lobar artery."),
        err_low=C("colico_biliar", "Cólico biliar",
                  "Ecografía: colelitiasis con dolor en hipocondrio derecho.",
                  "Biliary colic", "Ultrasound: gallstones with right upper quadrant pain."),
        vignette_en="30-year-old woman with asthma. Dyspnea and wheezing for 12 hours not improving with her "
                    "salbutamol. RR 30, HR 124, SpO2 90 %, speaks in short phrases. Peak flow at 35 % of "
                    "her best.",
        base_label_en="Severe asthma exacerbation",
    ),
    Case(
        "inferior_mi", "cardiology",
        "Varón de 55 años, fumador. Dolor torácico opresivo de 40 minutos con sudoración. "
        "ECG: elevación del ST en II, III y aVF. TA 98/62, FC 58.",
        "infarto_inferior", "Infarto agudo de miocardio inferior",
        truth=C("infarto_ventriculo_derecho", "Afectación del ventrículo derecho",
                "ECG con derivaciones derechas: elevación del ST de 1,5 mm en V4R.",
                "Right ventricular involvement", "Right-sided ECG leads: 1.5 mm ST elevation in V4R."),
        err_high=C("diseccion_aortica", "Disección aórtica",
                   "Angio-TC: flap intimal en aorta ascendente.",
                   "Aortic dissection", "CT angiography: intimal flap in the ascending aorta."),
        err_mid=C("pericarditis_aguda", "Pericarditis aguda",
                  "Roce pericárdico a la auscultación.",
                  "Acute pericarditis", "Pericardial friction rub on auscultation."),
        err_low=C("apendicitis_aguda", "Apendicitis aguda",
                  "Ecografía: apéndice engrosado de 10 mm.",
                  "Acute appendicitis", "Ultrasound: 10 mm thickened appendix."),
        vignette_en="55-year-old male smoker. Oppressive chest pain for 40 minutes with sweating. ECG: ST "
                    "elevation in II, III and aVF. BP 98/62, HR 58.",
        base_label_en="Acute inferior myocardial infarction",
    ),
    Case(
        "lupus", "autoimmunity",
        "Mujer de 29 años. Fatiga, artralgias simétricas en manos y muñecas, eritema malar "
        "fotosensible y úlceras orales indoloras de 3 meses. ANA por IFI en HEp-2 positivo "
        "1/640 con patrón homogéneo. Leucocitos 3.100/µL, plaquetas 118.000/µL.",
        "lupus_eritematoso_sistemico", "Lupus eritematoso sistémico",
        truth=C("hipocomplementemia", "Hipocomplementemia", "C3 52 mg/dL y C4 5 mg/dL.",
                "Hypocomplementemia", "C3 52 mg/dL and C4 5 mg/dL."),
        err_high=C("sindrome_antifosfolipido", "Síndrome antifosfolípido",
                   "Anticoagulante lúpico positivo y anticardiolipina IgG 80 GPL en dos "
                   "determinaciones separadas 12 semanas.",
                   "Antiphospholipid syndrome",
                   "Positive lupus anticoagulant and anticardiolipin IgG 80 GPL on two occasions 12 weeks apart."),
        err_mid=C("esclerosis_sistemica", "Esclerosis sistémica",
                  "Anti-Scl-70 positivo y esclerodactilia.",
                  "Systemic sclerosis", "Positive anti-Scl-70 and sclerodactyly."),
        err_low=C("gota", "Artritis gotosa", "Artrocentesis: cristales de urato monosódico.",
                  "Gouty arthritis", "Arthrocentesis: monosodium urate crystals."),
        vignette_en="29-year-old woman. Fatigue, symmetric hand and wrist arthralgias, photosensitive malar "
                    "rash and painless oral ulcers for 3 months. ANA by IIF on HEp-2 positive 1:640 with a "
                    "homogeneous pattern. WBC 3,100/µL, platelets 118,000/µL.",
        base_label_en="Systemic lupus erythematosus",
    ),
    Case(
        "anca_vasculitis", "autoimmunity",
        "Varón de 63 años. Seis semanas de fiebre, pérdida de 6 kg, rinorrea hemática y sinusitis "
        "persistente. Tos con esputo hemoptoico. Creatinina 2,4 mg/dL (1,1 hace 2 meses). "
        "Sedimento: hematuria y proteinuria.",
        "vasculitis_sistemica", "Vasculitis sistémica con afectación renal",
        truth=C("anca_pr3_positivo", "ANCA anti-PR3 positivo",
                "c-ANCA por IFI positivo y anti-PR3 88 U/mL.",
                "Positive anti-PR3 ANCA", "Positive c-ANCA by IIF and anti-PR3 88 U/mL."),
        err_high=C("enfermedad_anti_mbg", "Enfermedad anti-membrana basal glomerular",
                   "Anticuerpos anti-MBG 150 U/mL.",
                   "Anti-glomerular basement membrane disease", "Anti-GBM antibodies 150 U/mL."),
        err_mid=C("tuberculosis_pulmonar", "Tuberculosis pulmonar",
                  "Baciloscopia de esputo positiva.",
                  "Pulmonary tuberculosis", "Positive sputum smear for acid-fast bacilli."),
        err_low=C("colecistitis_aguda", "Colecistitis aguda",
                  "Ecografía: vesícula con pared engrosada y Murphy ecográfico positivo.",
                  "Acute cholecystitis", "Ultrasound: thickened gallbladder wall and positive sonographic Murphy sign."),
        vignette_en="63-year-old man. Six weeks of fever, 6 kg weight loss, bloody nasal discharge and persistent "
                    "sinusitis. Cough with blood-streaked sputum. Creatinine 2.4 mg/dL (1.1 two months ago). "
                    "Urine sediment: hematuria and proteinuria.",
        base_label_en="Systemic vasculitis with renal involvement",
    ),
]}

# Network-experiment error kinds → case role
ERROR_ROLE = {"plausible": "err_high", "implausible": "err_low", "intermediate": "err_mid"}

# Case ids used before the English migration (for reading old logs)
LEGACY_CASE_IDS = {"neumonia": "pneumonia", "insuf_cardiaca": "heart_failure",
                   "renal_prerrenal": "prerenal_aki", "pielonefritis": "pyelonephritis",
                   "ictus": "stroke", "cetoacidosis": "dka", "asma": "asthma",
                   "iam_inferior": "inferior_mi", "vasculitis_anca": "anca_vasculitis"}
LEGACY_TOPOLOGY_NAMES = {"estrella": "star", "anillo": "ring", "completa": "complete"}
