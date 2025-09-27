class Cols:
    id = "id"
    age_diagnose = "age_diagnose"
    sex = "sex"
    hunt_hess = "hunt_hess"
    fisher = "fisher"
    cns_infection = "cns_infection"
    EVD = "EVD"
    aneurysm = "aneurysm"
    aneurysm_diameter = "aneurysm_diameter"
    aneurysm_height = "aneurysm_height"
    aneurysm_location = "aneurysm_location"
    aneurysm_location_cat = "aneurysm_location_cat"
    aneurysm_treatment = "aneurysm_treatment"
    vasospasm = "vasospasm"
    vasospasm_severity = "vasospasm_severity"


# Features for vasospasm classification after processing
vasospasm_features = [
    Cols.age_diagnose,
    Cols.sex,
    Cols.hunt_hess,
    Cols.fisher,
    Cols.cns_infection,
    Cols.EVD,
    Cols.aneurysm,
    # Cols.aneurysm_location,
    Cols.aneurysm_location_cat,
    Cols.aneurysm_diameter,
    Cols.aneurysm_height,
    Cols.aneurysm_treatment,
]
