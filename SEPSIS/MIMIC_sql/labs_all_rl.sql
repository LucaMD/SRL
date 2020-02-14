-- This query pivots lab values taken since 2 days before the patient's ICU stay
-- Have already confirmed that the unit of measurement is always the same: null or the correct unit
set search_path to mimiciii;
DROP MATERIALIZED VIEW IF EXISTS labs_all_rl CASCADE;
CREATE materialized VIEW labs_all_rl AS
SELECT
  pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, pvt.label as lab_id, pvt.valuenum
FROM
( -- begin query that extracts the data
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
  , CASE

        WHEN itemid = 50818 THEN 'PACO2'
        WHEN itemid = 50821 THEN 'PAO2'
        WHEN itemid = 50868 THEN 'ANION GAP'
        WHEN itemid = 50862 THEN 'ALBUMIN'
        WHEN itemid = 51144 THEN 'BANDS'
        WHEN itemid = 50882 THEN 'BICARBONATE'
        WHEN itemid = 50885 THEN 'BILIRUBIN'
        WHEN itemid = 50912 THEN 'CREATININE'
        WHEN itemid = 50806 THEN 'CHLORIDE'
        WHEN itemid = 50902 THEN 'CHLORIDE'
        WHEN itemid = 50809 THEN 'GLUCOSE'
        WHEN itemid = 50931 THEN 'GLUCOSE'
        WHEN itemid = 50811 THEN 'HEMOGLOBIN'
        WHEN itemid = 51222 THEN 'HEMOGLOBIN'
        WHEN itemid = 50813 THEN 'LACTATE'
        WHEN itemid = 51265 THEN 'PLATELET'
        WHEN itemid = 50822 THEN 'POTASSIUM'
        WHEN itemid = 50971 THEN 'POTASSIUM'
        WHEN itemid = 51275 THEN 'PTT'
        WHEN itemid = 51274 THEN 'PT'
        WHEN itemid = 50824 THEN 'SODIUM'
        WHEN itemid = 50983 THEN 'SODIUM'
        WHEN itemid = 51006 THEN 'BUN'
        WHEN itemid = 51300 THEN 'WBC'
        WHEN itemid = 51301 THEN 'WBC'
        WHEN ITEMID = 50820 THEN 'PH'
        WHEN ITEMID = 50808 THEN 'ION_CALCIUM'
        WHEN ITEMID = 50889 THEN 'CRP'
        WHEN ITEMID = 50861 THEN 'ALAT'
        WHEN ITEMID = 50878 THEN 'ASAT'
        WHEN ITEMID = 50960 THEN 'MAGNESIUM'
        WHEN ITEMID = 50802 THEN 'BaseExcess'
        WHEN ITEMID = 50893 THEN 'CALCIUM'

      ELSE null
    END AS label
  , -- add in some sanity checks on the values
  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
    CASE
      WHEN ITEMID = 50802 THEN Valuenum -- Base Excess blood gas 
      WHEN itemid = 50818 and valuenum <     0 THEN null -- PaCO2, added by Stephen for BMI 215
      WHEN itemid = 50821 and valuenum <     0 THEN null -- mmHg 'PaO2' 
      WHEN ITEMID = 50861 AND valuenum <     0 THEN NULL -- ALAT
      WHEN ITEMID = 50878 AND valuenum <     0 THEN NULL -- ASAT
      WHEN itemid = 50862 and valuenum >    10 THEN null -- g/dL 'ALBUMIN'
      WHEN itemid = 50868 and valuenum > 10000 THEN null -- mEq/L 'ANION GAP'
      WHEN itemid = 51144 and valuenum <     0 THEN null -- immature band forms, %
      WHEN itemid = 51144 and valuenum >   100 THEN null -- immature band forms, %
      WHEN itemid = 50882 and valuenum > 10000 THEN null -- mEq/L 'BICARBONATE'
      WHEN itemid = 50885 and valuenum >   150 THEN null -- mg/dL 'BILIRUBIN'
      WHEN itemid = 50806 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
      WHEN itemid = 50902 and valuenum > 10000 THEN null -- mEq/L 'CHLORIDE'
      WHEN itemid = 50912 and valuenum >   150 THEN null -- mg/dL 'CREATININE'
      WHEN itemid = 50809 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
      WHEN itemid = 50931 and valuenum > 10000 THEN null -- mg/dL 'GLUCOSE'
      WHEN itemid = 50811 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
      WHEN itemid = 51222 and valuenum >    50 THEN null -- g/dL 'HEMOGLOBIN'
      WHEN itemid = 50813 and valuenum >    50 THEN null -- mmol/L 'LACTATE'
      WHEN itemid = 51265 and valuenum > 10000 THEN null -- K/uL 'PLATELET'
      WHEN itemid = 50822 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
      WHEN itemid = 50971 and valuenum >    30 THEN null -- mEq/L 'POTASSIUM'
      WHEN itemid = 51275 and valuenum >   150 THEN null -- sec 'PTT'
      WHEN itemid = 51274 and valuenum >   150 THEN null -- sec 'PT'
      WHEN itemid = 50824 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
      WHEN itemid = 50983 and valuenum >   200 THEN null -- mEq/L == mmol/L 'SODIUM'
      WHEN itemid = 51006 and valuenum >   300 THEN null -- 'BUN'
      WHEN itemid = 51300 and valuenum >  1000 THEN null -- 'WBC'
      WHEN itemid = 51301 and valuenum >  1000 THEN null -- 'WBC'
      WHEN ITEMID = 50820 and valuenum >    14 THEN null -- 'PH'
      WHEN ITEMID = 50808 and valuenum >   100 THEN null -- 'FREE_CALCIUM' mmol/L
      WHEN ITEMID = 50893 and Valuenum >   100 THEN null -- 'blood calcium mg/dl'
      WHEN ITEMID = 50960 and Valuenum >   100 then NULL --  'Magnesium' in mg/dl


    ELSE le.valuenum
    END AS valuenum

  FROM icustays ie

  LEFT JOIN labevents le
    ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
    AND le.charttime >= (ie.intime - interval '48' hour)
    AND le.ITEMID in
    (

      
      -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
      50802, -- Base excess
      50820, -- 'PH'
      50861, -- 'ALAT' i IU/L
      50878, -- 'ASAT' in IU/L
      50960, -- Magnesium in blood in mg/dl
      50808, -- 'FREE_CALCIUM' BLOOD GAS in mmol/L
      50893, -- calcium labevents: blood in mg/dl
      50821, -- PaO2 in labevents

      50868, -- ANION GAP | CHEMISTRY | BLOOD | 769895
      50862, -- ALBUMIN | CHEMISTRY | BLOOD | 146697
      51144, -- BANDS - hematology
      50882, -- BICARBONATE | CHEMISTRY | BLOOD | 780733
      50885, -- BILIRUBIN, TOTAL | CHEMISTRY | BLOOD | 238277
      50912, -- CREATININE | CHEMISTRY | BLOOD | 797476
      50902, -- CHLORIDE | CHEMISTRY | BLOOD | 795568
      50806, -- CHLORIDE, WHOLE BLOOD | BLOOD GAS | BLOOD | 48187
      50931, -- GLUCOSE | CHEMISTRY | BLOOD | 748981
      50809, -- GLUCOSE | BLOOD GAS | BLOOD | 196734
      51221, -- HEMATOCRIT | HEMATOLOGY | BLOOD | 881846
      50810, -- HEMATOCRIT, CALCULATED | BLOOD GAS | BLOOD | 89715
      51222, -- HEMOGLOBIN | HEMATOLOGY | BLOOD | 752523
      50811, -- HEMOGLOBIN | BLOOD GAS | BLOOD | 89712
      50813, -- LACTATE | BLOOD GAS | BLOOD | 187124
      51265, -- PLATELET COUNT | HEMATOLOGY | BLOOD | 778444
      50971, -- POTASSIUM | CHEMISTRY | BLOOD | 845825
      50822, -- POTASSIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 192946
      51275, -- PTT | HEMATOLOGY | BLOOD | 474937
      51237, -- INR(PT) | HEMATOLOGY | BLOOD | 471183
      51274, -- PT | HEMATOLOGY | BLOOD | 469090
      50983, -- SODIUM | CHEMISTRY | BLOOD | 808489
      50824, -- SODIUM, WHOLE BLOOD | BLOOD GAS | BLOOD | 71503
      51006, -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
      51301, -- WHITE BLOOD CELLS | HEMATOLOGY | BLOOD | 753301
      51300, -- WBC COUNT | HEMATOLOGY | BLOOD | 2371
      50818  -- PaCO2, added by Stephen for BMI 215
    )
    AND valuenum IS NOT null --AND valuenum > 0-- lab values cannot be 0 and cannot be negative --> luca: ever heard of base excess?
) pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, lab_id, pvt.valuenum
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, lab_id, pvt.valuenum;



-- MISSING:
-- ASAT
-- ALAT MAGNESIUM
-- PH CALCIUM
-- ION-ca  AKA Free-calcium?
-- mayby not labs table: FIO2