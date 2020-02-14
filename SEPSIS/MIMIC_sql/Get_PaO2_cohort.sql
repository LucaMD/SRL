SELECT
  pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, pvt.label as lab_id, pvt.valuenum
FROM
( -- begin query that extracts the data
  SELECT ie.subject_id, ie.hadm_id, ie.icustay_id, le.charttime
  -- here we assign labels to ITEMIDs
  -- this also fuses together multiple ITEMIDs containing the same data
  , CASE
        WHEN itemid = 50821 THEN 'PAO2'
      ELSE null
    END AS label
  , -- add in some sanity checks on the values
  -- the where clause below requires all valuenum to be > 0, so these are only upper limit checks
    CASE
      WHEN itemid = 50821 and valuenum <     0 THEN null -- mmHg 'PaO2' 
    ELSE le.valuenum
    END AS valuenum
  FROM icustays ie
  LEFT JOIN labevents le
    ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
    AND le.charttime >= (ie.intime - interval '48' hour)
    AND le.ITEMID in
    (
      50821, -- PaO2 in labevents
    )
    AND valuenum IS NOT null --AND valuenum > 0-- lab values cannot be 0 and cannot be negative --> luca: ever heard of base excess?
) pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, lab_id, pvt.valuenum
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime, lab_id, pvt.valuenum;
