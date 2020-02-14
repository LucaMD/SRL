DROP MATERIALIZED VIEW IF EXISTS martin_sepsis CASCADE;
CREATE MATERIALIZED VIEW martin_sepsis as
WITH co_dx AS
(
	SELECT subject_id, hadm_id
  , MAX(
    	CASE
        -- septicemia
    		WHEN substring(icd9_code,1,3) = '038' THEN 1
        -- septicemic, bacteremia, disseminated fungal infection, disseminated candida infection
				-- NOTE: the paper specifies 020.0 ... but this is bubonic plague
				-- presumably, they meant 020.2, which is septicemic plague
        WHEN substring(icd9_code,1,4) in ('0202','7907','1179','1125') THEN 1
        -- disseminated fungal endocarditis
        WHEN substring(icd9_code,1,5) = '11281' THEN 1
      ELSE 0 END
    ) AS sepsis
    , MAX(
      CASE
        WHEN substring(icd9_code,1,4) in ('7991') THEN 1
        WHEN substring(icd9_code,1,5) in ('51881','51882','51885','78609') THEN 1
      ELSE 0 END
    ) AS respiratory
    , MAX(
      CASE
        WHEN substring(icd9_code,1,4) in ('4580','7855','4580','4588','4589','7963') THEN 1
        WHEN substring(icd9_code,1,5) in ('785.51','785.59') THEN 1
      ELSE 0 END
    ) AS cardiovascular
    , MAX(
      CASE
        WHEN substring(icd9_code,1,3) in ('584','580','585') THEN 1
      ELSE 0 END
    ) AS renal
    , MAX(
      CASE
        WHEN substring(icd9_code,1,3) in ('570') THEN 1
        WHEN substring(icd9_code,1,4) in ('5722','5733') THEN 1
      ELSE 0 END
    ) AS hepatic
    , MAX(
      CASE
        WHEN substring(icd9_code,1,4) in ('2862','2866','2869','2873','2874','2875') THEN 1
      ELSE 0 END
    ) AS hematologic
    , MAX(
      CASE
        WHEN substring(icd9_code,1,4) in ('2762') THEN 1
      ELSE 0 END
    ) AS metabolic
    , MAX(
      CASE
        WHEN substring(icd9_code,1,3) in ('293') THEN 1
        WHEN substring(icd9_code,1,4) in ('3481','3483') THEN 1
        WHEN substring(icd9_code,1,5) in ('78001','78009') THEN 1
      ELSE 0 END
    ) AS neurologic
  FROM diagnoses_icd
  GROUP BY subject_id, hadm_id
)
, co_proc as
(
  SELECT subject_id, hadm_id
  , MAX(CASE WHEN substring(icd9_code,1,3) = '967' then 1 ELSE 0 END) as respiratory
  , MAX(CASE WHEN substring(icd9_code,1,4) = '3995' then 1 ELSE 0 END) as renal
  , MAX(CASE WHEN substring(icd9_code,1,4) = '8914' then 1 ELSE 0 END) as neurologic
  FROM procedures_icd
  GROUP BY subject_id, hadm_id
)
select adm.subject_id, adm.hadm_id
, co_dx.sepsis
, CASE
    WHEN co_dx.respiratory = 1 OR co_proc.respiratory = 1
      OR co_dx.cardiovascular = 1
      OR co_dx.renal = 1 OR co_proc.renal = 1
      OR co_dx.hepatic = 1
      OR co_dx.hematologic = 1
      OR co_dx.metabolic = 1
      OR co_dx.neurologic = 1 OR co_proc.neurologic = 1
    THEN 1
  ELSE 0 END as organ_failure
, case when co_dx.respiratory = 1 or co_proc.respiratory = 1 then 1 else 0 end as respiratory
, co_dx.cardiovascular
, case when co_dx.renal = 1 or co_proc.renal = 1 then 1 else 0 end as renal
, co_dx.hepatic
, co_dx.hematologic
, co_dx.metabolic
, case when co_dx.neurologic = 1 or co_proc.neurologic = 1 then 1 else 0 end as neurologic
FROM admissions adm
left join co_dx
  on adm.hadm_id = co_dx.hadm_id
left join co_proc
  on adm.hadm_id = co_proc.hadm_id;
