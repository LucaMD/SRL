
select 
	mimiciii.inputevents_cv.icustay_id, mimiciii.inputevents_cv.charttime, rate, rateuom,
	CASE WHEN itemid in(30047,30120)      THEN 'norepinephrine'
		 WHEN itemid in (30043,30307,30125) THEN 'dopamine'
		 WHEN itemid = 30051                THEN 'vasopressin'
		 WHEN itemid in (30127,30128)       THEN 'phenylephrine' 
         ELSE 'other'                   END	as Vasoactive_drug,
    CASE WHEN itemid in(30047,30120)    THEN rate * 1
		 WHEN itemid in (30043,30307,30125) THEN rate * 0.01
		 WHEN itemid=30051                  THEN rate * 5 -- * 5 is conversion from dose/min to mcg/kg/min (assuming ~100kg) 
		 WHEN itemid in (30127,30128)       THEN rate * 0.45 
         ELSE rate                      END as mcgkgmin
from mimiciii.inputevents_cv
INNER JOIN public.rl_cohort ON mimiciii.inputevents_cv.icustay_id = public.rl_cohort.icustay_id
where itemid in
(
  30047,30120 -- norepinephrine
  ,30044,30119,30309 -- epinephrine
  ,30127,30128 -- phenylephrine
  ,30051 -- vasopressin
  ,30043,30307,30125 -- dopamine
)
and rate is not NULL
order by icustay_id, charttime