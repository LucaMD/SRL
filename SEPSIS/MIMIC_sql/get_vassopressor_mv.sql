select 
    mimiciii.inputevents_mv.icustay_id, starttime, endtime, rate, rateuom, patientweight,
    CASE WHEN itemid=221906 THEN 'norepinephrine'
         WHEN itemid=221289 THEN 'epinephrine'
		 WHEN itemid=221662 THEN 'dopamine'
		 WHEN itemid=222315 THEN 'vasopressin'
		 WHEN itemid=221749 THEN 'phenylephrine' 
         ELSE 'other'   END	as Vasoactive_drug,
    CASE WHEN itemid=221906 THEN rate * 1
         WHEN itemid=221289 THEN rate * 1
		 WHEN itemid=221662 THEN rate * 0.01
		 WHEN itemid=222315 THEN rate * 5 / 60  --- /60 = convert to dose per minute | * 5 is conversion from dose/min to mcg/kg/min (assuming 100kg) 
		 WHEN itemid=221749 THEN rate * 0.45 
         ELSE rate      END as mcgkgmin,
    CASE WHEN itemid=221906 THEN rate * 1 * patientweight -- convert to mcg/min
         WHEN itemid=221289 THEN rate * 1 * patientweight -- convert to mcg/min
		 WHEN itemid=221662 THEN rate * 0.01 * patientweight -- convert to mcg/min
		 WHEN itemid=222315 THEN rate / 60  --- /60 = convert to dose per minute 
		 WHEN itemid=221749 THEN rate * 0.45 * patientweight -- convert to mcg/min
         ELSE rate      END as mcgmin
  from mimiciii.inputevents_mv
  INNER JOIN public.rl_cohort ON mimiciii.inputevents_mv.icustay_id = public.rl_cohort.icustay_id
  where itemid in
  (
  221906 -- norepinephrine
  --,221289 -- epinephrine
  ,221749 -- phenylephrine
  ,222315 -- vasopressin
  ,221662 -- dopamine
  )
  and statusdescription != 'Rewritten' -- only valid orders
Order by Icustay_id, StartTime