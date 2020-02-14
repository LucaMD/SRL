SELECT rl_cohort.subject_id, rl_cohort.hadm_id, rl_cohort.icustay_id,
       charttime, value
FROM public.rl_cohort
INNER JOIN public.urineoutput l
  ON l.icustay_id = rl_cohort.icustay_id
WHERE l.charttime >= rl_cohort.window_start AND 
    l.charttime <= rl_cohort.window_end;