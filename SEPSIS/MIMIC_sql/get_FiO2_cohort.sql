-- -- SOURCE: https://github.com/MIT-LCP/mimic-code/blob/fac79ba4ab3a3ca7fc37b24603c0f5021fcc4106/concepts/firstday/blood-gas-first-day-arterial.sql
select input.SUBJECT_ID, input.HADM_ID, rl.ICUSTAY_ID, input.CHARTTIME
    -- pre-process the FiO2s to ensure they are between 21-100%
    , max(
        case
          when itemid = 223835
            then case
              when valuenum > 0 and valuenum <= 1
                then valuenum * 100
              -- improperly input data - looks like O2 flow in litres
              when valuenum > 1 and valuenum < 21
                then null
              when valuenum >= 21 and valuenum <= 100
                then valuenum
              else null end -- unphysiological
        when itemid in (3420, 3422)
        -- all these values are well formatted
            then valuenum
        when itemid = 190 and valuenum > 0.20 and valuenum < 1
        -- well formatted but not in %
            then valuenum * 100
      else null end
    ) as FiO2
  from mimiciii.CHARTEVENTS input
  INNER JOIN public.rl_cohort rl on input.icustay_id =rl.icustay_id 
  where input.ITEMID in
  (
    3420 -- FiO2
  , 190 -- FiO2 set
  , 223835 -- Inspired O2 Fraction (FiO2)
  , 3422 -- FiO2 [measured]
  )
  -- exclude rows marked as error
  AND input.error IS DISTINCT FROM 1
  AND input.charttime >= rl.window_start AND 
    input.charttime <= rl.window_end
  group by input.SUBJECT_ID, input.HADM_ID, RL.ICUSTAY_ID, input.CHARTTIME;