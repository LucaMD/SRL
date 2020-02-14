-- As the script is generating many tables, it may take some time.
-- We assume the database and the search path are set correctly.
-- You can set the search path as follows:
-- SET SEARCH_PATH TO public,mimiciii;
-- This will create tables on public and read tables from mimiciii

BEGIN;
-- ----------------------------- --
-- ---------- STAGE 1 ---------- --
-- ----------------------------- --
-- -- Generate the views for the RL cohort
\i create_cohort_table.sql

\i create_UrineOutput_view.sql

-- ----------------------------- --
-- ---------- STAGE 2 ---------- --
-- ----------------------------- --
--- -- Generate the views for the RL dataset
\i gcs_all.sql
\i labs_all_rl.sql
\i vitals_all_rl.sql
COMMIT;
