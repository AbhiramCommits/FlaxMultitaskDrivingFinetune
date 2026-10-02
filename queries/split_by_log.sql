-- split_by_log.sql
-- Leakage-safe split BY log_id with fixed hash buckets, 70/15/15

CREATE OR REPLACE TABLE windows_split AS
SELECT 
    *,
    CASE 
        WHEN abs(hash(log_id)) % 100 < 70 THEN 'train'
        WHEN abs(hash(log_id)) % 100 < 85 THEN 'val'
        ELSE 'test'
    END AS split
FROM windows;
