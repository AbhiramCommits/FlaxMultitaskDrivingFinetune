-- class_stats.sql
-- Per-split, per-task class counts and positive rates

SELECT 
    split,
    COUNT(*) as total_windows,
    SUM(task_rare) as rare_positives,
    CAST(SUM(task_rare) AS DOUBLE) / COUNT(*) as rare_pos_rate,
    AVG(task_horizon) as mean_horizon,
    SUM(CASE WHEN task_event = 0 THEN 1 ELSE 0 END) as event_cruise,
    SUM(CASE WHEN task_event = 1 THEN 1 ELSE 0 END) as event_lane_change,
    SUM(CASE WHEN task_event = 2 THEN 1 ELSE 0 END) as event_hard_brake,
    SUM(CASE WHEN task_event = 3 THEN 1 ELSE 0 END) as event_yield
FROM windows_split
GROUP BY split;
