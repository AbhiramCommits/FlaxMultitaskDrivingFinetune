-- assemble_windows.sql
-- Window per-step rows into fixed-length sequences (seq_len=32, stride=8) using SQL window functions

CREATE OR REPLACE TABLE raw_ordered AS
SELECT *,
       ROW_NUMBER() OVER (PARTITION BY log_id ORDER BY t) as row_num
FROM raw;

CREATE OR REPLACE TABLE windows AS
WITH partitioned AS (
    SELECT 
        log_id,
        city,
        t,
        ego_vx,
        ego_vy,
        ego_yaw_rate,
        ego_accel,
        lead_dist,
        lead_rel_v,
        n_agents,
        lane_curvature,
        traffic_light_state,
        weather,
        time_of_day,
        task_event,
        task_horizon,
        task_rare,
        ROW_NUMBER() OVER (PARTITION BY log_id ORDER BY t) as rn
    FROM raw
),
window_starts AS (
    SELECT log_id, rn AS start_rn
    FROM partitioned
    WHERE (rn - 1) % 8 = 0
)
SELECT 
    w.log_id || '_' || w.start_rn AS window_id,
    p.log_id,
    p.city,
    LIST(p.ego_vx) OVER w_frame AS ego_vx_seq,
    LIST(p.ego_vy) OVER w_frame AS ego_vy_seq,
    LIST(p.ego_yaw_rate) OVER w_frame AS ego_yaw_rate_seq,
    LIST(p.ego_accel) OVER w_frame AS ego_accel_seq,
    LIST(p.lead_dist) OVER w_frame AS lead_dist_seq,
    LIST(p.lead_rel_v) OVER w_frame AS lead_rel_v_seq,
    LIST(p.n_agents) OVER w_frame AS n_agents_seq,
    LIST(p.lane_curvature) OVER w_frame AS lane_curvature_seq,
    LIST(p.traffic_light_state) OVER w_frame AS traffic_light_state_seq,
    LIST(p.weather) OVER w_frame AS weather_seq,
    LIST(p.time_of_day) OVER w_frame AS time_of_day_seq,
    -- Labels taken from the last step of the window
    FIRST_VALUE(p.task_event) OVER (PARTITION BY w.log_id, w.start_rn ORDER BY p.rn DESC) AS task_event,
    FIRST_VALUE(p.task_horizon) OVER (PARTITION BY w.log_id, w.start_rn ORDER BY p.rn DESC) AS task_horizon,
    MAX(p.task_rare) OVER w_frame AS task_rare
FROM window_starts w
JOIN partitioned p ON p.log_id = w.log_id AND p.rn >= w.start_rn AND p.rn < w.start_rn + 32
WINDOW w_frame AS (PARTITION BY w.log_id, w.start_rn ORDER BY p.rn ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING);
