-- Final fact table for machine learning.
-- Contains transformed data and engineered features from the intermediate layer.

SELECT *
FROM {{ ref('int_iot23_features') }}