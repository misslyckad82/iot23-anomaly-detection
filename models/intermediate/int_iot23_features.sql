-- Intermediate model for feature engineering.
-- Converts data types and creates features for machine learning.

SELECT

    -------------------------------------------------
    -- Identifier
    -------------------------------------------------

    TS,
    UID,

    -------------------------------------------------
    -- Network information
    -------------------------------------------------

    ID_ORIG_H,
    ID_ORIG_P,
    ID_RESP_H,
    ID_RESP_P,

    -------------------------------------------------
    -- Protocol
    -------------------------------------------------

    PROTO,
    SERVICE,
    CONN_STATE,

    -------------------------------------------------
    -- Convert numeric data types
    -------------------------------------------------

    COALESCE(TRY_TO_DOUBLE(DURATION), 0) AS DURATION,
    
    COALESCE(TRY_TO_NUMBER(ORIG_BYTES), 0) AS ORIG_BYTES,
    COALESCE(TRY_TO_NUMBER(RESP_BYTES), 0) AS RESP_BYTES,

    ORIG_PKTS,
    RESP_PKTS,

    ORIG_IP_BYTES,
    RESP_IP_BYTES,

    MISSED_BYTES,

    -------------------------------------------------
    -- Other columns
    -------------------------------------------------

    HISTORY,
    LOCAL_ORIG,
    LOCAL_RESP,

    TUNNEL_PARENTS_LABEL_DETAILED_LABEL,

    LABEL,
    LABEL_DETAILED,

    -------------------------------------------------
    -- Feature engineering
    -------------------------------------------------

    COALESCE(TRY_TO_NUMBER(ORIG_BYTES), 0)
    + COALESCE(TRY_TO_NUMBER(RESP_BYTES), 0)
        AS TOTAL_BYTES,

    COALESCE(ORIG_PKTS, 0)
    + COALESCE(RESP_PKTS, 0)
        AS TOTAL_PACKETS,

    CASE
    WHEN
        (
            COALESCE(ORIG_PKTS, 0)
            + COALESCE(RESP_PKTS, 0)
        ) > 0
    THEN
        (
            COALESCE(TRY_TO_NUMBER(ORIG_BYTES), 0)
            + COALESCE(TRY_TO_NUMBER(RESP_BYTES), 0)
        )
        /
        (
            COALESCE(ORIG_PKTS, 0)
            + COALESCE(RESP_PKTS, 0)
        )
    ELSE 0
END
AS BYTES_PER_PACKET,

    CASE
        WHEN COALESCE(MISSED_BYTES, 0) > 0
        THEN 1
        ELSE 0
    END
        AS HAS_MISSED_BYTES,

    CASE
        WHEN LOCAL_ORIG = 'T'
          OR LOCAL_RESP = 'T'
        THEN 1
        ELSE 0
    END
        AS IS_LOCAL_TRAFFIC,

    CASE
        WHEN LABEL = 'Malicious'
        THEN 1
        ELSE 0
    END
        AS IS_MALICIOUS

FROM {{ ref('stg_iot23') }}