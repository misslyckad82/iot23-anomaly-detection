SELECT

    -- Timestamp and identifier
    TS,
    UID,

    -- Network information
    ID_ORIG_H,
    ID_ORIG_P,
    ID_RESP_H,
    ID_RESP_P,

    -- Network protocol and service
    PROTO,
    SERVICE,
    CONN_STATE,

    -- Traffic statitics
    DURATION,
    ORIG_BYTES,
    RESP_BYTES,
    ORIG_PKTS,
    RESP_PKTS,
    ORIG_IP_BYTES,
    RESP_IP_BYTES,

    -- Other information
    LOCAL_ORIG,
    LOCAL_RESP,
    MISSED_BYTES,
    HISTORY,

    -- Original merged column
    TUNNEL_PARENTS_LABEL_DETAILED_LABEL,

    -- Classification
    CASE
        WHEN TUNNEL_PARENTS_LABEL_DETAILED_LABEL ILIKE '%Benign%'
        THEN 'Benign'
        ELSE 'Malicious'
    END AS LABEL,

    -- Attack type
    CASE
        WHEN TUNNEL_PARENTS_LABEL_DETAILED_LABEL ILIKE '%PartOfAHorizontalPortScan%'
        THEN 'PartOfAHorizontalPortScan'
        ELSE NULL
    END AS LABEL_DETAILED

FROM {{ source('iot23', 'IOT23_DATA') }}