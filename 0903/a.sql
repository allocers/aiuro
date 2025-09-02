select
    a.comnid
    ,COALESCE(b.dpay_settlement_amount_202406, 0) as dpay_settlement_amount_202406
    ,COALESCE(b.dpay_settlement_amount_202407, 0) as dpay_settlement_amount_202407
    ,COALESCE(b.dpay_settlement_amount_202408, 0) as dpay_settlement_amount_202408
    ,COALESCE(b.dpay_settlement_amount_202409, 0) as dpay_settlement_amount_202409
    ,COALESCE(c.dpoints_use_202406, 0) as dpoints_use_202406
    ,COALESCE(c.dpoints_use_202407, 0) as dpoints_use_202407
    ,COALESCE(c.dpoints_use_202408, 0) as dpoints_use_202408
    ,COALESCE(c.dpoints_use_202409, 0) as dpoints_use_202409
    ,COALESCE(c.dpoints_give_202406, 0) as dpoints_give_202406
    ,COALESCE(c.dpoints_give_202407, 0) as dpoints_give_202407
    ,COALESCE(c.dpoints_give_202408, 0) as dpoints_give_202408
    ,COALESCE(c.dpoints_give_202409, 0) as dpoints_give_202409
    ,COALESCE(d.dcard_use_amount_202406, 0) as dcard_use_amount_202406
    ,COALESCE(e.contract, 0) as contract
    ,COALESCE(e.CONTRACT_PERIOD, 0) as contract_period
    ,e.gender
    ,e.age
from
    control_user_random as a
left join
    dpay_settlement as b on a.COMNID = b.COMNID
left join
    dpoint_use_give as c on a.COMNID = c.COMNID
left join
    dcard_use as d on a.COMNID = d.COMNID
inner join
    contractor as e on a.COMNID = e.COMNID
where
    gender is not null
    and age is not null
;