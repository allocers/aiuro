with treatment_user as(
    -- 処遇群ユーザーを定義 (元のスクリプトより)
    ...
)
, control_user as (
    -- 対照群ユーザーを定義 (元のスクリプトより)
    ...
)
, control_user_random as (
    -- 無作為抽出された対照群ユーザーを定義 (元のスクリプトより)
    ...
)
, dpay_settlement_rfm as (
    select
        comnid
        ,datediff(day, max(settlement_date), '2024-07-09') as r_dpay -- Recency (最新d払い決済日からの経過日数)
        ,count(distinct case when to_varchar(settlement_date, 'YYYYMM') = '202406' then settlement_date end) as f_dpay -- Frequency (CP期間前1ヶ月のd払い利用日数)
        ,sum(case when to_varchar(settlement_date, 'YYYYMM') = '202406' then settlement_amount else 0 end) as m_dpay -- Monetary (CP期間前1ヶ月のd払い決済金額)
    from
        PROCODB_B_P01_LAK_D_MR_000_VW_ADM_DNHL_SETTLEMENT_ANALYSIS
    where
        comnid in (select comnid from treatment_user union all select comnid from control_user_random)
        and to_varchar(settlement_date, 'YYYYMM') = '202406'
    group by
        comnid
)
, dpoint_use_give_rfm as (
    select
        comnid
        ,datediff(day, max(to_timestamp(sedai_date, 'YYYYMMDD')), '2024-07-09') as r_dpoint -- Recency (最新dポイント利用・付与日からの経過日数)
        ,count(distinct case when to_varchar(sedai_month, 'YYYYMM') = '202406' then sedai_date end) as f_dpoint -- Frequency (CP期間前1ヶ月のdポイント利用・付与日数)
        ,sum(case when to_varchar(sedai_month, 'YYYYMM') = '202406' then dpoint_merchant_05_use_point_number else 0 end) as m_dpoint_use -- Monetary (CP期間前1ヶ月のdポイント利用量)
        ,sum(case when to_varchar(sedai_month, 'YYYYMM') = '202406' then dpoint_merchant_05_give_point_number else 0 end) as m_dpoint_give -- Monetary (CP期間前1ヶ月のdポイント付与量)
    from
        PROCODB_B_P01_LAK_L_MR_000_VW_DIC_MECB_MEMBER_PROFILE_POINT_GIVE_USE
    where
        comnid in (select comnid from treatment_user union all select comnid from control_user_random)
        and to_varchar(sedai_month, 'YYYYMM') = '202406'
    group by
        comnid
)
, dcard_use_rfm as (
    select
        comnid
        ,datediff(day, max(to_timestamp(card_use_nengetsu, 'YYYYMM')), '2024-07-09') as r_dcard -- Recency (最新dカード決済日からの経過日数)
        ,count(distinct case when to_varchar(card_use_nengetsu, 'YYYYMM') = '202406' then card_use_nengetsu end) as f_dcard -- Frequency (CP期間前1ヶ月のdカード利用回数)
        ,sum(case when to_varchar(card_use_nengetsu, 'YYYYMM') = '202406' then use_amount else 0 end) as m_dcard -- Monetary (CP期間前1ヶ月のdカード決済金額)
    from
        PROCODB_B_P01_LAK_D_MR_000_VW_DIC_MECB_DCARD_INTEGRATION_DCARD_SALES
    where
        comnid in (select comnid from treatment_user union all select comnid from control_user_random)
        and to_varchar(card_use_nengetsu, 'YYYYMM') = '202406'
    group by
        comnid
)
, dpay_settlement as (
    ...
)
, dpoint_use_give as (
    ...
)
, dcard_use as (
    ...
)
, contractor as (
    ...
)
select
    a.comnid
    -- 既存の列
    ,COALESCE(b.dpay_settlement_amount_202406, 0) as dpay_settlement_amount_202406
    ...
    ,e.gender
    ,e.age
    -- RFM共変量
    ,COALESCE(f.r_dpay, 9999) as r_dpay
    ,COALESCE(f.f_dpay, 0) as f_dpay
    ,COALESCE(f.m_dpay, 0) as m_dpay
    ,COALESCE(g.r_dpoint, 9999) as r_dpoint
    ,COALESCE(g.f_dpoint, 0) as f_dpoint
    ,COALESCE(g.m_dpoint_use, 0) as m_dpoint_use
    ,COALESCE(g.m_dpoint_give, 0) as m_dpoint_give
    ,COALESCE(h.r_dcard, 9999) as r_dcard
    ,COALESCE(h.f_dcard, 0) as f_dcard
    ,COALESCE(h.m_dcard, 0) as m_dcard
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
left join
    dpay_settlement_rfm as f on a.COMNID = f.COMNID
left join
    dpoint_use_give_rfm as g on a.COMNID = g.COMNID
left join
    dcard_use_rfm as h on a.COMNID = h.COMNID
where
    e.gender is not null
    and e.age is not null