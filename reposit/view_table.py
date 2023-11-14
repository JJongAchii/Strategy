def get_risk_score():
    with session_local() as session:
        subquery = (
            session.query(
                TbPortBook.stk_id,
                TbPortBook.trd_dt,
                sa.func.max(TbRiskScore.risk_score).label('max_risk_score')
            )
            .join(
                TbRiskScore,
                sa.and_(
                    TbRiskScore.stk_id == TbPortBook.stk_id,
                    TbRiskScore.trd_dt <= TbPortBook.trd_dt
                )
            )
            .group_by(TbPortBook.stk_id, TbPortBook.trd_dt)
            .subquery()
        )

        query = (
            session.query(
                TbPortBook.trd_dt,
                TbPortBook.port_id,
                TbPortBook.weights,
                subquery.c.max_risk_score.label('risk_score')
            )
            .join(
                subquery,
                sa.and_(
                    subquery.c.stk_id == TbPortBook.stk_id,
                    subquery.c.trd_dt == TbPortBook.trd_dt
                )
            )
        )

        create_view_query = sa.text(f"CREATE OR REPLACE VIEW tb_port_value_view AS {query}")
        session.execute(create_view_query)
        return read_sql_query(query)