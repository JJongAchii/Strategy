from hive_old import db

class TbView(Base, Mixins, Timestamp):
    __tablename__ = "tb_view"
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(255))
    count = sa.Column(sa.Integer)

    @classmethod
    def create_view(cls):
        with session_local() as session:
            session.execute("""
            CREATE OR REPLACE VIEW new_view AS SELECT name, COUNT(*) as count FROM tb_view GROUP BY name""")


db.create_all()
db.TbView.create_view()
