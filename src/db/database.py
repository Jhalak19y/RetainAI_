import os
from datetime import datetime
from pathlib import Path
import bcrypt
from sqlalchemy import create_engine, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, sessionmaker, relationship

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "retainai.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    probability: Mapped[float] = mapped_column(Float)
    risk: Mapped[str] = mapped_column(String(20))
    customer_json: Mapped[str] = mapped_column(Text)

def init_db():
    Base.metadata.create_all(engine)

def register_user(username, password):
    username = username.strip()
    if len(username) < 3 or len(password) < 8:
        raise ValueError("Username must be at least 3 characters and password at least 8.")
    with SessionLocal() as db:
        if db.query(User).filter_by(username=username).first():
            raise ValueError("Username already exists.")
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        db.add(User(username=username, password_hash=hashed))
        db.commit()

def authenticate(username, password):
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=username.strip()).first()
        if not user:
            return False
        return bcrypt.checkpw(password.encode(), user.password_hash.encode())

def save_prediction(username, probability, risk, customer_json):
    with SessionLocal() as db:
        db.add(Prediction(username=username, probability=float(probability),
                          risk=risk, customer_json=customer_json))
        db.commit()

def get_history(username):
    with SessionLocal() as db:
        rows = db.query(Prediction).filter_by(username=username).order_by(
            Prediction.created_at.desc()
        ).all()
        return [{
            "id": r.id, "created_at": r.created_at, "probability": r.probability,
            "risk": r.risk, "customer_json": r.customer_json
        } for r in rows]

def delete_prediction(username, prediction_id):
    with SessionLocal() as db:
        row = db.query(Prediction).filter_by(id=prediction_id, username=username).first()
        if row:
            db.delete(row)
            db.commit()
