# FAST API AUTH with SQLite

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# ---------------- Database Setup ----------------
DATABASE_URL = "sqlite:///./Database.db"  # Local DB file

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------- Employee Model ----------------
class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    department = Column(String)
    salary = Column(Float)

# Create the table
Base.metadata.create_all(bind=engine)

# ---------------- FastAPI App ----------------
app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- Routes ----------------
@app.get("/")
def root():
    return {"message": "🚀 FastAPI + SQLite Employees DB Ready"}

@app.post("/employees/")
def add_employee(name: str, department: str, salary: float, db: Session = Depends(get_db)):
    emp = Employee(name=name, department=department, salary=salary)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return emp

@app.get("/employees/")
def list_employees(db: Session = Depends(get_db)):
    return db.query(Employee).all()

@app.get("/employees/{emp_id}")
def get_employee(emp_id: int, db: Session = Depends(get_db)):
    emp = db.query(Employee).filter(Employee.id == emp_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    return emp
