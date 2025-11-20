# FAST API AUTH with JWT
 
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta

# FastAPI app
app = FastAPI()
 
# Secret key for JWT (in real apps keep this in env variable)
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
 
# Dummy user DB
fake_users_db = {
    "rahul": {
        "username": "rahul",
        "password": "1234",  # ❌ plain text just for demo (use hashing in real apps)
    },
    "sneha": {
        "username": "sneha",
        "password": "abcd",
    }
}
 
# Token scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
 
# Create JWT token
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
 
# Login API → returns JWT token
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
 
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
 
# Verify JWT
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise HTTPException(status_code=401, detail="Invalid token")
        return fake_users_db[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
 
# Protected route
@app.get("/profile")
def read_profile(current_user: dict = Depends(get_current_user)):
    return {"message": f"Hello {current_user['username']}! 🎉", "user": current_user}
