from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import joblib
from fancyimpute import SoftImpute
import traceback
from io import StringIO
import uvicorn
from sklearn import preprocessing

app = FastAPI()

templates = Jinja2Templates(directory="templates")

try:
    model = joblib.load('GBM_Model_version.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

def check_file(filename: str) -> bool:
    return filename.lower().endswith(".csv")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/eligibility")
async def eligibility_check(request: Request, file: UploadFile = File(...)):
    try:
        # Validate file type
        if not check_file(file.filename):
            raise HTTPException(
                status_code=400, detail="Invalid file format. Only CSV allowed.")

        # Read the CSV into a DataFrame
        contents = await file.read()
        test = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Categorical columns
        cat_cols = ['Term', 'Years in current job',
                    'Home Ownership', 'Purpose']
        for c in cat_cols:
            test[c] = pd.factorize(test[c])[0]

        updated_test_data = pd.DataFrame(
            SoftImpute().fit_transform(test.iloc[:, 3:19]),
            columns=test.columns[3:19],
            index=test.index
        )

        test_data = pd.get_dummies(updated_test_data, drop_first=True)
        test_data = preprocessing.scale(test_data)

        # Make predictions
        y_pred = model.predict(test_data)
        y_pred_labels = np.where(
            y_pred == 0, 'Loan Approved', 'Loan Regected')

        # Add predictions to DataFrame
        test['Loan Status'] = y_pred_labels

        json_data = test.replace({np.nan: None}).to_dict(orient='records')

        return templates.TemplateResponse("results.html", {"request": request, "results": json_data})

    except Exception as e:
        return {"code": 500, "msg": traceback.format_exc()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
