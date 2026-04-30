from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import predict_relevance

app = FastAPI()


class Study(BaseModel):
    study_id: str
    study_description: str = ""


class Case(BaseModel):
    case_id: str
    current_study: Study
    prior_studies: List[Study]


class Request(BaseModel):
    cases: List[Case]


@app.post("/predict")
def predict(request: Request):
    results = []

    for case in request.cases:
        preds = predict_relevance(case)

        for study_id, label in preds:
            results.append({
                "case_id": case.case_id,
                "study_id": study_id,
                "predicted_is_relevant": label
            })

    return {"predictions": results}