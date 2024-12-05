from pydantic import BaseModel


class Result(BaseModel):
    correct: int
    total: int


class TestResults(BaseModel):
    result: Result
    labels_results: dict[str, Result]
