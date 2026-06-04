from typing import List, TypedDict, Annotated
import operator

class PlanExecuteState(TypedDict):

    input: str

    plan: List[str]

    past_step: Annotated[List[tuple], operator.add]

    response: str
