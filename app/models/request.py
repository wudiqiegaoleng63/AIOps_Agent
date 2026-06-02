from pydantic import BaseModel, Field


class ChatRequest(BaseModel):

    id: str = Field(..., description="会话 ID", alias="Id")
    question: str = Field(..., description="用户输入的问题", alias="Question")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Id": "12345",
                "Question": "什么是AIOps？"
            }
        }

class ClearRequest(BaseModel):

    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        populate_by_name = True





