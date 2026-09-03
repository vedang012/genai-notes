from pydantic import BaseModel

class Education(BaseModel):
    degree: str
    institution: str
    year: str | None

class Experience(BaseModel):
    company: str
    role: str
    duration: str | None


class Resume(BaseModel):
    name: str
    mobile: str | None
    email: str | None
    skills: list[str]
    education: list[Education]
    experience: list[Experience]
    projects: list[str]



