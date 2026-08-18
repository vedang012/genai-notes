## Mini Project - AI Resume Parser

This is a simple mini project used to parse resume

PDF → PyPDF → Text → Groq LLM → JSON → Pydantic

### Setup
pip install -r requirements.txt

Add your API key to .env:

GROQ_API_KEY=your_api_key

in main.py 
resume_text = parser.extract_text("resume.pdf")

paste the resume link here

### Run:

python main.py