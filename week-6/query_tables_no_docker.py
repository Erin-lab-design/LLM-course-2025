import pdfplumber
import requests
from io import BytesIO
from llama_index.llms.ollama import Ollama

# Initialize LLM
llm = Ollama(model="llama3", request_timeout=60.0)

# Download PDF from URL
pdf_url = "https://s206.q4cdn.com/479360582/files/doc_financials/2024/q1/2024q1-alphabet-earnings-release-pdf.pdf"
print("Downloading PDF...")
response = requests.get(pdf_url)
pdf_file = BytesIO(response.content)

# Extract tables from PDF using pdfplumber
print("Extracting tables from PDF...")
sections_with_tables = []

with pdfplumber.open(pdf_file) as pdf:
    for page_num, page in enumerate(pdf.pages, 1):
        tables = page.extract_tables()
        if tables:
            text = page.extract_text()
            print(f"Found {len(tables)} table(s) on page {page_num}")
            
            for table_num, table in enumerate(tables, 1):
                # Convert table to HTML format
                html_table = "<table>\n"
                for row in table:
                    html_table += "<tr>"
                    for cell in row:
                        cell_text = cell if cell else ""
                        html_table += f"<td>{cell_text}</td>"
                    html_table += "</tr>\n"
                html_table += "</table>"
                
                sections_with_tables.append({
                    'title': f'Page {page_num} - Table {table_num}',
                    'content': html_table,
                    'text_context': text[:500]  # Include some context
                })

print(f"\nTotal sections with tables: {len(sections_with_tables)}\n")

# Combine all table contexts
all_contexts = "\n\n".join([
    f"Section: {s['title']}\n{s['content']}" 
    for s in sections_with_tables
])

# Test questions on all tables
question = "What was Google's operating margin for 2024"
resp = llm.complete(
    f"read these tables and answer question: {question}:\n{all_contexts}")
print(f"Q: {question}")
print(f"A: {resp.text}\n")

question = "What % Net income is of the Revenues?"
resp = llm.complete(
    f"read these tables and answer question: {question}:\n{all_contexts}")
print(f"Q: {question}")
print(f"A: {resp.text}\n")

# Task 1b: Test calculation capabilities
question = "What is the sum of Total revenues and Total costs and expenses?"
resp = llm.complete(
    f"read these tables and answer question: {question}:\n{all_contexts}")
print(f"Q: {question}")
print(f"A: {resp.text}\n")

question = "Calculate the difference between Q1 2024 revenues and Q1 2023 revenues"
resp = llm.complete(
    f"read these tables and answer question: {question}:\n{all_contexts}")
print(f"Q: {question}")
print(f"A: {resp.text}")
