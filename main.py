import json
from langchain.llms import OpenAI as LangChainOpenAI
from langchain_openai import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests
from fpdf import FPDF
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_PATH'] = 1000000

# Load OpenAI API key


# Function to extract text from PDF file
def get_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from URL (PDF, blog, or video transcript)
def extract_text_from_blog_url(url):
    if url.endswith('.pdf'):
        response = requests.get(url)
        with open('temp.pdf', 'wb') as f:
            f.write(response.content)
        text = get_pdf_text('temp.pdf')
    elif url.startswith('https://'):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = '\n'.join([p.get_text() for p in paragraphs])
        text = text[50:-200]
    else:
        text = ""
    return text

def extract_video_id(url):
    if 'youtu.be/' in url:
        video_id_index = url.index('youtu.be/') + len('youtu.be/')
        video_id = url[video_id_index:].split('?')[0]
        return video_id
    elif 'watch?v=' in url:
        video_id_index = url.index('watch?v=') + len('watch?v=')
        video_id = url[video_id_index:].split('&')[0]
        return video_id
    else:
        return None

def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return None

def sanitize_text(text):
    return text.encode('latin1', 'replace').decode('latin1')

# Initialize OpenAI language model
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the prompt template for generating quiz questions
template = """
Using the following JSON schema,
Please list {num_questions} quiz questions in {language} on {subject} for {schooling_level} and difficulty level of the quiz should be {level}.
Cover all of the topics given in the content while making questions.
Include only the following types of questions: {question_types}.
Make sure to return the data in JSON format exactly matching this schema.
Recipe = {{
    "question": "str",
    "options": "list",
    "answer": "list" if type == "multiple_select" else "str",
    "type": "str",  # Indicating question type (single_select / true_false / numeric / theory / multiple_select)
    "explanation": "str"  # Add an explanation for the answer
}}
Return: list[Recipe]

example:
[
    {{
        "question": "What is the largest ocean in the world?",
        "options": ["Atlantic Ocean", "Indian Ocean", "Pacific Ocean", "Arctic Ocean"],
        "answer": "Pacific Ocean",
        "type": "single_select",
        "explanation": "The Pacific Ocean is the largest and deepest ocean on Earth."
    }},
    {{
        "question": " J.K. Rowling is the author of the Harry Potter series.",
        "options": ["True", "False"],
        "answer": "True",
        "type": "true_false",
        "explanation": "J.K. Rowling is indeed the author of the Harry Potter series."
    }},
    {{
        "question": "What is 5 + 3?",
        "options": [],
        "answer": "8",
        "type": "numeric",
        "explanation": "The sum of 5 and 3 is 8."
    }},
    {{
        "question": "Explain the theory of relativity.",
        "options": [],
        "answer": "The theory of relativity is a scientific concept describing the relationship between space, time, and gravity.",
        "type": "theory",
        "explanation": "Einstein's theory of relativity includes both the special relativity and general relativity principles."
    }},
    {{
        "question": "Which of the following are programming languages?",
        "options": ["Python", "HTML", "Java", "CSS"],
        "answer": ["Python", "Java"],
        "type": "multiple_select",
        "explanation": "Python and Java are programming languages, while HTML and CSS are markup and style sheet languages, respectively."
    }}
]
"""

# Initialize LangChain LLMChain with the prompt template
llm_chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["num_questions", "language", "subject", "schooling_level", "level", "question_types"], template=template))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    input_type = request.form['input_type']
    schooling_level = request.form['schooling_level']
    num_questions = int(request.form['num_questions'])
    level = request.form['level']
    language = request.form['language']
    question_types = request.form.getlist('question_types')

    subject = ""
    if input_type == "PDF":
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            subject = get_pdf_text(file_path)
    elif input_type == "Text":
        subject = request.form['input_text']
    elif input_type == "Blog URL":
        url = request.form['url']
        subject = extract_text_from_blog_url(url)
    elif input_type == "Video URL":
        url = request.form['url']
        video_id = extract_video_id(url)
        subject = get_video_transcript(video_id)

    inputs = {
        "num_questions": num_questions,
        "language": language,
        "subject": subject,
        "schooling_level": schooling_level,
        "level": level,
        "question_types": ", ".join(question_types)
    }

    try:
        raw_response = llm_chain.run(inputs)
        json_start_idx = raw_response.find("[")
        json_end_idx = raw_response.rfind("]")
        if json_start_idx != -1 and json_end_idx != -1:
            json_response = raw_response[json_start_idx:json_end_idx + 1]
            data = json.loads(json_response)
        else:
            raise ValueError("No JSON part found in response")

        if len(data) < num_questions:
            flash(f"Only {len(data)} questions were generated. You may want to adjust the parameters.")
        
        sorted_data = []
        for qtype in question_types:
            sorted_data.extend([q for q in data if q['type'] == qtype])
        request.session['questions'] = sorted_data
        flash("Quiz generated successfully!")
        return redirect(url_for('show_quiz'))
    except json.JSONDecodeError as e:
        flash(f"Error decoding JSON from response: {e}")
        flash(f"Raw response: {raw_response}")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}")
        return redirect(url_for('index'))

@app.route('/quiz', methods=['GET', 'POST'])
def show_quiz():
    if request.method == 'POST':
        user_answers = {}
        for idx, question in enumerate(request.session['questions'], start=1):
            answer = request.form.get(f'answer_{idx}')
            user_answers[idx] = answer

        correct_answers = 0
        results = []

        for idx, question in enumerate(request.session['questions'], start=1):
            correct_answer = question['answer']
            user_answer = user_answers.get(idx)

            if isinstance(correct_answer, list):
                correct = set(user_answer) == set(correct_answer)
            else:
                correct = user_answer == correct_answer

            result = {
                "question": question['question'],
                "user_answer": user_answer,
                "correct_answer": correct_answer,
                "explanation": question['explanation'],
                "correct": correct
            }
            results.append(result)

            if correct:
                correct_answers += 1

        score = (correct_answers / len(request.session['questions'])) * 100
        flash(f"You scored {score}%")
        return render_template('results.html', results=results, score=score)

    return render_template('quiz.html', questions=request.session['questions'])

@app.route('/download_pdf')
def download_pdf():
    school_name = request.args.get('school_name')
    exam_title = request.args.get('exam_title')
    num_questions = request.args.get('num_questions')
    questions = request.session.get('questions', [])

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=school_name, ln=True, align="C")
    pdf.cell(200, 10, txt=exam_title, ln=True, align="C")

    for idx, question in enumerate(questions, start=1):
        pdf.cell(200, 10, txt=f"Q{idx}: {question['question']}", ln=True)
        for option in question['options']:
            pdf.cell(200, 10, txt=f"- {option}", ln=True)
        pdf.cell(200, 10, txt=f"Answer: {question['answer']}", ln=True)
        pdf.cell(200, 10, txt=f"Explanation: {question['explanation']}", ln=True)

    pdf_file = "quiz.pdf"
    pdf.output(pdf_file)
    return send_file(pdf_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
