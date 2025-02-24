import os

from pymed import PubMed
from typing import List, Dict, Any
from haystack import component, Document
from haystack.components.generators import HuggingFaceAPIGenerator
from dotenv import load_dotenv
import os
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
import gradio as gr
from haystack.utils import Secret

# Load API Key from .env
load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')


# Initialize PubMed API
pubmed = PubMed(tool="Haystack2.0Prototype", email="dummyemail@gmail.com")

def documentize(article):
    return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

@component
class PubMedFetcher():
    @component.output_types(articles=List[Document])
    def run(self, queries: list[str]):
        cleaned_queries = queries[0].strip().split('\n')
        articles = []
        try:
            for query in cleaned_queries:
                response = pubmed.query(query, max_results=1)
                documents = [documentize(article) for article in response]
                articles.extend(documents)
        except Exception as e:
            print(e)
            print(f"Couldn't fetch articles for queries: {queries}")
        return {'articles': articles}

@component
class ArticleFormatter():
    @component.output_types(template_variables=Dict[str, Any])
    def run(self, articles: List[Document], question: str):
        formatted_articles = [{"content": doc.content, "title": doc.meta['title'], "keywords": doc.meta.get('keywords', [])} for doc in articles]
        return {'template_variables': {"question": question, "articles": formatted_articles}}

# Initialize HuggingFace Generator
llm = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    token=Secret.from_env_var("HUGGINGFACE_API_KEY"),
    #token=HUGGINGFACE_API_KEY,
    generation_kwargs={"max_new_tokens": 500, "temperature": 0.6, "do_sample": True}
)

# Quiz Generation Prompt
quiz_prompt_template = """
Generate a set of 10 medical questions based on the given articles.

For each document, create the following based on the type of question specified:
- **Multiple-choice questions (MCQs):** Focus on clinical decision-making or disease pathophysiology. Provide a clear explanation of the correct answer.
- **Case-based scenario questions:** Generate 10 clinical case scenarios and ask for the most appropriate next step in management or diagnosis.
- **Short-answer questions:** Generate 10 concise, conceptually challenging questions, followed by a precise medical response.

If there is no relevant content for a specific question type, generate one based on general medical knowledge.

**Topic:** {{ question }}

**Articles:**
{% for article in articles %}
  {{ article.content }}
  keywords: {{ article.keywords }}
  title: {{ article.title }}
{% endfor %}

**Questions:**

{% if question == "quiz" %}
**Multiple-choice (MCQ):**
Q1: <clinical or pathophysiology-based question>
A. <option 1>
B. <option 2>
C. <option 3>
D. <option 4>
**Correct answer:** <correct option>
**Explanation:** <why the answer is correct, include relevant medical reasoning>

{% for i in range(2, 11) %}
Q{{i}}: <clinical or pathophysiology-based question>
A. <option 1>
B. <option 2>
C. <option 3>
D. <option 4>
**Correct answer:** <correct option>
**Explanation:** <why the answer is correct, include relevant medical reasoning>
{% endfor %}

{% elif question == "case-based" or question == "case study" %}
**Case-based scenario:**
{% for i in range(10) %}
Q{{i}}: A patient presents with <symptoms>. The patient has a history of <relevant medical history>. Based on the given information, what is the most appropriate next step in management?
**Answer:** <correct management approach>

{% endfor %}

{% elif question == "short-answer" %}
**Short-answer:**
{% for i in range(10) %}
Q{{i}}: <Short but conceptually challenging medical question>
**Answer:** <concise and precise medical response>

{% endfor %}
{% endif %}


  """


# Initialize Components
fetcher = PubMedFetcher()
formatter = ArticleFormatter()
prompt_builder = PromptBuilder(template=quiz_prompt_template)

# Create Pipeline
pipe = Pipeline()

pipe.add_component("pubmed_fetcher", fetcher)
pipe.add_component("article_formatter", formatter)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

# Connect Pipeline Components
pipe.connect("pubmed_fetcher.articles", "article_formatter.articles")
pipe.connect("article_formatter.template_variables", "prompt_builder.template_variables")
pipe.connect("prompt_builder.prompt", "llm.prompt")

# Function to Generate Quiz
def generate_quiz(topic):
    output = pipe.run(data={
        "pubmed_fetcher": {"queries": [topic]},
        "article_formatter": {"question": topic},
        "llm": {"generation_kwargs": {"max_new_tokens": 1000}}
    })

    first_batch = output['llm']['replies'][0]

    # Check if the first batch of questions is less than 10, if so, generate the next set of questions
    if len(first_batch.split('\n')) < 10:
        output_second_batch = pipe.run(data={
            "pubmed_fetcher": {"queries": [topic]},
            "article_formatter": {"question": topic},
            "llm": {"generation_kwargs": {"max_new_tokens": 1000}}  # Generate second batch
        })
        second_batch = output_second_batch['llm']['replies'][0]
        # Concatenate the first and second batches
        final_output = first_batch + "\n\n" + second_batch
    else:
        final_output = first_batch

    return final_output
    
import gradio as gr # Gradio Interface
css = """
body { background-color: white !important; color: black !important; }
.gradio-container { max-width: 700px; margin: auto; }
.input-textbox { background-color: #333; color: white; border: 1px solid #555; }
.output-markdown { background-color: #222; padding: 20px; border-radius: 10px; }
h1 { color: #1565c0; text-align: center; font-size: 30px; font-weight: bold; }
button { font-size: 16px; padding: 10px 15px; border-radius: 8px; cursor: pointer; }
button.primary { background-color: #1976d2; color: white; border: none; }
button.secondary { background-color: #e0e0e0; color: black; border: none; }

"""

iface = gr.Interface(
    fn=generate_quiz,
    inputs=gr.Textbox(
        value="Generate a quiz on COVID-19.",
        label="Enter a topic you need help with:",
        elem_id="input-textbox"
    ),
    outputs=gr.Markdown(elem_id="output-markdown"),
    title="🩺 Your AI Medical Tutor",
    description="Get quizzes encompassing MCQs, case-based questions and short answers based on PubMed research.",
    examples=[
        ["Generate a quiz on COVID-19."],
        ["Generate a quiz on Autoimmune Disorders."],
        ["Create a quiz about Pneumonia."],
        ["Generate a quiz on Diabetes."],
    ],
    theme=gr.themes.Monochrome(),
    allow_flagging="never",
    css=css
)

iface.launch(debug=True)