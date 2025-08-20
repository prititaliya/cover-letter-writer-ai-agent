from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
import vertexai

vertexai.init(project="Generative Language API Key", location="us-central1")
import hashlib
import os
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults

class gen_structure(TypedDict):
    content:Annotated[str, "The content of the generated cover letter"]
    critism:Annotated[str, "Any criticism or feedback on the generated cover letter"]
    

gen_model = init_chat_model(
    model="gpt-4o-mini",  
    temperature=0.2,
)
gen_model = gen_model.with_structured_output(gen_structure)
# review_model=init_chat_model(
#     model="gemini-2.5-flash"
# )
class cvState(TypedDict):
    resume_vectorstore: FAISS
    resume_text:str
    resume_retriver: FAISS
    resume_path: str
    cover_letter: str
    jd_url: str
    jd_text: str
    jd_vectorstore: FAISS
    jd_retriver: FAISS
    satisfied: bool
    message: add_messages
    iterations:int

def load_resume(state: cvState):
    # Load the resume content from the state
    persist_dir="fiass_index"
    is_available=False
    if os.path.exists(persist_dir):
        resume_vectorstore = FAISS.load_local(persist_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        is_available=True
        
        docs = resume_vectorstore.docstore._dict.values()

        all_text = " ".join([doc.page_content for doc in docs])
        
        return {
            "resume_vectorstore": resume_vectorstore,
            "resume_retriver": resume_vectorstore.as_retriever(),
            "resume_text": all_text
        }
    if not is_available:
        doc = PyPDFLoader(state['resume_path']).load()
        resume_text=" ".join([d.page_content for d in doc])
        resume_hash=hashlib.sha256(resume_text.encode('utf-8')).hexdigest()


        splitters=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks=splitters.split_documents(doc)
        
        for chunk in chunks:
            chunk.metadata['hash']=resume_hash

        if is_available:
            resume_vectorstore.add_documents(chunks)
        else:
            resume_vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

        # Persist index
        resume_vectorstore.save_local(persist_dir)
        docs = resume_vectorstore.docstore._dict.values()

        all_text = " ".join([doc.page_content for doc in docs])
        return {
            "resume_vectorstore": resume_vectorstore,
            "resume_retriver": resume_vectorstore.as_retriever(),
            "resume_text": all_text
        }

def jd_loader(state:cvState):
    # Load the job description content from the state
    jd_hash = hashlib.sha256(state['jd_url'].encode('utf-8')).hexdigest()
    persist_dir = f"jd_faiss_index_{jd_hash}"
    is_available=False
    if os.path.exists(persist_dir):
        jd_vectorstore = FAISS.load_local(persist_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        is_available=True

        docs = jd_vectorstore.docstore._dict.values()
        all_text = " ".join([doc.page_content for doc in docs])

        return {
            "jd_vectorstore": jd_vectorstore,
            "jd_retriver": jd_vectorstore.as_retriever(),
            "jd_text": all_text
        }
    
    if not is_available:
        doc=WebBaseLoader(state['jd_url']).load()
        jd_text=" ".join([d.page_content for d in doc])
        jd_hash=hashlib.sha256(jd_text.encode('utf-8')).hexdigest()

        splitters=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks=splitters.split_documents(doc)

        for chunk in chunks:
            chunk.metadata['hash']=jd_hash

        if is_available:
            jd_vectorstore.add_documents(chunks)
        else:
            jd_vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

        # Persist index
        jd_vectorstore.save_local(persist_dir)
        
        
        docs = jd_vectorstore.docstore._dict.values()
        all_text = " ".join([doc.page_content for doc in docs])
        
        return {
            "jd_vectorstore": jd_vectorstore,
            "jd_retriver": jd_vectorstore.as_retriever(),
            "jd_text": all_text
        }


def generate_cover_letter(state:cvState):
    resume_text = state['resume_text']
    jd_text = state['jd_text']

    system_msg = f"""You are an expert cover letter writer who helps job applicants create compelling, authentic cover letters that sound genuinely like them while getting noticed by hiring managers. Your approach is based on Harvard Business Review's proven methodology for effective cover letters.

    Core Principles

    1. Do Your Research First
    - Always research the company thoroughly before writing
    - Understand their mission, values, recent news, and culture
    - Research the specific role and its requirements
    - Identify the hiring manager's name when possible
    - Look for connections or mutual contacts

    2. Start Off Strong
    - Open with a compelling hook that immediately grabs attention
    - Avoid generic openings like "I am writing to apply for..."
    - Use a specific achievement, relevant story, or unique insight about the company
    - Make the first sentence memorable and relevant to the role

    3. Emphasize Your Value
    - Focus on what you can do for THEM, not what they can do for you
    - Use specific examples and quantifiable achievements
    - Connect your experience directly to their needs and challenges
    - Show how you'll solve their problems or contribute to their goals

    Structure and Format

    Opening Paragraph:
    - Strong, attention-grabbing first line
    - Mention the specific position and how you learned about it
    - Include a brief value proposition or key qualification

    Body Paragraphs (1-2):
    - Highlight 2-3 most relevant experiences or skills
    - Use the STAR method (Situation, Task, Action, Result) for examples
    - Connect each point to the job requirements
    - Show knowledge of the company and role

    Closing Paragraph:
    - Reiterate your interest and value proposition
    - Include a confident call to action
    - Express enthusiasm for next steps
    - Thank them for their consideration

    Writing Guidelines

    Tone and Voice:
    - Write in your authentic voice - let your personality shine through
    - Be professional but conversational
    - Show enthusiasm without being overly eager
    - Maintain confidence without arrogance

    Content Strategy:
    - Keep it concise (3-4 paragraphs maximum)
    - Focus on achievements, not just responsibilities
    - Use active voice and strong action verbs
    - Include specific metrics and results when possible
    - Address any obvious gaps or concerns proactively

    Technical Requirements:
    - Address to a specific person when possible
    - Use standard business letter format
    - Keep to one page
    - Use readable font (11-12pt)
    - Proofread meticulously for errors

    What to Avoid:
    - Generic, template-style language
    - Repeating everything from your resume
    - Focusing on what you want rather than what you offer
    - Being overly humble or self-deprecating
    - Making demands or assumptions about salary/benefits
    - Using clichés or buzzwords without substance

    Key Questions to Address:
    Before writing, ensure you can answer:
    - Why this company? (Show you've done research)
    - Why this role? (Connect to your goals and skills)
    - Why you? (What unique value do you bring?)
    - Why now? (What makes this the right timing?)

    CRITICAL: You must return your response in exactly this JSON structure:

    {{
        "content": "The complete, polished cover letter text ready to send",
        "criticism": "Self-assessment and feedback on the generated cover letter including strengths, potential weaknesses, and suggestions for improvement"
    }}

    Content Field Requirements:
    - Must contain the complete cover letter text
    - Should be professionally formatted and ready to send
    - Include proper salutation, body paragraphs, and closing
    - Maximum one page when printed
    - No additional commentary or metadata

    Criticism Field Requirements:
    - Provide honest self-evaluation of the cover letter's effectiveness
    - Identify specific strengths (what works well)
    - Point out potential weaknesses or areas for improvement
    - Assess how well it follows the HBR methodology
    - Rate authenticity and likelihood of getting noticed
    - Suggest specific improvements if any major issues exist
    - Include brief research summary and customization notes
    - Evaluate against the key questions: Why this company? Why this role? Why you? Why now?

    Quality Standards for Self-Criticism:
    - Be specific rather than generic in feedback
    - Reference actual content from the letter
    - Identify if the letter sounds too template-like or robotic
    - Assess whether it demonstrates genuine company research
    - Evaluate if the value proposition is compelling and specific
    - Check if the opening is strong enough to grab attention
    - Confirm the letter focuses on employer benefits, not candidate wants

    Remember: The goal is to create a cover letter that sounds authentically like the applicant while strategically positioning them as the ideal candidate for the specific role and company.

    Here is a resume and a job description. Use the information to generate a cover letter that is tailored to the job description and highlights the key strengths of the applicant.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {jd_text}"""
    cover_letter = gen_model.invoke(system_msg)

    return {
        "cover_letter": cover_letter,
        'iterations':state['iterations'] + 1
    }
def check_condition(state:cvState):
     if(state['iterations'] > 5 or state['satisfied']):
        return END
     else:
        return 'continue'
def review_cover_letter_condition(state:cvState):
    resume=state['resume_text']
    job_description=state['jd_text']
    cover_letter=state['cover_letter']
    class coverLetterAnalyser(TypedDict):
        overall: float
        research_and_customization: str
        opening_strength: str
        value_proposition: str
        authenticity_and_voice: str
        structure_and_execution: str

    model=init_chat_model(model='gpt-4o-mini').with_structured_output(coverLetterAnalyser)

    system_msg = f"""You are an expert cover letter analyst and career coach specializing in evaluating cover letters against Harvard Business Review's proven methodology. Your role is to provide comprehensive, actionable feedback to help improve cover letter effectiveness.

    Analysis Framework

    You will evaluate the cover letter across 5 key dimensions with specific scoring criteria:

    1. RESEARCH AND CUSTOMIZATION (Weight: 25%)
    Score 1-4 based on:
    - Evidence of company research (mission, values, recent news, culture)
    - Understanding of role requirements and responsibilities  
    - Personalization to specific company and position
    - Knowledge of industry context and challenges
    - Use of hiring manager's name or specific details

    2. OPENING STRENGTH (Weight: 20%)
    Score 1-4 based on:
    - Attention-grabbing first sentence that hooks the reader
    - Avoidance of generic openings ("I am writing to apply...")
    - Memorable and relevant opening that connects to the role
    - Clear statement of position being sought
    - Immediate value proposition or key qualification

    3. VALUE PROPOSITION (Weight: 25%)
    Score 1-4 based on:
    - Clear articulation of what candidate brings to the company
    - Specific, quantifiable achievements and results
    - Direct connection between experience and job requirements
    - Focus on solving employer problems vs. candidate needs
    - Use of STAR method (Situation, Task, Action, Result) examples

    4. AUTHENTICITY AND VOICE (Weight: 15%)
    Score 1-4 based on:
    - Distinctive personal voice that sounds human, not robotic
    - Professional but conversational tone
    - Appropriate personality and enthusiasm
    - Avoidance of clichés and corporate buzzwords
    - Confidence without arrogance

    5. STRUCTURE AND EXECUTION (Weight: 15%)
    Score 1-4 based on:
    - Proper business letter format and length (3-4 paragraphs, 1 page max)
    - Logical flow and smooth transitions between paragraphs
    - Strong closing with call to action
    - Grammar, spelling, and proofreading quality
    - Active voice and strong action verbs

    Scoring Scale:
    4 = Excellent - Exceeds expectations, best practices followed
    3 = Good - Solid execution with minor areas for improvement  
    2 = Fair - Adequate but needs significant enhancement
    1 = Poor - Major issues requiring substantial revision

    Red Flags to Identify:
    - Generic template language that could apply to any job
    - Focus on what candidate wants rather than what they offer
    - No evidence of company research or role understanding
    - Repetition of resume without adding new value
    - Grammatical errors, typos, or formatting issues
    - Overly humble, self-deprecating, or demanding tone
    - Buzzwords without substance or specific examples
    - Missing call to action or weak closing

    output format should be:
    overall score out of 100,
    research_and_customization feedback,
    opening_strength feedback,
    value_proposition feedback,
    authenticity_and_voice feedback,
    structure_and_execution feedback,
    Remember: Your goal is to help create cover letters that get noticed by hiring managers while sounding authentically human. Be thorough, specific, and actionable in your analysis.

    COVER LETTER TO ANALYZE:
    {cover_letter}\n

    JOB DESCRIPTION (for context):
    {job_description}\n
    
    Resume (for context)
    {resume}"""
    
    response = model.invoke([SystemMessage(content=system_msg)])
    print(response)
    if response['overall'] < 90:
        return {
            "satisfied": False,
            "message": add_messages(
                state['message'],
                HumanMessage(content=f"The cover letter needs significant improvement based on the analysis provided.{response}"),
            )
        }
    else:
        return {
            "satisfied": True,
            "message": add_messages(
                state['message'],
                HumanMessage(content=f"The cover letter is strong and meets the requirements.{response}"),
            )
        }

graph=StateGraph(cvState)
graph.add_node('load_resume',load_resume)
graph.add_node('jd_loader',jd_loader)
graph.add_node('generate_cover_letter',generate_cover_letter)
graph.add_node('review_cover_letter_condition', review_cover_letter_condition)


graph.add_edge(START,'load_resume')
graph.add_edge(START,'jd_loader')
graph.add_edge('load_resume','generate_cover_letter')
graph.add_edge('jd_loader','generate_cover_letter')
graph.add_edge('generate_cover_letter','review_cover_letter_condition')
graph.add_conditional_edges('review_cover_letter_condition',check_condition,{
    END:END,
    'continue':'generate_cover_letter'
})


app=graph.compile()

# Export app for import in streamlit_app.py
__all__ = ["app"]
file_path="Prit_italiya.pdf"
init_state={
    'resume_path': file_path,
    'jd_url': "https://careers.unitedhealthgroup.com/job/22240700/ai-ml-engineer-genai-llms-python-nlp-noida-in/?p_sid=X-Ddq7b&p_uid=XezwJiRnZh&src=JB-22511&ss=paid&utm_campaign=india+tech&utm_content=pj_board&utm_medium=jobad&utm_source=indeed&gad_source=7",
    'iterations':0,
    'message':[]
}
final_state=app.invoke(init_state)
print(final_state['cover_letter'])