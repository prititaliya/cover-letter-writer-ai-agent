from time import time
from dotenv import load_dotenv
from pydantic import BaseModel
load_dotenv()
from langgraph.types import interrupt,Command
import vertexai
from langgraph.checkpoint.memory import MemorySaver


vertexai.init(project="Generative Language API Key", location="us-central1")
import hashlib
import os
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated, TypedDict, final
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_community.tools import DuckDuckGoSearchResults

class gen_structure(TypedDict):
    content:Annotated[str, "The content of the generated cover letter"]
    critism:Annotated[str, "Any criticism or feedback on the generated cover letter"]
    
model_name="gpt-4o-mini"
gen_model = init_chat_model(
    model=model_name,
    temperature=0.2,
)
gen_model = gen_model.with_structured_output(gen_structure)
class cvState(TypedDict):
    resume_path: str                
    resume_text: str                
    resume_store_path: str          

    jd_url: str       
    jd_text: str      
    jd_store_path: str

    cover_letter: str
    satisfied: bool                 
    message: add_messages           
    iterations: int       
    
def load_resume(state: cvState):
    persist_dir="chroma_resume"
    is_available=False
    if os.path.exists(persist_dir):
        resume_vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        is_available=True
        
        data = resume_vectorstore.get()

        docs = data["documents"]
        all_text = " ".join([doc for doc in docs])
        return {
            "resume_store_path": persist_dir,
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
            resume_vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(),
                                                       persist_directory=persist_dir)

        # Persist index
        resume_vectorstore.persist()
        data = resume_vectorstore.get()

        docs = data["documents"]
        all_text = " ".join([doc for doc in docs])
        return {
            "resume_store_path": persist_dir,
            "resume_text": all_text
        }

def jd_loader(state:cvState):
    # Handle direct text input (when jd_text is provided and jd_url is empty)
    if state.get('jd_text') and not state.get('jd_url'):
        jd_text = state['jd_text']
        jd_hash = hashlib.sha256(jd_text.encode('utf-8')).hexdigest()
        persist_dir = f"jd_Chroma_index_{jd_hash}"
        
        # Create documents from the direct text
        from langchain.schema import Document
        doc = Document(page_content=jd_text, metadata={'source': 'direct_input'})
        
        splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitters.split_documents([doc])
        
        for chunk in chunks:
            chunk.metadata['hash'] = jd_hash
        
        jd_vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(),
                                               persist_directory=persist_dir)
        jd_vectorstore.persist()
        
        return {
            "jd_store_path": persist_dir,
            "jd_text": jd_text
        }
    
    # Handle URL-based input (existing functionality)
    elif state.get('jd_url'):
        jd_hash = hashlib.sha256(state['jd_url'].encode('utf-8')).hexdigest()
        persist_dir = f"jd_Chroma_index_{jd_hash}"
        is_available=False
        if os.path.exists(persist_dir):
            jd_vectorstore = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
            is_available=True
            data = jd_vectorstore.get()

            docs = data["documents"]
            all_text = " ".join([doc for doc in docs])
            return {
                "jd_store_path": persist_dir,
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
                jd_vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(),
                                                       persist_directory=persist_dir)

            # Persist index
            jd_vectorstore.persist()

            data = jd_vectorstore.get()

            docs = data["documents"]
            all_text = " ".join([doc for doc in docs])
            
            return {
                "jd_store_path": persist_dir,
                "jd_text": all_text
            }
    else:
        raise ValueError("Either jd_url or jd_text must be provided")


def generate_cover_letter(state:cvState):
    resume_text = state['resume_text']
    jd_text = state['jd_text']
    system_msg = f"""You are an expert cover letter writer following the CS Co-op Workshop guidelines from the University of Manitoba. Create personalized, compelling cover letters using this framework:
Core Principles:

Cover letters must showcase passion, personality, and demonstrate FIT between candidate and employer
Maximum one page, single-spaced, business block format
Each letter must be specifically tailored - no generic templates
Write authentically in the candidate's voice, not overly formal AI language
Focus on what the candidate offers the employer, not what they want from the job

Required Structure:
Header Format:
[Name]
[Address • City, Province Postal Code]
[Phone • Email • LinkedIn]

[Date]

[Company Name]
RE: [Position Title]

[Salutation - use specific name or omit, NEVER "To whom it may concern"]
Four-Paragraph Formula:
Paragraph 1 - Company Research & Interest (5-7 sentences)

Open with original statement about the position (avoid clichés)
Demonstrate specific knowledge about the company (recent news, projects, values)
Explain genuine interest in THIS employer specifically
Mention networking connections if applicable (Speed Meet & Greets, current employees)
Show you understand what they do and why it matters to you

Paragraph 2 - Technical/Hard Skills Match (5-7 sentences)

Directly match your technical skills to job requirements
Provide specific examples: "When programming, I ensure code is documented..."
Include programming languages, tools, methodologies with context
Mention relevant coursework with outcomes (grades only if exceptional)
Never mention skills you lack or apologize for missing qualifications
Vary sentence starters (avoid multiple "I" beginnings)

Paragraph 3 - Soft Skills & Experience (5-7 sentences)

Highlight transferable soft skills from the posting
Support with concrete examples from work/volunteer/academic experience
Include: teamwork, communication, problem-solving, leadership examples
Quantify impact where possible (number of people, percentage improvements)
Connect experiences directly to role requirements

Paragraph 4 - Confident Closing (4-5 sentences)

Restate fit: "I believe my [specific skills] combined with [attributes] make me an ideal candidate"
Reiterate genuine interest in the position
Express desire to discuss further: "I look forward to the opportunity to meet..."
Thank them for time and consideration
Sign with "Kind regards" or "Sincerely" with e-signature

Writing Guidelines:
Language Rules:

Write in active voice with strong action verbs
No first person pronouns to start multiple sentences in a row
Use professional but conversational tone - let personality show
Match formality level to company culture
Avoid buzzwords without substance

Content Rules:

Every claim must be backed with evidence/examples
Focus on accomplishments and results, not just duties
Show how you add value based on research
Demonstrate understanding of their challenges/needs
Connect your experience to their requirements explicitly

Quality Checks:

Zero typos or grammatical errors
Consistent formatting with resume
Appropriate white space and readability
Verify company name, position title, contact details
Ensure it sounds like a real person, not AI-generated

Common Mistakes to Avoid:

Generic openings: "I am writing to apply for..."
Listing responsibilities without impact
Focusing on what you want to gain
Restating resume without adding value
Using outdated formal language
Making unsupported claims
Being too modest or too boastful

Personality Integration:
The cover letter must:

Sound conversational yet professional
Include specific details that show genuine interest
Reflect the candidate's authentic voice
Demonstrate enthusiasm naturally
Show character through example selection
Balance confidence with humility

Final Review Questions:

Does this letter show specific knowledge about THIS company?
Have I proven I have what they're looking for with examples?
Does my personality and enthusiasm come through?
Would this make them want to meet me?
Is every sentence necessary and impactful?
Does it complement, not repeat, my resume?

Remember: The cover letter gets you 50% there - it must make them WANT to interview you by showing you're not just qualified, but genuinely interested and a great fit for their specific team and culture.

    RESUME:
    {resume_text}

    JOB DESCRIPTION:
    {jd_text}"""
    
    cover_letter = gen_model.invoke(system_msg)
    return {
        "cover_letter": cover_letter,
        "iterations": state['iterations'] + 1
    }
    
def check_condition(state:cvState):
     if(state['iterations'] > 5 or state['satisfied']):
        return END
     else:
        return 'continue'

def review_cover_letter_condition(state:cvState):
    human_approval=interrupt("Enhance the cover letter yes or no")
    if human_approval.lower() == "no":
       return Command(
            goto=END,
            update={
                "cover_letter": cover_letter,
                "iterations": state['iterations'] + 1,
                "satisfied":True
            }
        )
        
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

    model=init_chat_model(
    model=model_name,  
    temperature=0.2,
).with_structured_output(coverLetterAnalyser)

    system_msg = f"""
   You are an expert cover letter reviewer using the CS Co-op Workshop criteria from the University of Manitoba. Critically evaluate cover letters against professional standards and provide actionable feedback.
Review Framework:
INITIAL ASSESSMENT (First Impression - 6-7 second scan):

Does it look professional and well-formatted?
Is it exactly one page?
Does the opening grab attention?
Is the company name and position correct?
Are there any obvious typos in the first paragraph?

Detailed Evaluation Criteria:
1. FORMAT & STRUCTURE COMPLIANCE
Check for:

 Business block letter format
 Single-spaced with appropriate white space
 Consistent formatting with resume style
 Proper header with complete contact information
 Date and company name included
 Appropriate salutation (no "To whom it may concern")
 E-signature (not cursive font)
 Four distinct paragraphs
 Each paragraph 5-7 sentences maximum

2. PARAGRAPH 1 - RESEARCH & INTEREST
Evaluate:

 Original opening (not generic/template)
 Specific company knowledge demonstrated
 Clear position title mentioned
 Genuine interest articulated with reasons
 References to company news/projects/values
 Networking connections mentioned if applicable
 Avoids clichés like "I am writing to apply for..."

Rate: Weak / Adequate / Strong
Missing elements: _______
3. PARAGRAPH 2 - TECHNICAL SKILLS
Assess:

 Direct match to job requirements
 Specific examples provided
 Technical skills backed with context
 Avoids overstating qualifications
 Doesn't mention lacking skills
 Varied sentence structure
 Concrete evidence of capabilities

Rate: Weak / Adequate / Strong
Missing elements: _______
4. PARAGRAPH 3 - SOFT SKILLS
Review:

 Relevant soft skills highlighted
 Backed by concrete examples
 Quantified impact where possible
 Shows teamwork, communication, leadership
 Connects to role requirements
 Demonstrates transferable skills

Rate: Weak / Adequate / Strong
Missing elements: _______
5. PARAGRAPH 4 - CLOSING
Check:

 Confident statement of fit
 Reiterated interest
 Request for interview/meeting
 Professional gratitude
 Appropriate sign-off

Rate: Weak / Adequate / Strong
Missing elements: _______
Content Quality Assessment:
PERSONALITY & AUTHENTICITY (Score 1-10):

Does it sound like a real person or AI/template?
Is enthusiasm genuine or forced?
Does personality come through?
Are examples unique and personal?
Would you remember this candidate?

RESEARCH DEPTH (Score 1-10):

Generic mentions vs. specific insights?
Understanding of company challenges/needs?
Knowledge beyond basic website info?
Connection to company values/culture?
Industry awareness demonstrated?

VALUE PROPOSITION (Score 1-10):

Clear "what I offer you" focus?
Benefits to employer emphasized?
Problems they can solve identified?
Unique differentiators highlighted?
ROI for employer apparent?

Technical Review:
LANGUAGE & STYLE:

 Active voice predominant
 Strong action verbs used
 Minimal "I" sentence starters
 Professional yet conversational tone
 No buzzwords without substance
 Sentence variety (length and structure)
 Smooth flow between paragraphs

ERRORS TO FLAG:

 Spelling mistakes
 Grammar errors
 Punctuation issues
 Inconsistent formatting
 Wrong company/position names
 Repetitive phrasing
 Clichéd expressions

Red Flags to Identify:
CRITICAL ISSUES:

Generic template language
No company-specific content
Focus on what candidate wants vs. offers
Restating resume without added value
Over/understating qualifications
Negative statements about self or others
Too long (over one page)
Unprofessional tone or language

WARNING SIGNS:

Weak opening statement
No concrete examples
All telling, no showing
Passive voice throughout
No clear value proposition
Missing enthusiasm
Poor research evident
Formatting inconsistencies

Feedback Structure:
PROVIDE:

Overall Impression (2-3 sentences)

First impression impact
Overall professionalism
Memorability factor


Strengths (3-4 bullet points)

What works well
Effective examples
Strong elements to keep


Areas for Improvement (prioritized)

Critical fixes needed immediately
Important enhancements
Polish recommendations


Specific Revision Suggestions

Exact phrases to change
Examples to add/strengthen
Restructuring needs


Impact Assessment

Current state: Would this get an interview?
Potential after revisions
Key differentiators to emphasize



Scoring Rubric:
Rate each cover letter:
EXCEPTIONAL (90-100%):

Compelling and memorable
Perfect technical execution
Clear personality and fit
Would definitely get interview

STRONG (75-89%):

Solid and professional
Minor improvements needed
Good examples and research
Likely to get interview

ADEQUATE (60-74%):

Meets basic requirements
Several areas need work
Some generic elements
Might get interview

NEEDS WORK (Below 60%):

Major revisions required
Too generic or unfocused
Missing key elements
Unlikely to get interview

Final Review Questions:

The 6-Second Test: Does it hook the reader immediately?
The Fit Test: Is it obvious why THIS person for THIS job?
The Personality Test: Would you want to meet this person?
The Memory Test: What would you remember about this candidate?
The Competition Test: Does this stand out from other applicants?
The Authenticity Test: Does it sound genuine or manufactured?
The Value Test: Is it clear what the employer gains?

Review Output Format:
COVER LETTER REVIEW
==================
Position: [Job Title] at [Company]
Overall Score: [X/100]
Recommendation: [Ready to Send / Minor Revisions / Major Revisions / Complete Rewrite]

STRENGTHS:
- [Specific strength with example]
- [Specific strength with example]
- [Specific strength with example]

CRITICAL IMPROVEMENTS NEEDED:
1. [Most important fix with specific suggestion]
2. [Second priority with specific suggestion]
3. [Third priority with specific suggestion]

PARAGRAPH-BY-PARAGRAPH FEEDBACK:
[Detailed feedback for each paragraph]

TECHNICAL ISSUES:
[List any spelling, grammar, formatting problems]

FINAL VERDICT:
[Would this get an interview? Why or why not?]
Remember: Be constructive but honest. The goal is a cover letter that gets interviews, not just meets minimum requirements.
    COVER LETTER TO ANALYZE:
    {cover_letter}\n

    JOB DESCRIPTION (for context):
    {job_description}\n
    
    Resume (for context)
    {resume}"""
    
    response = model.invoke([SystemMessage(content=system_msg)])
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
graph.add_edge('generate_cover_letter',END)
graph.add_conditional_edges('review_cover_letter_condition',check_condition,{
    END:END,
    'continue':'generate_cover_letter'
})

config={
    'configurable':{
        'thread_id':10
    }
}
app=graph.compile(checkpointer=MemorySaver())

# Export app for import in streamlit_app.py
__all__ = ["app"]
file_path="Prit_italiya.pdf"
init_state={
    'resume_path': file_path,
    'jd_url': "https://careers.unitedhealthgroup.com/job/22240700/ai-ml-engineer-genai-llms-python-nlp-noida-in/?p_sid=X-Ddq7b&p_uid=XezwJiRnZh&src=JB-22511&ss=paid&utm_campaign=india+tech&utm_content=pj_board&utm_medium=jobad&utm_source=indeed&gad_source=7",
    'iterations':0,
    'message':[]
}
final_state=app.invoke(init_state,config=config,stream_mode='updates')
# print(final_state['cover_letter'])

while((app.get_state(config) !=())):
    user_input = input("Do you want to enhance your resume more? ")
    if user_input.lower() == "yes":
        final_state=app.invoke(Command(resume="yes"),config=config,stream_mode='updates')
    else:
      break

print(final_state[-2]['generate_cover_letter']['cover_letter']['content'])