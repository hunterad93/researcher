import re
import streamlit as st
import requests
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


def send_perplexity_message(message, conversation_history, model="llama-3-sonar-large-32k-online", system_prompt=""):
    url = "https://api.perplexity.ai/chat/completions"
    
    conversation_history.append({"role": "user", "content": message})
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt}
        ] + conversation_history
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {st.secrets['PPLX_API_KEY']}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    
    if 'choices' in response_data and len(response_data['choices']) > 0:
        ai_response = response_data['choices'][0]['message']['content']
        conversation_history.append({"role": "assistant", "content": ai_response})
        print(ai_response)
        return ai_response
    else:
        return "Error: Unable to get a response from the API"


def generate_subquestions(main_question):
    prompt = f"Given the question '{main_question}', provide three google search queries that will shed light on the question. Format the response as a numbered list."
    
    response = send_perplexity_message(
        prompt,
        [],
        model="llama-3-70b-instruct",
        system_prompt="You are a research assistant."
    )
    
    # Use regex to find numbered items
    subqueries = re.findall(r'\d+\.\s*(.*)', response)
    
    # If no numbered items found, return the whole response as a single subquestion
    if not subqueries:
        return [response.strip()]
    
    return subqueries

def research_subquestion(subquery):
    research_prompt = f"Research the following google search query: {subquery}"
    
    response = send_perplexity_message(
        research_prompt,
        [],
        model="llama-3-sonar-large-32k-online",
        system_prompt="Provide a concise and response. Include relevant facts, examples, and explanations."
    )
    
    return response

def summarize_research(main_question, subqueries, answers):
    summary_prompt = f"Summarize the following research to answer the main question: '{main_question}'\n\n"
    for q, a in zip(subqueries, answers):
        summary_prompt += f"Subquery: {q}\nAnswer: {a}\n\n"
    summary_prompt += "Provide a concise summary that addresses the main question based on this research."
    
    summary = send_perplexity_message(
        summary_prompt,
        [],
        model="llama-3-70b-instruct",
        system_prompt="Synthesize the information and provide a clear, concise summary."
    )
    
    return summary

def markdown_to_pdf(markdown_content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    # Convert list to string if necessary
    if isinstance(markdown_content, list):
        markdown_content = '\n'.join(markdown_content)

    # Split the markdown content into lines
    lines = markdown_content.split('\n')
    for line in lines:
        if line.startswith('# '):
            flowables.append(Paragraph(line[2:], styles['Title']))
        elif line.startswith('## '):
            flowables.append(Paragraph(line[3:], styles['Heading2']))
        else:
            flowables.append(Paragraph(line, styles['BodyText']))

    doc.build(flowables)
    buffer.seek(0)
    return buffer

def main():
    st.title("Focused Research Assistant")
    
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'main_question' not in st.session_state:
        st.session_state.main_question = ""

    main_question = st.text_input("Enter your research question:", value=st.session_state.main_question)
    st.session_state.main_question = main_question

    if st.button("Start Research"):
        try:
            with st.spinner("Researching..."):
                # Generate subquestions
                subqueries = generate_subquestions(main_question)
                
                # Research each subquestion
                answers = []
                for i, subq in enumerate(subqueries, 1):
                    st.text(f"Researching subquery {i}...")
                    answer = research_subquestion(subq)
                    answers.append(answer)
                
                # Summarize the research
                summary = summarize_research(main_question, subqueries, answers)
                
                st.session_state.research_results = list(zip(subqueries, answers))
                st.session_state.summary = summary
        except Exception as e:
            st.error(f"An error occurred during research: {str(e)}")

    # Display results if they exist in session state
    if st.session_state.research_results:
        st.markdown("## Research Results")
        for i, (subq, answer) in enumerate(st.session_state.research_results, 1):
            st.markdown(f"### Subquery {i}: {subq}")
            st.markdown(answer)
        
        st.markdown("## Summary")
        st.markdown(st.session_state.summary)
        
        # Create PDF for full research
        research_content = "# Research Results\n\n" + "\n\n".join([f"## Subquery: {subq}\n\n{answer}" for subq, answer in st.session_state.research_results])
        research_pdf = markdown_to_pdf(research_content)
        st.download_button(
            "Download Full Research (PDF)", 
            research_pdf, 
            file_name=f"{main_question}_full_research.pdf",
            mime="application/pdf"
        )
        
        # Create PDF for summary
        summary_pdf = markdown_to_pdf(f"# Summary for '{main_question}'\n\n{st.session_state.summary}")
        st.download_button(
            "Download Summary (PDF)", 
            summary_pdf, 
            file_name=f"{main_question}_summary.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()