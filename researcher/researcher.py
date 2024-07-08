import re
import streamlit as st
import requests
# import anthropic
from openai import OpenAI


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
        return ai_response
    else:
        return "Error: Unable to get a response from the API"


def extract_subtopics(text):
    # Extract numbered list items
    subtopics = re.findall(r'\d+\.\s*(.*)', text)
    return subtopics

def research_topic(main_topic, max_subtopics=10):
    conversation_history = []
    research_prompt = "Conclude your response with a list of URLs used from your search."
    
    # Step 1: Get overview
    overview = send_perplexity_message(f"Provide an overview of {main_topic} with a numbered list of around 5 subtopics.", conversation_history, system_prompt=research_prompt)
    
    # Step 2: Extract subtopics
    subtopics = extract_subtopics(overview)[:max_subtopics]
    
    # Step 3: Research each subtopic
    detailed_research = [overview]
    for subtopic in subtopics:
        subtopic_info = send_perplexity_message(f"Provide detailed information about {subtopic} in the context of {main_topic}.", conversation_history, system_prompt=research_prompt)
        detailed_research.append(subtopic_info)
    
    # Step 4: Create markdown document
    markdown_doc = create_markdown_document(main_topic, detailed_research)
    
    # Step 5: Generate summary
    summary_prompt = "Be precise and concise."
    summary = generate_summary(detailed_research, summary_prompt, main_topic)
    summary_markdown = create_summary_markdown(main_topic, summary)
    
    return markdown_doc, summary_markdown

def create_markdown_document(topic, research_data):
    markdown = f"# Research on {topic}\n\n"
    markdown += "## Overview\n\n"
    markdown += research_data[0] + "\n\n"
    
    for i, subtopic_data in enumerate(research_data[1:], 1):
        markdown += f"## Subtopic {i}\n\n"
        markdown += subtopic_data + "\n\n"
    
    return markdown

def generate_summary(research_data, summary_prompt, main_topic):
    summary_request = f"Can you extract the most relevant valuable information from the research that specifically addresses the main topic: '{main_topic}'.\n\n Here is the research:\n\n" + "\n\n".join(research_data)
    summary = send_perplexity_message(
        summary_request,
        [],
        model="llama-3-70b-instruct",
        system_prompt=summary_prompt
    )
    return summary

# def generate_summary_claude(research_data, summary_prompt):
#     client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
    
#     summary_request = "Summarize the following research information concisely:\n\n" + "\n\n".join(research_data)
    
#     message = client.messages.create(
#         model="claude-3-sonnet-20240229",
#         max_tokens=1000,
#         temperature=0.7,
#         system=summary_prompt,
#         messages=[
#             {"role": "user", "content": summary_request}
#         ]
#     )
    
#     return message.content

def generate_summary_openai(research_data, summary_prompt, main_topic):

    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    
    summary_request = f"Extract the single most relevant point from the research that specifically addresses the main topic: '{main_topic}'.\n\n Here is the research:\n\n" + "\n\n".join(research_data)
    response = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": summary_request}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content


def create_summary_markdown(topic, summary):
    return f"# Summary of Research on {topic}\n\n{summary}"

# Streamlit app
def main():
    st.title("Research Assistant")
    
    topic = st.text_input("Enter a research topic:")
    if st.button("Start Research"):
        with st.spinner("Researching..."):
            research_result, summary_result = research_topic(topic)
        
        st.markdown("## Full Research")
        st.markdown(research_result)
        st.download_button(
            "Download Full Research", 
            research_result, 
            file_name=f"{topic}_research.md",
            mime="text/markdown"
        )
        
        st.markdown("\n\n##Summary")
        st.markdown(summary_result)
        st.download_button(
            "Download Summary", 
            summary_result, 
            file_name=f"{topic}_summary.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()