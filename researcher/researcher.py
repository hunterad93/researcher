import re
import streamlit as st
import requests

def send_perplexity_message(message, conversation_history):
    url = "https://api.perplexity.ai/chat/completions"
    
    conversation_history.append({"role": "user", "content": message})
    
    payload = {
        "model": "llama-3-sonar-large-32k-online",
        "messages": [
            {"role": "system", "content": "Try to be specific in your responses. Conclude your response with a list of URLS used from your search."}
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

def research_topic(main_topic):
    conversation_history = []
    
    # Step 1: Get overview
    overview = send_perplexity_message(f"Provide an overview of {main_topic} with a numbered list of subtopics.", conversation_history)
    
    # Step 2: Extract subtopics
    subtopics = extract_subtopics(overview)
    
    # Step 3: Research each subtopic
    detailed_research = [overview]
    for subtopic in subtopics:
        subtopic_info = send_perplexity_message(f"Provide detailed information about {subtopic} in the context of {main_topic}.", conversation_history)
        detailed_research.append(subtopic_info)
    
    # Step 4: Create markdown document
    markdown_doc = create_markdown_document(main_topic, detailed_research)
    
    return markdown_doc

def create_markdown_document(topic, research_data):
    markdown = f"# Research on {topic}\n\n"
    markdown += "## Overview\n\n"
    markdown += research_data[0] + "\n\n"
    
    for i, subtopic_data in enumerate(research_data[1:], 1):
        markdown += f"## Subtopic {i}\n\n"
        markdown += subtopic_data + "\n\n"
    
    return markdown

# Streamlit app
def main():
    st.title("Research Assistant")
    
    topic = st.text_input("Enter a research topic:")
    if st.button("Start Research"):
        with st.spinner("Researching..."):
            research_result = research_topic(topic)
        
        st.markdown(research_result)
        st.download_button("Download Research", research_result, file_name=f"{topic}_research.md")

if __name__ == "__main__":
    main()