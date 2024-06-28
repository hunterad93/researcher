import re
import streamlit as st
import requests
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
    print(payload)
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

def select_best_subtopic(main_topic, subtopics):
    prompt = f"Given the main topic '{main_topic}', which of the following subtopics is most useful to understand further? Respond with only the number of the best subtopic.\n\n"
    for i, subtopic in enumerate(subtopics, 1):
        prompt += f"{i}. {subtopic}\n"
    
    response = send_perplexity_message(
        prompt,
        [],
        model="llama-3-70b-instruct",
        system_prompt="You are a research assistant. Select the most relevant subtopic."
    )
    
    try:
        selected_index = int(response.strip()) - 1
        return subtopics[selected_index]
    except (ValueError, IndexError):
        return subtopics[0]  # Default to the first subtopic if parsing fails

def research_topic(main_topic, max_iterations=3):
    research_prompt = "Conclude your message with a list of URLs used from your search."
    
    progress_bar = st.progress(0)
    progress_text = st.empty()

    detailed_research = []
    current_topic = main_topic

    for iteration in range(max_iterations):
        progress_text.text(f"Iteration {iteration + 1}: Getting subtopics...")
        subtopics_response = send_perplexity_message(
            f"Provide a numbered list of up to 5 key subtopics related to: {current_topic}",
            [],
            system_prompt=research_prompt
        )
        subtopics = extract_subtopics(subtopics_response)
        
        progress_text.text(f"Iteration {iteration + 1}: Selecting best subtopic...")
        best_subtopic = select_best_subtopic(main_topic, subtopics)
        
        progress_text.text(f"Iteration {iteration + 1}: Researching '{best_subtopic}'...")
        subtopic_info = send_perplexity_message(
            f"Provide concise, specific information about '{best_subtopic}' in the context of the question: '{main_topic}'.",
            [],
            system_prompt=research_prompt
        )
        detailed_research.append((current_topic, subtopic_info))
        
        current_topic = best_subtopic
        progress_bar.progress((iteration + 1) * 100 // max_iterations)

    markdown_doc = create_markdown_document(main_topic, detailed_research)
    
    progress_text.text("Generating summary...")
    summary = generate_summary(main_topic, detailed_research)
    summary_markdown = create_summary_markdown(main_topic, summary)
    
    progress_bar.progress(100)
    progress_text.text("Research complete!")
    
    return markdown_doc, summary_markdown

def create_markdown_document(topic, research_data):
    markdown = f"# Research on {topic}\n\n"
    
    for i, (subtopic, data) in enumerate(research_data, 1):
        markdown += f"## {i}. {subtopic}\n\n"
        markdown += data + "\n\n"
    
    return markdown

def generate_summary(main_topic, research_data):
    summary_prompt = f"Provide a concise, specific answer to the question: {main_topic}. Focus only on the most relevant information from the following research:\n\n"
    for subtopic, data in research_data:
        summary_prompt += f"- {subtopic}: {data}\n\n"
    summary_prompt += "Limit the response to 3-5 key points."
    
    summary = send_perplexity_message(
        summary_prompt,
        [],
        model="llama-3-70b-instruct",
        system_prompt="Be precise and concise."
    )
    
    return summary

def generate_summary_openai(summary_prompt):

    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    
    summary_system_prompt = "Be precise and concise."
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": summary_system_prompt},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content


# Update the create_summary_markdown function
def create_summary_markdown(topic, summary):
    return f"# Summary of Research on {topic}\n{summary}"

# Streamlit app
def main():
    st.title("Research Assistant")
    
    topic = st.text_input("Enter a research topic:")
    if st.button("Start Research"):
        try:
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
            
            st.markdown("\n\n## Summary")
            st.markdown(summary_result)
            st.download_button(
                "Download Summary", 
                summary_result, 
                file_name=f"{topic}_summary.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred during research: {str(e)}")

if __name__ == "__main__":
    main()