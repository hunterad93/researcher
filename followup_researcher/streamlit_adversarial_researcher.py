import streamlit as st
import requests

# Constants for system prompts
ONLINE_SYSTEM_PROMPT = """Act as an advocate for the company you are asked about. Conclude your response with a list of URLS used from your search."""

OFFLINE_SYSTEM_PROMPT = """You are an AI assistant who is trying to get specific information about data brokers 
from a conversation partner who is connected to the internet. Your **ONLY** concern is the accuracy of the data, because 
you are investigating on behalf of advertisers who are paying for the data."""

SUMMARY_PROMPT = "Be precise and concise. Only provide the summary, without restating the question, or giving additional context or explanation. Make the summary sound like a natural, human explanation rather than a marketing spiel."    


def send_perplexity_message(conversation_history, model, system_prompt):
    url = "https://api.perplexity.ai/chat/completions"
    
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {st.secrets['PPLX_API_KEY']}"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response_data = response.json()
    
    if 'choices' in response_data and len(response_data['choices']) > 0:
        return response_data['choices'][0]['message']['content']
    else:
        print(response_data)
        return "Error: Unable to get a response from the LLM"

def create_conversation(domain, data_type, num_iterations=3):
    online_model = "llama-3-sonar-large-32k-online"
    offline_model = "llama-3-sonar-large-32k-chat"
    
    online_conversation = []
    offline_conversation = []
    display_conversation = []
    
    # Initial query
    initial_prompt = f"Answer this question: how does {domain} collect {data_type} data that it sells to advertisers?"
    online_conversation.append({"role": "user", "content": initial_prompt})
    display_conversation.append({"role": "user", "content": initial_prompt})
    
    for i in range(num_iterations):
        # Get response from online model
        online_response = send_perplexity_message(online_conversation, online_model, ONLINE_SYSTEM_PROMPT)
        online_conversation.append({"role": "assistant", "content": online_response})
        display_conversation.append({"role": "assistant", "content": online_response})
        
        # Update offline conversation
        offline_conversation = online_conversation.copy()
        
        if i < num_iterations - 1:
            # Generate follow-up question using offline model
            follow_up_prompt = "Based on the previous conversation, generate a follow-up question to get more specific information. Phrase it as if you're the original user seeking clarification. Only provide the question, without any additional context or explanation."
            offline_conversation.append({"role": "user", "content": follow_up_prompt})
            follow_up_question = send_perplexity_message(offline_conversation, offline_model, OFFLINE_SYSTEM_PROMPT)
            
            # Add follow-up question to conversations
            online_conversation.append({"role": "user", "content": follow_up_question})
            display_conversation.append({"role": "user", "content": follow_up_question})
    
    return offline_conversation, initial_prompt

def summarize_conversation(initial_prompt, conversation_history):
    summary_prompt = f"""Based on the following conversation about '{initial_prompt}', provide a concise summary for a non-technical advertiser. 
    Focus on answering the initial question and find a single answer to satisfy the question. Keep it brief and easy to understand.

    Conversation:
    {conversation_history}

    Summary:"""
    
    summary = send_perplexity_message([{"role": "user", "content": summary_prompt}], "llama-3-sonar-large-32k-chat", SUMMARY_PROMPT)
    return summary
def create_markdown_document(initial_prompt, conversation_history):
    markdown = f"# Adversarial conversation on Question: {initial_prompt}\n\n"
    for message in conversation_history:
        role = "Online Model" if message["role"] == "assistant" else "Offline Model"
        markdown += f"## {role}\n\n{message['content']}\n\n"
    
    # Add summary section
    summary = summarize_conversation(initial_prompt, conversation_history)
    markdown += f"## Summary for Advertisers\n\n{summary}\n"
    
    return markdown

def main():
    st.title("Adversarial Research Agent")
    
    domain = st.text_input("Enter the data provider domain (e.g. lotame.com, acxiom.com):")
    data_type = st.text_input("Enter the data type (e.g. behavioral, demographic, etc.):")
    num_iterations = st.slider("Number of follow-up questions:", 1, 5, 3)
    
    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            conversation_history, initial_prompt = create_conversation(domain, data_type, num_iterations)
            qa_result = create_markdown_document(initial_prompt, conversation_history)
        
        st.markdown("## Question-Answer Conversation")
        st.markdown(qa_result)
        st.download_button(
            "Download Q&A Conversation", 
            qa_result, 
            file_name="question_answer_conversation.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()