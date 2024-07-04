import streamlit as st
import requests

# Constants for system prompts
ONLINE_SYSTEM_PROMPT = """Always provide a list of links used in your search after each response. 
Advocate for the accuracy of the data you are discussing, as you are speaking on behalf of the 3rd party data seller."""

OFFLINE_SYSTEM_PROMPT = """You are an AI assistant who is trying to get specific information about 3rd party 
data collectors from another AI who is connected to the internet. Your only concern is the accuracy of the data, because 
you are investigating on behalf of advertisers who are paying for the data."""

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

def create_conversation(domain, data_vertical, num_iterations=3):
    online_model = "llama-3-sonar-large-32k-online"
    offline_model = "llama-3-70b-instruct"
    
    online_conversation = []
    offline_conversation = []
    display_conversation = []
    
    # Initial query
    initial_prompt = f"Answer this question: how does {domain} collect {data_vertical} data that it sells to advertisers?"
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

def create_markdown_document(initial_prompt, conversation_history):
    markdown = f"# Adversarial conversation on Question: {initial_prompt}\n\n"
    for message in conversation_history:
        role = "Online Model" if message["role"] == "assistant" else "Offline Model"
        markdown += f"## {role}\n\n{message['content']}\n\n"
    return markdown

def main():
    st.title("Conversational Question Answering Assistant")
    
    domain = st.text_input("Enter the data provider domain (e.g. lotame.com):")
    data_vertical = st.text_input("Enter the data vertical (e.g. healthcare):")
    num_iterations = st.slider("Number of follow-up questions:", 1, 5, 5)
    
    if st.button("Get Answer"):
        with st.spinner("Processing..."):
            conversation_history, initial_prompt = create_conversation(domain, data_vertical, num_iterations)
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