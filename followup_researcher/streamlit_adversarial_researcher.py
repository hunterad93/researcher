import streamlit as st
import requests
from pinecone_utils import get_cached_summary, cache_summary


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
    online_model = "llama-3.1-sonar-large-128k-online"
    offline_model = "llama-3.1-sonar-large-128k-chat"
    
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
    # Generate summary first
    summary = summarize_conversation(initial_prompt, conversation_history)
    
    # Create markdown document with summary at the top
    markdown = f"## Summary for Advertisers\n\n{summary}\n\n"
    markdown += "## LLM Research Conversation\n\n"
    
    for message in conversation_history:
        role = "Online Model" if message["role"] == "assistant" else "Offline Model"
        markdown += f"### {role}\n\n{message['content']}\n\n"
    
    return markdown

def main():
    st.title("Data Broker Research")

    password = st.text_input("Enter password:", type="password")
    if password != st.secrets["app_password"]:  # Ensure this key exists in your secrets
        st.error("Incorrect password. Please try again.")
        return  # Exit the main function if the password is incorrect
    
    # Initialize session state
    if 'qa_result' not in st.session_state:
        st.session_state.qa_result = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'cached_summary' not in st.session_state:
        st.session_state.cached_summary = None
    if 'show_regenerate' not in st.session_state:
        st.session_state.show_regenerate = False

    domain = st.text_input("Enter the data provider name (e.g. Acxiom, Lotame, Oracle, Ameribase, Skydeo etc.):")
    data_type = st.text_input("Enter the data category (e.g. behavioral, demographic) or segment (e.g. coffee drinker enthusiast, frequent traveler, etc.):")
    num_iterations = 3
    
    initial_prompt = f"Answer this question: how does {domain} collect {data_type} data that it sells to advertisers?"
    
    if st.button("Research"):
        st.session_state.show_regenerate = False
        cached_summary = get_cached_summary(initial_prompt)
        
        if cached_summary:
            st.session_state.summary = cached_summary['summary']
            st.session_state.qa_result = None
            st.session_state.show_regenerate = True
            st.info("Displaying cached summary. Click 'Generate New Research' for fresh results and full research document.")
            st.markdown(st.session_state.summary)
        else:
            with st.spinner("Processing..."):
                conversation_history, _ = create_conversation(domain, data_type, num_iterations)
                st.session_state.qa_result = create_markdown_document(initial_prompt, conversation_history)
                st.session_state.summary = summarize_conversation(initial_prompt, conversation_history)
                # Cache the new summary
                cache_summary(domain, data_type, initial_prompt, st.session_state.summary)

    if st.session_state.show_regenerate:
        if st.button("Generate New Research"):
            with st.spinner("Processing..."):
                conversation_history, _ = create_conversation(domain, data_type, num_iterations)
                st.session_state.qa_result = create_markdown_document(initial_prompt, conversation_history)
                st.session_state.summary = summarize_conversation(initial_prompt, conversation_history)
                # Cache the new summary
                cache_summary(domain, data_type, initial_prompt, st.session_state.summary)
            st.session_state.show_regenerate = False

    
    if st.session_state.qa_result:
        st.markdown(st.session_state.qa_result)

if __name__ == "__main__":
    main()