import gradio as gr
from langchain_ollama import OllamaLLM, ChatOllama


model_id = "qwen2.5-coder:3b"

llm = OllamaLLM(
    model=model_id,
    temperature=0.5,
    num_predict=256
)

def get_response(prompt):
    return llm.invoke(prompt)
    

demo = gr.Interface(
    fn=get_response,
    inputs = [gr.Text(placeholder="Enter the prompt")],
    outputs=gr.Text()
)

demo.launch(share=True)

