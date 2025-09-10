from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import login
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch
app = Flask(__name__)


# Load token từ .env
load_dotenv()
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Login trực tiếp bằng code
login(token=token)



embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


# llm = ChatOpenAI()



model = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    quantization_config=bnb_config,
    token=token
)
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})




memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


