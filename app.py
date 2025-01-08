from crewai_tools import BaseTool, CSVSearchTool
import streamlit as st
from PIL import Image
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from crewai.agent import Agent
from crewai.task import Task
from crewai.crew import Crew
from crewai.process import Process
# retrieve purchase history tool 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import openai
from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import os
import tiktoken
import re
from gradio_client import Client as GradioClient, file, handle_file
import cv2


# Please fill your openai api key

openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

#csv_search_tool_history = CSVSearchTool("Dataset/Customer_Interaction_Data.csv")

#csv_search_tool_product = CSVSearchTool("Dataset/final_product_catalog.csv")
df = pd.read_csv('Dataset/Customer_Interaction_Data_v2.csv')
df_products = pd.read_csv('Dataset/final_product_catalog.csv')

# 1. Create Vector Database
def load_vector_db():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)
    #vector_db = FAISS.from_documents(documents, embeddings)
    vector_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    return vector_db

def retrieve_transcation(cust_id):
    return df[df['Customer_ID'] == cust_id].head(3).to_dict()

# 2. Retrieval Agent
def retrieve_documents(query, vector_db, top_k=5):
    return vector_db.similarity_search(query, top_k)

# 3. Generator Agent
def generate_response_openai(query, docs, purchase_hist):
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Answer the following question based on the context:\n\nContext: {context}\n by providing a similarity with user purchase history:\n\n History : {purchase_hist}\n\n Question: {query}. "
        "Provide detailed and accurate answer with maximum 3 products. "
        "Always include the reason. "
        "If the question is product related, always attach product id"
    )

    completion = openai.chat.completions.create(
        model="gpt-4o",  # Adjust the model name as per availability
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer accurately and give reason"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# 4. Multi-Agent System
def multi_agent_rag(query, vector_db, purchase_hist):
    retrieved_docs = retrieve_documents(query, vector_db)
    return generate_response_openai(query, retrieved_docs, purchase_hist)

# Streamlit Interface
st.title("💬 Product Recommendation Chatbot")

# Inisialisasi sesi untuk menyimpan percakapan dan ID pelanggan
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! Please provide your Customer ID to start."}]
if "customer_id" not in st.session_state:
    st.session_state["customer_id"] = None
if "clicked_button" not in st.session_state:
    st.session_state.clicked_button = None
if "product_ids" not in st.session_state:
    st.session_state.product_ids = []
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "product_url" not in st.session_state:
    st.session_state.product_url = None
if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None
if "waiting_for_image" not in st.session_state:
    st.session_state.waiting_for_image = False

# log debug for button click
    
def handle_click(action, product_id, url):
    st.session_state.clicked_button = f"{action} for {product_id}"
    
    st.session_state.product_id = product_id
    st.session_state.product_url = url

    st.session_state.waiting_for_image = True

# virtual try on function
def virtual_tryon(garment_img_path, person_img_path):
    gradio_client = GradioClient("Nymbo/Virtual-Try-On")
    result = gradio_client.predict(
                dict={"background": file(person_img_path), "layers": [], "composite": None},
                garm_img=handle_file(garment_img_path),
                garment_des="",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
    try_on_image_path = result[0]
    img = cv2.imread(try_on_image_path)
    cv2.imwrite("result.png", img)

    return Image.open("result.png")

# Menampilkan percakapan sebelumnya
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input pengguna
if prompt := st.chat_input(placeholder="Type here for recommend product..."):
    # Simpan dan tampilkan input pengguna
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Respons dari asisten
    if not st.session_state["customer_id"]:
        # Jika belum ada Customer ID, minta pengguna memasukkan ID
        st.session_state["customer_id"] = prompt
        if st.session_state["customer_id"] not in df["Customer_ID"].values:
            response = "Customer ID not found. Please try again."
            st.session_state["customer_id"] = None  # Reset ID
        else:
            response = f"Thank you! Customer ID '{st.session_state['customer_id']}' has been verified. How can I assist you?"
    else:
        # Proses permintaan dengan Crew
        try:
            inputs = {"query": prompt, "customer": st.session_state["customer_id"]}
            vector_db = load_vector_db()
            transaction_data = retrieve_transcation(st.session_state["customer_id"])
            response = multi_agent_rag(inputs['query'], vector_db, transaction_data)
        except Exception as e:
            response = f"An error occurred: {e}"

    # Simpan dan tampilkan respons dari asisten
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
    
    # Extract raw output
    output = response
    print(output)
    # Regular expression pattern to extract Product ID
    pattern = r"(?i)\b(PROD\d+)\b"
    product_ids = re.findall(pattern, output)
    print(product_ids)
    # Menghilangkan duplikasi
    unique_product_ids = list(set(product_ids))
    st.session_state.product_ids = unique_product_ids
    print("Produk ID: ",unique_product_ids)

if st.session_state.product_ids:   
    for product_id in st.session_state.product_ids:
        # validation product_ids in df_product
        filtered_df = df_products[df_products['Product_ID'] == product_id]
        if not filtered_df.empty:
            # Get image URL or file path
            url = filtered_df['Url_Image'].iloc[0]
            img = Image.open(url)
            img.thumbnail((300, 600)) # Maksimum lebar dan tinggi
            
            # Menggunakan tiga kolom untuk memusatkan elemen
            col1, col2, col3 = st.columns([1, 2, 1])  # Rasio kolom: kiri, tengah, kanan
            with col2:  # Konten di kolom tengah
                st.subheader(f"Product ID: {product_id}")
                st.image(img, caption=f"Product ID: {product_id}")
                st.button(
                    f"Virtual Try-On for {product_id}",
                    key=f"try_{product_id}",
                    on_click=handle_click,
                    args=("Try ", product_id, url),
                    )

    # if it's waiting for image
    if st.session_state.waiting_for_image:
        # upload image
        uploaded_image = st.file_uploader("Please upload person image:", type=["jpg", "png", "jpeg"])
        print("tes upliad:", uploaded_image)
        if uploaded_image:
            st.session_state.uploaded_image = uploaded_image  # Store uploaded image in session state
            st.session_state.uploaded_image_name = uploaded_image.name  # Store image name
            image = Image.open(uploaded_image)
            image.thumbnail((300,600))
            st.image(image, caption="Your image")
            with open(uploaded_image.name, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.success(f"Image uploaded and saved successfully!")
            
            with st.spinner('Virtual try on is running...'):
                print(st.session_state.product_url)
                result_vto = virtual_tryon(st.session_state.product_url, uploaded_image.name)
            st.success("Done!")

            # display the result
            result_vto.thumbnail((300, 600))
            st.image(result_vto, caption=f"Try on for {st.session_state.product_id}")
            
            # print details
            # print(f"Customer ID: {st.session_state['customer_id']}")
            # print(f"Product ID: {st.session_state.product_id}")
            # print(f"Product URL: {st.session_state.product_url}")
            # print(f"Uploaded Image: {uploaded_image.name}")
            
            # stop asking for the image
            st.session_state.waiting_for_image = False
