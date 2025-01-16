import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
import openai
from langchain.vectorstores import FAISS
import os
import re
from io import StringIO, BytesIO
from google.cloud import storage
import replicate

load_dotenv()
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

# credential_path = "D:\RL\IYKRA\Capstone\application_default_credentials.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# Please fill your openai api key
client = storage.Client()
bucket_name = "capstone_iykra"
bucket = client.bucket(bucket_name)
openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

#csv_search_tool_history = CSVSearchTool("Dataset/Customer_Interaction_Data.csv")

#csv_search_tool_product = CSVSearchTool("Dataset/final_product_catalog.csv")
blob_cust_interaction = bucket.blob('Dataset/Customer_Interaction_Data_v2.csv')
blob_product_catalog = bucket.blob('Dataset/final_product_catalog_v2.csv')
cust_interaction = blob_cust_interaction.download_as_text()
prod_catalog = blob_product_catalog.download_as_text()
df = pd.read_csv(StringIO(cust_interaction))
df_products = pd.read_csv(StringIO(prod_catalog))

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
# def generate_response_openai(query, docs, purchase_hist):
#     # Combine retrieved documents into context
#     context = "\n\n".join([doc.page_content for doc in docs])
#     prompt = (
#         f"Answer the following question based on the context:\n\nContext: {context}\n by providing a similarity with user purchase history:\n\n History : {purchase_hist}\n\n Question: {query}. "
#         "Provide detailed and accurate answer with maximum 3 products. "
#         "Always include the reason. "
#         "If the question is product related, always attach product id"
#     )

#     completion = openai.chat.completions.create(
#         model="gpt-4o",  # Adjust the model name as per availability
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant. Answer accurately and give reason"},
#             {"role": "user", "content": prompt}
#         ],
#         stream=True
#     )
#     for chunk in completion:
#         if chunk.choices[0].delta.content is not None:
#             yield chunk.choices[0].delta.content
    # return completion.choices[0].message.content

def generate_streaming_response_openai(query, docs, purchase_hist):
    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        f"Answer the following question based on the context:\n\nContext: {context}\n by adding similarity witout adding the product id from history:\n\n History : {purchase_hist}\n\n Question: {query}. "
        "Provide detailed and accurate answer with maximum 3 products. "
        "Always include the reason. "
        "If the question is product related, always attach product id. "
    )

    # Call OpenAI API with streaming
    response = openai.chat.completions.create(
        model="gpt-4",  # Adjust the model name as per availability
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer accurately and give reason."},
            {"role": "user", "content": prompt}
        ],
        stream=True  # Enable streaming
    )
     
    # Placeholder for the response
    output_placeholder = st.empty()
    collected_messages = []

    # Stream the response chunks
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            collected_messages.append(chunk_message)
            # Update the placeholder with the current response
            output_placeholder.markdown("".join(collected_messages))
    output_placeholder.empty() 
    # Return the full response
    return "".join(collected_messages)

# 4. Multi-Agent System
def multi_agent_rag(query, vector_db, purchase_hist):
    retrieved_docs = retrieve_documents(query, vector_db)
    #return generate_response_openai(query, retrieved_docs, purchase_hist)
    return generate_streaming_response_openai(query, retrieved_docs, purchase_hist)

# log debug for button click
    
def handle_click(action, product_id, url):
    st.session_state.clicked_button = f"{action} for {product_id}"
    st.session_state.product_id = product_id
    st.session_state.product_url = url
    st.session_state.waiting_for_image = True

# virtual try on function
def virtual_tryon(garment_img_path, person_img_path, prod_id):
    print("download from gcs")
    garment_image = download_image_from_gcs(garment_img_path)
    person_image = download_image_from_gcs(person_img_path)
    print("to bytes")
    garment_file = BytesIO()
    person_file = BytesIO()
    garment_image.save(garment_file, format="JPEG")
    person_image.save(person_file, format="JPEG")
    print("to file")
    garment_file.seek(0)
    person_file.seek(0)
    type = df_products[df_products['Product_ID'] == prod_id]["Type"].to_string(index=False)
    short_desc = df_products[df_products['Product_ID'] == prod_id]["Category"].to_string(index=False)
    print("vton")
    input = {
        "seed": 42,
        "steps": 30,
        "category": type,
        "garm_img": garment_file,
        "human_img": person_file,
        "garment_des": short_desc,
    }

    output = replicate.run(
        "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
        input=input
    )
    print("done")
    return str(output)

def upload_image_to_gcs(image_bytes, image_name):
    blob = bucket.blob(image_name)
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    #return f"gs://{bucket.name}/{image_name}"

def download_image_from_gcs(img_path):
    # Construct the full GCS URL by prepending 'gs://'
    #gcs_url = f"gs://{bucket_name}/{gcs_path}"
    
    # Download the image as bytes from GCS
    blob = client.bucket(bucket_name).blob(img_path)
    image_bytes = blob.download_as_bytes()
    # Open the image from the downloaded bytes
    image = Image.open(BytesIO(image_bytes))
    return image

def render_product(product_id):
    filtered_df = df_products[df_products['Product_ID'] == product_id]
    if not filtered_df.empty:
        # Mendapatkan URL gambar atau file path
        url = filtered_df['Url_Image'].iloc[0]
        img = download_image_from_gcs(url)
        img.thumbnail((300, 600))  # Maksimum lebar dan tinggi

        # Menggunakan tiga kolom untuk memusatkan elemen
        col1, col2, col3 = st.columns([1, 2, 1])  # Rasio kolom: kiri, tengah, kanan
        with col2:  # Konten di kolom tengah
            st.subheader(f"{product_id}")
            st.image(img)
            st.button(
                f"Virtual Try-On for {product_id}",
                key=f"try_{product_id}",
                on_click=handle_click,
                args=("Try ", product_id, url),
            )

# Fungsi utama chatbot
def chatbot_function():
    # Streamlit Interface
    st.header("💬 Product Recommendation Chatbot")

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
                # inputs = {"query": prompt, "customer": st.session_state["customer_id"]}
                # vector_db = load_vector_db()
                # transaction_data = retrieve_transcation(st.session_state["customer_id"])
                #response = multi_agent_rag(inputs['query'], vector_db, transaction_data)
                print(df_products[df_products['Product_ID'] == "PROD4100"])
                response = "PROD4100, PROD2104, and PROD3659"
            except Exception as e:
                response = f"An error occurred: {e}"

        # Simpan dan tampilkan respons dari asisten
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

        # Extract raw output
        output = response
        # Regular expression pattern to extract Product ID
        pattern = r"(?i)\b(PROD\d+)\b"
        product_ids = re.findall(pattern, output)
        unique_product_ids = list(set(product_ids))
        st.session_state.product_ids = unique_product_ids

    if st.session_state.product_ids:
        for product_id in st.session_state.product_ids:
            # Validasi product_ids di df_product
            with st.container():
                render_product(product_id)

        # Jika sedang menunggu gambar
    if st.session_state.waiting_for_image:
        # Unggah gambar
        uploaded_image = st.file_uploader("Please upload person image:", type=["jpg", "png", "jpeg"])
        if uploaded_image:
            st.session_state.uploaded_image = uploaded_image  # Simpan gambar yang diunggah di session state
            st.session_state.uploaded_image_name = uploaded_image.name  # Simpan nama gambar
            image = Image.open(uploaded_image)
            image.thumbnail((300, 600))
            st.image(image, caption="Your image")
            image_bytes = uploaded_image.getvalue()
            upload_image_to_gcs(image_bytes, uploaded_image.name)
            st.success(f"Image uploaded and saved successfully!")

            with st.spinner('Virtual try on is running...'):
                print(st.session_state.product_url)
                result_vto = virtual_tryon(st.session_state.product_url, uploaded_image.name, st.session_state.product_id)
            st.success("Done!")

            # tampilkan hasilnya
            # result_vto.thumbnail((300, 600))
            st.image(result_vto, caption=f"Try on for {st.session_state.product_id}")

            # Stop asking for the image
            st.session_state.waiting_for_image = False
