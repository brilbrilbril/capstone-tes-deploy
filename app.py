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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import re
from gradio_client import Client as GradioClient, file, handle_file
import cv2


# Please fill your openai api key

openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

#csv_search_tool_history = CSVSearchTool("Dataset/Customer_Interaction_Data.csv")

csv_search_tool_product = CSVSearchTool("Dataset/final_product_catalog.csv")
df = pd.read_csv('Dataset/Customer_Interaction_Data_v2.csv')
df_products = pd.read_csv('Dataset/final_product_catalog.csv')

llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0.1)
llm_product_recommender = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-4o-mini', temperature=0.1)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Accept context that shows about data, besides that say that the context is prohibited. "
    "Give detailed answer, but don't make it a table. "
    "Use this following format: "
    "   Purchased history: "
        "- Customer ID: "
        "- Product ID: "
        "- Transaction ID: "
        "- Purchase Date: "
        "- Order Value: "
        "- Product Details: "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{context}"),
    ]
)

csv_agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    "Dataset/Customer_Interaction_Data_v2.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True,
    prompt=prompt
)

class PurchaseHistoryInfo(BaseTool):
    name: str = "Purchase History Information Tool"
    description: str = "Search past history based on specific task and customer"

    def _run(self, query: str) -> str:
        return csv_agent.run(query)

# agent to retrieve purchase history
purchase_history_retriever_agent = Agent(
    role='Purchase history retriever',
    goal='Find the purchase history specifically for the {customer}.',
    backstory=(
        "You are a meticulous Purchase History Retrieval working at Fashion company "
        "ensuring accurate and up-to-date data gotten from the database. "
        "is available for solving customers queries."
        "You need to make sure that the data is relevant and accurate."
        "Prodive the Customer_ID, Product_ID, Transaction_ID, Purchase_Date, Order_Value, and Product_Details."
        ),
    llm=llm,
    verbose=True,
    allow_delegation=True,
    tools=[PurchaseHistoryInfo()]
)

#agent to retrieve products
product_catalog_retriever_agent = Agent(
    role='Product Catalog Retriever',
    goal='Find the related product based on customer task: {query}',
    backstory=(
        "You are working on a task: {query} to provide the customer. "
        "Retrieve the specific output from the specific task: {query}. Provide the list of products, including Product_ID, Rating, Category, Size, Color, Price, Weather, and Event Type."
        "You need to make sure that the data is relevant and accurate. "
        "Provide with at least 5 products."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=True,
    tools=[csv_search_tool_product]
)

review_agent = Agent(
    role='Result Reviewer',
    goal="Review the response drafted by the Purchase history retriever and Product Catalog Retriever for {customer}'s inquiry: {query}. ",
    backstory=(
        "You are working on a task: {query} to provide the customer. "
        "Review the response drafted by the Purchase history retriever and Product Catalog Retriever for {customer}'s inquiry: {query}. "
    ),
    llm=llm,
    verbose=True
)

product_recommender_agent = Agent(
    role='Product Recommendation Provider',
    goal="Give product recommendation based on review from Result Reviewer. ",
    backstory=(
        "You are working on a task: {query} to give product recommendation. "
        "Analyze the product details from customer purchase history and product catalog. "
        "Find the correlation between them to provide best product recommendation. "
    ),
    llm=llm_product_recommender,
    verbose=True
)

# purchase history task
retrieve_purchase_history_task = Task(
    description=(
        "Gather all relevant {customer} data from the dataset, focusing "
        "on crucial data which will be great to know when addressing the "
        "customer's inquiry."
        "Give accurate answer and no assumption. Return '{customer} is not available' if the customer is not in the dataset."
        ),
    expected_output=("A list of the customer's information. "
      "Turn them into bullets. "
      "Highlighting key info of the customer that will be helpful "
      "to the team when addressing the customer's query. "
    ),
    agent=purchase_history_retriever_agent
)

#product catalog task
retrieve_product_catalog_task = Task(
    description=("customer is asking for specific task: {query}. "
      "Analyze the Product_Catalog_Data.csv and make sure the task from customer is answered accurately. "
      "Make sure that you provide the best and relevant answer. "
      "Give accurate answer and no assumption."
      "Use 'search_query: {query}' for tool input."
    ),
    expected_output=("List of 5 product(s) with product ID that match user task."
    ),
    agent=product_catalog_retriever_agent,
)

#Review Result Task
review_result_task = Task(
    description=(
        "Review the purchase history from Purchase History Retriever and the products"
        "from Product Catalog Retriever."
    ),
    expected_output=(
        "List all of received purchase history from Purchase History Retriever with following format: "
        "1. Purchased history: "
        "- Customer ID: "
        "- Product ID: "
        "- Transaction ID: "
        "- Purchase Date: "
        "- Order Value: "
        "and list of prodcut from Product Catalog Retriever with following format: "
        "2. Product Catalog: "
    ),
    agent=review_agent,
    context=[retrieve_purchase_history_task, retrieve_product_catalog_task]
)

#Product Recommendation TAsk
product_recommendation_task = Task(
    description=(
        "Analyze the product details from user purchase history and product catalog received only from Result Reviewer "
        "to give best product recommendation to user."
        "You are not allowed to search any product besides from Result Reviewer."
    ),
    expected_output=(
        "List of maximum 3 product recommendation after find the similarity or relevance between  product details from user purchase history and product catalog from Result Reviewer. "
        "Give the reason to convince customer. "
        "Do not make any assumption, just based on received user purchase history and product catalog. "
        "If the product is not available, return that the product is not in product catalog."
    ),
    agent=product_recommender_agent,
    context=[review_result_task]
)

crew = Crew(
    agents=[purchase_history_retriever_agent, product_catalog_retriever_agent, review_agent, product_recommender_agent],
    tasks=[retrieve_purchase_history_task, retrieve_product_catalog_task, review_result_task, product_recommendation_task],
    verbose=True,
    memory=False,
    process=Process.sequential,
)

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
    
def handle_click(action, product_id):
    st.session_state.clicked_button = f"{action} for {product_id}"
    
    st.session_state.product_id = product_id
    st.session_state.product_url = f"image/{product_id}.jpg"

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
            response = crew.kickoff(inputs=inputs)
            response = response.raw
        except Exception as e:
            response = f"An error occurred: {e}"

    # Simpan dan tampilkan respons dari asisten
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
    
    # Extract raw output
    output = response
    print(output)
    # Regular expression pattern to extract Product ID
    pattern = r"Product[_ ]ID: (\w+)"
    product_ids = re.findall(pattern, output)
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
                    args=("Try ", product_id),
                    )

    # if it's waiting for image
    if st.session_state.waiting_for_image:
        # upload image
        uploaded_image = st.file_uploader("Please upload person image:", type=["jpg", "png", "jpeg"])
        print("tes upliad:", uploaded_image)
        if uploaded_image:
            st.session_state.uploaded_image = uploaded_image  # Store uploaded image in session state
            st.session_state.uploaded_image_name = uploaded_image.name  # Store image name

            with open(uploaded_image.name, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.success(f"Image uploaded and saved successfully!")
            
            with st.spinner('Virtual try on is running...'):
                result_vto = virtual_tryon(st.session_state.product_url, uploaded_image.name)
            st.success("Done!")

            # display the result
            st.image(result_vto, caption=f"Try on for {st.session_state.product_id}")
            
            # print details
            # print(f"Customer ID: {st.session_state['customer_id']}")
            # print(f"Product ID: {st.session_state.product_id}")
            # print(f"Product URL: {st.session_state.product_url}")
            # print(f"Uploaded Image: {uploaded_image.name}")
            
            # stop asking for the image
            st.session_state.waiting_for_image = False
