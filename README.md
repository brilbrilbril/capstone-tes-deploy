# Contributor
| Name  | Job Description |
| ------------- | ------------- |
| Brillyando Magathan Achmad  | Data preparation, data cleaning, and preprocessing data. Create, build, and develope multi-agent system, implement virtual try on and integrate to streamlit  |
| Putra Al Farizi  | Data Preparation, Data Preprocessing, Develop Multi-agent system, Analyze Retention Rate, Average Order Value and Conversion Rate, display recommended product images, Build Streamlit Application Chatbot |
| Khalid Destiawan  | Data Preparation, Develop Multi-agent system, Analyze Retention Rate, Average Order Value and Conversion Rate. Dashboard development |

# Business Understanding

## Problem Description
An e-commerce platform faces challenges in delivering a personalized shopping experience. The lack of tailored product recommendations leads to reduced customer engagement, lower conversion rates, and a suboptimal average order value. Customers expect intuitive and customized support when navigating extensive product catalogs, but the platform currently lacks the capability to provide this level of personalization.

## Goals
Develop a chatbot that leverages customer data to provide personalized product recommendations, enhancing customer engagement, satisfaction, and trust.

## Project Objectives
1. Build an AI-powered chatbot capable of analyzing customer behavior and customer preference to deliver personalized product recommendations in real-time.
2. Create a dashboard for tracking chatbot performance and customer behavior insights, aiding in data-driven decision-making.

# Tool
- Deployment: Streamlit
- Vector database: Crewai Tool
- Chatbot Pipeline: Crewai
- LLM (Large Language Model): GPT 3.5- turbo and GPT 4.0 mini
- Data Visualization: Looker

# Project instruction

## Installation

Follow these steps to install the project from GitHub:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PutraAlFarizi15/capstone-project-personalized-shopping-copilot.git
2. **Navigate to the project directory**
    ```bash
    cd capstone-project-personalized-shopping-copilot
3. **Create a virtual environment (optional but recommended)**
    ```bash
    python -m venv venv
4. **Activate the virtual environment**
    ```bash
    # Windows
    venv\Scripts\activate
    # MacOS/Linux
    source venv/bin/activate
5. **Install the project dependencies**
    ```bash
    pip install -r requirements.txt
6. **Add your OpenAI API key to the app.py file**: Find the following line in the app.py file and replace it with your API key
    ```bash
    openai_api_key = 'your openai api key'
7. **Run the application using Streamlit**
    ```bash
    streamlit run app.py
    ```

## How to use Application
Once the application is running, you will see the following message:
```bash
💬 Product Recommendation Chatbot
Welcome! Please provide your Customer ID to start.
```
**Steps:**

1. **Enter the Customer ID:**
   - Customer IDs are obtained from the customer interaction dataset.
   - As an example, you can use `CUST1001`.\
   after entering the Customer ID, the chatbot will respond:
   ```bash
   Thank you! Customer ID 'CUST1001' has been verified. How can I assist you?
    ```

2. **Request Product Recommendations**
    - Example prompt: white shirt with friend
    - The chatbot will recommend up to 3 products along with Buy and Virtual Try On buttons.
    after entering the Query, expectation that the chatbot response will be:
        ```bash
        based on the analysis of your purchase history and the product catalog, I recommend the following blue shirts for your vacation:

        1.Product ID: PROD4632

        Rating: 3.66\
        Size: L\
        Color: Blue\
        Price: 2,207,297.76\
        Weather: Sunny\
        Event Type: Vacation\
        Reason: This shirt is specifically designed for sunny weather, making it ideal for vacation. Its blue color aligns with your preference, and it falls within a similar price range to your previous purchases, indicating a potential fit for your budget.
        ```
        ![Struktur Database](image/PROD1457.jpg)\
        ```bash
        Virtual Try On
        ```

3. Virtual Try-On  
    - Click the **Virtual Try On** button to try the product virtually.  
    - You will then be prompted to upload a picture of a model or a person.  
    - The uploaded image will be processed, and the result will display the person wearing the selected clothing item. \
    **Important:** The Virtual Try-On feature has a usage limit. Please use it wisely. 

## NOTES (IMPORTANT!)
This app is under development. For frist execute, it maybe will take a lot of time because 
the dataset is wrote to the chromadb. The product recommendation is not perfect because the tool used is still experimental.

The uploaded image and result from virtual try on will be saved on local, so take a look at your drive.
