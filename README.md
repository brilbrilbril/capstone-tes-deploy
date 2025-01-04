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
```bash
# Clone the repository
git clone https://github.com/PutraAlFarizi15/capstone-project-personalized-shopping-copilot.git

# Navigate to the directory
cd capstone-project-personalized-shopping-copilot

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environmentV
venv\Scripts\activate

# Install the project dependencies
pip install -r requirements.txt

# fill openai api key in file app.py
openai_api_key = 'your openai api key'

# Start the development server
streamlit run app.py
```

