# Data Extraction From RECEIPT Images Using Multi Modal LLM GEMMA3

This is just fun trial project for multi modal LLM GEMMA3 and it's capability in extracting information from images.
The whole idea is the utilization of LLM in extracting data/information from the mobile banking transaction receipt.
With extraction such data such as the name of the recipient, the amount of the money and date, we can build simple
dashboard, hence it'll be a lot easier to monitor spending. The model used in this project is Gemma3 with 4b parameters
running locally via OLLAMA.

## General Procedure For OLLAMA
1. Head to https://ollama.com/ & register
2. Download ollama that suits your OS (My case is Mac)
3. Install ollama
4. Go to terminal and type: ollama run (your desired model), for example: ollama run gemma3:4b
5. Make sure the ollama run on the background

## How to Use the Scripts
1. Head to https://github.com/Rian021102/ai_receipt_extraction# and copy https://github.com/Rian021102/ai_receipt_extraction.git
2. Go to your terminal and type git clone https://github.com/Rian021102/ai_receipt_extraction.git
3. Go to the directory and installing dependencies by typing pip install -r requirements.txt
4. Type python trial01.py or trial02.py


