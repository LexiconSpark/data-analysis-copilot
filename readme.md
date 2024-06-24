# How to Navigate through this project
- Use Branch "V0.1_CSV_Analysis_Report_Generation" to see our demo for CSV Analysis Report Generation
- Use Branch "V0.2_Data_Analysis_Copilot" to see our demo for for CSV Analysis Copilot with 4 views

### For Hackathon Hackers of week of 20240624:
- Please use the ```V0.2_Data_Analysis_Copilot_20240624_hackathon``` branch of the code
- Run 'pip install -r requirements.txt' before running code. Ensure you have python and pip3/pip installed.
 - Add the .env file into the root directory and 2 variables in there:
    ```  
    OPENAI_API_KEY="REPLACE THIS WITH YOUR OPENAI API KEY FROM https://platform.openai.com/api-keys"
    ```

### For the Branches under the family of V0.2_Data_Analysis_Copilot do
 - Run 'pip install -r requirements.txt' before running code. Ensure you have python and pip3/pip installed.
 - Add the .env file into the root directory and 2 variables in there:
    ```  
    OPENAI_API_KEY="REPLACE THIS WITH YOUR OPENAI API KEY FROM https://platform.openai.com/api-keys"
    ```

 - Optional improvement if you like to overserve your langchain agent activities in more details
   - add this into the .env file
      ```
      LANGCHAIN_API_KEY="(Optional) REPLACE THIS WITH YOUR LANGSMITH API KEY FROM https://smith.langchain.com > go to SETTINGS > API Keys"
      ```
   - and add this into the streamlit_app.py file
      ```
         ## using langsmith
         # video for how langsmith is used in this demo code: https://share.descript.com/view/k4b3fyvaESB
         # To learn about this: https://youtu.be/tFXm5ijih98
         # To check the running result of langsmith, please go to: https://smith.langchain.com/
         os.environ["LANGCHAIN_TRACING_V2"] = "true"
         os.environ["LANGCHAIN_PROJECT"] = "data_analysis_copilot"
         os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
         # Initialize LangSmith client
         langsmith_client = LangSmithClient()

      ```
