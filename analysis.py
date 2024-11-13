import pandas as pd
import openai

openai.api_key = "33d4c5bf7f124d05b6c20a849864a752"
openai.base_url= "https://genai-openai-profitsentinel.openai.azure.com/openai/deployments/gpt-4o/chat/completions?"
#openai.base_url= "https://ai-ibsooraj8752ai916045283496.openai.azure.com/"
openai.api_version = "2024-08-01-preview"
openai.api_type = "azure"

def createPromptforoutput(df):
    columns_to_select=['Snapshot Month','Customer Age', 'Income Category', 'Month on Book', 'Credit_Limit', 'Revolving_Bal', 'Utilization',
            'external_bank_credit_card_max_util_greater_than_90', 'external_bank_credit_card_max_util_greater_than_50',
            'FICO', 'Total_Debt', 'Debt_to_Income_Ratio', 'Credit_Inquiries', 'Delinquency',
            'Monthly_Interest_Revenue', 'Late_Fee_Revenue', 'Annual_Fee','ECL MoM Charge','Cumulative Profit']#,'MoM Cumulative Profit']#,'Total Revenue']
    x=df[columns_to_select]
    y=x
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)



    prompt = f"""
                        "Assume you are a data scientist working for a major bank with access to customer month-end financial and risk performance metrics. Your goal is to categorize the customer’s financial risk, calculate their probability of default, and identify any significant deviations from expected financial behavior that could indicate the need for interventio using MOM cumulative profit decrease as major criteria.
    Risk Categorization: Classify the customer based on a weighted composite of financial indicators. These are as follows:

    MoM Cumulative Profit
    Revolving Balance (Revolving_Bal)
    ECL (Expected Credit Loss)
    ECL MoM Change
    Delinquency
    Utilization
    FICO
    Credit_Inquiries
    Total_Debt
    Debt_to_Income_Ratio
    external_bank_credit_card_max_util_greater_than_90
    external_bank_credit_card_max_util_greater_than_50
    Revolving_Bal
    Credit_Limit
    Income Category
    Education_Level
    Marital Status
    Customer Age
    Month on Book
    Use this scale for categorization:

    [<10%] Low Risk
    [10%-25%] Low-Medium Risk
    [25%-50%] Medium Risk
    [50%-80%] High Risk
    [80%-100%] Very High Risk
    Probability of Default: Calculate the probability of default based on a weighted composite of the metrics above. You may use the following as a guideline for weight assignment (example):

    Delinquency(22%)
    Utilization(22%)
    FICO(20%)
    Credit_Inquiries(6%)
    Total_Debt(3%)
    Debt_to_Income_Ratio(3%)
    external_bank_credit_card_max_util_greater_than_90(6%)
    external_bank_credit_card_max_util_greater_than_50(6%)
    Revolving_Bal(5%)
    Credit_Limit(2%)
    Income Category(1%)
    Education_Level(1%)
    Marital Status(1%)
    Customer Age(1%)
    Month on Book(1%)

    Significant Deviation Indicators: Pay attention to any month-on-month changes that signal a need for intervention. For example:

    A decrease in Cumulative Profit of >1%
    ECL increase of >10%
    Increase in Utilization of >10%
    Increase in Debt-to-Income ratio >5%
    Intervention Trigger: Identify the earliest month in which MoM Cumulative Profit is negative excluding first month with 50% weight. This month will trigger an intervention and recommend whether intervention or monitorning needed.
    Your output should be a single row table with the following columns:
    probabilityofdefault
    riskcategory
    earliest_month
    comments (recommendations)
    analysis (200-word detailed analysis)
    Please provide only the table without additional text.
                        """
    return y,prompt

def parseStringtodf(st):
    lines=st.split('\n')
    data=str(lines[2])[1:-1].split('|')
    columnsq=str(lines[0])[1:-1].split('|')
    data = pd.DataFrame([data])
    data.columns=columnsq
    return data

def get_openai_response_evaluator(context, sysPrompt,prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Replace with your model name,
            max_tokens=250,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are an expert data scientist who answers from given context"},
               # {"role": "system", "content": sysPrompt},
                {"role": "user", "content": "Context: " +  context.to_json(orient="records") + "\n\n Query: " + prompt}
            ]
        )
        x=response.choices[0].message.content
        content = x
        #print(content)
        return  content
    except Exception as e:
        return f"Error: {str(e)}"


def get_openai_response_for_parameters(context, param_list):

    res={}
    try:
        for param in param_list:
            prompt = f"""
                Based on the time series data for the performance and the risk metrics of a customer account based on different factors create a summary analysis of customer profile
                in 200 words for {param} giving it max weightage for analysis.
                Please use the facts to cross check and make sure that the summary is accurately reflecting the facts provided by the contextual snapshot month on month data 
    
                """
            print(prompt)
            response = openai.chat.completions.create(
                model="gpt-4o",  # Replace with your model name,
                max_tokens=250,
                temperature=0.1,
                messages=[
                    {"role": "system",
                     "content": "You are an expert data scientist who answers from given context of the month on month data"},
                    # {"role": "system", "content": sysPrompt},
                    {"role": "user", "content": "Context: " + context.to_json(orient="records") + "\n\n Query: " + prompt}
                ]
            )
            x = response.choices[0].message.content
            res[param] = x
        # print(content)
        return res
    except Exception as e:
        return f"Error: {str(e)}"

with pd.ExcelFile(r'C:\Users\PXSharma\Downloads\SecretSecret-main\SecretSecret-main\Corrected Account Examples.xlsx') as f:
    sheets = f.sheet_names
    param_list=['FICO', 'Credit_Inquiries']
    df_res=pd.DataFrame()
    for sht in sheets:
        print(sht)
        if 'Customer B' in sht:
            df = f.parse(sht)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.expand_frame_repr', False)
            df.drop(index=0)
            df = df.iloc[:,0:27]
            header=df.iloc[0].tolist()
            df=df.iloc[1:,:]
            df.columns=header
            df.dropna(inplace=True)
            #df = df.iloc[-36:, :]
            #df['Snapshot Month']=pd.to_datetime(df['Snapshot Month'])
            #df=df[pd.to_datetime(df['Snapshot Month'])>'2018-01-01']
            context,prompt=createPromptforoutput(df)
            # print(context)
            # print(prompt)print
            # x=get_openai_response_evaluator(context,"",prompt)
            # x1=parseStringtodf(x)
            x=get_openai_response_for_parameters(context, param_list)
            print(x)
            print("********")
