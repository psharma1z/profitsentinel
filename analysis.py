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
    Please provide only the table without any additional plain text in table format.
                        """
    return y,prompt

def parseStringtodf(st,cust):
    lines=st.split('\n')
    data=str(lines[2])[1:-1].split('|')
    data.append(cust)
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

def get_embedding(text):
    client = OpenAI()
    result = client.embeddings.create(
        model='text-embedding-ada-002',
        input=text
    )
    return result.data[0].embedding


def vector_similarity(vec1, vec2):
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(vec1), np.array(vec2))
def get_openai_response_for_embedding(context):
    # openai.api_key = "33d4c5bf7f124d05b6c20a849864a752"
    # openai.base_url = "https://genai-openai-profitsentinel.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?"
    # openai.api_version = "api-version=2023-05-15"
    # openai.api_type = "azure"
    res = {}
    try:
        for param in param_list:
            prompt = f"""
                Based on the time series data for the performance and the risk metrics of a customer account based on different factors create a summary analysis of customer profile and behavior
                in 300 words maximum for {param} giving it max weightage for analysis.
                Please use the facts to cross check and make sure that the summary is accurately reflecting the facts provided by the contextual snapshot month on month data in only 300 words 

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
                    {"role": "user",
                     "content": "Context: " + context.to_json(orient="records") + "\n\n Query: " + prompt}
                ]
            )
            x = response.choices[0].message.content
            res[param] = x
        # print(content)
        return res
    except Exception as e:
        return f"Error: {str(e)}"


def get_openai_response_for_parameters(context, param_list,customer):

    res={"customer":customer}
    try:
        for param in param_list:
            prompt = f"""
                          "You are a data scientist working for a major bank with access to monthly customer financial and risk performance data. Your goal is to categorize the customer’s financial risk, calculate their probability of default (PD), and analyze significant deviations from their expected financial behavior. For this task, focus on Month-over-Month (MoM) cumulative profit as the primary criterion for assessing financial risk.

Snapshot Month Data:
Customer Age
Income Category
Months on Book
Credit Limit
Revolving Balance (Revolving_Bal)
Utilization
External Bank Credit Card Utilization (>90% and >50%)
FICO Score
Total Debt
Debt-to-Income Ratio
Credit Inquiries
Delinquency
Monthly Interest Revenue
Late Fee Revenue
Annual Fee
ECL MoM Change
Cumulative Profit
Risk Categorization Instructions:
Financial Risk Classification:
Categorize the customer’s financial risk based on the following data points, with a particular focus on how {param} impact MoM cumulative profit. Specifically, analyze how {param} correlate with the changes in MoM cumulative profit.
{context}
Summary Profile:
Construct a summary profile for the customer, focusing on the most relevant parameters from the list above. Be sure to highlight how Credit Inquiries influence the MoM cumulative profit. Ensure that the summary reflects the data provided for each month in the snapshot.

Cross-Check Data:
Cross-check the summary to ensure it aligns with the given financial data. Verify the summary against the values for Credit Inquiries and the MoM Cumulative Profit to ensure consistency and accuracy.

Conciseness:
Limit the summary to 300 words, ensuring that it remains focused on the impact of Credit Inquiries on financial behavior and the MoM cumulative profit change.

Key Focus:
Prioritize analyzing the impact of {param} on MoM Cumulative Profit.
Be precise and ensure the summary is data-driven, using actual numbers from the snapshot month."
                           """
            print(prompt)
            # response = openai.chat.completions.create(
            #     model="gpt-4o",  # Replace with your model name,
            #     max_tokens=250,
            #     temperature=0.1,
            #     messages=[
            #         {"role": "system",
            #          "content": "You are an expert data scientist who answers from given context of the month on month data"},
            #         # {"role": "system", "content": sysPrompt},
            #         {"role": "user", "content": "Context: " + context.to_json(orient="records") + "\n\n Query: " + prompt}
            #     ]
            # )
            # x = response.choices[0].message.content
            # print(param,x)
            # res[param] = x
        # print(content)
        return res
    except Exception as e:
        return f"Error: {str(e)}"
def create_evaluator_response():
    with pd.ExcelFile(r'C:\Users\PXSharma\Downloads\SecretSecret-main\SecretSecret-main\Corrected Account Examples.xlsx') as f:
        sheets = f.sheet_names
        param_list=['FICO', 'Credit_Inquiries', 'FICO, Credit_Inquiries']
        df_res=[]
        for sht in sheets:
            print(sht)
            if 'C' in sht:
                df = f.parse(sht)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.expand_frame_repr', False)
                df.drop(index=0)
                df = df.iloc[:,0:27]
                header=df.iloc[0].tolist()
                df=df.iloc[1:,:]
                df.columns=header
                df.dropna(inplace=True)
                context,prompt=createPromptforoutput(df)
                x=get_openai_response_evaluator(context,"",prompt)
                df_res.append(parseStringtodf(x,sht))
                #x=get_openai_response_for_parameters(context, param_list,sht)
                print(x)
                print("********")
        final_result=pd.DataFrame(data=df_res,columns=["probabilityofdefault","riskcategory","earliest_month" ,"comments" ,"analysis","customer"])
        final_result=final_result.apply(lambda x: x.str.strip())
        final_result.to_csv(r'C:\Users\PXSharma\Downloads\New folder\pradeep.csv',index=False,sep="|")




##Working code below
def get_openai_response_evaluator_topnreason(data,prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",  # Replace with your model name,
            max_tokens=300,
            temperature=0.8,
            messages=[
                {"role": "system", "content": "You are an expert data scientist who answers from given credit card profile analysis of the time series of the performance metric of a customer account"},
               # {"role": "system", "content": sysPrompt},
                {"role": "user", "content": "Context: " +  data + "\n\n Query: " + prompt}
            ]
        )
        x=response.choices[0].message.content
        content = x
        return  content
    except Exception as e:
        return f"Error: {str(e)}"


def analyzeEvaluatorResponse(file_name):
    eval_response=pd.read_csv(file_name,header=0,sep='|')
    df=eval_response[['riskcategory','analysis', 'customer']]
    list=[]
    for index, row in df.iterrows():
        #if 'High Risk' in row['riskcategory']:
        print(row['riskcategory'], row['customer'],row['analysis'])
        tmp=[]
        prompt=f"""
            "You are an expert data scientist who answers from given credit card profile analysis of the time series of the performance metric of a customer account
             Based on the textual analysis of the narration please print top 1 reasons why this customer became delinquent or becomes high risk.
             Please note that the reason for customer account to become delinquent can not be delinquincy itself or due to ecl increase as they are the results and not reasons
             Print reason string only as output. Please produce no other data except the reason.
            """
        x= get_openai_response_evaluator_topnreason(row['analysis'],prompt)
        tmp.append(row['customer'])
        tmp.append(x)
        list.append(tmp)
        print(x)
    df=pd.DataFrame(list,columns=['customer','reason'])
    df.to_csv(r'C:\Users\PXSharma\Downloads\New folder\test_data_seg1.csv',index=False,sep="|")
file_name=r'C:\Users\PXSharma\Downloads\New folder\eval_test_data_final.csv'
# df_data= pd.read_csv(file_name,header=0,sep='|')
# df_obj = df_data.select_dtypes('object')
# df_data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
# print(df_data)
# df_data.to_csv(r'C:\Users\PXSharma\Downloads\New folder\eval_test_data1.csv',index=False,sep="|")

analyzeEvaluatorResponse(file_name)


####final file creation for testing data file:
# df1= pd.read_csv(r'C:\Users\PXSharma\Downloads\Testing Data.csv',sep=',',header=0)
# cus_list =df1['Customer ID'].unique()
# df_res=[]
# for cus in cus_list:
#     df=df1[df1['Customer ID'] == cus]
#     pd.set_option('display.max_columns', None)
#     pd.set_option('display.expand_frame_repr', False)
#     df.dropna(inplace=True)
#     print(cus)
#     context,prompt=createPromptforoutput(df)
#     x=get_openai_response_evaluator(context,"",prompt)
#     try:
#         df_res.append(parseStringtodf(x,cus))
#     except:
#         print("Error"+str(cus)+str(x))
#     #print(cus,x)
#     print("********")
#
# final_result=pd.DataFrame(data=df_res,columns=["probabilityofdefault","riskcategory","earliest_month" ,"comments" ,"analysis","customer"])
# final_result.to_csv(r'C:\Users\PXSharma\Downloads\New folder\pradeep11.csv',index=False,sep="|")
# final_result=final_result.apply(lambda x: x.str.strip())
# final_result.to_csv(r'C:\Users\PXSharma\Downloads\New folder\pradeep1.csv',index=False,sep="|")
#

# def getColumnEmbeddings(df):
#     columns_to_select = ['Snapshot Month', 'Customer Age', 'Income Category', 'Month on Book', 'Credit_Limit',
#                          'Revolving_Bal', 'Utilization',
#                          'external_bank_credit_card_max_util_greater_than_90',
#                          'external_bank_credit_card_max_util_greater_than_50',
#                          'FICO', 'Total_Debt', 'Debt_to_Income_Ratio', 'Credit_Inquiries',
#                          'Monthly_Interest_Revenue', 'Late_Fee_Revenue', 'Annual_Fee',
#                          'Cumulative Profit']
#     map={}
#     for e in columns_to_select:
#         x=get_embedding(str(e))
#         map[e]=x
#     print(map)
#
#
#
#
# def vector_similarity(vec1, vec2):
#     """
#     Returns the similarity between two vectors.
#
#     Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
#     """
#     return np.dot(np.array(vec1), np.array(vec2))
#
# def get_embedding(text):
#     openai.api_key = "33d4c5bf7f124d05b6c20a849864a752"
#     openai.base_url = "https://genai-openai-profitsentinel.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?"
#     # openai.base_url= "https://ai-ibsooraj8752ai916045283496.openai.azure.com/"
#     openai.api_version = "2023-05-15"
#     openai.api_type = "azure"
#     result = client.embeddings.create(
#         model='text-embedding-ada-002',
#         input=text
#     )
#     return result.data[0].embedding
# def vector_similarity(vec1, vec2):
#     """
#     Returns the similarity between two vectors.
#
#     Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
#     """
#     return np.dot(np.array(vec1), np.array(vec2))
#
# def getColumnEmbeddings(df):
#     columns_to_select = ['Snapshot Month', 'Customer Age', 'Income Category', 'Month on Book', 'Credit_Limit',
#                          'Revolving_Bal', 'Utilization',
#                          'external_bank_credit_card_max_util_greater_than_90',
#                          'external_bank_credit_card_max_util_greater_than_50',
#                          'FICO', 'Total_Debt', 'Debt_to_Income_Ratio', 'Credit_Inquiries',
#                          'Monthly_Interest_Revenue', 'Late_Fee_Revenue', 'Annual_Fee',
#                          'Cumulative Profit']
#     map={}
#     for e in columns_to_select:
#         x=get_embedding(str(e))
#         map[e]=x
#     print(map)




