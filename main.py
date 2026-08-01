from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()



def main():
    print("Agentic AI running....!")

    information = """
    Sri Sivasubramaniya Nadar College of Engineering (SSN), popularly known as SSN College of Engineering or simply SSN, is a private engineering college located in Chennai, Tamil Nadu, India.[1] It is an autonomous college affiliated with Anna University founded by Shiv Nadar.[2] The college is certified to ISO 9001:2000 standard[3] by the National Board of Accreditation.

In March 2018, the college was granted autonomous status by UGC.[4]

In Sep 19 2025, the college applied for progressive closure from the next academic year and merge with Shiv Nadar University according to TOI.[5]

History
Sri Sivasubramaniya Nadar College of Engineering(SSN) was started in 1996 by Padma Bhushan Dr Shiv Nadar. The college was opened in 1996 at a temporary location in Thoraipakkam in the suburbs of Chennai, Tamil Nadu as an affiliate of Anna University. It moved to a 230-acre campus at Kalavakkam (Near Thiruporur) on Rajiv Gandhi Salai (Old Mahabalipuram Road, Chennai) in 1998. Shiv Nadar took an active role in the college activities, including gifting Rs. 1 million worth of HCL Technologies shares to the college. Starting from the 2026 academic year, SSN College of Engineering will be merged into Shiv Nadar University Chennai. It will be renamed as the SSN School of Engineering under Shiv Nadar University. Admissions will shift from Anna University’s counseling system to SNU’s entrance exam and interview process.[6]

In association with Carnegie Mellon University, the SSN School of Advanced Software Engineering was started in 2001.[7][8]

Courses offered
B.Tech Chemical Engineering
B.Tech Mechanical Engineering
B.Tech Electronics and Communication Engineering
B.Tech Electrical and Electronics Engineering
B.Tech Information Technology
B.Tech Bio-Medical Engineering
B.Tech Computer Science and Engineering
B.Tech Civil Engineering

PG

M.Tech Energy Engineering
M.Tech Environmental Science and Technology
M.Tech Information Technology
M.Tech Medical Electronics
M.Tech Manufacturing Engineering
M.Tech Power Electronics and Drives
M.Tech VLSI Design
M.Tech Computer Science and Engineering
M.B.A
PhD

All streams of Engineering | Science | Management
    """
    summary_template = f"""
    given information {information}, 
    1. summarize the information in 3-4 lines and provide the summary in a JSON format with the following keys: "summary", "key_points", "important_dates", "courses_offered", "history". The value of each key should be a string containing the relevant information from the given text.
    2. List of UG courses
    3. Established year in DD/MM/YYYY format
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(
        temperature=0, model="qwen3:latest "
    )

    chain = summary_prompt_template | llm
    prompt_response = chain.invoke(input={"information": information})

    print(prompt_response.content)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

