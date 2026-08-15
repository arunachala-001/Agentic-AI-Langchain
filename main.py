from dotenv import load_dotenv
from prompts.raw_prompt import INFORMATION, JOB_SEARCH
from web_job_search.job_search import JobSearch
from agent_loop_tool_calling.ecommerce_assistant import ECommerceAssistant

load_dotenv()




def main():
    print("Agentic AI running....!")

    # Use the constant information text from the package so the entrypoint remains clean
    information = INFORMATION
    job_search = JOB_SEARCH


    # summarizer = TextSummarizer()
    # prompt_response = summarizer.summarize(information)

    # search = JobSearch()
    # result = search.agent_invoke(content=job_search)

    assistant = ECommerceAssistant()
    response = assistant.run_agent(question="What is the price of a laptop and apply a gold discount to it?")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

