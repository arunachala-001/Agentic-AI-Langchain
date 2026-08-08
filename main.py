from dotenv import load_dotenv
from prompts.raw_prompt import INFORMATION, JOB_SEARCH
from web_job_search.job_search import JobSearch

load_dotenv()



def main():
    print("Agentic AI running....!")

    # Use the constant information text from the package so the entrypoint remains clean
    information = INFORMATION
    job_search = JOB_SEARCH


    # summarizer = TextSummarizer()
    # prompt_response = summarizer.summarize(information)

    search = JobSearch()
    result = search.agent_invoke(content=job_search)
    print(result)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

