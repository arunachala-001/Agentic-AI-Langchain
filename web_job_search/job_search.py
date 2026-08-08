from typing import Any, List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

tavily = TavilySearch(
    include_domains=["linkedin.com"]
)

"""
LLM Decides when to call the tool and how to use it.
The tool is used to search for actual job postings on LinkedIn.
The LLM will invoke the tool with a query string, and the tool will return job posting pages only,
excluding LinkedIn member/profile pages and URLs containing /in/. 
The LLM will then process the results and return exactly 3 actual job postings in the specified format.
"""
@tool
def search_jobs(query: str) -> Any:
    """
    Search for actual job postings on LinkedIn.

    IMPORTANT:
    - Return job posting pages only.
    - Do NOT return LinkedIn member/profile pages.
    - Do not return URLs containing /in/.
    - Search for jobs, not people.
    """
    return tavily.invoke({
        "query": query
    })


class JobPosting(BaseModel):
    title: str = Field(description="The exact job title")
    company: str = Field(description="The company hiring for the position")
    location: str = Field(description="Job location")
    url: str = Field(description="URL of the actual LinkedIn job posting")
    description: str = Field(description="Short description of the job")


class AgentResponse(BaseModel):
    jobs: List[JobPosting] = Field(
        description="Exactly 3 actual job postings"
    )


class JobSearch:
    def __init__(self):
        self.llm = ChatOllama(model="qwen3:8b")
        self.agent = create_agent(model=self.llm, tools=[search_jobs], response_format=AgentResponse)


    def agent_invoke(self, content: str) -> dict[str, Any] | Any:
        response = self.agent.invoke({"messages": HumanMessage(content=content)})
        return response



