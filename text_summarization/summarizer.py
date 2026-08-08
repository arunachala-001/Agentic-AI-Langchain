from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


class PromptBuilder:
    """Builds and returns a PromptTemplate for summarization.

    Single responsibility: manage the prompt template content and construction.
    """

    def __init__(self, template_str: str):
        self.template_str = template_str

    def build(self) -> PromptTemplate:
        return PromptTemplate(input_variables=["information"], template=self.template_str)


class LLMClient:
    """Adapter for the LLM provider (ChatOllama).

    This isolates the rest of the code from the concrete LLM implementation (dependency inversion).
    """

    def __init__(self, model: str = "qwen3:8b", temperature: float = 0):
        self.model = model
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(temperature=self.temperature, model=self.model)
        return self._llm

    def generate(self, prompt_template: PromptTemplate, information: str):
        """Generate a response from the LLM given a PromptTemplate and input information.

        Returns the raw response object from the chain.invoke call so callers can access .content.
        """
        chain = prompt_template | self._get_llm()
        return chain.invoke(input={"information": information})


class TextSummarizer:
    """High level summarizer class.

    Responsible for orchestrating prompt creation and LLM calls. Depends on abstractions
    (PromptBuilder and LLMClient) which can be swapped for testing or other providers.
    """

    DEFAULT_TEMPLATE = (
        """
    given information {information}, 
    1. summarize the information in 3-4 lines and provide the summary in a JSON format with the following keys: \"summary\", \"key_points\", \"important_dates\", \"courses_offered\", \"history\". The value of each key should be a string containing the relevant information from the given text.
    2. List of UG courses
    3. Established year in DD/MM/YYYY format
    """
    )

    def __init__(self, llm_client: LLMClient = None, prompt_builder: PromptBuilder = None):
        self.llm_client = llm_client or LLMClient()
        self.prompt_builder = prompt_builder

    def summarize(self, information: str):
        """Return the summarization result for the provided information string.

        The method returns the .content attribute of the LLM response when present,
        otherwise returns the raw response object.
        """
        if not self.prompt_builder:
            self.prompt_builder = PromptBuilder(self.DEFAULT_TEMPLATE)

        prompt = self.prompt_builder.build()
        response = self.llm_client.generate(prompt, information)
        return getattr(response, "content", response)

