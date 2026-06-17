import logging
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


class RAGChain:

    # Chain Class Constructor
    def __init__(
        self,
        oLLM: ChatOpenAI,
        oRetriever: Any,
        oPrompt: PromptTemplate
    ):
        self.oLLM = oLLM
        self.oRetriever = oRetriever
        self.oPrompt = oPrompt


    def invokeRAGChain(self, sQuery: str):

        # pre-retrieval query rewriting
        sRewrittenQuery = self._preRetrievalQueryRewriting(sQuery, self.oLLM)

        logging.info(f"Rewritten query: {sRewrittenQuery.content}")

        # retrieval of the relevant documents with post-retrieval re-ranking
        aoDocs = self.oRetriever.invoke(sRewrittenQuery.content)

        # prompt template
        oFinalPrompt = self.oPrompt.format(context=aoDocs, query=sQuery)

        # invoke the LLM with the final prompt
        return self.oLLM.invoke(oFinalPrompt)


        

    def _preRetrievalQueryRewriting(self, sQuery: str, oLLM: ChatOpenAI):

        sQueryRewritePrompt = f"""You are a helpful assistant that takes a user's query and
        turns it into a short statement or paragraph so that it can
        be used in a semantic similarity search on a vector database
        to return the most similar chunks of content based on the
        rewritten query. Please make no comments, just return the
        rewritten query.
        
        user question: {sQuery}
        
        ai: """

        oRewrittenQuery = oLLM.invoke(sQueryRewritePrompt)

        return oRewrittenQuery
    

    def _formatDocs(aoDocs):
        return "\n\n".join(oDoc.page_content for oDoc in aoDocs)