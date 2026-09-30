import logging
from typing import Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


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


    def invokeRAGChain(self, sQuery: str, oMetadataFilter: Optional[dict] = None):

        # pre-retrieval query rewriting
        sRewrittenQuery = self._preRetrievalQueryRewriting(sQuery, self.oLLM)

        logging.info(f"Rewritten query: {sRewrittenQuery.content}")

        aoReconstructedDocs = self._retrieve_and_reconstruct(
            sQuery=sRewrittenQuery.content, 
            oVectorStore=self.oRetriever.vectorstore, 
            k=self.oRetriever.search_kwargs.get("k", 4),
            oMetadataFilter=oMetadataFilter  # <-- Filter explicitly passed
        )

        sFormattedContext = self._formatDocs(aoReconstructedDocs)

        # prompt template
        oFinalPrompt = self.oPrompt.format(context=sFormattedContext, query=sQuery)

        # invoke the LLM with the final prompt
        oAIMessage = self.oLLM.invoke(oFinalPrompt)

        print("Reply" + oAIMessage.pretty_repr())

        return oAIMessage
        

    def _preRetrievalQueryRewriting(self, sQuery: str, oLLM: ChatOpenAI):

        sQueryRewritePrompt = f"""You are a search query generator for a vector database.
        Convert the user's question into a short semantic search query focused on CONCEPTS, not code.
        Do not write code in your output. Use keywords related to platform architecture and documentation. 
        Please make no comments, just return the rewritten query.
        
        user question: {sQuery}
        
        ai: """

        oRewrittenQuery = oLLM.invoke(sQueryRewritePrompt)

        return oRewrittenQuery


    def _retrieve_and_reconstruct(self, sQuery: str, oVectorStore: Any, k: int = 4, oMetadataFilter: Optional[dict] = None) -> List[Document]:
        """
        Dynamically reconstructs full files from small chunks using the sourcePath metadata.
        """
        if oMetadataFilter:
            aoChildDocs = oVectorStore.similarity_search(sQuery, k=k, filter=oMetadataFilter)
        else:
            aoChildDocs = oVectorStore.similarity_search(sQuery, k=k)
        
        aoReconstructedParents: List[Document] = []
        setProcessedFiles = set()
        
        for oChildDoc in aoChildDocs:
            # Use 'sourcePath' instead of 'fileName'
            sSourcePath = oChildDoc.metadata.get("sourcePath")
            
            if not sSourcePath or sSourcePath in setProcessedFiles:
                continue
                
            setProcessedFiles.add(sSourcePath)
            
            # Query the underlying Chroma collection using 'sourcePath'
            oRawResults = oVectorStore._collection.get(
                where={"sourcePath": {"$eq": sSourcePath}}
            )
            
            if not oRawResults or not oRawResults.get('ids'):
                continue
                
            # Sort the extracted chunks using your dedicated 'chunkIndex' key!
            try:
                aoSortedDocs = sorted(
                    zip(oRawResults['metadatas'], oRawResults['documents']),
                    key=lambda x: int(x[0].get('chunkIndex', 0))
                )
            except Exception as e:
                logging.warning(f"Failed to sort chunks for {sSourcePath}: {e}")
                aoSortedDocs = zip(oRawResults['metadatas'], oRawResults['documents'])
            
            # Stitch the text back together in the correct order
            sFullText = "\n\n".join([doc[1] for doc in aoSortedDocs])
            
            # Append the reconstructed document
            aoReconstructedParents.append(
                Document(
                    page_content=sFullText, 
                    metadata={"sourcePath": sSourcePath, "source_type": oChildDoc.metadata.get("source_type")}
                )
            )
            
        return aoReconstructedParents

    def _formatDocs(self, aoDocs):
        return "\n\n".join(oDoc.page_content for oDoc in aoDocs)