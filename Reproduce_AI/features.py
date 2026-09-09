from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from core import format_docs, get_sources, get_tavily_client
import prompts

parser = StrOutputParser()


class PaperMetadata(BaseModel):
    title: str = Field(description="Title of the research paper")
    authors: str = Field(description="Authors")
    dataset: str = Field(description="Dataset used")
    architecture: str = Field(description="Model architecture")
    loss_function: str = Field(description="Loss function")
    optimizer: str = Field(description="Optimizer")
    learning_rate: str = Field(description="Learning rate")
    weight_decay: str = Field(description="Weight decay")
    scheduler: str = Field(description="Learning rate scheduler")
    batch_size: str = Field(description="Batch size")
    epochs: str = Field(description="Number of epochs")
    hardware: str = Field(description="Hardware used")
    parameters: str = Field(description="Number of model parameters")
    metrics: str = Field(description="Evaluation metrics")


class CRAGEvaluation(BaseModel):
    action: str = Field(description="CORRECT, AMBIGUOUS, or INCORRECT")
    score: float = Field(description="Overall retrieval quality from 0.0 to 1.0")
    reason: str = Field(description="Short explanation")


def generate_overview(llm, docs):  # Generate a high-level overview of the research paper
    chain = (
        prompts.overview_prompt
        | llm
        | parser
    )
    overview = chain.invoke(
        {
            "paper": docs[0].page_content
        }
    )
    return overview


def extract_metadata(llm, retriever):
    chain = (
        prompts.extract_prompt
        | llm
        | parser
    )
    results = {}
    citations = {}

    for field, question in prompts.FIELDS.items():
        docs = retriever.invoke(question)
        context = format_docs(docs)
        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )
        results[field] = answer.strip()
        citations[field] = get_sources(docs)
    metadata = PaperMetadata(**results)
    return metadata, citations


def summarize_paper(llm, retriever):  # Generate a summary of the research paper.
    chain = (
        prompts.paper_summary_prompt
        | llm
        | parser
    )
    docs = retriever.invoke(
        "Summarize this research paper.",
    )
    summary = chain.invoke(
        {
            "context": format_docs(docs)
        }
    )
    return summary


def generate_plan(llm, metadata):  # Generate an implementation plan from the extracted metadata
    chain = (
        prompts.planner_prompt
        | llm
        | parser
    )
    plan = chain.invoke(
        {
            "metadata": metadata.model_dump_json(indent=2)
        }
    )
    return plan


def analyze_risk(llm, metadata):  # Analyze the difficulty and risks of reproducing the paper.
    chain = (
        prompts.risk_prompt
        | llm
        | parser
    )
    risk_report = chain.invoke(
        {
            "metadata": metadata.model_dump_json(indent=2)
        }
    )
    return risk_report


def generate_executive_summary(llm, overview, metadata, plan, risk):
    # Generate the final executive summary.
    chain = (
        prompts.executive_summary_prompt
        | llm
        | parser
    )
    executive_summary = chain.invoke(
        {
            "overview": overview,
            "metadata": metadata.model_dump_json(indent=2),
            "plan": plan,
            "risk": risk,
        })
    return executive_summary


# ---------------- CRAG ----------------

def evaluate_retrieval(llm, retriever, question):
    # Retrieve chunks and score how relevant they are to the question
    docs = retriever.invoke(question)
    evaluator = llm.with_structured_output(CRAGEvaluation, method="json_mode")
    documents_str = "\n\n".join(
        f"DOCUMENT {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    result = (prompts.crag_eval_prompt | evaluator).invoke({
        "question": question,
        "documents": documents_str
    })
    return docs, result


def refine_documents(llm, docs, question):
    # Drop chunks that individually score low, keep the useful ones
    evaluator = llm.with_structured_output(CRAGEvaluation, method="json_mode")
    refined = []
    for doc in docs:
        result = (prompts.crag_eval_prompt | evaluator).invoke({
            "question": question,
            "documents": doc.page_content
        })
        if result.score >= 0.5:
            refined.append(doc.page_content)
    return refined


def web_search_context(question, paper_title=None, max_results=5):
    # Fallback external knowledge when the paper doesn't have the answer.
    # Anchor the query to the paper's title, otherwise a generic question
    # ("what thresholds were used?") pulls back completely unrelated results.
    query = f"{paper_title}: {question}" if paper_title else question
    tavily = get_tavily_client()
    results = tavily.search(query=query, max_results=max_results)
    return "\n\n".join(r["content"] for r in results["results"])


def chat_with_paper(llm, retriever, question: str, paper_title: str = None) -> str:
    # Corrective RAG: evaluate retrieval quality, then route to the right context
    docs, evaluation = evaluate_retrieval(llm, retriever, question)
    action = evaluation.action.strip().upper()

    if "CORRECT" in action and "INCORRECT" not in action:
        refined = refine_documents(llm, docs, question)
        context = "\n\n".join(refined) if refined else format_docs(docs)

    elif "INCORRECT" in action:
        context = web_search_context(question, paper_title)
        if not context:
            # web search came back empty too — fall back to whatever was
            # retrieved rather than answering from nothing
            context = format_docs(docs)

    else:  # AMBIGUOUS
        refined = refine_documents(llm, docs, question)
        internal_context = "\n\n".join(refined) if refined else format_docs(docs)
        web_context = web_search_context(question, paper_title, max_results=3)
        context = f"FROM THE PAPER:\n{internal_context}\n\nFROM THE WEB:\n{web_context}"

    chain = (
        prompts.chat_prompt
        | llm
        | parser
    )
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )
    return answer