# all intelligence here

from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from core import format_docs, get_sources
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

def generate_overview(llm, docs): # Generate a high-level overview of the research paper
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
    chain= (
        prompts.extract_prompt
        | llm
        | parser
    )
    results= {}
    citations= {}

    for field, question in prompts.FIELDS.items():
        docs= retriever.invoke(question)
        context= format_docs(docs)
        answer= chain.invoke(
            {
                "context": context,
                "question": question
            }
        )
        results[field]= answer.strip()
        citations[field]= get_sources(docs)
    metadata= PaperMetadata(**results)
    return metadata, citations


def summarize_paper(llm, retriever): # Generate a summary of the research paper.
    chain= (
        prompts.paper_summary_prompt
        | llm
        | parser
    )
    docs= retriever.invoke(
        "Summarize this research paper.",
    )
    summary= chain.invoke(
        {
            "context": format_docs(docs)
        }
    )
    return summary


def generate_plan(llm, metadata):# Generate an implementation plan from the extracted metadata
    chain= (
        prompts.planner_prompt
        | llm
        | parser
    )
    plan= chain.invoke(
        {
            "metadata": metadata.model_dump_json(indent=2)
        }
    )
    return plan


def analyze_risk(llm, metadata): # Analyze the difficulty and risks of reproducing the paper.
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


def generate_executive_summary(llm,overview,metadata,plan,risk):
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


def chat_with_paper(llm, retriever, question: str) -> str:
    # Answer a free-form question about the paper using RAG.
    chain = (
        prompts.chat_prompt
        | llm
        | parser
    )
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )
    return answer