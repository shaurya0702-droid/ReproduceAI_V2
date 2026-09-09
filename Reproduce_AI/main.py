# Sequential pipeline for ReproduceAI

from core import load_pdf, split_documents, get_embedding_model, build_vectorstore, get_retriever, get_llm
from features import generate_overview, extract_metadata, summarize_paper, generate_plan, analyze_risk


def run_pipeline(pdf_path: str):
    # returns the report AND the retriever/llm so chat can reuse them without rebuilding the vector store.
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    embeddings = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    overview = generate_overview(llm, docs)
    metadata, citations = extract_metadata(llm, retriever)
    summary = summarize_paper(llm, retriever)
    plan = generate_plan(llm, metadata)
    risk_report = analyze_risk(llm, metadata)

    report = {
        "overview": overview,
        "metadata": metadata.model_dump(),
        "citations": citations,
        "summary": summary,
        "plan": plan,
        "risk_report": risk_report,
    }
    return report, retriever, llm


if __name__ == "__main__":
    report, retriever, llm = run_pipeline("data/paper.pdf")
    print("\n========== ReproduceAI Report ==========\n")
    for section, value in report.items():
        print(f"\n{'=' * 15} {section.upper()} {'=' * 15}\n")
        print(value)
    print("\nPipeline completed successfully!")
