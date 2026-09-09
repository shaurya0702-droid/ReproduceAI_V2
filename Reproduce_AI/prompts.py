# every prompt here

from langchain_core.prompts import PromptTemplate

overview_prompt = PromptTemplate(
    template="""
You are an AI Research Assistant.

Extract the following information from the research paper.

Return only the information.

Title:

Authors:

Conference:

Year:

Research Domain:

One-line Description:

Paper:

{paper}
""",
    input_variables=["paper"],
)


extract_prompt = PromptTemplate(
    template="""
You are an AI Research Assistant.

Answer ONLY from the given context.

Do not infer information.

Do not use cited papers.

If multiple papers are mentioned, only answer using the uploaded research paper.

If the answer is missing, return exactly:
Not specified.

Context:

{context}

Question:

{question}

Answer:
""",
    input_variables=["context", "question"],
)


FIELDS = {
    "title": "What is the title of the paper?",
    "authors": "Who are the authors?",
    "dataset": "What dataset was used?",
    "architecture": "What model architecture was proposed?",
    "loss_function": "What loss function was used?",
    "optimizer": "What optimizer was used?",
    "learning_rate": "What learning rate was used?",
    "weight_decay": "What weight decay was used?",
    "scheduler": "What learning rate scheduler was used?",
    "batch_size": "What batch size was used?",
    "epochs": "How many training epochs were used?",
    "hardware": "What hardware was used for training?",
    "parameters": "How many model parameters are there?",
    "metrics": "What evaluation metrics were reported?",
}


paper_summary_prompt = PromptTemplate(
    template="""
You are an AI Research Assistant.

Using ONLY the provided context,

Generate:

1. A short summary (4–5 sentences)

2. Five key bullet points

3. Main contribution

Context:

{context}
""",
    input_variables=["context"],
)


planner_prompt = PromptTemplate(
    template="""
You are an expert Machine Learning Engineer.

Below is the extracted metadata from a research paper.

Metadata:
{metadata}

Generate:

1. Step-by-step implementation plan.
2. Required libraries.
3. Suggested project folder structure.
4. Training workflow.
5. Evaluation workflow.

Generate an implementation roadmap for reproducing ONLY the uploaded paper.

Do not suggest unrelated models or architectures.

If any information is missing, clearly mention it.
""",
    input_variables=["metadata"],
)


risk_prompt = PromptTemplate(
    template="""
You are an ML Research Engineer.

Below is metadata extracted from a research paper.

Metadata:

{metadata}

Analyse:

1. Missing information.
2. Reproduction difficulty.
3. Possible risks.
4. Suggestions before implementation.

Only analyze risks related to reproducing this paper.

Return a structured report.
""",
    input_variables=["metadata"],
)


executive_summary_prompt = PromptTemplate(
    template="""
You are an AI Research Assistant.

Based on the following information

Overview:

{overview}

Metadata:

{metadata}

Implementation Plan:

{plan}

Risk Report:

{risk}

Write an executive summary.

Maximum 250 words.
""",
    input_variables=[
        "overview",
        "metadata",
        "plan",
        "risk",
    ],
)


chat_prompt = PromptTemplate(
    template="""
You are an AI Research Assistant helping a user understand and reproduce a research paper.

Answer ONLY using the provided context and related to it.

If the context is genuinely empty, reply exactly:
"I couldn't find that information in the paper."

Do not make assumptions beyond the given context.

Context:

{context}

Question:

{question}

Answer:
""",
    input_variables=["context", "question"],
)


# CRAG: scores how well the retrieved chunks actually answer the question
crag_eval_prompt = PromptTemplate(
    template="""
You are the retrieval evaluator in a Corrective Retrieval Augmented Generation system.

Question:
{question}

Retrieved documents:
{documents}

Judge ONLY the retrieved documents, not outside knowledge.

- CORRECT: the documents contain enough relevant information to answer.
- AMBIGUOUS: the documents have some relevant information but are incomplete or uncertain.
- INCORRECT: the documents do not contain useful information for answering.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"action": "CORRECT" or "AMBIGUOUS" or "INCORRECT", "score": <float between 0.0 and 1.0>, "reason": "<short explanation>"}}
""",
    input_variables=["question", "documents"],
)