"""
run_eval - RAG + LLaMA 3.1 pipeline for medical QA evaluation
"""

import json
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from Bio import Entrez

Entrez.email = ""  # Required

# ---------- PubMed Acquire ----------
def fetch_pubmed(query, max_results=5):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="xml")
    papers = Entrez.read(handle)

    results = []
    for article in papers["PubmedArticle"]:
        title = article["MedlineCitation"]["Article"]["ArticleTitle"]
        abstract = ""
        if "Abstract" in article["MedlineCitation"]["Article"]:
            abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
        results.append(f"{title}. {abstract}")
    return results

# ---------- RAG Pipeline ----------
class RAGPipeline:
    def __init__(self, base_docs=None, llm_model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = base_docs if base_docs else []
        self.index = None
        self.llm = pipeline(
            "text-generation",
            model=llm_model,
            token="",  # Yours HF token
            torch_dtype="auto",
            device_map="auto"
        )

    def build_index(self, new_docs):
        self.documents.extend(new_docs)
        doc_embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        dim = doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(doc_embeddings)

    def retrieve(self, question, k=3):
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        return [self.documents[i] for i in I[0]]

    def answer(self, question, options, k=3):
        retrieved = self.retrieve(question, k)
        context = "\n".join(retrieved)
        opts = "\n".join([f"{k}: {v}" for k, v in options.items()])
        prompt = f"""You are a medical assistant.
Context:
{context}

Question: {question}
Options:
{opts}

Answer with the most likely option letter (A/B/C/D) and explain briefly.
"""
        output = self.llm(prompt, max_new_tokens=256, do_sample=False)
        return output[0]["generated_text"], retrieved

# ---------- MAIN Program ----------
def main():
    # medical docs from your dataset
    base_docs = []

    rag = RAGPipeline(base_docs=base_docs)

    print("Fetching PubMed docs for query: thyroid")
    pubmed_docs = fetch_pubmed("thyroid", 5)
    rag.build_index(pubmed_docs)

    # Read questions
    with open("xxxxx/benchmark.json", "r") as f:# questions location rewrite
        all_questions = json.load(f)

    qa_pairs = list(all_questions["thyroid"].values())

    correct, total = 0, len(qa_pairs)

    for i, qa in enumerate(qa_pairs):
        print("=" * 80)
        print(f"Q{i + 1}: {qa['question']}")
        result, evidence = rag.answer(qa["question"], qa["options"])

        print("Model Answer:\n", result.strip())
        print("Ground Truth Answer:", qa["answer"])

        match = re.search(r"\b([A-D])\b", result)
        predicted = match.group(1) if match else "?"

        if predicted == qa["answer"]:
            print("Correct")
            correct += 1
        else:
            print("Incorrect (Predicted:", predicted, ")")

        print("Evidence used:")
        for doc in evidence:
            print("-", doc)
        print()

    accuracy = correct / total * 100
    print("=" * 80)
    print(f"Final Accuracy: {correct}/{total} = {accuracy:.2f}%")

if __name__ == "__main__":
    main()

