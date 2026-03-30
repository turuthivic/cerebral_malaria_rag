"""
Phase 3: Gradio UI for the cerebral malaria RAG system.

Usage:
    python -m src.app

Opens a web interface at http://localhost:7860
"""

import gradio as gr

from src.rag import RAGPipeline

EXAMPLE_QUESTIONS = [
    "What causes cerebral malaria?",
    "How is cerebral malaria diagnosed?",
    "What is the mortality rate of cerebral malaria in children?",
    "What are the neurological sequelae of cerebral malaria?",
    "What is the role of malarial retinopathy in diagnosis?",
    "How does Plasmodium falciparum affect the blood-brain barrier?",
    "What treatments are available for cerebral malaria?",
]


def create_app():
    rag = RAGPipeline()

    def ask(question: str) -> tuple[str, str]:
        if not question.strip():
            return "Please enter a question.", ""

        result = rag.query(question)

        # Format sources
        sources_md = "### Sources\n\n"
        for i, src in enumerate(result["sources"], 1):
            year = f" ({src['year']})" if src["year"] else ""
            sources_md += (
                f"{i}. **[{src['source'].upper()}]** "
                f"{src['title']}{year} — "
                f"similarity: {src['similarity']}\n"
            )

        return result["answer"], sources_md

    with gr.Blocks(title="Cerebral Malaria RAG") as app:
        gr.Markdown(
            "# Cerebral Malaria Q&A\n"
            "Ask questions about cerebral malaria. Answers are grounded in "
            "medical literature from PubMed, PMC, WHO, and CDC.\n\n"
            "*Powered by Qwen2.5-7B + BGE embeddings + ChromaDB*"
        )

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What causes cerebral malaria?",
                    lines=2,
                )
                ask_btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Examples")
                for q in EXAMPLE_QUESTIONS:
                    gr.Button(q, size="sm").click(
                        fn=lambda q=q: q, outputs=question
                    )

        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Sources")

        ask_btn.click(fn=ask, inputs=question, outputs=[answer, sources])
        question.submit(fn=ask, inputs=question, outputs=[answer, sources])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
