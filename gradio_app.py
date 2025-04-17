import gradio as gr
from qa_pipeline import qa_pipeline

# Load QA pipeline and vector data
pipeline = qa_pipeline()
pipeline.load("faiss_index.index", "faiss_metadata.json")

# Gradio handler
def ask(query, top_k, threshold):
    try:
        answer, final_context, final_ordered_sections, grouped_chunks, group_scores, final_metadatas, final_scores = pipeline.ask_question(
            query=query, top_k=top_k, threshold=threshold)

        # Prepare metadata explanation
        metadata_str = "\n\n**🔎 Selected Chunks & Reranker Scores:**\n"
        for metadata, score in zip(final_metadatas, final_scores):
            title = metadata.get("title", "")
            subheading = metadata.get("subheading", "")
            place = metadata.get("text_place", "")
            text = metadata.get("text", "")
            metadata_str += f"\n---\n**Title:** {title}\n**Subheading:** {subheading}\n**Position:** {place}\n**Score:** {score:.2f}\n\n**Text:** {text}\n"

        return answer, metadata_str

    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# Build interface
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 FAISS-Powered QA System")
    gr.Markdown("Ask a question and tune the number of chunks + reranker threshold.")

    query = gr.Textbox(label="Enter your question")
    top_k = gr.Slider(1, 20, step=1, value=5, label="Top-K Retrieved Chunks")
    threshold = gr.Slider(0.0, 1.0, step=0.05, value=0.4, label="Re-ranker Threshold")

    answer_output = gr.Textbox(label="Answer")
    metadata_output = gr.Markdown(label="Metadata")

    run_btn = gr.Button("Get Answer")
    run_btn.click(fn=ask, inputs=[query, top_k, threshold], outputs=[answer_output, metadata_output])

# Launch the app
demo.launch()
