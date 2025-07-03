import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.openai import OpenAIEmbedder
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing in your .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# PostgreSQL connection URL
db_url = "postgresql+psycopg://ai:admin@localhost:5432/ai"

# Initialize OpenAI embedder (1536-dim embeddings)
embedder = OpenAIEmbedder(model="text-embedding-ada-002")

# Define the PDF Knowledge Base
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(
        collection="recipes",
        db_url=db_url,
        embedder=embedder
    )
)

def initialize_knowledge_base():
    """Initialize or recreate the knowledge base if needed."""
    try:
        print("Loading knowledge base...")
        knowledge_base.load(recreate=False)
        print("Knowledge base loaded successfully.")
        return True
    except Exception as e:
        print(f"Initial load failed: {e}")
        print("Attempting to recreate knowledge base...")
        try:
            knowledge_base.load(recreate=True)
            print("Knowledge base created and loaded successfully.")
            return True
        except Exception as e2:
            print(f"Failed to create knowledge base: {e2}")
            return False

def pdf_assistant(new: bool = False, user: str = 'user'):
    """Main function to run PDF assistant via CLI."""
    if not initialize_knowledge_base():
        print("Knowledge base setup failed. Exiting.")
        return

    storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)
    run_id: Optional[str] = None

    if not new:
        try:
            existing_run_ids: List[str] = storage.get_all_run_ids(user)
            if existing_run_ids:
                run_id = existing_run_ids[0]
        except Exception as e:
            print(f"Failed to get previous run: {e}. Starting new...")

    try:
        assistant = Assistant(
            run_id=run_id,
            user_id=user,
            knowledge_base=knowledge_base,
            storage=storage,
            show_tool_calls=True,
            search_knowledge=True,
            read_chat_history=True
        )

        print(f"{'Started' if run_id is None else 'Continuing'} Run: {assistant.run_id}\n")
        assistant.cli_app(markdown=True)

    except Exception as e:
        print(f"Assistant failed to start: {e}")

if __name__ == "__main__":
    typer.run(pdf_assistant)

