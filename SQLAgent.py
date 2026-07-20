import os
import pathlib
import logging
import time

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

LOG_DIR = pathlib.Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "agent.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY was not found. Add it to your .env file."
    )

model = init_chat_model(
    "openai:gpt-5.4",
    temperature=0,
)

DATABASE_URL = (
    "https://storage.googleapis.com/"
    "benchmarks-artifacts/chinook/Chinook.db"
)

DATABASE_PATH = pathlib.Path("Chinook.db")


def download_database() -> None:
    """Download the Chinook database if it is not already present."""

    if DATABASE_PATH.exists():
        logger.info("Database already exists")
        return

    logger.info("Downloading Chinook database")

    response = requests.get(DATABASE_URL, timeout=60)
    response.raise_for_status()

    DATABASE_PATH.write_bytes(response.content)

    logger.info("Database downloaded successfully")


download_database()

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)

tools = toolkit.get_tools()

PROMPT_VERSION = "1.0"

SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct {dialect}
query to run, then look at the results of the query and return the
answer.

Unless the user specifies a specific number of examples they wish to
obtain, always limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most
interesting examples in the database.

Never query for all columns from a table. Only select the columns
relevant to the user's question.

You MUST double-check your query before executing it. If you receive
an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements such as INSERT, UPDATE, DELETE, DROP,
ALTER, or CREATE.

To start, you should ALWAYS inspect the tables in the database.
Do not skip this step.

Then inspect the schemas of the most relevant tables before writing
the SQL query.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

def run(question: str) -> str:
    """Run the SQL agent and display every step."""

    start = time.perf_counter()

    logger.info(
        f"Prompt version: {PROMPT_VERSION}"
    )
    logger.info(
        f"User question: {question}"
    )

    final_answer = ""

    for step in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        stream_mode="values",
    ):
        latest_message = step["messages"][-1]

        # Shows the user question, tool calls, tool outputs,
        # generated SQL, and final answer.
        latest_message.pretty_print()

        if (
            latest_message.__class__.__name__ == "AIMessage"
            and latest_message.content
            and not latest_message.tool_calls
        ):
            final_answer = latest_message.content

    elapsed = time.perf_counter() - start

    logger.info(
        f"Execution time: {elapsed:.2f} seconds"
    )

    logger.info(
        f"Final answer: {final_answer}"
    )

    return final_answer