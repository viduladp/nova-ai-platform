"""
task5_nova_platform.py
NOVA AI Support Platform — Multi-Agent System
Built with LangGraph | Tasks 1-4 integrated
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional
import json, time, datetime, random, re, jsonlines

# Import RAG from Task 3
from rag_module import retrieve_and_answer

# Initialize with your config
def build_nova_platform(groq_client, db, faqs,
                          system_prompt, brand_voice_system):
    """Builds and returns the compiled NOVA LangGraph app."""

    orders_index    = {o["order_id"]:   o for o in db["orders"]}
    customers_index = {c["customer_id"]:c for c in db["customers"]}

    workflow = StateGraph(dict)

    # Add nodes here (same as notebook)
    # See task5_notebook.ipynb for full implementation

    workflow.set_entry_point("intent_classifier")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
