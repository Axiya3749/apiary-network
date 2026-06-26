"""Minimal HTTP wrapper so the agent can be deployed (Cloud Run / Agent Runtime).

Why this exists
---------------
The notebook proves the agent runs; *deployability* means it can also serve
traffic. This FastAPI app exposes a single POST /query endpoint backed by the
same ADK App built in agent.py, so the identical workflow demonstrated in the
notebook is what would be containerised and deployed -- no divergence between
demo and production.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from apiary_network.app.agent import app as adk_app

api = FastAPI(title="Backyard Apiary Network")
runner = InMemoryRunner(app=adk_app)


class Query(BaseModel):
    text: str
    keeper_id: str = "keeper"


@api.post("/query")
async def query(q: Query) -> dict:
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(
        app_name=adk_app.name, user_id=q.keeper_id, session_id=session_id
    )
    msg = types.Content(role="user", parts=[types.Part(text=q.text)])
    final_text = ""
    async for event in runner.run_async(user_id=q.keeper_id, session_id=session_id, new_message=msg):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text = part.text
        elif isinstance(getattr(event, "output", None), str) and event.output:
            # Deterministic @node terminal responses (security_block, handle_unrelated,
            # alert_keeper) return a plain string, which ADK wraps as Event(output=...)
            # rather than Event(content=...) -- without this branch their response is
            # silently dropped.
            final_text = event.output
    return {"answer": final_text, "session_id": session_id}


@api.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}
