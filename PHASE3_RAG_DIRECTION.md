# Phase 3 RAG Direction

## Purpose
Define the Phase 3 RAG direction before implementation work starts.

This file is intentionally product-first.
It captures what retrieval should do for this project, which users it serves,
and how it should fit into the current weather + meeting workflow.

## Core Idea
This project should not start with generic weather-document RAG.

The stronger direction is role-aware retrieval over user-provided documents.
That means the assistant should retrieve context from files such as:

- site PDFs
- project briefs
- meeting notes
- logistics documents
- safety instructions
- access requirements
- PPE/site rules
- client documents
- travel notes
- operation documents

The weather-agent can then combine:

- calendar event context
- weather forecast/risk
- user profile context
- retrieved user knowledge

## User Model
Primary design principle:

- the user selects a role in their profile
- the user can add their own documents/data
- retrieval should adapt using role + meeting context + uploaded knowledge

This means the system should not be hardcoded to one profession.
The same retrieval system should work across roles by changing profile fields
and documents, not by changing code.

Example roles:

- architect
- contractor
- project manager
- site supervisor
- sales manager
- event coordinator
- operations lead

## Why This Direction
This gives the project bigger scope while keeping the same core architecture.

Benefits:

- one system can support many user types
- data comes from the user instead of one fixed domain
- retrieval becomes personalized and practical
- the assistant can adapt without role-specific code branches

The key product idea is:

- same meeting
- same weather
- different role
- different retrieved context
- different final explanation/actions

## What Retrieval Should Answer
Retrieval should help answer questions like:

- What does this meeting/site/client/project involve?
- Is the visit indoor, outdoor, or mixed?
- Are there access, logistics, or safety constraints?
- Does the user data mention PPE, weather sensitivity, exposed areas, or travel notes?
- Is there any document detail that changes how weather risk should be explained?
- What preparation guidance should be surfaced for this user's role?

## What Retrieval Should Not Decide
Retrieval should not decide the weather-risk label itself.

Keep this boundary:

- deterministic logic decides `low`, `moderate`, `high`, `blocked`, `unknown`
- retrieval adds project/site context for explanation and preparation guidance

This keeps the system easier to test and safer to reason about.

## Product Use
When the user talks to the voice assistant, the system should eventually be
able to answer with better context such as:

- this meeting is tied to an exposed site or weather-sensitive location
- the uploaded notes mention outdoor inspection work
- the user documents say PPE is required
- the meeting location has restricted access or travel instructions
- the project/client notes imply extra preparation steps

That allows both:

- better voice responses
- better weather recommendations

## Phase 3 Scope Suggestion
First version of RAG should stay narrow:

1. ingest a small set of user-provided PDFs/documents
2. extract and chunk them
3. embed and store them
4. retrieve relevant snippets for one meeting/user context
5. use retrieved context only in the explanation/recommendation layer

Do not use RAG yet for:

- risk scoring
- weather lookup
- route selection in the graph

## Retrieval Trigger
Good first trigger:

- only retrieve when an in-person meeting has enough user/document context
- prioritize retrieval for `moderate` and `high` weather-risk meetings

That keeps cost and complexity controlled.

## Retrieval Inputs
Likely retrieval signals for the first version:

- meeting title
- meeting location
- project/site/client name
- city
- user role

Optional later signals:

- attendees
- project code
- date window
- meeting notes

## Retrieval Output
Retrieved output should stay compact and structured.

Good first output:

- top 1 to 3 snippets
- document name
- short citation label
- metadata such as role, project/site/client name

## Suggested Repo Direction
When implementation begins, Phase 3 can likely add:

- `apps/rag/loader.py`
- `apps/rag/chunker.py`
- `apps/rag/index.py`
- `apps/rag/retriever.py`
- `data/knowledge/` for local sample docs

## Design Principle
Keep the current architecture rule:

- rules decide risk
- retrieval adds context
- LLM turns risk + context into better guidance

## Decision Summary
Phase 3 RAG direction for this project:

- role-aware
- user-provided document focused
- retrieval used for context and explanation
- not generic weather-doc RAG as the first implementation
