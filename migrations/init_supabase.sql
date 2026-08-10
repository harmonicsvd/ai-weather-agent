-- Supabase Database Initialization Script
-- Run this in Supabase SQL Editor to set up the database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create user_profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    sub TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    default_city TEXT,
    timezone TEXT DEFAULT 'Europe/Berlin',
    role TEXT,
    commute_mode TEXT,
    ppe_required BOOLEAN DEFAULT FALSE,
    risk_tolerance TEXT,
    google_refresh_token TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email);

-- Create knowledge_chunk_vectors table (for RAG)
CREATE TABLE IF NOT EXISTS knowledge_chunk_vectors (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    user_sub TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_path TEXT,
    chunk_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(3072) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for RAG queries
CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_vectors_user_sub ON knowledge_chunk_vectors(user_sub);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_vectors_document_id ON knowledge_chunk_vectors(document_id);

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to auto-update updated_at
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
