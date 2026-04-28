#!/usr/bin/env python3
"""
Migration script to update existing database schema to enhanced version.
Run this once before using the enhanced diarization script.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

MIGRATION_DDL = """
-- Add missing columns to existing tables
DO $$ 
BEGIN
    -- Add columns to episodes table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='episodes' AND column_name='processing_metadata') THEN
        ALTER TABLE episodes ADD COLUMN processing_metadata JSONB DEFAULT '{}';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='episodes' AND column_name='updated_at') THEN
        ALTER TABLE episodes ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
    END IF;

    -- Add columns to utterances table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='utterances' AND column_name='speaker_confidence') THEN
        ALTER TABLE utterances ADD COLUMN speaker_confidence FLOAT DEFAULT 1.0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='utterances' AND column_name='word_count') THEN
        ALTER TABLE utterances ADD COLUMN word_count INT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='utterances' AND column_name='processing_notes') THEN
        ALTER TABLE utterances ADD COLUMN processing_notes TEXT;
    END IF;

    -- Add columns to mentions table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='mentions' AND column_name='confidence') THEN
        ALTER TABLE mentions ADD COLUMN confidence FLOAT DEFAULT 0.0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='mentions' AND column_name='extraction_method') THEN
        ALTER TABLE mentions ADD COLUMN extraction_method TEXT DEFAULT 'pattern';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='mentions' AND column_name='context') THEN
        ALTER TABLE mentions ADD COLUMN context TEXT;
    END IF;

    -- Add columns to chunks table
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chunks' AND column_name='chunk_type') THEN
        ALTER TABLE chunks ADD COLUMN chunk_type TEXT DEFAULT 'utterance';
    END IF;
END $$;

-- Create missing indexes
CREATE INDEX IF NOT EXISTS idx_utterances_episode_start ON utterances(episode_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_mentions_episode_speaker ON mentions(episode_id, speaker);
CREATE INDEX IF NOT EXISTS idx_mentions_about_person ON mentions(about_person);
CREATE INDEX IF NOT EXISTS idx_mentions_confidence ON mentions(confidence DESC);

-- Update existing word_count values if they're NULL
UPDATE utterances SET word_count = array_length(string_to_array(trim(text), ' '), 1) WHERE word_count IS NULL AND text IS NOT NULL AND trim(text) != '';
"""

def with_conn():
    """Database connection helper."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Try individual components
        host = os.environ.get("PGHOST")
        db = os.environ.get("PGDATABASE")
        user = os.environ.get("PGUSER")
        password = os.environ.get("PGPASSWORD")
        port = int(os.environ.get("PGPORT", "5432"))
        
        if not all([host, db, user, password]):
            raise RuntimeError("Missing database configuration. Set DATABASE_URL or individual PG* variables.")
        
        return psycopg2.connect(
            host=host,
            dbname=db,
            user=user,
            password=password,
            port=port,
            sslmode="require"
        )
    
    # Ensure SSL for cloud databases
    if "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=require"
    
    return psycopg2.connect(db_url)

def run_migration():
    """Run the database migration."""
    print("Starting database migration...")
    
    try:
        with with_conn() as conn, conn.cursor() as cur:
            # Run the migration
            cur.execute(MIGRATION_DDL)
            conn.commit()
            
        print("✅ Migration completed successfully!")
        print("Your database is now compatible with the enhanced diarization script.")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()
