-- Table to store synthetic inspiration pairs
CREATE TABLE IF NOT EXISTS nfl_synthetic_inspirations (
    id SERIAL PRIMARY KEY,
    synthetic_redzone_query JSONB NOT NULL,
    inspiration_play_id UUID NOT NULL,
    reasoning TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_inspiration_play 
        FOREIGN KEY(inspiration_play_id) 
        REFERENCES nfl_plays(id)
);

CREATE INDEX idx_synthetic_inspiration_play ON nfl_synthetic_inspirations(inspiration_play_id);
