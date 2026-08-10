-- Add google_refresh_token column to user_profiles table
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS google_refresh_token TEXT;
