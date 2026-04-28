#!/usr/bin/env python3
"""
Enhanced Armchair Expert transcription with improved diarization and guest mention extraction.
Key improvements:
1. Multi-pass speaker identification with voice consistency scoring
2. Enhanced guest mention extraction with context awareness
3. Better handling of cross-talk and overlapping speakers
4. Confidence scoring for all extractions
5. Ad break detection and quality assessment
"""

import os
import re
import csv
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import difflib
from collections import defaultdict, Counter
import statistics

import requests
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------
# Enhanced Configuration
# -----------------------
load_dotenv()

BASE_URL = "https://api.assemblyai.com"
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Missing ASSEMBLYAI_API_KEY in environment (.env).")

HOSTS = ["Dax Shepard", "Monica Padman"]
SKIP_TITLES_RE = re.compile(r"^\s*armchair\s+anonymous\b", re.IGNORECASE)

# Enhanced embedding model
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_emb_model = None

def emb_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return _emb_model

# -----------------------
# Enhanced Speaker Analysis
# -----------------------
class SpeakerAnalyzer:
    def __init__(self):
        self.host_indicators = {
            "dax": [
                "i'm dax", "dax shepard", "this is dax", "i'm your host dax",
                "my wife", "kristen", "bell", "shepard", "recovery", "sobriety",
                "motorcycle", "detroit", "michigan", "chips", "without a paddle"
            ],
            "monica": [
                "i'm monica", "monica padman", "this is monica", "fact check",
                "synesthesia", "georgia", "disney", "house of lies", "parks and rec",
                "the good place", "fact checking", "monica fact"
            ]
        }
        
        self.guest_indicators = [
            "thank you for having me", "thanks for having me", "glad to be here",
            "excited to be here", "honored to be here", "pleasure to be here",
            "my new", "my latest", "my book", "my movie", "my show", "my album",
            "coming out", "just released", "premieres", "available now"
        ]
    
    def analyze_speaker_patterns(self, utterances: List[Dict]) -> Dict[str, Dict]:
        """Enhanced speaker analysis with pattern recognition."""
        patterns = defaultdict(lambda: {
            "total_words": 0,
            "avg_length": 0,
            "host_score": 0,
            "guest_score": 0,
            "speaking_time": 0,
            "interruptions": 0,
            "question_ratio": 0,
            "self_references": 0,
            "typical_phrases": Counter()
        })
        
        for i, utt in enumerate(utterances):
            speaker = utt.get("speaker", "")
            text = (utt.get("text", "") or "").lower()
            duration = max(0, (utt.get("end", 0) or 0) - (utt.get("start", 0) or 0))
            
            if not speaker or not text:
                continue
                
            words = text.split()
            patterns[speaker]["total_words"] += len(words)
            patterns[speaker]["speaking_time"] += duration
            patterns[speaker]["question_ratio"] += text.count("?") / max(len(words), 1)
            patterns[speaker]["self_references"] += text.count(" i ") + text.count("i'm") + text.count("my ")
            
            # Host indicator scoring
            for host, indicators in self.host_indicators.items():
                score = sum(1 for ind in indicators if ind in text)
                if score > 0:
                    patterns[speaker]["host_score"] += score * (2 if host == "dax" else 1.5)
            
            # Guest indicator scoring
            guest_score = sum(1 for ind in self.guest_indicators if ind in text)
            patterns[speaker]["guest_score"] += guest_score
            
            # Track typical phrases (3-word combinations)
            for j in range(len(words) - 2):
                phrase = " ".join(words[j:j+3])
                patterns[speaker]["typical_phrases"][phrase] += 1
            
            # Interruption detection (short utterances following quickly)
            if i > 0 and duration < 2000 and len(words) < 10:  # Less than 2 seconds and 10 words
                prev_end = utterances[i-1].get("end", 0) or 0
                curr_start = utt.get("start", 0) or 0
                if curr_start - prev_end < 1000:  # Within 1 second
                    patterns[speaker]["interruptions"] += 1
        
        # Calculate averages
        for speaker_data in patterns.values():
            if speaker_data["total_words"] > 0:
                speaker_data["avg_length"] = speaker_data["speaking_time"] / speaker_data["total_words"]
        
        return dict(patterns)
    
    def identify_hosts_and_guests(self, utterances: List[Dict], parsed_guests: List[str]) -> Dict[str, str]:
        """Multi-pass speaker identification with confidence scoring."""
        patterns = self.analyze_speaker_patterns(utterances)
        
        if not patterns:
            return {}
        
        # Sort speakers by total speaking time
        speakers_by_time = sorted(patterns.keys(), 
                                key=lambda s: patterns[s]["speaking_time"], 
                                reverse=True)
        
        mapping = {}
        used_names = set()
        
        # Phase 1: Identify hosts more robustly
        host_candidates = []
        
        for speaker in speakers_by_time:
            data = patterns[speaker]
            total_speaking_time = sum(p["speaking_time"] for p in patterns.values())
            
            # Enhanced host scoring
            host_confidence = (
                data["host_score"] * 0.3 +  # Direct host indicators
                (data["speaking_time"] / max(total_speaking_time, 1)) * 0.4 +  # Speaking time ratio
                data["interruptions"] / max(data["total_words"] / 50, 1) * 0.1 +  # Interruption behavior
                data["question_ratio"] * 0.2  # Questions (hosts ask more)
            )
            
            # Only consider speakers with substantial participation (top 4 by time)
            if speaker in speakers_by_time[:4]:
                host_candidates.append((speaker, host_confidence, data))
        
        # Sort by confidence
        host_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Assign hosts: look for Dax indicators first
        dax_assigned = False
        monica_assigned = False
        
        for speaker, confidence, data in host_candidates:
            if len(mapping) >= 2:  # Already have both hosts
                break
                
            # Strong Dax indicators
            dax_score = (
                data["self_references"] / max(data["total_words"], 1) * 100 +  # Scale up
                sum(1 for phrase in data["typical_phrases"] if "dax" in phrase or "shepard" in phrase) * 0.3 +
                sum(1 for phrase in data["typical_phrases"] if any(word in phrase for word in ["wife", "kristen", "recovery", "detroit"])) * 0.2
            )
            
            # Strong Monica indicators  
            monica_score = (
                sum(1 for phrase in data["typical_phrases"] if "monica" in phrase or "padman" in phrase) * 0.3 +
                sum(1 for phrase in data["typical_phrases"] if any(word in phrase for word in ["fact", "synesthesia", "disney", "georgia"])) * 0.2
            )
            
            # Assign based on strongest indicators
            if not dax_assigned and (dax_score > monica_score or confidence > 0.4):
                if dax_score > 0.1 or confidence > 0.5 or not monica_assigned:  # Prefer Dax for highest confidence
                    mapping[speaker] = "Dax Shepard"
                    used_names.add("Dax Shepard")
                    dax_assigned = True
                    continue
            
            if not monica_assigned and speaker not in mapping:
                mapping[speaker] = "Monica Padman"
                used_names.add("Monica Padman")
                monica_assigned = True
        
        # If we only assigned one host, look for the other host more carefully
        if len(mapping) == 1:
            assigned_host = list(mapping.values())[0]
            remaining_candidates = [s for s in speakers_by_time if s not in mapping]
            
            # Look for the missing host among remaining speakers with host-like behavior
            for speaker in remaining_candidates:
                data = patterns[speaker]
                
                # Calculate host likelihood (not just speaking time)
                host_likelihood = (
                    data["host_score"] * 0.4 +  # Direct indicators
                    data["interruptions"] / max(data["total_words"] / 50, 1) * 0.2 +  # Host behavior
                    data["question_ratio"] * 0.3 +  # Asking questions
                    min(data["speaking_time"] / max(sum(p["speaking_time"] for p in patterns.values()), 1), 0.4) * 0.1  # Cap time influence
                )
                
                # Only assign as host if they show host-like behavior
                # AND have reasonable participation (not necessarily the most talkative)
                if (host_likelihood > 0.2 and 
                    data["speaking_time"] > 30000 and  # At least 30 seconds
                    data["total_words"] > 50):  # At least 50 words
                    
                    # Assign the missing host
                    if assigned_host == "Dax Shepard":
                        mapping[speaker] = "Monica Padman"
                    else:
                        mapping[speaker] = "Dax Shepard"
                    break
            
            # If still no second host found with host-like behavior, 
            # only then fall back to speaking time (but with more restrictions)
            if len(mapping) == 1:
                for speaker in remaining_candidates:
                    data = patterns[speaker]
                    # Must have substantial participation but not dominate
                    time_ratio = data["speaking_time"] / max(sum(p["speaking_time"] for p in patterns.values()), 1)
                    if (0.15 < time_ratio < 0.5 and  # Between 15%-50% of total time
                        data["total_words"] > 100):  # Reasonable word count
                        
                        if assigned_host == "Dax Shepard":
                            mapping[speaker] = "Monica Padman"
                        else:
                            mapping[speaker] = "Dax Shepard"
                        break
        
        # Phase 2: Identify guests from remaining speakers
        remaining_speakers = [s for s in speakers_by_time if s not in mapping]
        guest_candidates = []
        
        for speaker in remaining_speakers:
            data = patterns[speaker]
            # Guests should have meaningful participation
            if data["speaking_time"] > 60000 and data["total_words"] > 100:  # 1 min + 100 words
                guest_confidence = (
                    data["guest_score"] * 0.5 +
                    (data["speaking_time"] / max(sum(p["speaking_time"] for p in patterns.values()), 1)) * 0.4 +
                    (1 - min(data["interruptions"] / max(data["total_words"] / 100, 1), 1)) * 0.1
                )
                guest_candidates.append((speaker, guest_confidence))
        
        # Map guests to parsed names
        guest_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (speaker, confidence) in enumerate(guest_candidates):
            if i < len(parsed_guests):
                mapping[speaker] = f"Guest: {parsed_guests[i]}"
            else:
                mapping[speaker] = f"Speaker {speaker}"
        
        # Handle any remaining unmapped speakers
        for speaker in speakers_by_time:
            if speaker not in mapping:
                mapping[speaker] = f"Speaker {speaker}"
        
        return mapping

# -----------------------
# Enhanced Mention Extraction
# -----------------------
class MentionExtractor:
    def __init__(self, global_guest_catalog: List[str]):
        self.guest_catalog = global_guest_catalog
        self.full_lower, self.first_to_full, self.last_to_full = self._build_indexes()
        
        # Contextual patterns that indicate mentions
        self.mention_patterns = [
            r"\b(with|about|from|like|unlike|similar to|different from)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",
            r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(said|told|mentioned|explained|talked about)\b",
            r"\bwhen\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(was|did|came|went)\b",
            r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+and\s+(I|we|they)\b",
            r"\bremember\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b",
            r"\bknow\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"
        ]
    
    def _build_indexes(self) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
        """Build name resolution indexes with fuzzy matching capabilities."""
        full_lower = {}
        first_to_full = defaultdict(list)
        last_to_full = defaultdict(list)
        
        for name in self.guest_catalog:
            normalized = self._normalize_name(name)
            full_lower[normalized] = name
            
            tokens = self._name_tokens(name)
            if tokens:
                first_to_full[tokens[0].lower()].append(name)
                last_to_full[tokens[-1].lower()].append(name)
                
                # Also index middle names for better matching
                for token in tokens[1:-1]:
                    first_to_full[token.lower()].append(name)
        
        return full_lower, dict(first_to_full), dict(last_to_full)
    
    def _normalize_name(self, name: str) -> str:
        return re.sub(r"\s+", " ", name.strip().lower())
    
    def _name_tokens(self, name: str) -> List[str]:
        return [t for t in re.split(r"[^A-Za-z]+", name) if t and len(t) > 1]
    
    def extract_contextual_mentions(self, utterances: List[Dict], 
                                  speaker_mapping: Dict[str, str],
                                  fact_check_start: Optional[int] = None) -> List[Tuple[str, str, str, int, int, float]]:
        """Extract mentions using contextual patterns and NLP techniques."""
        mentions = []
        
        for i, utt in enumerate(utterances):
            speaker_label = utt.get("speaker", "")
            speaker_name = speaker_mapping.get(speaker_label, speaker_label)
            text = utt.get("text", "") or ""
            start_ms = utt.get("start", 0) or 0
            end_ms = utt.get("end", 0) or 0
            
            # Skip if speaker is a host
            if speaker_name.lower() in {"dax shepard", "monica padman"}:
                continue
            
            # Skip if this is in the fact check segment
            if fact_check_start and start_ms >= fact_check_start:
                continue
            
            # Extract potential mentions using patterns
            potential_mentions = self._find_name_patterns(text)
            
            # Get context from surrounding utterances
            context = self._get_context(utterances, i, window=2)
            
            for mention_text, quote_span, confidence in potential_mentions:
                resolved_name = self._resolve_guest_name(mention_text)
                if not resolved_name:
                    continue
                
                # Avoid self-mentions
                guest_name = speaker_name.replace("Guest: ", "").strip()
                if resolved_name.lower() == guest_name.lower():
                    continue
                
                # Calculate context-aware confidence
                context_confidence = self._calculate_context_confidence(
                    mention_text, quote_span, context, resolved_name
                )
                
                final_confidence = confidence * 0.7 + context_confidence * 0.3
                
                if final_confidence > 0.5:  # Confidence threshold
                    mentions.append((
                        speaker_name,
                        resolved_name,
                        quote_span,
                        start_ms,
                        end_ms,
                        final_confidence
                    ))
        
        return mentions
    
    def _find_name_patterns(self, text: str) -> List[Tuple[str, str, float]]:
        """Find potential name mentions using regex patterns."""
        results = []
        
        # Direct name extraction with patterns
        for pattern in self.mention_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                for group in groups:
                    if self._looks_like_name(group):
                        # Extract a broader quote span around the match
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        quote_span = text[start:end].strip()
                        
                        confidence = 0.8 if len(group.split()) > 1 else 0.6
                        results.append((group, quote_span, confidence))
        
        # Also look for capitalized sequences (potential names)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(name_pattern, text):
            name = match.group()
            if self._looks_like_name(name) and len(name.split()) >= 2:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                quote_span = text[start:end].strip()
                
                results.append((name, quote_span, 0.5))
        
        # Deduplicate while preserving highest confidence
        name_to_best = {}
        for name, quote, conf in results:
            if name not in name_to_best or conf > name_to_best[name][1]:
                name_to_best[name] = (quote, conf)
        
        return [(name, quote, conf) for name, (quote, conf) in name_to_best.items()]
    
    def _looks_like_name(self, text: str) -> bool:
        """Heuristics to determine if text looks like a person's name."""
        if not text or len(text) < 2:
            return False
        
        # Skip common words that aren't names
        common_words = {
            'the', 'and', 'but', 'or', 'for', 'nor', 'so', 'yet', 'a', 'an',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers',
            'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself',
            'itself', 'ourselves', 'yourselves', 'themselves'
        }
        
        words = text.lower().split()
        if len(words) == 1 and words[0] in common_words:
            return False
        
        # Must start with capital and have reasonable length
        if not text[0].isupper():
            return False
        
        # Should contain only letters, spaces, hyphens, apostrophes
        if not re.match(r"^[A-Za-z\s\-']+$", text):
            return False
        
        return True
    
    def _get_context(self, utterances: List[Dict], current_index: int, window: int = 2) -> str:
        """Get context from surrounding utterances."""
        start = max(0, current_index - window)
        end = min(len(utterances), current_index + window + 1)
        
        context_parts = []
        for i in range(start, end):
            if i != current_index:
                text = utterances[i].get("text", "") or ""
                context_parts.append(text)
        
        return " ".join(context_parts)
    
    def _calculate_context_confidence(self, mention_text: str, quote_span: str, 
                                    context: str, resolved_name: str) -> float:
        """Calculate confidence based on context analysis."""
        confidence = 0.5
        
        # Check if the resolved name appears in context
        if resolved_name.lower() in context.lower():
            confidence += 0.2
        
        # Check for relationship words
        relationship_words = [
            'friend', 'colleague', 'co-star', 'worked with', 'met', 'knows',
            'told me', 'said', 'mentioned', 'talked about', 'with', 'and'
        ]
        
        combined_text = (quote_span + " " + context).lower()
        relationship_count = sum(1 for word in relationship_words if word in combined_text)
        confidence += min(relationship_count * 0.1, 0.3)
        
        # Penalty for very short mentions
        if len(mention_text.split()) == 1:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _resolve_guest_name(self, mention_text: str, fuzzy_threshold: float = 0.85) -> Optional[str]:
        """Resolve a mention to a canonical guest name."""
        if not mention_text or not mention_text.strip():
            return None
        
        normalized = self._normalize_name(mention_text)
        
        # Exact match
        if normalized in self.full_lower:
            return self.full_lower[normalized]
        
        # Try partial matches
        tokens = self._name_tokens(mention_text)
        if len(tokens) == 1:
            # Single name - try last name match
            candidates = self.last_to_full.get(tokens[0].lower(), [])
            if len(candidates) == 1:
                return candidates[0]
            
            # Try first name match if unique
            candidates = self.first_to_full.get(tokens[0].lower(), [])
            if len(candidates) == 1:
                return candidates[0]
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for canonical_normalized, canonical_name in self.full_lower.items():
            score = difflib.SequenceMatcher(None, normalized, canonical_normalized).ratio()
            if score > best_score and score >= fuzzy_threshold:
                best_score = score
                best_match = canonical_name
        
        return best_match

# -----------------------
# Enhanced Database Schema
# -----------------------
ENHANCED_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodes (
  id BIGSERIAL PRIMARY KEY,
  file_stem TEXT UNIQUE,
  file_path TEXT,
  guests JSONB,
  processing_metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS utterances (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  speaker_confidence FLOAT DEFAULT 1.0,
  start_ms INT,
  end_ms INT,
  text TEXT,
  word_count INT,
  processing_notes TEXT
);

CREATE TABLE IF NOT EXISTS mentions (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  about_person TEXT,
  quote TEXT,
  start_ms INT,
  end_ms INT,
  confidence FLOAT DEFAULT 0.0,
  extraction_method TEXT DEFAULT 'pattern',
  context TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  episode_id BIGINT REFERENCES episodes(id) ON DELETE CASCADE,
  speaker TEXT,
  start_ms INT,
  end_ms INT,
  text TEXT,
  embedding vector(384),
  chunk_type TEXT DEFAULT 'utterance'
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_utterances_episode_start ON utterances(episode_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_mentions_episode_speaker ON mentions(episode_id, speaker);
CREATE INDEX IF NOT EXISTS idx_mentions_about_person ON mentions(about_person);
CREATE INDEX IF NOT EXISTS idx_mentions_confidence ON mentions(confidence DESC);
CREATE INDEX IF NOT EXISTS chunks_embedding_ivf ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

# -----------------------
# Fact Check Detection
# -----------------------
def detect_fact_check_segment(utterances: List[Dict], speaker_mapping: Dict[str, str]) -> Tuple[Optional[int], List[str]]:
    """
    Detect the start of the fact check segment.
    Returns: (start_timestamp_ms, fact_check_indicators_found)
    """
    fact_check_indicators = [
        "fact check with my soulmate",
        "fact check with monica padman", 
        "fact checking with dax shepard",
        "time for fact check",
        "let's fact check",
        "now it's time to fact check",
        "fact check time",
        "monica fact",
        "dax fact"
    ]
    
    music_indicators = [
        "music", "♪", "♫", "🎵", "🎶",
        "[music]", "(music)", "theme music",
        "transition music", "outro music"
    ]
    
    found_indicators = []
    potential_start = None
    
    # Look in the last 45 minutes of the episode for fact check start
    if not utterances:
        return None, []
    
    # Get total episode duration
    last_utterance = max(utterances, key=lambda u: u.get("end", 0) or 0)
    total_duration = last_utterance.get("end", 0) or 0
    
    # Start looking from 30 minutes before the end
    fact_check_search_start = max(0, total_duration - (45 * 60 * 1000))  # 45 minutes in ms
    
    # Look for fact check indicators in the last portion
    for i, utt in enumerate(utterances):
        start_ms = utt.get("start", 0) or 0
        text = (utt.get("text", "") or "").lower()
        speaker = speaker_mapping.get(utt.get("speaker", ""), "")
        
        if start_ms < fact_check_search_start:
            continue
            
        # Look for fact check phrases
        for indicator in fact_check_indicators:
            if indicator in text:
                if potential_start is None or start_ms < potential_start:
                    potential_start = start_ms
                    found_indicators.append(f"'{indicator}' at {ms_to_time(start_ms)}")
        
        # Look for music cues (often precede fact check)
        for music_cue in music_indicators:
            if music_cue in text and start_ms >= fact_check_search_start:
                # Check if fact check language appears soon after music
                for j in range(i, min(i + 10, len(utterances))):  # Check next 10 utterances
                    next_text = (utterances[j].get("text", "") or "").lower()
                    if any(fc in next_text for fc in fact_check_indicators[:5]):  # Main fact check phrases
                        next_start = utterances[j].get("start", 0) or 0
                        if potential_start is None or next_start < potential_start:
                            potential_start = next_start
                            found_indicators.append(f"Music cue + fact check at {ms_to_time(next_start)}")
                        break
    
    return potential_start, found_indicators

# -----------------------
# Ad Detection
# -----------------------
def detect_ad_breaks(utterances: List[Dict], speaker_mapping: Dict[str, str]) -> Tuple[List[Tuple[int, int, str]], bool]:
    """
    Detect ad breaks in the episode.
    Returns: (list of (start_ms, end_ms, indicator_text), ads_present_bool)
    """
    ad_indicators = [
        "we'll be right back with armchair expert",
        "we'll be right back",
        "we will be right back",
        "and we're back",
        "we're back with",
        "brought to you by",
        "this episode is brought to you",
        "today's episode is sponsored",
        "our sponsors",
        "word from our sponsor",
        "quick break",
        "take a quick break",
        "after the break",
        "before we get back",
        "let's hear from our sponsor",
        "speaking of sponsors"
    ]
    
    music_transitions = [
        "[music]", "(music)", "♪", "♫", "🎵", "🎶",
        "theme music", "transition music"
    ]
    
    detected_breaks = []
    
    for i, utt in enumerate(utterances):
        text = (utt.get("text", "") or "").lower()
        start_ms = utt.get("start", 0) or 0
        end_ms = utt.get("end", 0) or 0
        speaker = speaker_mapping.get(utt.get("speaker", ""), "")
        
        # Check for ad indicators from hosts
        if speaker.lower() in {"dax shepard", "monica padman"}:
            for indicator in ad_indicators:
                if indicator in text:
                    # Look for music transition nearby (within 5 utterances)
                    has_music = False
                    break_end = end_ms
                    
                    for j in range(max(0, i-2), min(len(utterances), i+5)):
                        nearby_text = (utterances[j].get("text", "") or "").lower()
                        if any(music in nearby_text for music in music_transitions):
                            has_music = True
                            break_end = utterances[j].get("end", 0) or end_ms
                            break
                    
                    detected_breaks.append((
                        start_ms,
                        break_end,
                        f"'{indicator}' at {ms_to_time(start_ms)}" + (" + music" if has_music else "")
                    ))
                    break
    
    # Deduplicate breaks that are close together (within 30 seconds)
    unique_breaks = []
    for break_start, break_end, indicator in sorted(detected_breaks):
        if not unique_breaks or break_start - unique_breaks[-1][0] > 30000:
            unique_breaks.append((break_start, break_end, indicator))
    
    ads_present = len(unique_breaks) > 0
    
    return unique_breaks, ads_present

def analyze_ad_removal_quality(utterances: List[Dict], speaker_mapping: Dict[str, str], 
                               total_duration_ms: int) -> Dict:
    """
    Analyze whether ads were successfully removed and provide quality metrics.
    """
    ad_breaks, ads_present = detect_ad_breaks(utterances, speaker_mapping)
    
    # Check for suspicious gaps in timestamps (might indicate removed ads)
    gaps = []
    for i in range(len(utterances) - 1):
        current_end = utterances[i].get("end", 0) or 0
        next_start = utterances[i + 1].get("start", 0) or 0
        gap_duration = next_start - current_end
        
        # Gaps longer than 5 seconds might be removed content
        if gap_duration > 5000:
            gaps.append((current_end, next_start, gap_duration))
    
    # Large gaps might be removed ads
    suspicious_gaps = [g for g in gaps if g[2] > 30000]  # Gaps > 30 seconds
    
    return {
        "ads_detected": ads_present,
        "ad_break_count": len(ad_breaks),
        "ad_breaks": [
            {
                "timestamp": ms_to_time(start),
                "indicator": indicator,
                "duration_ms": end - start
            }
            for start, end, indicator in ad_breaks
        ],
        "suspicious_gaps_count": len(suspicious_gaps),
        "suspicious_gaps": [
            {
                "start": ms_to_time(start),
                "end": ms_to_time(end),
                "duration_seconds": duration / 1000
            }
            for start, end, duration in suspicious_gaps
        ],
        "total_gap_time_ms": sum(g[2] for g in gaps),
        "ad_removal_quality": "CLEAN" if not ads_present and len(suspicious_gaps) == 0 
                             else "ADS_PRESENT" if ads_present 
                             else "POSSIBLE_GAPS"
    }

def ms_to_time(milliseconds: int) -> str:
    """Convert milliseconds to MM:SS format."""
    if milliseconds <= 0:
        return "0:00"
    total_seconds = milliseconds // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"

# -----------------------
# Enhanced Processing Functions
# -----------------------
def enhanced_process_episode(file_path: str,
                           global_guest_catalog: List[str],
                           skip_anonymous: bool = True,
                           reprocess: bool = False) -> Dict:
    """Enhanced episode processing with improved diarization and mention extraction."""
    
    file_path = os.path.abspath(file_path)
    stem = safe_stem(file_path)
    
    if skip_anonymous and SKIP_TITLES_RE.match(stem):
        return {"file": file_path, "stem": stem, "skipped": True, "reason": "Armchair Anonymous"}
    
    existing_id = episode_exists(stem)
    if existing_id and not reprocess:
        return {"file": file_path, "stem": stem, "skipped": True, "episode_id": existing_id, "reason": "already_ingested"}
    
    print(f"Processing: {stem}")
    
    # Transcribe with enhanced settings
    result, transcript_id = enhanced_transcribe(file_path)
    
    utterances = result.get("utterances") or []
    entities = [e for e in result.get("entities", []) if e.get("entity_type") == "person_name"]
    
    # Get total duration
    total_duration = max((u.get("end", 0) or 0) for u in utterances) if utterances else 0
    
    # Enhanced speaker analysis
    speaker_analyzer = SpeakerAnalyzer()
    parsed_guests = parse_guests_from_filename(stem)
    speaker_mapping = speaker_analyzer.identify_hosts_and_guests(utterances, parsed_guests)
    
    print(f"Speaker mapping: {speaker_mapping}")
    
    # Detect ads
    ad_analysis = analyze_ad_removal_quality(utterances, speaker_mapping, total_duration)
    
    if ad_analysis["ads_detected"]:
        print(f"⚠️  ADS DETECTED - Found {ad_analysis['ad_break_count']} ad breaks:")
        for ad_break in ad_analysis["ad_breaks"]:
            print(f"   - {ad_break['timestamp']}: {ad_break['indicator']}")
    else:
        print(f"✓ No ads detected")
        if ad_analysis["suspicious_gaps_count"] > 0:
            print(f"   Note: {ad_analysis['suspicious_gaps_count']} suspicious gaps found (possible removed ads)")
    
    # Detect fact check segment
    fact_check_start, fact_check_indicators = detect_fact_check_segment(utterances, speaker_mapping)
    
    if fact_check_start:
        print(f"Fact check detected starting at {ms_to_time(fact_check_start)} - {', '.join(fact_check_indicators)}")
    else:
        print("No fact check segment detected")
    
    # Enhanced mention extraction (exclude fact check segment)
    mention_extractor = MentionExtractor(global_guest_catalog)
    contextual_mentions = mention_extractor.extract_contextual_mentions(
        utterances, speaker_mapping, fact_check_start
    )
    
    # Combine with entity-based mentions
    entity_mentions = extract_entity_mentions(
        entities, utterances, speaker_mapping, mention_extractor, 
        global_guest_catalog, fact_check_start
    )
    
    # Merge and deduplicate mentions
    all_mentions = merge_mentions(contextual_mentions, entity_mentions)
    
    print(f"Found {len(contextual_mentions)} contextual mentions, {len(entity_mentions)} entity mentions")
    print(f"Total unique mentions: {len(all_mentions)}")
    
    # Prepare database rows
    utt_rows = []
    for utt in utterances:
        speaker = speaker_mapping.get(utt.get("speaker", ""), utt.get("speaker", "SPK"))
        text = (utt.get("text") or "").strip()
        word_count = len(text.split()) if text else 0
        
        # Mark fact check utterances
        processing_notes = None
        if fact_check_start and (utt.get("start", 0) or 0) >= fact_check_start:
            processing_notes = "fact_check_segment"
        
        utt_rows.append((
            speaker,
            1.0,  # speaker_confidence - could be enhanced
            utt.get("start") or -1,
            utt.get("end") or -1,
            text,
            word_count,
            processing_notes
        ))
    
    # Mention rows with confidence scores
    mention_rows = []
    for speaker, about, quote, start_ms, end_ms, confidence in all_mentions:
        # Get some context for storage
        context = get_utterance_context(utterances, start_ms, end_ms)
        mention_rows.append((
            speaker,
            about,
            quote,
            start_ms,
            end_ms,
            confidence,
            'enhanced',  # extraction_method
            context[:500]  # truncate context
        ))
    
    # Store in database
    processing_metadata = {
        "speaker_patterns": speaker_analyzer.analyze_speaker_patterns(utterances),
        "total_utterances": len(utterances),
        "total_mentions": len(all_mentions),
        "avg_mention_confidence": statistics.mean([m[5] for m in all_mentions]) if all_mentions else 0.0,
        "fact_check_detected": fact_check_start is not None,
        "fact_check_start_ms": fact_check_start,
        "fact_check_indicators": fact_check_indicators,
        "ad_analysis": ad_analysis  # Add ad analysis to metadata
    }
    
    episode_id = upsert_enhanced_episode(stem, file_path, parsed_guests, processing_metadata)
    clear_episode_rows(episode_id)
    insert_enhanced_utterances(episode_id, utt_rows)
    insert_enhanced_mentions(episode_id, mention_rows)
    insert_chunks_with_embeddings(episode_id, [(r[0], r[2], r[3], r[4]) for r in utt_rows])
    
    return {
        "file": file_path,
        "stem": stem,
        "skipped": False,
        "episode_id": episode_id,
        "utterances": len(utt_rows),
        "mentions": len(mention_rows),
        "avg_confidence": processing_metadata["avg_mention_confidence"],
        "fact_check_detected": fact_check_start is not None,
        "ad_removal_quality": ad_analysis["ad_removal_quality"],
        "ads_detected": ad_analysis["ads_detected"]
    }

def enhanced_transcribe(file_path: str, max_speakers: int = 6) -> Tuple[dict, str]:
    """Enhanced transcription with optimized settings for podcast diarization."""
    headers = {"authorization": API_KEY}
    
    # Upload
    with open(file_path, "rb") as f:
        up = requests.post(f"{BASE_URL}/v2/upload", headers=headers, data=f, timeout=600)
        up.raise_for_status()
    audio_url = up.json()["upload_url"]
    
    # Enhanced transcription settings
    payload = {
        "audio_url": audio_url,
        "speech_model": "universal",  # or "nano" for faster processing
        "speaker_labels": True,
        "speakers_expected": max_speakers,  # Allow for more speakers
        "entity_detection": True,
        "sentiment_analysis": True,  # Could be useful for context
        "auto_highlights": True,  # Might help identify key moments
        "punctuate": True,
        "format_text": True,
        "language_detection": True
    }
    
    resp = requests.post(f"{BASE_URL}/v2/transcript", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    tid = resp.json()["id"]
    
    # Poll for completion
    endpoint = f"{BASE_URL}/v2/transcript/{tid}"
    while True:
        r = requests.get(endpoint, headers=headers, timeout=60)
        r.raise_for_status()
        res = r.json()
        if res.get("status") == "completed":
            return res, tid
        if res.get("status") == "error":
            raise RuntimeError(res.get("error"))
        time.sleep(3)

# -----------------------
# Helper Functions
# -----------------------
def safe_stem(path_or_str: str) -> str:
    stem = os.path.splitext(os.path.basename(path_or_str))[0]
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"[^\w\-\.\(\) ]", "", stem)
    return stem or "audio"

def parse_guests_from_filename(stem: str) -> List[str]:
    """Enhanced guest name parsing from filename."""
    s = stem
    # Remove episode numbers at start
    s = re.sub(r"^\s*\d+\s*[-–—]\s*", "", s)
    
    # Remove trailing metadata (Interview, Part, version, etc.)
    s = re.sub(r"\s*[-–—]\s*(Interview.*|Part\s*\d+|v\d+|Remaster.*|Live|Best Of)$", "", s, flags=re.I)
    
    # Check for guests in parentheses first (like "with Ben Gilbert and David Rosenthal")
    parentheses_guests = []
    paren_match = re.search(r"\(with\s+([^)]+)\)", s, re.IGNORECASE)
    if paren_match:
        # Extract names from parentheses
        paren_content = paren_match.group(1)
        # Split on "and", "&", ","
        paren_parts = re.split(r"\s*(?:and|&|,)\s+", paren_content, flags=re.I)
        for part in paren_parts:
            cleaned = part.strip()
            if cleaned and len(cleaned) > 2:
                parentheses_guests.append(cleaned)
        # Remove the parentheses part from the main string
        s = re.sub(r"\s*\(with[^)]+\)", "", s, flags=re.I)
    
    # Remove any remaining parenthetical descriptions
    s = re.sub(r"\s*\([^)]+\)\s*", "", s)
    
    # Handle multiple separators for guest names in main title
    main_parts = re.split(r"\s*(?:&|,| and | with | feat\.? | ft\.? )\s*", s, flags=re.I)
    
    bad_words = {"armchair", "expert", "live", "best of", "bonus", "interview", "remaster", "part", "on", "about", "podcast"}
    main_guests = []
    
    for part in main_parts:
        cleaned = part.strip(" -—""'")
        if cleaned and not any(bad in cleaned.lower() for bad in bad_words):
            # Additional cleanup
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            # Must be reasonable name length and not just numbers
            if len(cleaned) > 2 and not cleaned.isdigit() and len(cleaned.split()) <= 4:
                main_guests.append(cleaned)
    
    # Combine parentheses guests (higher priority) with main guests
    all_guests = parentheses_guests + main_guests
    
    # Deduplicate case-insensitive
    seen, unique = set(), []
    for name in all_guests:
        key = name.lower()
        if key not in seen:
            unique.append(name)
            seen.add(key)
    
    return unique

def extract_entity_mentions(entities: List[Dict], utterances: List[Dict], 
                          speaker_mapping: Dict[str, str], mention_extractor: MentionExtractor,
                          global_guest_catalog: List[str],
                          fact_check_start: Optional[int] = None) -> List[Tuple[str, str, str, int, int, float]]:
    """Extract mentions from entity detection results."""
    mentions = []
    
    for entity in entities:
        entity_text = (entity.get("text") or "").strip()
        start_ms = entity.get("start")
        end_ms = entity.get("end")
        
        if not entity_text or start_ms is None or end_ms is None:
            continue
        
        # Skip if this entity is in the fact check segment
        if fact_check_start and start_ms >= fact_check_start:
            continue
        
        # Find the best overlapping utterance
        best_utterance = None
        best_overlap = 0
        
        for utt in utterances:
            utt_start = utt.get("start", 0) or 0
            utt_end = utt.get("end", 0) or 0
            overlap = max(0, min(utt_end, end_ms) - max(utt_start, start_ms))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_utterance = utt
        
        if not best_utterance:
            continue
        
        speaker_label = best_utterance.get("speaker", "")
        speaker_name = speaker_mapping.get(speaker_label, speaker_label)
        
        # Skip host speakers
        if speaker_name.lower() in {"dax shepard", "monica padman"}:
            continue
        
        # Resolve entity to canonical guest name
        resolved_name = mention_extractor._resolve_guest_name(entity_text)
        if not resolved_name:
            continue
        
        # Avoid self-mentions
        guest_name = speaker_name.replace("Guest: ", "").strip()
        if resolved_name.lower() == guest_name.lower():
            continue
        
        quote_text = (best_utterance.get("text") or "").strip()
        confidence = entity.get("confidence", 0.7)  # AssemblyAI provides confidence
        
        mentions.append((
            speaker_name,
            resolved_name,
            quote_text,
            best_utterance.get("start", 0) or 0,
            best_utterance.get("end", 0) or 0,
            confidence
        ))
    
    return mentions

def merge_mentions(contextual_mentions: List[Tuple[str, str, str, int, int, float]],
                   entity_mentions: List[Tuple[str, str, str, int, int, float]]) -> List[Tuple[str, str, str, int, int, float]]:
    """Merge and deduplicate mentions from different extraction methods."""
    all_mentions = []
    seen_keys = set()
    
    # Process contextual mentions first (usually higher quality)
    for mention in contextual_mentions:
        speaker, about, quote, start_ms, end_ms, confidence = mention
        key = (speaker.lower(), about.lower(), start_ms)  # Use start time to distinguish
        
        if key not in seen_keys:
            all_mentions.append(mention)
            seen_keys.add(key)
    
    # Add entity mentions that aren't duplicates
    for mention in entity_mentions:
        speaker, about, quote, start_ms, end_ms, confidence = mention
        key = (speaker.lower(), about.lower(), start_ms)
        
        if key not in seen_keys:
            all_mentions.append(mention)
            seen_keys.add(key)
    
    # Sort by confidence score (highest first)
    all_mentions.sort(key=lambda x: x[5], reverse=True)
    
    return all_mentions

def get_utterance_context(utterances: List[Dict], start_ms: int, end_ms: int, window_ms: int = 30000) -> str:
    """Get context around a specific time range."""
    context_parts = []
    
    for utt in utterances:
        utt_start = utt.get("start", 0) or 0
        utt_end = utt.get("end", 0) or 0
        
        # Include utterances within the time window
        if (utt_start >= start_ms - window_ms and utt_start <= end_ms + window_ms):
            text = (utt.get("text") or "").strip()
            if text:
                context_parts.append(text)
    
    return " ".join(context_parts)

# -----------------------
# Enhanced Database Functions
# -----------------------
def ensure_enhanced_schema():
    """Create enhanced database schema."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute(ENHANCED_DDL)
        conn.commit()

def episode_exists(file_stem: str) -> Optional[int]:
    """Check if episode already exists."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM episodes WHERE file_stem=%s", (file_stem,))
        row = cur.fetchone()
        return row[0] if row else None

def upsert_enhanced_episode(file_stem: str, file_path: str, guests: List[str], 
                          processing_metadata: Dict) -> int:
    """Insert or update episode with enhanced metadata."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO episodes (file_stem, file_path, guests, processing_metadata, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (file_stem) DO UPDATE SET 
                file_path=EXCLUDED.file_path, 
                guests=EXCLUDED.guests,
                processing_metadata=EXCLUDED.processing_metadata,
                updated_at=EXCLUDED.updated_at
            RETURNING id
        """, (file_stem, file_path, json.dumps(guests), json.dumps(processing_metadata)))
        episode_id = cur.fetchone()[0]
        conn.commit()
        return episode_id

def clear_episode_rows(episode_id: int):
    """Clear existing data for re-processing."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM utterances WHERE episode_id=%s", (episode_id,))
        cur.execute("DELETE FROM mentions WHERE episode_id=%s", (episode_id,))
        cur.execute("DELETE FROM chunks WHERE episode_id=%s", (episode_id,))
        conn.commit()

def insert_enhanced_utterances(episode_id: int, rows: List[Tuple]):
    """Insert utterances with enhanced metadata."""
    with with_conn() as conn, conn.cursor() as cur:
        if not rows:
            return
        execute_values(cur, """
            INSERT INTO utterances (episode_id, speaker, speaker_confidence, start_ms, end_ms, text, word_count, processing_notes)
            VALUES %s
        """, [(episode_id, *r) for r in rows])
        conn.commit()

def insert_enhanced_mentions(episode_id: int, rows: List[Tuple]):
    """Insert mentions with confidence scores and metadata."""
    with with_conn() as conn, conn.cursor() as cur:
        if not rows:
            return
        execute_values(cur, """
            INSERT INTO mentions (episode_id, speaker, about_person, quote, start_ms, end_ms, confidence, extraction_method, context)
            VALUES %s
        """, [(episode_id, *r) for r in rows])
        conn.commit()

def insert_chunks_with_embeddings(episode_id: int, rows: List[Tuple[str,int,int,str]]):
    """Insert semantic chunks with embeddings."""
    if not rows:
        return
    
    texts = [r[3] for r in rows if r[3]]  # Filter out empty texts
    if not texts:
        return
        
    vecs = emb_model().encode(texts, normalize_embeddings=True).tolist()
    vec_literals = ["[" + ",".join(f"{x:.6f}" for x in v) + "]" for v in vecs]
    
    with with_conn() as conn, conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO chunks (episode_id, speaker, start_ms, end_ms, text, embedding, chunk_type)
            VALUES %s
        """, [(episode_id, r[0], r[1], r[2], r[3], vec_literals[i], 'utterance') 
              for i, r in enumerate(rows) if r[3]])
        conn.commit()

# -----------------------
# Database Connection
# -----------------------
def with_conn():
    """Enhanced database connection with better error handling."""
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

# -----------------------
# Guest Catalog Management
# -----------------------
def build_comprehensive_guest_catalog(input_dir: Optional[Path] = None, 
                                    csv_path: Optional[str] = None) -> List[str]:
    """Build a comprehensive guest catalog from multiple sources."""
    all_guests = []
    
    # From filenames
    if input_dir:
        patterns = ["*.mp3", "*.m4a", "*.wav", "*.flac"]
        for pattern in patterns:
            for file_path in input_dir.glob(pattern):
                stem = safe_stem(str(file_path))
                if not SKIP_TITLES_RE.match(stem):
                    guests = parse_guests_from_filename(stem)
                    all_guests.extend(guests)
    
    # From existing database
    try:
        with with_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT jsonb_array_elements_text(guests) FROM episodes")
            db_guests = [row[0] for row in cur.fetchall() if row[0]]
            all_guests.extend(db_guests)
    except Exception as e:
        print(f"Warning: Could not load guests from database: {e}")
    
    # From CSV file
    if csv_path and Path(csv_path).exists():
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                # Skip header if present
                first_row = next(reader, [])
                if first_row and not any(char.isdigit() for char in first_row[0]):
                    pass  # Skip header row
                else:
                    all_guests.append(first_row[0] if first_row else "")
                
                # Read remaining rows
                for row in reader:
                    if row and row[0].strip():
                        all_guests.append(row[0].strip())
        except Exception as e:
            print(f"Warning: Could not load CSV guest list: {e}")
    
    # Deduplicate and clean
    host_names_lower = {host.lower() for host in HOSTS}
    unique_guests = []
    seen = set()
    
    for guest in all_guests:
        if not guest or not guest.strip():
            continue
        
        cleaned = guest.strip()
        normalized = cleaned.lower()
        
        # Skip hosts and duplicates
        if normalized in host_names_lower or normalized in seen:
            continue
        
        # Basic validation
        if len(cleaned) > 2 and not cleaned.isdigit():
            unique_guests.append(cleaned)
            seen.add(normalized)
    
    print(f"Built guest catalog with {len(unique_guests)} unique guests")
    return sorted(unique_guests)

# -----------------------
# Analysis and Reporting Functions
# -----------------------
def analyze_mention_quality(episode_id: int) -> Dict:
    """Analyze the quality of extracted mentions for an episode."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 
                COUNT(*) as total_mentions,
                AVG(confidence) as avg_confidence,
                COUNT(DISTINCT about_person) as unique_guests_mentioned,
                COUNT(DISTINCT speaker) as speakers_with_mentions,
                extraction_method,
                COUNT(*) as method_count
            FROM mentions 
            WHERE episode_id = %s
            GROUP BY extraction_method
        """, (episode_id,))
        
        results = cur.fetchall()
        
        # Get overall stats
        cur.execute("""
            SELECT 
                COUNT(*) as total_mentions,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                COUNT(DISTINCT about_person) as unique_guests,
                COUNT(DISTINCT speaker) as speakers_mentioning
            FROM mentions 
            WHERE episode_id = %s
        """, (episode_id,))
        
        overall = cur.fetchone()
        
    return {
        "overall": {
            "total_mentions": overall[0],
            "avg_confidence": float(overall[1]) if overall[1] else 0.0,
            "min_confidence": float(overall[2]) if overall[2] else 0.0,
            "max_confidence": float(overall[3]) if overall[3] else 0.0,
            "unique_guests_mentioned": overall[4],
            "speakers_with_mentions": overall[5]
        },
        "by_method": {row[4]: {"count": row[5], "avg_confidence": float(row[1])} 
                     for row in results}
    }

def export_mentions_for_game(episode_id: int, min_confidence: float = 0.6) -> List[Dict]:
    """Export high-quality mentions in format suitable for the game."""
    with with_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 
                m.speaker,
                m.about_person,
                m.quote,
                m.confidence,
                m.start_ms,
                m.end_ms,
                e.file_stem
            FROM mentions m
            JOIN episodes e ON m.episode_id = e.id
            WHERE m.episode_id = %s AND m.confidence >= %s
            ORDER BY m.confidence DESC
        """, (episode_id, min_confidence))
        
        results = []
        for row in cur.fetchall():
            results.append({
                "speaker": row[0],
                "mentioned_guest": row[1],
                "quote": row[2],
                "confidence": float(row[3]),
                "timestamp_ms": row[4],
                "duration_ms": row[5] - row[4],
                "episode": row[6]
            })
        
    return results

# -----------------------
# Main Processing Function
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced Armchair Expert transcription and mention extraction")
    parser.add_argument("--input-dir", type=str, help="Directory containing audio files")
    parser.add_argument("--guest-csv", type=str, help="CSV file with guest names")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess existing episodes")
    parser.add_argument("--include-anonymous", action="store_true", help="Include Armchair Anonymous episodes")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    parser.add_argument("--analyze-only", type=str, help="Just analyze mentions for given episode stem")
    
    args = parser.parse_args()
    
    # Initialize database
    ensure_enhanced_schema()
    
    if args.analyze_only:
        episode_id = episode_exists(args.analyze_only)
        if episode_id:
            quality = analyze_mention_quality(episode_id)
            mentions = export_mentions_for_game(episode_id)
            print(f"\nAnalysis for {args.analyze_only}:")
            print(f"Overall: {quality['overall']}")
            print(f"By method: {quality['by_method']}")
            print(f"High-quality mentions for game: {len(mentions)}")
        else:
            print(f"Episode {args.analyze_only} not found")
        return
    
    # Build guest catalog
    input_path = Path(args.input_dir).expanduser() if args.input_dir else None
    guest_catalog = build_comprehensive_guest_catalog(input_path, args.guest_csv)
    
    # Get files to process
    files = []
    if args.input_dir:
        patterns = ["*.mp3", "*.m4a", "*.wav", "*.flac"]
        for pattern in patterns:
            files.extend(list(input_path.glob(pattern)))
        files.sort()
    
    if args.limit and len(files) > args.limit:
        files = files[:args.limit]
    
    print(f"Processing {len(files)} files with guest catalog of {len(guest_catalog)} names")
    
    # Process files
    processed = 0
    for i, file_path in enumerate(files, 1):
        try:
            result = enhanced_process_episode(
                str(file_path),
                guest_catalog,
                skip_anonymous=not args.include_anonymous,
                reprocess=args.reprocess
            )
            
            if result.get("skipped"):
                print(f"[{i}/{len(files)}] SKIP {result['stem']} - {result.get('reason', 'unknown')}")
            else:
                avg_conf = result.get('avg_confidence', 0.0)
                ad_quality = result.get('ad_removal_quality', 'UNKNOWN')
                
                # Add ad status indicator to output
                ad_indicator = ""
                if result.get('ads_detected'):
                    ad_indicator = " [ADS DETECTED]"
                elif ad_quality == "POSSIBLE_GAPS":
                    ad_indicator = " [GAPS DETECTED]"
                else:
                    ad_indicator = " [CLEAN]"
                
                print(f"[{i}/{len(files)}] ✓ {result['stem']}{ad_indicator} - "
                      f"{result['utterances']} utterances, {result['mentions']} mentions "
                      f"(avg confidence: {avg_conf:.2f})")
                processed += 1
                
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCompleted! Processed {processed}/{len(files)} files successfully.")

if __name__ == "__main__":
    main()