# Voice - Audio to Text Memory Aide

Convert audio recordings to searchable text transcripts using AI-powered transcription and semantic search.

## Overview

This project provides an offline solution for transcribing voice recordings and making them searchable. Perfect for daily voice memos, meeting notes, or any audio content you want to preserve and search later.

### Key Features

- **Offline Processing**: Runs entirely on your local machine (Intel i7 Gen 11 optimized)
- **High Accuracy**: Uses OpenAI's Whisper via faster-whisper (4× faster than original)
- **Semantic Search**: FAISS-powered search finds related concepts, not just keywords
- **Scalable**: Handles years of daily recordings (100,000+ transcripts)
- **Recording Metadata**: Preserves original recording date/time from audio files
- **AI-Searchable**: Structured JSON format enables AI assistants to query transcripts

## Technology Stack

- **Transcription**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Fast, accurate speech-to-text
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) - Semantic search at scale
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) - Offline embedding generation
- **Metadata**: mutagen - Extract recording timestamps from audio files

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Transcribe a file
python -m src.cli recording.mp3

# Search transcripts
python -m src.search "what did I say about the dentist?"

# Watch directory for new uploads
python -m src.watcher /path/to/uploads/ --output-dir ./transcripts/
```

## Performance

**Intel i7 Gen 11 CPU:**
- Transcription: 10 min audio → 2-3 min processing (small model)
- Search: <10ms across 10,000+ recordings
- Memory: ~20MB for 10,000 indexed transcripts

## Documentation

See [CLAUDE.md](CLAUDE.md) for complete development documentation including:
- Architecture and design decisions
- Development commands
- Implementation guidelines
- Performance optimization
- Troubleshooting

## Project Structure

```
voice/
├── src/                 # Source code
│   ├── transcriber.py   # Core transcription engine
│   ├── vectorstore.py   # FAISS vector search
│   ├── search.py        # Search interface
│   └── utils/           # Utilities (metadata, embeddings, formats)
├── transcripts/         # Output directory
│   ├── metadata/        # FAISS index and metadata
│   └── json/            # Transcripts organized by date
├── tests/               # Unit tests
├── CLAUDE.md           # Development guide
└── README.md           # This file
```

## License

MIT (or specify your license)

## Contributing

See [CLAUDE.md](CLAUDE.md) for development workflow and guidelines.
