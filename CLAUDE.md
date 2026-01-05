# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Voice** - An offline audio-to-text transcription tool for creating searchable text from voice recordings as a memory aide.

**Target Hardware:** Intel i7 Gen 11 CPU (optimized for Intel/AMD x86-64 processors)

**Core Technology:** faster-whisper (OpenAI Whisper via CTranslate2 inference engine)

**Workflow:** Audio files are uploaded to a directory daily and must be transcribed with the recording date/time preserved alongside the text.

### Why faster-whisper?

After comparing whisper.cpp, original Whisper, and faster-whisper:
- **30-200% faster** than whisper.cpp on Intel/AMD CPUs (due to oneDNN/MKL optimizations)
- **4× faster** than original OpenAI Whisper with same accuracy
- **50% less memory** usage than original Whisper
- **Easy Python integration** for rapid development
- **Active maintenance** by SYSTRAN
- **INT8 quantization** support for 2× additional speedup

Note: whisper.cpp would be faster on Apple Silicon with CoreML, but this project targets Intel CPUs.

## Project Architecture

```
voice/
├── src/
│   ├── transcriber.py       # Core faster-whisper transcription engine
│   ├── cli.py                # Command-line interface
│   ├── batch.py              # Batch processing for directories
│   ├── watcher.py            # Directory monitoring for new uploads
│   ├── vectorstore.py        # FAISS vector store for semantic search
│   ├── search.py             # Search interface (semantic + metadata)
│   └── utils/
│       ├── audio.py          # Audio format handling/validation
│       ├── metadata.py       # Extract recording date/time from audio
│       ├── embeddings.py     # Generate embeddings for FAISS
│       └── formats.py        # Output formatters (txt, json, srt)
├── transcripts/
│   ├── metadata/
│   │   ├── index.json        # Master metadata index
│   │   ├── faiss_index.bin   # FAISS vector index
│   │   └── id_mapping.json   # Map FAISS IDs to filenames
│   └── json/                 # Organized by date
│       └── YYYY/MM/          # transcript_*.json files
├── tests/
│   └── test_transcriber.py
├── pyproject.toml
├── README.md
└── .gitignore
```

## Development Commands

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_transcriber.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Fix linting issues automatically
ruff check --fix src/ tests/
```

### Usage

```bash
# Transcribe single file (preserves recording date/time in output)
python -m src.cli audio_file.mp3

# Specify model size (tiny/base/small/medium/large-v2)
python -m src.cli audio_file.mp3 --model small

# Use INT8 quantization for 2× speed
python -m src.cli audio_file.mp3 --quantization int8

# Specify output file
python -m src.cli audio_file.mp3 --output transcript.txt

# Batch process directory (processes all audio files)
python -m src.batch /path/to/audio/files/ --output-dir ./transcripts/

# Watch directory for new uploads (daily workflow)
python -m src.watcher /path/to/upload/dir/ --output-dir ./transcripts/ --model small

# Process only new files since last run
python -m src.batch /path/to/audio/files/ --output-dir ./transcripts/ --only-new

# Specify language (auto-detect if not specified)
python -m src.cli audio_file.mp3 --language en

# Semantic search using FAISS (fast even with years of data)
python -m src.search "dentist appointment"

# Search with date filter
python -m src.search "project deadline" --date-from 2026-01-01 --date-to 2026-01-31

# Find similar recordings (semantic similarity)
python -m src.search --similar-to transcript_2026-01-04_143022.json

# Rebuild FAISS index (if corrupted or after bulk import)
python -m src.vectorstore rebuild ./transcripts/

# Check index stats
python -m src.vectorstore stats
```

## Key Implementation Details

### Model Selection

Models are downloaded automatically on first use and cached locally:

- **tiny** (39MB): Fast, lower accuracy - good for quick tests
- **base** (74MB): Balanced - good for simple recordings
- **small** (244MB): **RECOMMENDED** - best accuracy/speed balance
- **medium** (769MB): High accuracy - for important recordings
- **large-v2** (1.5GB): Best accuracy - may be slow on CPU

### Expected Performance (Intel i7 Gen 11)

Using **small** model (recommended):
- 1 hour audio → ~15-20 minutes processing
- 10 min voice memo → ~2-3 minutes

Using **tiny** model (fastest):
- 1 hour audio → ~5-8 minutes
- 10 min voice memo → ~45-90 seconds

With **INT8 quantization**: approximately 2× faster with minimal accuracy loss

### Output Formats

1. **Plain text** (`.txt`): Simple transcription
2. **JSON** (`.json`): Includes timestamps and word-level data for searchability
3. **SRT** (`.srt`): Subtitle format with timecodes

### Audio Format Support

faster-whisper supports common formats through ffmpeg:
- MP3, WAV, M4A, FLAC, OGG, WMA
- Automatically resamples to 16kHz mono (Whisper requirement)

## Core Dependencies

```toml
[project.dependencies]
faster-whisper>=1.0.0         # Core transcription engine
torch>=2.0.0                  # Required by faster-whisper
torchaudio>=2.0.0             # Audio processing
mutagen>=1.47.0               # Audio metadata extraction (recording date/time)
watchdog>=3.0.0               # Directory monitoring for daily uploads
faiss-cpu>=1.7.4              # Fast vector search (CPU-optimized for i7 Gen 11)
sentence-transformers>=2.2.0  # Generate embeddings for semantic search
numpy>=1.24.0                 # Required by FAISS

[project.optional-dependencies.dev]
pytest>=7.0.0             # Testing
black>=23.0.0             # Code formatting
ruff>=0.1.0               # Fast Python linter
```

## Important Patterns

### Recording Date/Time Extraction (CRITICAL)

**Priority order for extracting recording date/time:**

1. **Audio file metadata** (preferred): Extract from ID3, MP4, FLAC tags
   - Use `mutagen` library to read embedded metadata
   - Look for: `recording_date`, `creation_time`, `date`, `TDRC` (ID3)

2. **File creation timestamp**: Use `os.path.getctime()` as fallback
   - More reliable than modification time
   - May not reflect actual recording time if file was copied

3. **Filename parsing**: Extract date/time if filename contains it
   - Common patterns: `recording_2026-01-04_143022.mp3`
   - Parse using regex or dateutil

**Always include recording date/time in output:**
- JSON format: `{"recording_datetime": "2026-01-04T14:30:22", ...}`
- Plain text: First line should be `Recording Date: 2026-01-04 14:30:22`
- SRT: Include as metadata comment

### Error Handling

- Handle corrupted audio files gracefully in batch mode
- Validate audio format before processing
- Provide clear error messages for unsupported formats
- Log failures without stopping batch processing
- Continue processing even if metadata extraction fails (use fallback)

### Memory Management

- Process long audio files in chunks if needed
- Clear model cache between batch operations if memory constrained
- Monitor memory usage for large models (medium/large)

### Configuration

Use environment variables or config file for:
- Default model size
- Input directory for daily uploads
- Output directory
- Default language
- Quantization preference
- Watch mode settings (polling interval)

## Memory Aide Specific Features

### Daily Upload Workflow

**Typical usage pattern:**
1. Audio files uploaded to designated directory daily
2. Watcher monitors directory or batch processor runs on schedule
3. Each file transcribed with recording date/time preserved
4. Transcripts stored with searchable metadata

### File Organization

- **Output filename format**: `transcript_<recording-date>_<recording-time>.txt`
  - Example: `transcript_2026-01-04_143022.txt`
- **Metadata file**: Alongside each transcript, store `.json` with:
  - Original filename
  - Recording date/time (extracted from metadata)
  - Transcription date/time
  - Model used
  - Processing duration
  - Audio duration
  - Language detected

### Recording Date/Time Storage

**JSON output format (recommended for memory aide):**
```json
{
  "recording_datetime": "2026-01-04T14:30:22",
  "original_filename": "voice_memo_001.m4a",
  "transcription_datetime": "2026-01-04T15:45:10",
  "audio_duration_seconds": 180,
  "model": "small",
  "language": "en",
  "text": "Full transcription text...",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "First segment..."
    }
  ]
}
```

**Plain text format:**
```
Recording Date: 2026-01-04 14:30:22
Original File: voice_memo_001.m4a
Duration: 3:00
Language: English
Model: small
---

[Transcription text here...]
```

### AI Searchability (CRITICAL)

**Date/time must be searchable by AI** - Use FAISS vector store + structured JSON for fast semantic search across years of recordings.

**Why FAISS:**
- **Scales to years of data**: Fast even with 10,000+ recordings
- **Semantic understanding**: Finds "dentist" when you search "doctor"
- **Speed**: 5-10ms search time vs 2-5s reading thousands of JSON files
- **CPU-optimized**: Uses AVX2/AVX-512 on Intel i7 Gen 11
- **Memory efficient**: ~20MB for 10,000 recordings

**Architecture (Hybrid Approach):**

1. **FAISS vector index** (`faiss_index.bin`):
   - Stores 384-dimensional embeddings of all transcripts
   - Enables semantic search (meaning-based, not just keywords)
   - Search time: <10ms even with 100,000 recordings

2. **JSON metadata index** (`index.json`):
   - Master list of all transcripts with metadata
   - AI can read directly without FAISS
   - Fast date/duration filtering

3. **Individual JSON files** (`json/YYYY/MM/transcript_*.json`):
   - Full transcripts organized by date
   - Human and AI readable
   - Backup if FAISS index corrupts

**FAISS Index Types by Scale:**

```python
# For < 10,000 recordings (up to ~15 years daily):
# Use IndexFlatL2 (exact search, no training needed)
index = faiss.IndexFlatL2(384)
Search time: 5-10ms

# For 10,000+ recordings (15+ years daily):
# Use IndexIVFFlat (approximate, 95-99% accurate)
index = faiss.IndexIVFFlat(quantizer, 384, nlist=100)
Search time: 1-3ms
```

**Embedding Model:**
- **all-MiniLM-L6-v2** (80MB, runs offline)
- Fast on CPU: ~100 transcripts/second
- Good quality for general text

**Search Workflow:**

```
User query: "What did I say about dentist appointments?"

1. Generate query embedding (all-MiniLM-L6-v2)
2. FAISS finds top 10 most similar transcripts (5-10ms)
3. Filter by date if specified
4. Return results with recording date/time

Finds:
- "dentist appointment" ✓
- "dental checkup" ✓
- "going to tooth doctor" ✓
```

**AI Integration:**

AI assistants can search in two ways:

1. **Direct JSON reading** (slower but simple):
   - Read `index.json` for metadata
   - Filter by date/duration
   - Read matching transcript files

2. **FAISS search** (faster, semantic):
   - Call search API with natural language query
   - Get semantically similar results
   - Results include recording_datetime

**Master Index Format (`metadata/index.json`):**
```json
{
  "total_recordings": 1247,
  "embedding_model": "all-MiniLM-L6-v2",
  "faiss_index_type": "IndexFlatL2",
  "transcripts": [
    {
      "file": "json/2026/01/transcript_2026-01-04_143022.json",
      "recording_datetime": "2026-01-04T14:30:22",
      "duration_seconds": 180,
      "language": "en",
      "model": "small",
      "faiss_id": 42
    }
  ]
}
```

### FAISS Performance (Intel i7 Gen 11)

**Index Building:**
```
1,000 recordings:   ~10 seconds
10,000 recordings:  ~90 seconds
100,000 recordings: ~15 minutes
```

**Search Time:**
```
IndexFlatL2 (exact search):
- 1,000 recordings:   1-2ms
- 10,000 recordings:  5-10ms ✓
- 100,000 recordings: 50-100ms

IndexIVFFlat (approximate, 95-99% accurate):
- 10,000 recordings:  1-3ms
- 100,000 recordings: 3-8ms ✓
```

**Memory Usage:**
```
10,000 recordings: ~20MB (vectors + metadata)
Very lightweight!
```

**When to Switch Index Types:**

- **< 10 years daily recordings**: Use `IndexFlatL2` (exact, no training)
- **10+ years daily recordings**: Use `IndexIVFFlat` (faster, needs training)

### Future Enhancements

Consider implementing:
- Automatic tagging/categorization using embeddings
- Summary generation (using LLM)
- Speaker diarization (via WhisperX integration)
- Web interface for browsing by date with semantic search
- Clustering similar recordings by topic
- Temporal search ("show trend of mentions over time")

## FAISS Implementation Guidelines

### Initial Setup (IndexFlatL2 - Exact Search)

**Start with this for simplicity:**

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384  # all-MiniLM-L6-v2 embedding size
index = faiss.IndexFlatL2(dimension)

# Add transcript
text = "Full transcription text..."
embedding = model.encode([text])[0]
index.add(np.array([embedding]))

# Save index
faiss.write_index(index, "metadata/faiss_index.bin")

# Load index
index = faiss.read_index("metadata/faiss_index.bin")

# Search
query_embedding = model.encode(["dentist appointment"])[0]
distances, ids = index.search(np.array([query_embedding]), k=5)
```

**When to upgrade:** If you have >10,000 recordings and search feels slow

### Advanced Setup (IndexIVFFlat - Approximate Search)

**Switch to this for 10+ years of data:**

```python
# Create index
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)

# IMPORTANT: Must train on sample data first
training_embeddings = get_sample_embeddings(1000)  # Sample of recordings
index.train(training_embeddings)

# Then add all embeddings
index.add(all_embeddings)

# Set search parameters (nprobe = how many clusters to search)
index.nprobe = 10  # Higher = more accurate but slower
```

### Key Implementation Patterns

**1. Incremental Updates:**
```python
# When new recording is transcribed:
# 1. Generate embedding
# 2. Add to FAISS index
# 3. Update id_mapping.json
# 4. Save index periodically (not every time - batch saves)

# Save index every N additions or on shutdown
if num_additions % 100 == 0:
    faiss.write_index(index, "metadata/faiss_index.bin")
```

**2. ID Mapping:**
```python
# FAISS uses sequential integer IDs (0, 1, 2, ...)
# Map to your transcript filenames:

id_mapping = {
    0: "json/2026/01/transcript_2026-01-04_143022.json",
    1: "json/2026/01/transcript_2026-01-05_090015.json",
    # ...
}

# After search:
distances, faiss_ids = index.search(query_embedding, k=5)
filenames = [id_mapping[id] for id in faiss_ids[0]]
```

**3. Date Filtering:**
```python
# FAISS doesn't support metadata filtering
# So: get more results, then filter

# Get 10x results
distances, ids = index.search(query_embedding, k=50)

# Filter by date
filtered_results = []
for dist, faiss_id in zip(distances[0], ids[0]):
    metadata = get_metadata(faiss_id)
    if is_in_date_range(metadata['recording_datetime'], date_range):
        filtered_results.append((dist, faiss_id))
        if len(filtered_results) >= 5:  # Want 5 results
            break
```

### Error Handling

```python
# Handle missing/corrupted index
try:
    index = faiss.read_index("metadata/faiss_index.bin")
except:
    print("Index corrupted, rebuilding from JSON files...")
    index = rebuild_index_from_json()

# Validate index size matches metadata
assert index.ntotal == len(id_mapping), "Index/mapping mismatch!"
```

### Memory Optimization

**For systems with limited RAM:**

```python
# Use IndexIVFPQ (Product Quantization) for larger datasets
# Reduces memory by ~4x with minimal accuracy loss
index = faiss.IndexIVFPQ(
    quantizer,
    dimension,
    nlist=100,    # clusters
    m=8,          # subquantizers
    nbits=8       # bits per subquantizer
)
```

## Performance Optimization

### CPU Optimization

faster-whisper automatically uses:
- **Intel oneDNN** on Intel CPUs (significant speedup)
- **OpenBLAS** on AMD CPUs
- **Multi-threading** for parallel processing

FAISS CPU optimizations:
- **AVX2/AVX-512** vector instructions (automatic on i7 Gen 11)
- **OpenMP** for multi-threading
- **BLAS** integration for matrix operations

### Speedup Options

1. **Use smaller models**: tiny/base for quick notes, small for important recordings
2. **Enable INT8 quantization**: `--quantization int8` for 2× speedup
3. **Adjust beam size**: Lower beam size (1-3) for faster processing
4. **Batch processing**: Process multiple files to amortize model loading time

## Troubleshooting

### Model Download Issues

Models download from HuggingFace on first use. If download fails:
- Check internet connection
- Models cached in `~/.cache/huggingface/`
- Can manually download and place in cache directory

### Performance Issues

If transcription is slower than expected:
- Verify oneDNN/MKL is being used (check faster-whisper logs)
- Try INT8 quantization
- Use smaller model
- Close other CPU-intensive applications

### Accuracy Issues

If transcription quality is poor:
- Use larger model (small → medium → large)
- Specify correct language with `--language`
- Check audio quality (background noise, clarity)
- Ensure audio is clear speech (not music or multiple overlapping speakers)

### FAISS Search Issues

**Search returns irrelevant results:**
- Check if embedding model is loaded correctly
- Verify FAISS index is not corrupted: `python -m src.vectorstore stats`
- Rebuild index if needed: `python -m src.vectorstore rebuild`
- For IndexIVFFlat: increase `nprobe` (more clusters searched)

**"Index is not trained" error:**
- Only happens with IndexIVFFlat
- Must call `index.train(sample_embeddings)` before adding vectors
- Need at least `nlist` vectors for training (e.g., 100 vectors for nlist=100)

**Search is slow:**
- If using IndexFlatL2 with >10,000 recordings, switch to IndexIVFFlat
- Check CPU usage - should use multiple cores
- Verify FAISS is using AVX2: check installation with `import faiss; print(faiss.__version__)`

**Index/mapping mismatch:**
- FAISS index ntotal != id_mapping length
- Rebuild both: `python -m src.vectorstore rebuild`
- Always update id_mapping.json when adding to FAISS

**Out of memory:**
- Use IndexIVFPQ instead of IndexIVFFlat (4× less memory)
- Process embeddings in batches
- Close other applications

## Development Workflow

1. **Adding new features**: Write tests first, implement, ensure tests pass
2. **Modifying transcription logic**: Test with various audio samples
3. **Performance changes**: Benchmark before/after with real audio files
4. **New output formats**: Add formatter in `utils/formats.py`

## Related Resources

**Transcription:**
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper Documentation](https://github.com/openai/whisper)
- [CTranslate2 Documentation](https://opennmt.net/CTranslate2/)

**Vector Search:**
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [all-MiniLM-L6-v2 Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

**Audio Metadata:**
- [Mutagen Documentation](https://mutagen.readthedocs.io/)
