"""
curious_ai.py — The Initiative Gap: MAIL Framework v2
======================================================
Meta-cognitive Autonomous Inquiry Loop (MAIL)

Full pipeline:
    - all-MiniLM-L6-v2          — 384-dim embedding space
    - Phi-4 14B Q4_K_M          — reasoning agent (T4 GPU via Ollama)
    - ChromaDB                  — persistent vector knowledge store
    - MAIL loop                 — gap detection, hypothesis, validation
    - PyBullet                  — physics simulation
    - QTE + PRM                 — philosophical reasoning
    - ExperimentalHypothesisGenerator — researcher-facing output

Colab T4 setup:
    Cell 1:
        !pip install sentence-transformers scikit-learn pybullet requests chromadb -q
        !curl -fsSL https://ollama.com/install.sh | sh

    Cell 2:
        import subprocess, threading, time
        threading.Thread(target=lambda: subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )).start()
        time.sleep(5)
        !ollama pull phi4:14b-q4_K_M

    Cell 3:
        exec(open('curious_ai.py').read())

Author: Harshal Adari
Date:   2026
Repo:   https://github.com/harshalworkai64254-bit/initiative_gap
"""

import numpy as np
import json
import time
import requests
import os
import uuid
from typing import Optional
from datetime import datetime

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed.")

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not installed.")

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[WARNING] PyBullet not installed.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[WARNING] ChromaDB not installed. Run: pip install chromadb")

try:
    from google.colab import drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False


# ==============================================================================
# SECTION 1 — OLLAMA CLIENT
# ==============================================================================

class OllamaClient:
    def __init__(self, model: str = "phi4:14b-q4_K_M",
                 base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models    = [m['name'] for m in r.json().get('models', [])]
                available = any(self.model in m for m in models)
                if available:
                    print(f"[OLLAMA] {self.model} ready.")
                else:
                    print(f"[OLLAMA] {self.model} not found. "
                          f"Run: ollama pull {self.model}")
                return available
        except Exception:
            print("[OLLAMA] Server not running. Run: ollama serve")
            return False
        return False

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 800) -> str:
        if not self.available:
            return f"[OLLAMA UNAVAILABLE]"
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "system":  system,
            "stream":  False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        }
        try:
            r = requests.post(f"{self.base_url}/api/generate",
                              json=payload, timeout=240)
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            return f"[OLLAMA ERROR] {r.status_code}"
        except Exception as e:
            return f"[OLLAMA ERROR] {e}"


# ==============================================================================
# SECTION 2 — INITIAL KNOWLEDGE BASE
# ==============================================================================

INITIAL_KNOWLEDGE = [
    # Physics
    "objects fall due to gravity",
    "water flows downhill",
    "fire requires oxygen to burn",
    "ice melts above zero degrees celsius",
    "sound travels through air as pressure waves",
    "light travels faster than sound",
    "metals conduct electricity",
    "magnets attract iron",
    "friction causes heat",
    "pressure causes liquids to flow",
    "temperature affects the speed of chemical reactions",
    "force causes acceleration",
    "heat causes expansion in most materials",
    # Biology
    "plants convert sunlight into energy",
    "animals need food and water to survive",
    "cells are the basic unit of life",
    "DNA carries genetic information",
    "evolution occurs through natural selection",
    "the heart pumps blood through the body",
    "the brain processes sensory information",
    "mutations in DNA can be passed to offspring",
    "the immune system defends against pathogens",
    "proteins are built from amino acid sequences",
    # Chemistry
    "atoms bond to form molecules",
    "chemical reactions conserve mass",
    "acids and bases neutralise each other",
    "catalysts speed up reactions without being consumed",
    "entropy tends to increase in isolated systems",
    # History / social
    "the assassination of Franz Ferdinand triggered World War One",
    "economic inequality can lead to social unrest",
    "technological change alters the structure of society",
    "colonialism had long lasting effects on colonised nations",
    # Mathematics
    "prime numbers have no factors other than one and themselves",
    "the sum of angles in a triangle is one hundred and eighty degrees",
    "correlation does not imply causation",
    "exponential growth accelerates over time",
    "probability quantifies uncertainty",
    # Philosophy / contested
    "consciousness arises from physical brain processes",
    "free will may be compatible with determinism",
    "moral judgements can be objective or subjective",
    "the existence of God is contested",
    "the nature of time is debated in physics",
    # Neuroscience / cognition
    "the brain forms memories through synaptic strengthening",
    "sleep is essential for memory consolidation",
    "stress hormones affect cognitive performance",
    "reward circuits in the brain reinforce repeated behaviours",
]


# ==============================================================================
# SECTION 3 — EMBEDDING MODEL
# ==============================================================================

class EmbeddingModel:
    def __init__(self):
        if not ST_AVAILABLE:
            print("[EMBEDDING] Using random vectors (install sentence-transformers).")
            self.model = None
            self.dim   = 384
        else:
            print("[EMBEDDING] Loading all-MiniLM-L6-v2...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dim   = 384
            print("[EMBEDDING] Ready.")

    def encode(self, texts: list) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(texts), self.dim).astype(np.float32)
        return self.model.encode(texts, convert_to_numpy=True,
                                  show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


# ==============================================================================
# SECTION 4 — VECTOR DATABASE KNOWLEDGE STORE
# ==============================================================================
#
# Replaces the simple text list from v1.
#
# Why ChromaDB:
#   - Every confirmed fact is stored as an embedding + metadata
#   - Semantic search: find relevant knowledge by meaning not keyword
#   - Persistent across Colab sessions (save to Google Drive)
#   - Scales to millions of facts without slowing down
#
# Two collections:
#   'empirical_facts'   — confirmed by human oracle or simulation
#   'belief_states'     — philosophical/contested, stored with confidence
#   'experimental_results' — results fed back from real lab experiments
# ==============================================================================

class VectorKnowledgeStore:
    """
    ChromaDB-backed persistent knowledge store.
    Replaces the in-memory text list from v1.
    """

    def __init__(self, embedding_model: EmbeddingModel,
                 persist_path: str = "./mail_knowledge_db"):
        self.embedding_model = embedding_model
        self.persist_path    = persist_path

        if not CHROMA_AVAILABLE:
            print("[KNOWLEDGE STORE] ChromaDB unavailable. "
                  "Falling back to in-memory list.")
            self.client        = None
            self._fallback_texts   = []
            self._fallback_vectors = None
            return

        self.client = chromadb.PersistentClient(path=persist_path)

        # Three collections
        self.empirical   = self.client.get_or_create_collection(
            "empirical_facts",
            metadata={"description": "Confirmed empirical knowledge"})
        self.beliefs     = self.client.get_or_create_collection(
            "belief_states",
            metadata={"description": "Philosophical/contested positions"})
        self.experiments = self.client.get_or_create_collection(
            "experimental_results",
            metadata={"description": "Results from real-world experiments"})

        print(f"[KNOWLEDGE STORE] ChromaDB ready at {persist_path}")
        print(f"  Empirical facts:       {self.empirical.count()}")
        print(f"  Belief states:         {self.beliefs.count()}")
        print(f"  Experimental results:  {self.experiments.count()}")

    def _safe_id(self, text: str) -> str:
        """Generate a safe ChromaDB ID from text."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text[:200]))

    def add_empirical(self, text: str, metadata: dict = None) -> bool:
        """Add a confirmed empirical fact."""
        if not text.strip() or len(text) > 300:
            return False

        if self.client is None:
            if text not in self._fallback_texts:
                self._fallback_texts.append(text)
                self._fallback_vectors = None
            return True

        doc_id = self._safe_id(text)
        meta   = metadata or {}
        meta.update({
            'timestamp':  datetime.now().isoformat(),
            'type':       'empirical',
            'text':       text[:200]
        })
        embedding = self.embedding_model.encode_single(text).tolist()
        try:
            self.empirical.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta]
            )
            return True
        except Exception as e:
            print(f"  [STORE ERROR] {e}")
            return False

    def add_belief(self, text: str, framework: str,
                   confidence: float, metadata: dict = None) -> bool:
        """Add a philosophical belief state with confidence."""
        if self.client is None:
            return self.add_empirical(text)

        doc_id = self._safe_id(text)
        meta   = metadata or {}
        meta.update({
            'timestamp':  datetime.now().isoformat(),
            'type':       'belief',
            'framework':  framework,
            'confidence': confidence,
            'text':       text[:200]
        })
        embedding = self.embedding_model.encode_single(text).tolist()
        try:
            self.beliefs.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta]
            )
            return True
        except Exception as e:
            print(f"  [STORE ERROR] {e}")
            return False

    def add_experimental_result(self, hypothesis: str, result: str,
                                 confirmed: bool, experiment_type: str,
                                 researcher_notes: str = "") -> bool:
        """Add a result fed back from a real-world experiment."""
        if self.client is None:
            return self.add_empirical(f"[EXPERIMENT] {hypothesis}: {result}")

        text   = f"Experiment: {hypothesis} | Result: {result}"
        doc_id = self._safe_id(text)
        meta   = {
            'timestamp':        datetime.now().isoformat(),
            'type':             'experimental',
            'hypothesis':       hypothesis[:200],
            'result':           result[:200],
            'confirmed':        confirmed,
            'experiment_type':  experiment_type,
            'researcher_notes': researcher_notes[:300],
        }
        embedding = self.embedding_model.encode_single(text).tolist()
        try:
            self.experiments.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[meta]
            )
            print(f"  [STORE] Experimental result saved: "
                  f"{'✓ confirmed' if confirmed else '✗ denied'}")
            return True
        except Exception as e:
            print(f"  [STORE ERROR] {e}")
            return False

    def get_all_texts(self) -> list:
        """Get all knowledge as text list (for embedding space rebuild)."""
        if self.client is None:
            return self._fallback_texts.copy()
        texts = []
        for collection in [self.empirical, self.beliefs, self.experiments]:
            if collection.count() > 0:
                results = collection.get(include=['documents'])
                texts.extend(results['documents'])
        return texts

    def get_all_vectors(self, embedding_model) -> tuple:
        """Get all knowledge as (texts, vectors) for MAIL operations."""
        texts = self.get_all_texts()
        if not texts:
            return [], np.array([])
        vectors = embedding_model.encode(texts)
        return texts, vectors

    def semantic_search(self, query: str, n: int = 5,
                        collection: str = 'empirical') -> list:
        """Find most semantically similar stored knowledge."""
        if self.client is None:
            return []
        col = getattr(self, collection, self.empirical)
        if col.count() == 0:
            return []
        embedding = self.embedding_model.encode_single(query).tolist()
        results   = col.query(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
            include=['documents', 'metadatas', 'distances']
        )
        out = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            out.append({'text': doc, 'metadata': meta,
                        'distance': dist})
        return out

    def count(self) -> dict:
        if self.client is None:
            return {'empirical': len(self._fallback_texts),
                    'beliefs': 0, 'experiments': 0}
        return {
            'empirical':   self.empirical.count(),
            'beliefs':     self.beliefs.count(),
            'experiments': self.experiments.count(),
        }

    def save_to_drive(self, drive_path: str = "/content/drive/MyDrive/mail_db"):
        """Save ChromaDB to Google Drive for persistence across sessions."""
        if not COLAB_AVAILABLE:
            print("[SAVE] Not in Colab — DB already persisted locally.")
            return
        try:
            drive.mount('/content/drive')
            import shutil
            if os.path.exists(drive_path):
                shutil.rmtree(drive_path)
            shutil.copytree(self.persist_path, drive_path)
            print(f"[SAVE] Knowledge saved to Google Drive: {drive_path}")
        except Exception as e:
            print(f"[SAVE ERROR] {e}")

    def load_from_drive(self, drive_path: str = "/content/drive/MyDrive/mail_db"):
        """Load ChromaDB from Google Drive."""
        if not COLAB_AVAILABLE:
            return
        try:
            drive.mount('/content/drive')
            import shutil
            if os.path.exists(drive_path):
                if os.path.exists(self.persist_path):
                    shutil.rmtree(self.persist_path)
                shutil.copytree(drive_path, self.persist_path)
                print(f"[LOAD] Knowledge loaded from Google Drive.")
            else:
                print("[LOAD] No saved DB found on Drive. Starting fresh.")
        except Exception as e:
            print(f"[LOAD ERROR] {e}")


# ==============================================================================
# SECTION 5 — EXPERIMENTAL HYPOTHESIS GENERATOR
# ==============================================================================
#
# This is the key new component.
#
# Old Stage 3 output:
#   "Is this concept related to X but distinct from Y?"  (yes/no question)
#
# New Stage 3 output (researcher-facing):
#   Hypothesis:          specific falsifiable claim
#   Predicted outcome:   what you'd observe if true
#   Null hypothesis:     what you'd observe if false
#   Suggested method:    concrete experimental approach
#   Required equipment:  what a lab would need
#   Expected timescale:  rough estimate
#   Confidence:          how strongly the vector math supports this
#   Related knowledge:   what we already know that's relevant
#
# The structure is deterministic (from vector math + probe fingerprint).
# Phi-4 fills in the natural language.
# ==============================================================================

# Domain-specific experiment templates
# Keyed by which probe axes score highest in the fingerprint
EXPERIMENT_TEMPLATES = {
    'biology': {
        'methods': [
            "cell culture assay", "PCR analysis", "protein gel electrophoresis",
            "microscopy imaging", "RNA sequencing", "animal model study",
            "controlled growth experiment", "genetic knockout study"
        ],
        'equipment': [
            "PCR machine", "microscope", "centrifuge",
            "cell culture hood", "spectrophotometer"
        ],
        'timescale': "weeks to months"
    },
    'physics': {
        'methods': [
            "controlled measurement with calibrated instruments",
            "particle collision experiment", "spectroscopy analysis",
            "computational fluid dynamics simulation",
            "materials stress test", "electromagnetic field measurement"
        ],
        'equipment': [
            "spectrometer", "oscilloscope", "vacuum chamber",
            "high-precision scales", "laser interferometer"
        ],
        'timescale': "days to weeks"
    },
    'chemistry': {
        'methods': [
            "titration experiment", "chromatography separation",
            "mass spectrometry", "NMR spectroscopy",
            "controlled reaction under varying conditions",
            "crystallisation study"
        ],
        'equipment': [
            "mass spectrometer", "NMR machine", "HPLC system",
            "fume hood", "analytical balance"
        ],
        'timescale': "days to weeks"
    },
    'neuroscience': {
        'methods': [
            "fMRI brain imaging study", "EEG measurement",
            "behavioural experiment with control group",
            "patch clamp electrophysiology",
            "optogenetics study", "longitudinal cognitive assessment"
        ],
        'equipment': [
            "fMRI scanner", "EEG headset", "eye tracker",
            "reaction time apparatus", "electrode array"
        ],
        'timescale': "months"
    },
    'general': {
        'methods': [
            "randomised controlled experiment",
            "systematic observation with controls",
            "meta-analysis of existing literature",
            "computational modelling study",
            "longitudinal observational study"
        ],
        'equipment': [
            "standard lab equipment",
            "data collection apparatus",
            "statistical analysis software"
        ],
        'timescale': "weeks to months"
    }
}


class ExperimentalHypothesisGenerator:
    """
    Converts MAIL's vector-derived hypothesis into a researcher-actionable
    experimental brief.

    Uses probe fingerprint to infer domain → select appropriate methods.
    Uses Phi-4 to write the actual hypothesis text and experimental design.
    All structural decisions (domain, method type, confidence) are deterministic.
    """

    def __init__(self, ollama: OllamaClient,
                 embedding_model: EmbeddingModel,
                 knowledge_store: VectorKnowledgeStore):
        self.ollama          = ollama
        self.embedding_model = embedding_model
        self.store           = knowledge_store

    def infer_domain(self, fingerprint: dict,
                     nearest_concepts: list) -> str:
        """
        Infer research domain from probe fingerprint and nearest concepts.
        Used to select appropriate experimental methods.
        """
        concept_str = ' '.join(nearest_concepts).lower()

        # Keyword-based domain detection from nearest concepts
        if any(w in concept_str for w in [
            'cell', 'dna', 'protein', 'gene', 'evolution',
            'brain', 'neuron', 'immune', 'organism', 'biological'
        ]):
            if any(w in concept_str for w in [
                'brain', 'neuron', 'memory', 'cognitive', 'sleep'
            ]):
                return 'neuroscience'
            return 'biology'

        if any(w in concept_str for w in [
            'force', 'gravity', 'energy', 'wave', 'particle',
            'magnetic', 'electric', 'quantum', 'velocity', 'pressure'
        ]):
            return 'physics'

        if any(w in concept_str for w in [
            'acid', 'molecule', 'reaction', 'catalyst', 'bond',
            'entropy', 'atom', 'compound', 'chemical'
        ]):
            return 'chemistry'

        return 'general'

    def compute_confidence(self, hypothesis: np.ndarray,
                           gap_vector: np.ndarray,
                           similarities: np.ndarray) -> float:
        """
        Confidence score for the hypothesis.
        Based on:
        - cosine similarity between gap and hypothesis (how well does H fit the gap?)
        - spread of top similarities (narrow = more focused = higher confidence)
        """
        def cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        gap_alignment = max(0.0, cos(gap_vector, hypothesis))
        top_sims      = np.sort(similarities)[-10:]
        focus_score   = float(np.mean(top_sims))
        confidence    = (gap_alignment * 0.6 + focus_score * 0.4)
        return round(min(1.0, max(0.0, confidence)), 3)

    def generate(self, hypothesis: np.ndarray,
                 gap_vector: np.ndarray,
                 similarities: np.ndarray,
                 nearest_concepts: list,
                 fingerprint: dict) -> dict:
        """
        Generate a full experimental brief from the hypothesis vector.
        Returns structured dict + Phi-4 natural language.
        """
        import random

        domain     = self.infer_domain(fingerprint, nearest_concepts)
        confidence = self.compute_confidence(hypothesis, gap_vector, similarities)
        template   = EXPERIMENT_TEMPLATES.get(domain,
                                               EXPERIMENT_TEMPLATES['general'])

        # Pick methods appropriate to domain
        methods  = random.sample(template['methods'],
                                  min(3, len(template['methods'])))
        equip    = template['equipment']
        timescale = template['timescale']

        # Retrieve related knowledge from store
        related_query = ' '.join(nearest_concepts[:3])
        related       = self.store.semantic_search(
            related_query, n=4, collection='empirical')
        related_texts = [r['text'] for r in related]

        # Use Phi-4 to generate the hypothesis text and experimental design
        system = (
            "You are a scientific hypothesis generator. "
            "Given information about a gap in knowledge, generate a precise, "
            "falsifiable scientific hypothesis and experimental design. "
            "Be specific. Use concrete measurements. "
            "Do not be vague. Output exactly the JSON structure requested."
        )

        prompt = f"""
A knowledge gap has been detected in the following region:
Nearest known concepts: {', '.join(nearest_concepts[:5])}
Research domain: {domain}
Confidence score: {confidence:.3f}
Related known facts: {'; '.join(related_texts[:3]) if related_texts else 'none yet'}

Generate a research brief with exactly this JSON structure (no other text):
{{
  "hypothesis": "one specific falsifiable claim about what might be true in this gap",
  "rationale": "one sentence explaining why this gap exists based on nearby concepts",
  "predicted_outcome": "specific observable measurement if hypothesis is true",
  "null_outcome": "specific observable measurement if hypothesis is false",
  "suggested_method": "concrete experimental approach for a lab to test this",
  "key_variables": ["independent variable", "dependent variable", "control variable"],
  "predicted_mechanism": "brief explanation of underlying mechanism if true",
  "implications": "what confirming this would mean for the field"
}}

Output only valid JSON. No preamble. No explanation outside the JSON.
"""

        raw = self.ollama.generate(prompt, system=system, max_tokens=600)

        # Parse JSON from Phi-4 response
        brief = {}
        try:
            # Strip markdown fences if present
            clean = raw.replace('```json', '').replace('```', '').strip()
            brief = json.loads(clean)
        except Exception:
            # Fallback if JSON parsing fails
            brief = {
                "hypothesis":          f"A previously unknown relationship exists between {nearest_concepts[0]} and {nearest_concepts[1]}",
                "rationale":           f"The embedding space shows a gap between {domain} concepts",
                "predicted_outcome":   "Measurable change in dependent variable under experimental conditions",
                "null_outcome":        "No significant change under experimental conditions",
                "suggested_method":    methods[0],
                "key_variables":       ["treatment", "measured outcome", "baseline"],
                "predicted_mechanism": "Unknown — requires empirical investigation",
                "implications":        f"Would extend understanding in {domain}"
            }

        # Add structural metadata (deterministic, not from LLM)
        brief.update({
            'domain':           domain,
            'confidence':       confidence,
            'suggested_methods': methods,
            'equipment':        equip,
            'timescale':        timescale,
            'nearest_concepts': nearest_concepts,
            'related_knowledge': related_texts,
            'generated_at':     datetime.now().isoformat(),
        })

        return brief


# ==============================================================================
# SECTION 6 — RESEARCHER INTERFACE
# ==============================================================================
#
# Formats the experimental brief into something a real researcher can read,
# act on, and feed results back into.
# ==============================================================================

class ResearcherInterface:
    """
    Human-facing interface for the experimental hypothesis pipeline.

    Two modes:
    1. Display mode  — show the brief to a researcher
    2. Feedback mode — researcher feeds experimental results back into MAIL
    """

    def display_brief(self, brief: dict):
        """Print a formatted experimental brief."""
        print(f"\n{'═'*70}")
        print(f"  EXPERIMENTAL BRIEF — MAIL Research Accelerator")
        print(f"  Generated: {brief.get('generated_at', 'now')}")
        print(f"{'═'*70}")

        print(f"\n  DOMAIN:      {brief.get('domain', 'unknown').upper()}")
        print(f"  CONFIDENCE:  {brief.get('confidence', 0):.3f} / 1.000")
        print(f"  TIMESCALE:   {brief.get('timescale', 'unknown')}")

        print(f"\n  ── HYPOTHESIS ──────────────────────────────────────────")
        print(f"  {brief.get('hypothesis', 'N/A')}")

        print(f"\n  ── RATIONALE ───────────────────────────────────────────")
        print(f"  {brief.get('rationale', 'N/A')}")

        print(f"\n  ── PREDICTED OUTCOME (if true) ─────────────────────────")
        print(f"  {brief.get('predicted_outcome', 'N/A')}")

        print(f"\n  ── NULL OUTCOME (if false) ──────────────────────────────")
        print(f"  {brief.get('null_outcome', 'N/A')}")

        print(f"\n  ── SUGGESTED EXPERIMENTAL METHOD ───────────────────────")
        print(f"  {brief.get('suggested_method', 'N/A')}")

        kv = brief.get('key_variables', [])
        if kv:
            print(f"\n  ── KEY VARIABLES ───────────────────────────────────────")
            labels = ['Independent', 'Dependent', 'Control']
            for i, v in enumerate(kv[:3]):
                print(f"  {labels[i]:<12}: {v}")

        methods = brief.get('suggested_methods', [])
        if methods:
            print(f"\n  ── ALTERNATIVE METHODS ─────────────────────────────────")
            for m in methods:
                print(f"  • {m}")

        equip = brief.get('equipment', [])
        if equip:
            print(f"\n  ── EQUIPMENT ───────────────────────────────────────────")
            print(f"  {', '.join(equip)}")

        print(f"\n  ── PREDICTED MECHANISM ─────────────────────────────────")
        print(f"  {brief.get('predicted_mechanism', 'N/A')}")

        print(f"\n  ── IMPLICATIONS IF CONFIRMED ───────────────────────────")
        print(f"  {brief.get('implications', 'N/A')}")

        related = brief.get('related_knowledge', [])
        if related:
            print(f"\n  ── RELATED KNOWLEDGE IN DATABASE ───────────────────────")
            for r in related[:3]:
                print(f"  • {r}")

        print(f"\n{'═'*70}")

    def collect_feedback(self, brief: dict,
                          store: VectorKnowledgeStore) -> Optional[dict]:
        """
        Ask researcher to feed experimental results back in.
        Stores result in ChromaDB experimental_results collection.
        """
        print(f"\n[RESEARCHER FEEDBACK]")
        print(f"  Has this hypothesis been tested? (yes / no / skip)")
        ans = input("  Answer: ").strip().lower()

        if ans == 'skip' or ans not in ('yes', 'y', 'no', 'n'):
            return None

        confirmed = ans in ('yes', 'y')

        print(f"\n  Describe the experimental result in one sentence:")
        result_text = input("  Result: ").strip()
        if not result_text:
            return None

        print(f"  Any notes on method or conditions? (press Enter to skip)")
        notes = input("  Notes: ").strip()

        print(f"  Experiment type (e.g. 'in vitro', 'computational', "
              f"'observational', 'RCT'):")
        exp_type = input("  Type: ").strip() or "unspecified"

        store.add_experimental_result(
            hypothesis    = brief.get('hypothesis', ''),
            result        = result_text,
            confirmed     = confirmed,
            experiment_type = exp_type,
            researcher_notes = notes
        )

        feedback = {
            'hypothesis': brief.get('hypothesis', ''),
            'result':     result_text,
            'confirmed':  confirmed,
            'exp_type':   exp_type,
            'notes':      notes,
        }

        print(f"\n  ✓ Result stored in knowledge database.")
        print(f"  Knowledge base now has "
              f"{store.count()['experiments']} experimental results.")

        return feedback

    def export_brief_json(self, brief: dict,
                           path: str = "mail_brief.json"):
        """Export brief as JSON for sharing with collaborators."""
        with open(path, 'w') as f:
            # Convert numpy types to Python types for JSON serialisation
            clean = {}
            for k, v in brief.items():
                if isinstance(v, np.floating):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            json.dump(clean, f, indent=2)
        print(f"  [EXPORT] Brief saved to {path}")
        return path


# ==============================================================================
# SECTION 7 — SEMANTIC PROBE SYSTEM (unchanged from v1)
# ==============================================================================

SEMANTIC_PROBES = {
    "physical_metaphysical":       ("a rock falling to the ground",
                                     "the soul of a person after death"),
    "observable_unobservable":     ("measuring temperature with a thermometer",
                                     "the feeling of consciousness from inside"),
    "falsifiable_unfalsifiable":   ("evolution can be tested with fossil records",
                                     "God exists outside the physical universe"),
    "objective_subjective":        ("the boiling point of water is 100 degrees",
                                     "this painting is beautiful"),
    "deterministic_probabilistic": ("a ball rolling off a table will fall",
                                     "a radioactive atom may or may not decay"),
    "empirical_normative":         ("humans evolved from earlier primates",
                                     "humans ought to treat each other fairly"),
    "simple_paradoxical":          ("the sky is blue during daytime",
                                     "this statement is false"),
    "consensus_contested":         ("the earth orbits the sun",
                                     "whether free will exists is unresolved"),
}

PROBE_THRESHOLDS = {
    "unfalsifiable": 0.3,
    "paradoxical":   0.3,
    "metaphysical":  0.25,
    "normative":     0.25,
}


class SemanticProbeSystem:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.probe_axes      = {}
        self._build_axes()

    def _build_axes(self):
        print("[PROBES] Building semantic axes...")
        for name, (neg, pos) in SEMANTIC_PROBES.items():
            v_neg  = self.embedding_model.encode_single(neg)
            v_pos  = self.embedding_model.encode_single(pos)
            axis   = v_pos - v_neg
            norm   = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
            self.probe_axes[name] = {
                'axis': axis, 'neg_concept': neg, 'pos_concept': pos}
        print(f"[PROBES] {len(self.probe_axes)} axes ready.")

    def fingerprint(self, hypothesis: np.ndarray) -> dict:
        h = hypothesis / (np.linalg.norm(hypothesis) + 1e-8)
        return {
            name: float(np.dot(h, probe['axis']))
            for name, probe in self.probe_axes.items()
        }

    def classify_reasoning_mode(self, fingerprint: dict) -> str:
        phys_meta   = fingerprint.get("physical_metaphysical", 0)
        falsifiable = fingerprint.get("falsifiable_unfalsifiable", 0)
        paradox     = fingerprint.get("simple_paradoxical", 0)
        normative   = fingerprint.get("empirical_normative", 0)
        unfalsifiable_score = -falsifiable

        if paradox > PROBE_THRESHOLDS["paradoxical"]:
            return 'paradox'
        if (unfalsifiable_score > PROBE_THRESHOLDS["unfalsifiable"] or
            phys_meta           > PROBE_THRESHOLDS["metaphysical"] or
            normative           > PROBE_THRESHOLDS["normative"]):
            return 'philosophical'
        return 'empirical'

    def explain_fingerprint(self, fingerprint: dict):
        print(f"\n[SEMANTIC FINGERPRINT]")
        for name, score in fingerprint.items():
            probe   = self.probe_axes[name]
            bar     = "█" * int(abs(score) * 20)
            pole    = probe['pos_concept'][:35] if score > 0 \
                      else probe['neg_concept'][:35]
            arrow   = "→" if score > 0 else "←"
            print(f"  {name:<32} {score:+.3f}  {bar:<20}  {arrow} {pole}")


# ==============================================================================
# SECTION 8 — QUESTION TRANSFORMATION ENGINE (unchanged from v1)
# ==============================================================================

PHILOSOPHICAL_FRAMEWORKS = {
    "classical_theism":   "God exists as a personal creator of the universe",
    "atheism":            "no gods or supernatural beings exist",
    "agnosticism":        "the existence of God cannot be known",
    "pantheism":          "God and the universe are identical",
    "pragmatism":         "truth is what works in practice",
    "empiricism":         "all knowledge comes from sensory experience",
    "rationalism":        "reason is the primary source of knowledge",
    "existentialism":     "existence precedes essence and meaning is created",
    "determinism":        "all events are caused by prior events",
    "compatibilism":      "free will is compatible with determinism",
    "moral_realism":      "moral facts exist independently of human opinion",
    "moral_relativism":   "morality is relative to culture and context",
    "physicalism":        "everything that exists is physical",
    "dualism":            "mind and body are fundamentally different substances",
    "buddhism":           "suffering arises from attachment and can be ended",
    "stoicism":           "virtue is the only true good and reason guides action",
    "utilitarianism":     "actions are right if they maximise overall happiness",
    "deontology":         "some actions are inherently right or wrong regardless of outcome",
}


class QuestionTransformationEngine:
    def __init__(self, embedding_model, probe_system, ollama):
        self.embedding_model = embedding_model
        self.probes          = probe_system
        self.ollama          = ollama
        print("[QTE] Pre-embedding philosophical frameworks...")
        fw_texts   = list(PHILOSOPHICAL_FRAMEWORKS.values())
        fw_names   = list(PHILOSOPHICAL_FRAMEWORKS.keys())
        fw_vecs    = self.embedding_model.encode(fw_texts)
        self.frameworks = {
            name: {'text': text, 'vector': vec}
            for name, text, vec in zip(fw_names, fw_texts, fw_vecs)
        }
        print(f"[QTE] {len(self.frameworks)} frameworks ready.")

    def _cos(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def decompose(self, hypothesis, vectors, labels, n=4):
        sims  = np.array([self._cos(hypothesis, v) for v in vectors])
        top_n = np.argsort(sims)[-n:][::-1]
        return [(labels[i], float(sims[i])) for i in top_n]

    def find_assumption(self, fingerprint):
        axis_name, score = max(fingerprint.items(), key=lambda x: abs(x[1]))
        probe = self.probes.probe_axes[axis_name]
        assumption = probe['pos_concept'] if score > 0 else probe['neg_concept']
        challenge  = probe['neg_concept'] if score > 0 else probe['pos_concept']
        return axis_name, assumption, challenge, score

    def triangulate(self, hypothesis, top_n=6):
        results = sorted(
            [(n, fw['text'], self._cos(hypothesis, fw['vector']))
             for n, fw in self.frameworks.items()],
            key=lambda x: x[2], reverse=True
        )
        top     = results[:top_n]
        std     = float(np.std([r[2] for r in top]))
        return top, std

    def find_empirical_proxy(self, hypothesis, vectors, labels, fingerprint):
        axis        = self.probes.probe_axes["empirical_normative"]['axis']
        scores      = []
        for v in vectors:
            vn   = v / (np.linalg.norm(v) + 1e-8)
            emp  = float(np.dot(vn, -axis)) + 1.0
            sim  = self._cos(hypothesis, v)
            scores.append(sim * emp)
        best    = int(np.argmax(scores))
        dist    = 1.0 - self._cos(hypothesis, vectors[best])
        return labels[best], vectors[best], dist

    def extract_need(self, hypothesis, vectors, labels):
        sims       = np.array([self._cos(hypothesis, v) for v in vectors])
        generality = np.array([1.0 / (len(l.split()) + 1) for l in labels])
        top_n      = np.argsort(sims * generality)[-3:][::-1]
        return [(labels[i], float(sims[i])) for i in top_n]

    def transform(self, hypothesis, vectors, labels, fingerprint,
                  original_question=""):
        print(f"\n[QTE] Running transformations...")
        sub_qs      = self.decompose(hypothesis, vectors, labels)
        assumption  = self.find_assumption(fingerprint)
        frameworks, conv = self.triangulate(hypothesis)
        proxy, pv, pdist = self.find_empirical_proxy(
            hypothesis, vectors, labels, fingerprint)
        needs       = self.extract_need(hypothesis, vectors, labels)
        print(f"  [1] Decomposed | [2] Assumption found | "
              f"[3] Triangulated | [4] Proxy found | [5] Need extracted")
        return {
            'sub_questions': sub_qs,
            'assumption':    assumption,
            'frameworks':    frameworks,
            'convergence':   conv,
            'proxy':         (proxy, pdist),
            'needs':         needs,
        }


# ==============================================================================
# SECTION 9 — PHILOSOPHICAL REASONING MODE (unchanged from v1)
# ==============================================================================

class PhilosophicalReasoningMode:
    def __init__(self, ollama, probe_system):
        self.ollama  = ollama
        self.probes  = probe_system

    def compute_framework_distribution(self, frameworks):
        names  = [f[0] for f in frameworks]
        scores = np.array([f[2] for f in frameworks])
        exp_s  = np.exp(scores - scores.max())
        probs  = exp_s / exp_s.sum()
        return {n: float(p) for n, p in zip(names, probs)}

    def question_wellformedness(self, fingerprint):
        return max(0.0, 1.0 - fingerprint.get("simple_paradoxical", 0) * 2)

    def reason(self, qte_output, fingerprint, reasoning_mode,
               original_question, hypothesis):
        print(f"\n[PRM] Mode: {reasoning_mode}")
        fw_dist     = self.compute_framework_distribution(qte_output['frameworks'])
        wellformed  = self.question_wellformedness(fingerprint)
        top_fw      = sorted(fw_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        proxy, pdist = qte_output['proxy']
        axis, assumption, challenge, score = qte_output['assumption']
        conv        = qte_output['convergence']

        system = (
            "You are a rigorous philosophical reasoning engine. "
            "Synthesise the given analysis into a clear, honest 3-4 paragraph response. "
            "Do not claim to answer the unanswerable. Be specific about what IS tractable."
        )
        prompt = f"""
Question: "{original_question}"
Well-formedness: {wellformed:.2f}
Load-bearing assumption: '{assumption}' — challenging: '{challenge}'
Top frameworks: {', '.join([f'{n}:{p:.3f}' for n,p in top_fw])}
Framework convergence std: {conv:.4f} ({'agree' if conv < 0.05 else 'diverge'})
Nearest empirical proxy: '{proxy}' (distance {pdist:.3f})
Underlying needs: {', '.join([n[0] for n in qte_output['needs']])}

Write a rigorous philosophical response covering:
1. What the question is really asking
2. The load-bearing assumption and what changes when you challenge it
3. Where frameworks converge (if anywhere)
4. The most tractable version of the question
"""
        response = self.ollama.generate(prompt, system=system, max_tokens=600)
        return {
            'reasoning_mode':   reasoning_mode,
            'wellformedness':   wellformed,
            'framework_dist':   fw_dist,
            'top_frameworks':   top_fw,
            'frameworks_agree': conv < 0.05,
            'assumption':       assumption,
            'challenge':        challenge,
            'empirical_proxy':  proxy,
            'proxy_distance':   pdist,
            'underlying_needs': qte_output['needs'],
            'phi4_response':    response,
        }

    def display_result(self, result):
        print(f"\n{'═'*70}")
        print(f"  PHILOSOPHICAL REASONING MODE")
        print(f"{'═'*70}")
        print(f"\n  Well-formedness: {result['wellformedness']:.2f}/1.00")
        print(f"\n  Framework distribution:")
        for name, prob in result['top_frameworks']:
            bar = "█" * int(prob * 40)
            print(f"    {name:<25} {prob:.3f}  {bar}")
        print(f"\n  Load-bearing assumption: '{result['assumption']}'")
        print(f"  Transformed question:    '{result['challenge']}'")
        print(f"  Nearest empirical proxy: '{result['empirical_proxy']}'")
        print(f"\n{'─'*70}")
        print(f"\n{result['phi4_response']}")
        print(f"{'═'*70}")


# ==============================================================================
# SECTION 10 — REAL MODEL INTERFACE (updated for v2)
# ==============================================================================

class RealModelInterface:
    def __init__(self, knowledge_base: list = None,
                 llm_model: str = "phi4:14b-q4_K_M",
                 db_path: str = "./mail_knowledge_db"):
        self.embedding_model = EmbeddingModel()
        self.dim             = self.embedding_model.dim
        self.ollama          = OllamaClient(model=llm_model)

        # Vector database (replaces text list)
        self.store = VectorKnowledgeStore(self.embedding_model, db_path)

        # Seed with initial knowledge if DB is empty
        if self.store.count()['empirical'] == 0:
            print("[INIT] Seeding knowledge base...")
            seed = knowledge_base or INITIAL_KNOWLEDGE.copy()
            for fact in seed:
                self.store.add_empirical(fact, {'source': 'seed'})
            print(f"[INIT] {len(seed)} concepts seeded.")

        # Build probe system, QTE, PRM
        self.probes     = SemanticProbeSystem(self.embedding_model)
        self.qte        = QuestionTransformationEngine(
            self.embedding_model, self.probes, self.ollama)
        self.prm        = PhilosophicalReasoningMode(self.ollama, self.probes)

        # Experimental hypothesis generator + researcher interface
        self.exp_gen    = ExperimentalHypothesisGenerator(
            self.ollama, self.embedding_model, self.store)
        self.researcher = ResearcherInterface()

        counts = self.store.count()
        print(f"\n[MODEL INTERFACE] v2 Ready.")
        print(f"  Empirical facts:      {counts['empirical']}")
        print(f"  Belief states:        {counts['beliefs']}")
        print(f"  Experimental results: {counts['experiments']}")
        print(f"  LLM: {llm_model}")

    def get_embedding_space(self):
        texts, vectors = self.store.get_all_vectors(self.embedding_model)
        if len(texts) == 0:
            # Fallback to initial knowledge
            texts   = INITIAL_KNOWLEDGE.copy()
            vectors = self.embedding_model.encode(texts)
        return vectors

    def get_concept_labels(self):
        texts = self.store.get_all_texts()
        return texts if texts else INITIAL_KNOWLEDGE.copy()

    def add_confirmed_knowledge(self, text: str, metadata: dict = None):
        if not text or not text.strip():
            return
        ok = self.store.add_empirical(text.strip(), metadata)
        if ok:
            print(f"  [KNOWLEDGE] Added: '{text[:60]}'")
            counts = self.store.count()
            print(f"  [KNOWLEDGE] DB: {counts['empirical']} empirical, "
                  f"{counts['experiments']} experimental")

    def vector_to_natural_language(self, top_positive, top_negative):
        pos_str = ', '.join(top_positive[:5])
        neg_str = ', '.join(top_negative[:5])
        system  = (
            "You are a curious AI. Generate ONE specific yes/no question "
            "probing a hidden concept. Do NOT use the word 'hypothesis'. "
            "Output only the question."
        )
        prompt  = (f"Concept is related to: {pos_str}\n"
                   f"Concept is unlike: {neg_str}\n"
                   f"Ask one specific yes/no question.")
        q = self.ollama.generate(prompt, system=system, max_tokens=80)
        if not q.strip() or '[OLLAMA' in q:
            q = (f"Is this concept related to {top_positive[0]} "
                 f"but distinct from {top_negative[0]}?")
        if not q.strip().endswith("?"):
            q = q.strip() + "?"
        return q

    def get_fisher_information(self):
        return np.ones(self.dim, dtype=np.float32)


# ==============================================================================
# SECTION 11 — GAP DETECTOR (unchanged from v1)
# ==============================================================================

class GapDetector:
    def __init__(self, n_clusters=5, n_probes=20, probe_steps=10, k=5):
        self.n_clusters  = n_clusters
        self.n_probes    = n_probes
        self.probe_steps = probe_steps
        self.k           = k

    def cosine_sim(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compute_density_at(self, point, vectors):
        k    = min(self.k, len(vectors))
        nrms = np.linalg.norm(vectors, axis=1)
        nrms = np.where(nrms == 0, 1e-8, nrms)
        pn   = np.linalg.norm(point)
        if pn == 0: return 0.0
        sims = (vectors @ point) / (nrms * pn + 1e-8)
        return 1.0 / (np.sort(1.0 - sims)[:k].mean() + 1e-8)

    def find_clusters(self, vectors):
        if not SKLEARN_AVAILABLE:
            return vectors.copy(), np.arange(len(vectors))
        n      = min(self.n_clusters, len(vectors))
        km     = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        centres = km.cluster_centers_
        norms   = np.linalg.norm(centres, axis=1, keepdims=True)
        norms   = np.where(norms == 0, 1e-8, norms)
        return centres / norms, labels

    def find_interdisciplinary_midpoints(self, centres):
        mps = []
        n   = len(centres)
        for i in range(n):
            for j in range(i+1, n):
                mid  = (centres[i] + centres[j]) / 2.0
                norm = np.linalg.norm(mid)
                if norm > 0: mid = mid / norm
                mps.append({'vector': mid, 'cluster_a': i, 'cluster_b': j,
                             'gap_size': 1.0 - self.cosine_sim(centres[i], centres[j])})
        mps.sort(key=lambda x: x['gap_size'], reverse=True)
        return mps

    def probe_frontier(self, midpoint, vectors):
        centroid = vectors.mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn > 0: centroid = centroid / cn
        direction = midpoint - centroid
        dn = np.linalg.norm(direction)
        if dn > 0: direction = direction / dn
        best = midpoint.copy()
        lowest = self.compute_density_at(midpoint, vectors)
        for step in range(1, self.probe_steps + 1):
            pp = midpoint + direction * (step * 0.1)
            pn = np.linalg.norm(pp)
            if pn > 0: pp = pp / pn
            d = self.compute_density_at(pp, vectors)
            if d < lowest:
                lowest = d
                best   = pp.copy()
        return best, lowest

    def find_gaps(self, vectors):
        print(f"\n[GAP DETECTION]  KB size: {len(vectors)}")
        centres, labels = self.find_clusters(vectors)
        mps = self.find_interdisciplinary_midpoints(centres)
        gap_vectors, gap_metadata, densities = [], [], []
        for mp in mps:
            frontier, density = self.probe_frontier(mp['vector'], vectors)
            gap_vectors.append(frontier)
            gap_metadata.append({'cluster_a': mp['cluster_a'],
                                  'cluster_b': mp['cluster_b'],
                                  'gap_size':  mp['gap_size'],
                                  'density':   density})
            densities.append(density)
        gap_vectors = np.array(gap_vectors)
        densities   = np.array(densities)
        idx         = np.argsort(densities)
        print(f"  {len(gap_vectors)} frontier gaps found.")
        return gap_vectors[idx], [gap_metadata[i] for i in idx], densities[idx]


# ==============================================================================
# SECTION 12 — HYPOTHESIS GENERATOR (unchanged from v1)
# ==============================================================================

class HypothesisGenerator:
    def __init__(self, n_neighbours=10):
        self.n_neighbours = n_neighbours

    def _cos(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def generate(self, gap_vector, all_vectors):
        n    = min(self.n_neighbours, len(all_vectors))
        sims = np.array([self._cos(gap_vector, v) for v in all_vectors])
        idx  = np.argsort(sims)
        top_pos, top_neg = idx[-n:][::-1], idx[:n]
        weights = np.zeros(len(all_vectors))
        for i in top_pos: weights[i] = sims[i]
        for i in top_neg: weights[i] = sims[i]
        ws = np.sum(np.abs(weights))
        if ws > 0: weights = weights / ws
        hypothesis = sum(
            weights[i] * all_vectors[i] for i in range(len(all_vectors))
            if weights[i] != 0
        )
        print(f"\n[HYPOTHESIS]  norm={np.linalg.norm(hypothesis):.4f}  "
              f"cos(gap,H)={self._cos(gap_vector, hypothesis):.4f}")
        return hypothesis, sims, idx


# ==============================================================================
# SECTION 13 — SPARSE FEATURE EXPLAINER (unchanged from v1)
# ==============================================================================

class SparseFeatureExplainer:
    def __init__(self, k=10):
        self.k = k

    def explain(self, hypothesis, concept_labels, similarities=None):
        if similarities is not None and len(similarities) == len(concept_labels):
            idx = np.argsort(similarities)
            return ([concept_labels[i] for i in idx[-self.k:][::-1]
                     if i < len(concept_labels)],
                    [concept_labels[i] for i in idx[:self.k]
                     if i < len(concept_labels)])
        k   = min(self.k, len(hypothesis))
        idx = np.argsort(hypothesis)
        return ([concept_labels[i] for i in idx[-k:][::-1]
                 if i < len(concept_labels)],
                [concept_labels[i] for i in idx[:k]
                 if i < len(concept_labels)])

    def ask_oracle(self, hypothesis, concept_labels, model_interface,
                   similarities=None):
        top_pos, top_neg = self.explain(hypothesis, concept_labels, similarities)
        question = model_interface.vector_to_natural_language(top_pos, top_neg)
        print(f"\n[ORACLE QUERY]")
        print(f"  Most similar: {', '.join(top_pos[:3])}")
        print(f"  Most unlike:  {', '.join(top_neg[:3])}")
        print(f"\n  Question: {question}")
        while True:
            ans = input("\n  Answer (yes / no / idk): ").strip().lower()
            if ans in ('yes', 'y'):   return True
            elif ans in ('no', 'n'): return False
            elif ans in ('idk', "i don't know", 'dunno', '?'): return None
            else: print("  Please answer: yes, no, or idk")


# ==============================================================================
# SECTION 14 — PHYSICS ORACLE (unchanged from v1)
# ==============================================================================

class PhysicsOracle:
    def __init__(self, epsilon=0.7):
        self.epsilon = epsilon
        if PYBULLET_AVAILABLE:
            self.client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self.client)
            print("[PHYSICS] PyBullet ready.")
        else:
            self.client = None

    def _cos(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def run_simulation(self, hypothesis):
        print("[SIMULATION] Running...")
        if PYBULLET_AVAILABLE:
            col  = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5,
                                          physicsClientId=self.client)
            body = p.createMultiBody(baseMass=1.0,
                                     baseCollisionShapeIndex=col,
                                     basePosition=[0, 0, 5],
                                     physicsClientId=self.client)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.client)
            p.removeBody(body, physicsClientId=self.client)
        return hypothesis + np.random.randn(*hypothesis.shape) * 0.1

    def validate(self, hypothesis):
        result    = self.run_simulation(hypothesis)
        sim       = self._cos(hypothesis, result)
        confirmed = sim >= self.epsilon
        print(f"  cos(H,R)={sim:.4f} ε={self.epsilon} "
              f"→ {'CONFIRMED ✓' if confirmed else 'DENIED ✗'}")
        return confirmed, result, sim


class GPCalibrator:
    def __init__(self, epsilon_dynamic=0.3):
        self.epsilon_dynamic = epsilon_dynamic
        self.residuals       = []

    def record_residual(self, h, r):
        self.residuals.append(float(np.linalg.norm(h - r)))

    def check_discrepancy_frontier(self):
        if len(self.residuals) < 5: return False
        avg = np.mean(self.residuals[-5:])
        if avg > self.epsilon_dynamic:
            print(f"[DISCREPANCY FRONTIER] avg residual {avg:.4f} — "
                  f"unknown property detected.")
            return True
        return False


# ==============================================================================
# SECTION 15 — KNOWLEDGE INTEGRATOR (updated for ChromaDB)
# ==============================================================================

class KnowledgeIntegrator:
    def __init__(self, lambda_ekc=0.5):
        self.lambda_ekc = lambda_ekc
        self.integrated = []

    def integrate(self, confirmed_hypothesis, model_interface,
                  concept_text=None, source='human'):
        print(f"\n[INTEGRATION]")
        if concept_text:
            model_interface.add_confirmed_knowledge(
                concept_text, {'source': source})
            self.integrated.append(concept_text)
        anchor = model_interface.get_embedding_space().mean(axis=0)
        fisher = model_interface.get_fisher_information()
        delta  = confirmed_hypothesis - anchor[:len(confirmed_hypothesis)]
        ekc_p  = self.lambda_ekc * np.sum(fisher[:len(delta)] * delta**2)
        print(f"  EKC penalty: {ekc_p:.4f} | Total integrated: {len(self.integrated)}")
        return True


# ==============================================================================
# SECTION 16 — KNOWLEDGE TRACKER
# ==============================================================================

class KnowledgeTracker:
    def __init__(self):
        self.facts                    = []
        self.frontier_events          = []
        self.discrepancy_frontiers    = []
        self.philosophical_encounters = []
        self.experimental_briefs      = []

    def add_fact(self, source, confirmed, detail=""):
        self.facts.append({'source': source, 'confirmed': confirmed,
                           'detail': detail, 'timestamp': time.time()})

    def add_philosophical(self, question, mode, top_frameworks):
        self.philosophical_encounters.append(
            {'question': question, 'mode': mode,
             'top_frameworks': top_frameworks, 'timestamp': time.time()})

    def add_brief(self, brief):
        self.experimental_briefs.append(brief)

    def frontier_ratio(self, window=20):
        recent = self.facts[-window:]
        if not recent: return 0.0
        return sum(1 for f in recent if f['source'] == 'simulation') / len(recent)

    def summary(self):
        confirmed = sum(1 for f in self.facts if f['confirmed'])
        print(f"\n[SESSION SUMMARY]")
        print(f"  Facts processed:          {len(self.facts)}")
        print(f"  Confirmed:                {confirmed}")
        print(f"  Philosophical:            {len(self.philosophical_encounters)}")
        print(f"  Experimental briefs:      {len(self.experimental_briefs)}")
        print(f"  Discrepancy frontiers:    {len(self.discrepancy_frontiers)}")
        if self.experimental_briefs:
            print(f"\n  Experimental hypotheses generated:")
            for b in self.experimental_briefs:
                print(f"    [{b.get('domain','?').upper()}] "
                      f"{b.get('hypothesis','')[:60]}")


# ==============================================================================
# SECTION 17 — COMPETITIVE KNOWLEDGE ARENA (unchanged from v1)
# ==============================================================================

class CompetitiveKnowledgeArena:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self.scores   = {i: 0.0 for i in range(n_agents)}
        print(f"\n[CKA] {n_agents} agents | zero-sum rewards")

    def award(self, winner, delta_I):
        self.scores[winner] += delta_I
        penalty = delta_I / max(1, self.n_agents - 1)
        for j in range(self.n_agents):
            if j != winner: self.scores[j] -= penalty
        print(f"  [CKA] Agent {winner} +{delta_I:.4f} | "
              f"{{{', '.join([f'{k}:{v:.3f}' for k,v in self.scores.items()])}}}")

    def leaderboard(self):
        print(f"\n[CKA LEADERBOARD]")
        for rank, (agent, score) in enumerate(
            sorted(self.scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            print(f"  #{rank} Agent {agent}: {score:.4f}")


# ==============================================================================
# SECTION 18 — THE FULL MAIL LOOP v2
# ==============================================================================

def run_mail_loop(
    model_interface: RealModelInterface,
    n_cycles: int = 5,
    use_competition: bool = False,
    n_agents: int = 2,
    researcher_mode: bool = True
):
    """
    MAIL v2 loop.

    New in v2:
    - Experimental hypothesis generator at Stage 3 (researcher_mode=True)
    - Researcher feedback collection after each brief
    - Results stored in ChromaDB experimental_results collection
    - Knowledge persists across sessions via Google Drive

    Routing:
        oracle yes/no → normal MAIL
        oracle idk + empirical fingerprint → simulation → experimental brief
        oracle idk + philosophical fingerprint → QTE → PRM
    """

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP — MAIL v2 (Research Accelerator)")
    print("  Embedding: all-MiniLM-L6-v2 | LLM: Phi-4 14B Q4_K_M")
    print("  + ChromaDB | + Experimental Hypothesis Generator")
    print("  + Researcher Interface | + QTE | + PRM")
    print("="*70)

    gap_detector  = GapDetector(n_clusters=5, n_probes=20, probe_steps=10, k=5)
    hyp_generator = HypothesisGenerator(n_neighbours=10)
    sfe           = SparseFeatureExplainer(k=10)
    physics       = PhysicsOracle(epsilon=0.7)
    gp            = GPCalibrator(epsilon_dynamic=0.3)
    integrator    = KnowledgeIntegrator(lambda_ekc=0.5)
    tracker       = KnowledgeTracker()
    cka           = CompetitiveKnowledgeArena(n_agents) if use_competition else None
    visited_gaps  = set()

    for cycle in range(1, n_cycles + 1):
        print(f"\n{'─'*70}")
        print(f"  CYCLE {cycle}/{n_cycles}")
        print(f"{'─'*70}")

        # ── Stage 1: Gap Detection ─────────────────────────────────────────
        vectors = model_interface.get_embedding_space()
        labels  = model_interface.get_concept_labels()

        gap_vectors, gap_metadata, gap_densities = gap_detector.find_gaps(vectors)
        if len(gap_vectors) == 0:
            print("  No gaps found.")
            break

        unvisited = [
            i for i, m in enumerate(gap_metadata)
            if f"gap_{m['cluster_a']}_{m['cluster_b']}" not in visited_gaps
        ]
        if not unvisited:
            print("  All frontiers explored. Resetting.")
            visited_gaps.clear()
            unvisited = list(range(len(gap_vectors)))

        ti         = unvisited[0]
        gap_vector = gap_vectors[ti]
        gap_meta   = gap_metadata[ti]
        gap_key    = f"gap_{gap_meta['cluster_a']}_{gap_meta['cluster_b']}"

        sims_to_gap = np.array([gap_detector.cosine_sim(gap_vector, v)
                                 for v in vectors])
        top3        = np.argsort(sims_to_gap)[-3:][::-1]
        nearest     = [labels[i] for i in top3]

        print(f"\n  Target: {gap_key}")
        print(f"  Nearest: {', '.join(nearest)}")

        # Fingerprint the gap
        gap_fp   = model_interface.probes.fingerprint(gap_vector)
        gap_mode = model_interface.probes.classify_reasoning_mode(gap_fp)
        model_interface.probes.explain_fingerprint(gap_fp)
        print(f"\n  Predicted mode: {gap_mode.upper()}")

        # ── Stage 2: Hypothesis Generation ────────────────────────────────
        hypothesis, similarities, sorted_idx = hyp_generator.generate(
            gap_vector, vectors)
        hyp_fp   = model_interface.probes.fingerprint(hypothesis)
        hyp_mode = model_interface.probes.classify_reasoning_mode(hyp_fp)
        print(f"  Hypothesis mode: {hyp_mode.upper()}")

        # ── Stage 3: Human Oracle ──────────────────────────────────────────
        oracle_response = sfe.ask_oracle(
            hypothesis, labels, model_interface, similarities)

        confirmed    = False
        concept_text = None

        if oracle_response is True:
            print(f"\n  ✓ Confirmed by oracle")
            confirmed    = True
            concept_text = input("  Name this concept (max 150 chars): ").strip()
            tracker.add_fact('human', True, concept_text)

        elif oracle_response is False:
            print(f"\n  ✗ Denied")
            tracker.add_fact('human', False, gap_key)
            visited_gaps.add(gap_key)

        else:
            # Knowledge Frontier
            print(f"\n  ? Frontier detected → routing: {hyp_mode.upper()}")

            if hyp_mode == 'empirical':
                # ── 4a: Simulation + Experimental Brief ────────────────────
                print(f"\n→ Stage 4a: Empirical Validation")
                confirmed, result, sim_score = physics.validate(hypothesis)
                tracker.add_fact('simulation', confirmed, gap_key)
                gp.record_residual(hypothesis, result)

                if not confirmed:
                    visited_gaps.add(gap_key)

                if gp.check_discrepancy_frontier():
                    tracker.discrepancy_frontiers.append(
                        {'cycle': cycle, 'gap': gap_key})

                # ── Generate experimental brief regardless of sim result ───
                if researcher_mode:
                    print(f"\n→ Generating Experimental Brief...")
                    brief = model_interface.exp_gen.generate(
                        hypothesis   = hypothesis,
                        gap_vector   = gap_vector,
                        similarities = similarities,
                        nearest_concepts = nearest,
                        fingerprint  = hyp_fp
                    )
                    model_interface.researcher.display_brief(brief)
                    tracker.add_brief(brief)

                    # Ask researcher for feedback
                    feedback = model_interface.researcher.collect_feedback(
                        brief, model_interface.store)

                    if feedback and feedback['confirmed']:
                        # Feed confirmed result back into knowledge base
                        confirmed_text = (
                            f"Experimentally confirmed: {feedback['hypothesis'][:100]} "
                            f"| Result: {feedback['result'][:80]}"
                        )
                        model_interface.add_confirmed_knowledge(
                            confirmed_text,
                            {'source': 'experiment', 'type': feedback['exp_type']}
                        )
                        confirmed    = True
                        concept_text = confirmed_text

                    # Export brief as JSON
                    export_path = f"brief_cycle_{cycle}.json"
                    model_interface.researcher.export_brief_json(
                        brief, export_path)

            else:
                # ── 4b: Philosophical Reasoning Mode ───────────────────────
                print(f"\n→ Stage 4b: Philosophical Reasoning Mode")
                top_pos, top_neg = sfe.explain(hypothesis, labels, similarities)
                original_q = model_interface.vector_to_natural_language(
                    top_pos, top_neg)
                print(f"  Question: {original_q}")

                qte_out = model_interface.qte.transform(
                    hypothesis, vectors, labels, hyp_fp, original_q)
                prm_result = model_interface.prm.reason(
                    qte_out, hyp_fp, hyp_mode, original_q, hypothesis)
                model_interface.prm.display_result(prm_result)

                tracker.add_philosophical(
                    original_q, hyp_mode, prm_result['top_frameworks'])
                tracker.add_fact('philosophical', True, original_q[:80])

                # Store as belief state with top framework confidence
                top_fw_name, top_fw_prob = prm_result['top_frameworks'][0]
                model_interface.store.add_belief(
                    text       = original_q,
                    framework  = top_fw_name,
                    confidence = top_fw_prob,
                    metadata   = {'mode': hyp_mode}
                )
                visited_gaps.add(gap_key)

                if cka:
                    cka.award(cycle % cka.n_agents, 0.05)
                continue

        # ── Stage 5: Integration ───────────────────────────────────────────
        if confirmed:
            print(f"\n→ Stage 5: Integration")
            if concept_text and len(concept_text) > 150:
                concept_text = input(
                    "  Shorter (max 150 chars): ").strip()
            integrator.integrate(hypothesis, model_interface, concept_text)
            if cka:
                cka.award(cycle % cka.n_agents, 0.1)

        print(f"\n  Frontier ratio: {tracker.frontier_ratio():.2f}")

    print(f"\n{'='*70}")
    tracker.summary()
    if cka:
        cka.leaderboard()

    # Offer to save to Google Drive
    if COLAB_AVAILABLE:
        save = input("\nSave knowledge to Google Drive? (yes/no): ").strip().lower()
        if save in ('yes', 'y'):
            model_interface.store.save_to_drive()

    return tracker


# ==============================================================================
# SECTION 19 — ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP — MAIL v2")
    print("  Research Accelerator Edition")
    print("  all-MiniLM-L6-v2 | Phi-4 14B Q4_K_M | ChromaDB")
    print("="*70)

    # Offer to load existing knowledge from Google Drive
    if COLAB_AVAILABLE:
        load = input("\nLoad saved knowledge from Google Drive? (yes/no): ").strip().lower()
        if load in ('yes', 'y'):
            # Temp store to trigger load
            tmp_emb   = EmbeddingModel()
            tmp_store = VectorKnowledgeStore(tmp_emb)
            tmp_store.load_from_drive()

    model = RealModelInterface(
        knowledge_base = INITIAL_KNOWLEDGE.copy(),
        llm_model      = "phi4:14b-q4_K_M",
        db_path        = "./mail_knowledge_db"
    )

    print("\nOptions:")
    print("  1. Run MAIL loop — research accelerator mode")
    print("  2. Run MAIL loop — with Competitive Knowledge Arena")
    print("  3. Add knowledge manually")
    print("  4. Gap detection demo")
    print("  5. Test philosophical reasoning")
    print("  6. Search knowledge database")
    print("  7. View knowledge database stats")

    choice = input("\nChoice (1-7): ").strip()

    if choice == '1':
        n = int(input("Cycles? (3-5 recommended): ").strip() or "3")
        run_mail_loop(model, n_cycles=n, researcher_mode=True)

    elif choice == '2':
        n      = int(input("Cycles?: ").strip() or "3")
        agents = int(input("Agents? (2-4): ").strip() or "2")
        run_mail_loop(model, n_cycles=n, use_competition=True,
                      n_agents=agents, researcher_mode=True)

    elif choice == '3':
        print("\nAdd knowledge (type 'done' to finish):")
        while True:
            fact = input("  Fact: ").strip()
            if fact.lower() == 'done': break
            if fact: model.add_confirmed_knowledge(fact)

    elif choice == '4':
        vectors  = model.get_embedding_space()
        labels   = model.get_concept_labels()
        detector = GapDetector(n_clusters=5)
        gvecs, gmeta, dens = detector.find_gaps(vectors)
        print(f"\nFrontier gaps:")
        for i, (meta, density) in enumerate(zip(gmeta, dens)):
            sims = np.array([detector.cosine_sim(gvecs[i], v)
                              for v in vectors])
            top3 = np.argsort(sims)[-3:][::-1]
            near = [labels[j] for j in top3]
            fp   = model.probes.fingerprint(gvecs[i])
            mode = model.probes.classify_reasoning_mode(fp)
            print(f"\n  Gap {i+1}: clusters "
                  f"{meta['cluster_a']} ↔ {meta['cluster_b']}")
            print(f"    Density: {density:.4f}  Mode: {mode.upper()}")
            print(f"    Nearest: {', '.join(near)}")
            if i >= 6: break

    elif choice == '5':
        q   = input("\nEnter a question: ").strip()
        hyp = model.embedding_model.encode_single(q)
        fp  = model.probes.fingerprint(hyp)
        model.probes.explain_fingerprint(fp)
        mode = model.probes.classify_reasoning_mode(fp)
        print(f"\n  Mode: {mode.upper()}")
        vectors = model.get_embedding_space()
        labels  = model.get_concept_labels()
        qte_out = model.qte.transform(hyp, vectors, labels, fp, q)
        result  = model.prm.reason(qte_out, fp, mode, q, hyp)
        model.prm.display_result(result)

    elif choice == '6':
        q      = input("\nSearch query: ").strip()
        n      = int(input("Number of results (default 5): ").strip() or "5")
        col    = input("Collection (empirical/beliefs/experiments): ").strip() or "empirical"
        results = model.store.semantic_search(q, n=n, collection=col)
        print(f"\nTop {len(results)} results from '{col}':")
        for i, r in enumerate(results, 1):
            print(f"\n  {i}. {r['text'][:80]}")
            print(f"     Distance: {r['distance']:.4f}")
            if r['metadata']:
                ts = r['metadata'].get('timestamp', '')[:10]
                src = r['metadata'].get('source', '')
                if ts or src:
                    print(f"     Source: {src}  Date: {ts}")

    elif choice == '7':
        counts = model.store.count()
        print(f"\n[KNOWLEDGE DATABASE]")
        print(f"  Empirical facts:       {counts['empirical']}")
        print(f"  Belief states:         {counts['beliefs']}")
        print(f"  Experimental results:  {counts['experiments']}")
        print(f"  Total:                 {sum(counts.values())}")

    else:
        run_mail_loop(model, n_cycles=3, researcher_mode=True)
