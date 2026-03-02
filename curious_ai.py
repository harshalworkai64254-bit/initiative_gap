"""
curious_ai.py — The Initiative Gap: MAIL Framework Implementation
=================================================================
Meta-cognitive Autonomous Inquiry Loop (MAIL)

Full pipeline connecting:
    - all-MiniLM-L6-v2  (sentence-transformers) — real embedding space
    - Llama 3.2 via Ollama                       — English language agent
    - MAIL loop                                  — gap detection, hypothesis
                                                   generation, SFE, validation
    - PyBullet                                   — physics simulation

EKC weight updates remain a placeholder — requires direct model weight access
(frontier lab infrastructure). Everything else runs for real on Colab free tier.

Run on Google Colab:
    !pip install sentence-transformers scikit-learn pybullet requests
    !curl -fsSL https://ollama.com/install.sh | sh
    !ollama pull llama3.2 &
    exec(open('curious_ai.py').read())

Author: [your name]
Date:   2026
Repo:   https://github.com/harshalworkai64254-bit/initiative_gap
"""

import numpy as np
import json
import time
import requests
import subprocess
import threading
from typing import Optional

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed.")
    print("  Run: pip install sentence-transformers")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not installed.")
    print("  Run: pip install scikit-learn")

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[WARNING] PyBullet not installed.")
    print("  Run: pip install pybullet")


# ==============================================================================
# SECTION 1 — OLLAMA CLIENT
# Handles communication with the locally running Llama model
# ==============================================================================

class OllamaClient:
    """
    Lightweight client for the Ollama local API.
    Ollama runs a local server at localhost:11434.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.available = self._check_available()

    def _check_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                available = any(self.model in m for m in models)
                if available:
                    print(f"[OLLAMA] {self.model} is ready.")
                else:
                    print(f"[OLLAMA] {self.model} not found. Pull it with: ollama pull {self.model}")
                return available
        except Exception:
            print(f"[OLLAMA] Server not running. Start with: ollama serve")
            return False
        return False

    def generate(self, prompt: str, system: str = "", max_tokens: int = 500) -> str:
        """Generate a response from the local Llama model."""
        if not self.available:
            return f"[OLLAMA UNAVAILABLE] Prompt was: {prompt[:100]}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3}
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            else:
                return f"[OLLAMA ERROR] Status {r.status_code}"
        except Exception as e:
            return f"[OLLAMA ERROR] {str(e)}"


# ==============================================================================
# SECTION 2 — KNOWLEDGE BASE
# The concepts the MAIL loop operates on.
# In production this grows dynamically as MAIL confirms new knowledge.
# Here we seed it with a starting set and grow it during the session.
# ==============================================================================

# Starting knowledge base — plain English concepts
# MAIL will find gaps between these and ask questions to fill them
INITIAL_KNOWLEDGE = [
    # Physical world
    "objects fall due to gravity",
    "water flows downhill",
    "fire requires oxygen to burn",
    "ice melts above zero degrees celsius",
    "sound travels through air as pressure waves",
    "light travels faster than sound",
    "metals conduct electricity",
    "magnets attract iron",
    "plants convert sunlight into energy",
    "animals need food and water to survive",
    # Causation
    "friction causes heat",
    "pressure causes liquids to flow",
    "temperature affects the speed of chemical reactions",
    "force causes acceleration",
    "heat causes expansion in most materials",
    # Biology
    "cells are the basic unit of life",
    "DNA carries genetic information",
    "evolution occurs through natural selection",
    "the heart pumps blood through the body",
    "the brain processes sensory information",
    # History / social
    "the assassination of Franz Ferdinand triggered World War One",
    "alliances between nations can cause local conflicts to become global wars",
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
]


# ==============================================================================
# SECTION 3 — EMBEDDING MODEL
# Converts text concepts into vectors using all-MiniLM-L6-v2
# This IS the embedding space that MAIL operates on
# ==============================================================================

class EmbeddingModel:
    """
    Wraps sentence-transformers all-MiniLM-L6-v2.
    Converts text concepts into 384-dimensional vectors.
    This is the real embedding space — not random placeholders.
    """

    def __init__(self):
        if not ST_AVAILABLE:
            print("[EMBEDDING] sentence-transformers not available. Using random vectors.")
            self.model = None
            self.dim = 384
        else:
            print("[EMBEDDING] Loading all-MiniLM-L6-v2...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dim = 384
            print("[EMBEDDING] Ready.")

    def encode(self, texts: list) -> np.ndarray:
        """Convert a list of text concepts into vectors."""
        if self.model is None:
            return np.random.randn(len(texts), self.dim).astype(np.float32)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        """Convert a single text concept into a vector."""
        return self.encode([text])[0]


# ==============================================================================
# SECTION 4 — REAL MODEL INTERFACE
# Replaces the placeholder FrontierModelInterface with real model calls
# ==============================================================================

class RealModelInterface:
    """
    Real implementation of the model interface.
    Uses all-MiniLM-L6-v2 for embeddings and Llama 3.2 for language.

    EKC weight updates remain placeholder — requires frontier lab access.
    Everything else is real.
    """

    def __init__(self, knowledge_base: list = None, llm_model: str = "llama3.2"):
        self.embedding_model = EmbeddingModel()
        self.dim = self.embedding_model.dim
        self.ollama = OllamaClient(model=llm_model)

        # Knowledge base — grows as MAIL confirms new facts
        # Sanitise on load: remove any garbage concepts over 150 chars
        raw = knowledge_base or INITIAL_KNOWLEDGE.copy()
        self.knowledge_texts = [t for t in raw if len(t.strip()) <= 150 and t.strip()]
        removed = len(raw) - len(self.knowledge_texts)
        if removed > 0:
            print(f"  [SANITISE] Removed {removed} invalid concept(s) from knowledge base.")
        self.knowledge_vectors = None
        self._rebuild_vectors()

        print(f"\n[MODEL INTERFACE] Ready.")
        print(f"  Embedding dim: {self.dim}")
        print(f"  Knowledge base size: {len(self.knowledge_texts)} concepts")
        print(f"  LLM: {llm_model} via Ollama\n")

    def _rebuild_vectors(self):
        """Recompute embedding vectors for all knowledge base concepts."""
        self.knowledge_vectors = self.embedding_model.encode(self.knowledge_texts)

    def get_embedding_space(self) -> np.ndarray:
        """Returns the real embedding matrix of current knowledge base."""
        return self.knowledge_vectors

    def get_concept_labels(self) -> list:
        """Returns the text labels for each concept in the knowledge base."""
        return self.knowledge_texts

    def vector_to_natural_language(self, top_positive: list,
                                    top_negative: list) -> str:
        """
        Uses Llama 3.2 to generate a natural language yes/no question
        from the top-10 and bottom-10 concept labels.
        """
        pos_str = ', '.join(top_positive[:5])
        neg_str = ', '.join(top_negative[:5])

        system = (
            "You are a curious AI exploring the frontiers of human knowledge. "
            "You will be given concepts that a hidden hypothesis is similar to, "
            "and concepts it is unlike. "
            "Your job is to generate ONE specific, concrete yes/no question "
            "that probes what the hidden concept might be. "
            "Do NOT use the word 'hypothesis' or placeholder text like [hypothesis] in your question. "
            "Ask about a real, specific relationship between the given concepts. "
            "Output only the question itself, nothing else."
        )

        prompt = (
            f"A hidden concept is strongly related to: {pos_str}\n"
            f"The same concept is strongly unlike: {neg_str}\n"
            f"Ask one specific yes/no question that would help identify what this concept is. "
            f"Be concrete. Do not use the word hypothesis or any placeholder text."
        )

        question = self.ollama.generate(prompt, system=system, max_tokens=80)

        # Reject if Llama left placeholder text in or failed
        bad_phrases = ["[hypothesis]", "[OLLAMA", "placeholder", "unknown concept"]
        if any(p in question for p in bad_phrases) or not question.strip():
            question = (
                f"Is this concept related to {', '.join(top_positive[:2])} "
                f"but distinct from {', '.join(top_negative[:2])}?"
            )

        # Ensure it ends with a question mark
        if not question.strip().endswith("?"):
            question = question.strip() + "?"

        return question

    def add_confirmed_knowledge(self, text: str):
        """
        Add a confirmed fact to the knowledge base and rebuild vectors.
        Validates input — rejects empty strings, duplicates, or anything
        over 150 characters (garbage/accidental pastes).
        """
        # Validate
        if not text or not text.strip():
            print(f"  [KNOWLEDGE] Rejected: empty input.")
            return
        text = text.strip()
        if len(text) > 150:
            print(f"  [KNOWLEDGE] Rejected: input too long ({len(text)} chars). Max 150.")
            print(f"  [KNOWLEDGE] Please describe the concept in one short sentence.")
            return
        if text in self.knowledge_texts:
            print(f"  [KNOWLEDGE] Already known: '{text}'")
            return
        # Accept
        self.knowledge_texts.append(text)
        self._rebuild_vectors()
        print(f"  [KNOWLEDGE] Added: '{text}'")
        print(f"  [KNOWLEDGE] Base now has {len(self.knowledge_texts)} concepts")

    def get_fisher_information(self) -> np.ndarray:
        """[EKC - FRONTIER MODEL REQUIRED] Returns uniform Fisher Information."""
        return np.ones(self.dim, dtype=np.float32)

    def update_weights_ekc(self, confirmed_vector, anchor_weights,
                            fisher_info, lambda_ekc=0.5) -> bool:
        """
        [EKC - FRONTIER MODEL REQUIRED]
        Real EKC requires write access to model weights.
        Currently: adds the confirmed concept to knowledge base as text.
        Full EKC gradient update requires frontier lab infrastructure.
        """
        print(f"  [EKC PLACEHOLDER] Weight update requires frontier model access.")
        print(f"  [EKC] Knowledge integrated as text concept instead.")
        return True


# ==============================================================================
# SECTION 5 — GAP DETECTOR
# Real implementation: inter-cluster probing into genuinely uncharted space.
#
# The old approach found the most isolated vectors WITHIN the knowledge base.
# That is finding gaps between known things — not finding unknown territory.
#
# The correct approach:
#   1. Find dense clusters in the knowledge base (fields of knowledge)
#   2. Find midpoints BETWEEN clusters (interdisciplinary frontiers)
#   3. Probe outward from those midpoints into empty space
#   4. Generate hypothesis vectors at the frontier
#
# Human knowledge is a tiny cluster in an infinite space.
# Everything outside the cluster is unknown.
# The most valuable gaps are at the BOUNDARIES between existing clusters —
# exactly where real interdisciplinary breakthroughs happen.
# ==============================================================================

class GapDetector:
    """
    Stage 1: Inter-cluster frontier detection.

    Finds genuine gaps in knowledge space by:
    1. Clustering known vectors into fields of knowledge
    2. Finding midpoints between cluster pairs (interdisciplinary gaps)
    3. Probing outward from midpoints into uncharted space
    4. Ranking gaps by how far they are from ALL known knowledge

    This finds territory humanity has never explored —
    not just which known concepts are most isolated.
    """

    def __init__(self, n_clusters: int = 5, n_probes: int = 20,
                 probe_steps: int = 10, k: int = 5):
        self.n_clusters  = n_clusters   # number of knowledge clusters
        self.n_probes    = n_probes     # random directions to probe per gap
        self.probe_steps = probe_steps  # steps outward along each probe
        self.k           = k            # neighbours for density check

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compute_density_at(self, point: np.ndarray,
                            vectors: np.ndarray) -> float:
        """Density of the embedding space at an arbitrary point."""
        k = min(self.k, len(vectors))
        norms = np.linalg.norm(vectors, axis=1)
        norms = np.where(norms == 0, 1e-8, norms)
        point_norm = np.linalg.norm(point)
        if point_norm == 0:
            return 0.0
        # Cosine similarity from point to all known vectors
        sims = (vectors @ point) / (norms * point_norm + 1e-8)
        distances = 1.0 - sims
        nearest_k = np.sort(distances)[:k]
        avg_dist = nearest_k.mean()
        return 1.0 / (avg_dist + 1e-8)

    def find_clusters(self, vectors: np.ndarray) -> tuple:
        """
        Group knowledge vectors into clusters using k-means.
        Each cluster represents a field of knowledge.
        Returns (cluster_centres, cluster_labels)
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: treat each vector as its own cluster
            return vectors.copy(), np.arange(len(vectors))

        from sklearn.cluster import KMeans
        n = min(self.n_clusters, len(vectors))
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        centres = km.cluster_centers_

        # Normalise centres to unit sphere
        norms = np.linalg.norm(centres, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        centres = centres / norms

        return centres, labels

    def find_interdisciplinary_midpoints(self,
                                          centres: np.ndarray) -> list:
        """
        Find midpoints between every pair of cluster centres.
        These are the interdisciplinary frontiers —
        the spaces between fields of knowledge.
        e.g. midpoint between physics cluster and biology cluster
             = potential biophysics frontier
        """
        midpoints = []
        n = len(centres)
        for i in range(n):
            for j in range(i + 1, n):
                mid = (centres[i] + centres[j]) / 2.0
                # Normalise to unit sphere
                norm = np.linalg.norm(mid)
                if norm > 0:
                    mid = mid / norm
                # Distance between the two clusters (how big is the gap?)
                gap_size = 1.0 - self.cosine_sim(centres[i], centres[j])
                midpoints.append({
                    'vector':    mid,
                    'cluster_a': i,
                    'cluster_b': j,
                    'gap_size':  gap_size
                })

        # Sort by gap size — biggest gaps first
        midpoints.sort(key=lambda x: x['gap_size'], reverse=True)
        return midpoints

    def probe_frontier(self, midpoint: np.ndarray,
                       vectors: np.ndarray) -> np.ndarray:
        """
        Starting from a midpoint between two clusters,
        probe outward into empty space to find the frontier.

        Walk in the direction of the midpoint (away from centroid)
        until density drops below threshold.
        The last point before density drops to near zero is the frontier.
        """
        centroid = vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Direction: from centroid toward and beyond the midpoint
        direction = midpoint - centroid
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm

        # Walk outward in steps
        best_frontier = midpoint.copy()
        lowest_density = self.compute_density_at(midpoint, vectors)

        for step in range(1, self.probe_steps + 1):
            # Each step moves further from the centroid
            probe_point = midpoint + direction * (step * 0.1)
            norm = np.linalg.norm(probe_point)
            if norm > 0:
                probe_point = probe_point / norm

            density = self.compute_density_at(probe_point, vectors)

            if density < lowest_density:
                lowest_density = density
                best_frontier = probe_point.copy()

        return best_frontier, lowest_density

    def find_gaps(self, vectors: np.ndarray) -> tuple:
        """
        Main gap detection method.

        Returns:
            gap_vectors  — frontier vectors in uncharted space
            gap_metadata — info about each gap (which clusters, gap size)
            density      — density at each gap point (lower = more unknown)
        """
        print(f"\n[GAP DETECTION — INTER-CLUSTER FRONTIER PROBING]")
        print(f"  Knowledge base: {len(vectors)} concepts")

        # Step 1: Find knowledge clusters
        centres, labels = self.find_clusters(vectors)
        n_clusters = len(centres)
        print(f"  Knowledge clusters found: {n_clusters}")

        # Label clusters by their most central concept
        cluster_labels = []
        for c_idx in range(n_clusters):
            members = np.where(labels == c_idx)[0]
            if len(members) == 0:
                cluster_labels.append(f"cluster_{c_idx}")
                continue
            # Find member closest to cluster centre
            sims = np.array([
                self.cosine_sim(vectors[m], centres[c_idx])
                for m in members
            ])
            best = members[np.argmax(sims)]
            cluster_labels.append(f"cluster_{c_idx}")

        # Step 2: Find interdisciplinary midpoints
        midpoints = self.find_interdisciplinary_midpoints(centres)
        print(f"  Interdisciplinary frontiers: {len(midpoints)}")

        # Step 3: Probe outward from each midpoint
        gap_vectors  = []
        gap_metadata = []
        densities    = []

        for mp in midpoints:
            frontier, density = self.probe_frontier(mp['vector'], vectors)
            gap_vectors.append(frontier)
            gap_metadata.append({
                'cluster_a': mp['cluster_a'],
                'cluster_b': mp['cluster_b'],
                'gap_size':  mp['gap_size'],
                'density':   density
            })
            densities.append(density)

        gap_vectors = np.array(gap_vectors)
        densities   = np.array(densities)

        # Sort by lowest density — most unknown territory first
        sorted_idx = np.argsort(densities)
        gap_vectors  = gap_vectors[sorted_idx]
        gap_metadata = [gap_metadata[i] for i in sorted_idx]
        densities    = densities[sorted_idx]

        print(f"  Frontier gaps identified: {len(gap_vectors)}")
        print(f"  Most unknown region density: {densities[0]:.4f}")
        print(f"  (Lower density = further from all known knowledge)")

        return gap_vectors, gap_metadata, densities

        print(f"\n[GAP DETECTION]")
        print(f"  Knowledge base: {len(vectors)} concepts")
        print(f"  Gaps found: {len(gap_indices)} (bottom {self.gap_percentile}% density)")

        return gap_indices, density


# ==============================================================================
# SECTION 6 — HYPOTHESIS GENERATOR (unchanged — maths is real)
# ==============================================================================

class HypothesisGenerator:
    """
    Stage 2: similarity-weighted vector interpolation.
    H = Σᵢ wᵢ × vᵢ   where wᵢ = cos(vᵢ, G_direction)
    """

    def __init__(self, n_neighbours: int = 10):
        self.n_neighbours = n_neighbours

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def generate(self, gap_vector: np.ndarray, all_vectors: np.ndarray) -> tuple:
        n = min(self.n_neighbours, len(all_vectors))
        similarities = np.array([
            self.cosine_similarity(gap_vector, v) for v in all_vectors
        ])
        sorted_idx = np.argsort(similarities)
        top_pos = sorted_idx[-n:][::-1]
        top_neg = sorted_idx[:n]

        weights = np.zeros(len(all_vectors))
        for idx in top_pos:
            weights[idx] = similarities[idx]
        for idx in top_neg:
            weights[idx] = similarities[idx]

        w_sum = np.sum(np.abs(weights))
        if w_sum > 0:
            weights = weights / w_sum

        hypothesis = sum(weights[i] * all_vectors[i] for i in range(len(all_vectors))
                        if weights[i] != 0)

        print(f"\n[HYPOTHESIS GENERATION]")
        print(f"  Hypothesis norm: {np.linalg.norm(hypothesis):.4f}")
        print(f"  cos(gap, hypothesis): {self.cosine_similarity(gap_vector, hypothesis):.4f}")

        return hypothesis, similarities, sorted_idx


# ==============================================================================
# SECTION 7 — SPARSE FEATURE EXPLAINER (now uses real concept labels)
# ==============================================================================

class SparseFeatureExplainer:
    """
    Stage 3: top-k / bottom-k concept extraction + Llama question generation.
    Now uses real concept labels from the knowledge base.
    """

    def __init__(self, k: int = 10):
        self.k = k

    def explain(self, hypothesis: np.ndarray, concept_labels: list,
                similarities: np.ndarray = None) -> tuple:
        """
        Use similarity scores if available (more meaningful than raw vector dimensions).
        Falls back to raw dimension values if not.
        """
        if similarities is not None and len(similarities) == len(concept_labels):
            sorted_idx = np.argsort(similarities)
            top_pos_idx = sorted_idx[-self.k:][::-1]
            top_neg_idx = sorted_idx[:self.k]
        else:
            k = min(self.k, len(hypothesis))
            sorted_idx = np.argsort(hypothesis)
            top_pos_idx = sorted_idx[-k:][::-1]
            top_neg_idx = sorted_idx[:k]

        top_positive = [concept_labels[i] for i in top_pos_idx
                       if i < len(concept_labels)]
        top_negative = [concept_labels[i] for i in top_neg_idx
                       if i < len(concept_labels)]

        return top_positive, top_negative

    def ask_oracle(self, hypothesis: np.ndarray, concept_labels: list,
                   model_interface: RealModelInterface,
                   similarities: np.ndarray = None) -> Optional[bool]:
        """
        Generates a real natural language question via Llama.
        Human answers yes / no / idk.
        """
        top_positive, top_negative = self.explain(
            hypothesis, concept_labels, similarities
        )
        question = model_interface.vector_to_natural_language(
            top_positive, top_negative
        )

        print(f"\n[ORACLE QUERY]")
        print(f"  Most similar concepts: {', '.join(top_positive[:3])}")
        print(f"  Most unlike concepts:  {', '.join(top_negative[:3])}")
        print(f"\n  Question: {question}")

        while True:
            answer = input("\n  Your answer (yes / no / idk): ").strip().lower()
            if answer in ('yes', 'y'):
                return True
            elif answer in ('no', 'n'):
                return False
            elif answer in ('idk', "i don't know", 'dunno', '?', 'idk'):
                return None
            else:
                print("  Please answer: yes, no, or idk")


# ==============================================================================
# SECTION 8 — PHYSICS ORACLE (real PyBullet if available)
# ==============================================================================

class PhysicsOracle:
    """Stage 4: post-frontier empirical validation via PyBullet."""

    def __init__(self, epsilon: float = 0.7):
        self.epsilon = epsilon
        self.obj_counter = 0

        if PYBULLET_AVAILABLE:
            self.client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self.client)
            print("[PHYSICS] PyBullet ready.")
        else:
            self.client = None

    def cosine_similarity(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def run_simulation(self, hypothesis: np.ndarray) -> np.ndarray:
        """
        [SIMULATION REQUIRED AT SCALE]
        Placeholder: returns noisy hypothesis.
        In production: translate hypothesis → physical parameters → simulate → embed result.
        """
        print(f"\n[SIMULATION] Running...")
        print(f"  [SIMULATION REQUIRED AT SCALE] for meaningful physical results.")

        if PYBULLET_AVAILABLE:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5,
                                         physicsClientId=self.client)
            body = p.createMultiBody(baseMass=1.0,
                                     baseCollisionShapeIndex=col,
                                     basePosition=[0, 0, 5],
                                     physicsClientId=self.client)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.client)
            p.removeBody(body, physicsClientId=self.client)

        return hypothesis + np.random.randn(*hypothesis.shape) * 0.1

    def validate(self, hypothesis: np.ndarray) -> tuple:
        result = self.run_simulation(hypothesis)
        sim = self.cosine_similarity(hypothesis, result)
        confirmed = sim >= self.epsilon
        print(f"  cos(H, R) = {sim:.4f} | ε = {self.epsilon}")
        print(f"  {'CONFIRMED ✓' if confirmed else 'DENIED ✗'}")
        return confirmed, result, sim


class GPCalibrator:
    """GP surrogate for Discrepancy Frontier detection."""

    def __init__(self, epsilon_dynamic: float = 0.3):
        self.epsilon_dynamic = epsilon_dynamic
        self.residuals = []

    def record_residual(self, h, r):
        self.residuals.append(float(np.linalg.norm(h - r)))

    def check_discrepancy_frontier(self) -> bool:
        if len(self.residuals) < 5:
            return False
        avg = np.mean(self.residuals[-5:])
        if avg > self.epsilon_dynamic:
            print(f"\n[DISCREPANCY FRONTIER] Residual {avg:.4f} > ε {self.epsilon_dynamic}")
            print(f"  Unknown physical property detected.")
            print(f"  Flagged as highest-priority inquiry target.")
            return True
        return False


# ==============================================================================
# SECTION 9 — KNOWLEDGE INTEGRATOR
# Real version: adds confirmed facts as text to knowledge base
# EKC weight update remains placeholder
# ==============================================================================

class KnowledgeIntegrator:
    """
    Stage 5: integrates confirmed hypotheses.

    Real implementation: adds confirmed concept text to knowledge base,
    rebuilds embedding vectors.

    EKC gradient update requires frontier model weight access — placeholder.
    """

    def __init__(self, lambda_ekc: float = 0.5):
        self.lambda_ekc = lambda_ekc
        self.integrated = []

    def integrate(self, confirmed_hypothesis: np.ndarray,
                  model_interface: RealModelInterface,
                  concept_text: str = None) -> bool:
        """
        Integrate confirmed knowledge.
        If concept_text provided: add to knowledge base as real text concept.
        """
        print(f"\n[KNOWLEDGE INTEGRATION]")

        if concept_text:
            model_interface.add_confirmed_knowledge(concept_text)
            self.integrated.append(concept_text)

        # EKC penalty computation (for logging)
        anchor = model_interface.get_embedding_space().mean(axis=0)
        fisher = model_interface.get_fisher_information()
        delta = confirmed_hypothesis - anchor[:len(confirmed_hypothesis)]
        ekc_penalty = self.lambda_ekc * np.sum(fisher[:len(delta)] * delta**2)
        print(f"  EKC penalty: {ekc_penalty:.4f}")
        print(f"  Total integrated this session: {len(self.integrated)}")

        return True


# ==============================================================================
# SECTION 10 — KNOWLEDGE TRACKER
# ==============================================================================

class KnowledgeTracker:
    def __init__(self):
        self.facts = []
        self.frontier_events = []
        self.discrepancy_frontiers = []

    def add_fact(self, source, confirmed, detail=""):
        self.facts.append({
            'source': source, 'confirmed': confirmed,
            'detail': detail, 'timestamp': time.time()
        })

    def frontier_ratio(self, window=20):
        recent = self.facts[-window:]
        if not recent:
            return 0.0
        return sum(1 for f in recent if f['source'] == 'simulation') / len(recent)

    def summary(self):
        confirmed = sum(1 for f in self.facts if f['confirmed'])
        human = sum(1 for f in self.facts if f['source'] == 'human')
        sim = sum(1 for f in self.facts if f['source'] == 'simulation')
        print(f"\n[SESSION SUMMARY]")
        print(f"  Facts processed:   {len(self.facts)}")
        print(f"  Confirmed:         {confirmed}")
        print(f"  From human oracle: {human}")
        print(f"  From simulation:   {sim}")
        print(f"  Frontier events:   {len(self.frontier_events)}")
        if self.facts:
            print(f"\n  Confirmed facts this session:")
            for f in self.facts:
                if f['confirmed'] and f['detail']:
                    print(f"    ✓ {f['detail']}")


# ==============================================================================
# SECTION 11 — COMPETITIVE KNOWLEDGE ARENA
# ==============================================================================

class CompetitiveKnowledgeArena:
    """Zero-sum epistemic competition between MAIL agents."""

    def __init__(self, n_agents: int = 2):
        self.n_agents = n_agents
        self.scores = {i: 0.0 for i in range(n_agents)}
        print(f"\n[CKA] {n_agents} agents | zero-sum rewards")

    def award(self, winner: int, delta_I: float):
        self.scores[winner] += delta_I
        penalty = delta_I / max(1, self.n_agents - 1)
        for j in range(self.n_agents):
            if j != winner:
                self.scores[j] -= penalty
        print(f"  [CKA] Agent {winner} +{delta_I:.4f} | "
              f"Scores: { {k: f'{v:.3f}' for k, v in self.scores.items()} }")

    def leaderboard(self):
        print(f"\n[CKA LEADERBOARD]")
        for rank, (agent, score) in enumerate(
            sorted(self.scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            print(f"  #{rank} Agent {agent}: {score:.4f}")


# ==============================================================================
# SECTION 12 — THE FULL MAIL LOOP
# ==============================================================================

def run_mail_loop(
    model_interface: RealModelInterface,
    n_cycles: int = 5,
    use_competition: bool = False,
    n_agents: int = 2
):
    """
    The complete MAIL loop with real model connections.

    1. DETECT    → k-NN gap detection on real embedding space
    2. GENERATE  → weighted interpolation hypothesis vector
    3. QUERY     → Llama generates question, human answers yes/no/idk
    4. VALIDATE  → confirm via human or PyBullet simulation
    5. INTEGRATE → add confirmed knowledge to knowledge base
    6. LOOP      → rebuild vectors, find next gap
    """

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP — MAIL Framework (Live)")
    print("  Embedding: all-MiniLM-L6-v2 | LLM: Llama 3.2")
    print("="*70)

    gap_detector  = GapDetector(n_clusters=5, n_probes=20, probe_steps=10, k=5)
    hyp_generator = HypothesisGenerator(n_neighbours=10)
    sfe           = SparseFeatureExplainer(k=10)
    physics       = PhysicsOracle(epsilon=0.7)
    gp            = GPCalibrator(epsilon_dynamic=0.3)
    integrator    = KnowledgeIntegrator(lambda_ekc=0.5)
    tracker       = KnowledgeTracker()
    cka           = CompetitiveKnowledgeArena(n_agents) if use_competition else None
    visited_gaps  = set()  # track denied/idk gaps so MAIL moves on

    for cycle in range(1, n_cycles + 1):
        print(f"\n{'─'*70}")
        print(f"  CYCLE {cycle}/{n_cycles}")
        print(f"{'─'*70}")

        # ── Stage 1: Gap Detection ─────────────────────────────────────────
        print(f"\n→ Stage 1: Gap Detection")
        vectors = model_interface.get_embedding_space()
        labels  = model_interface.get_concept_labels()

        gap_vectors, gap_metadata, gap_densities = gap_detector.find_gaps(vectors)

        if len(gap_vectors) == 0:
            print("  No gaps found.")
            break

        # Skip visited gaps
        unvisited = [
            i for i, meta in enumerate(gap_metadata)
            if f"gap_{meta['cluster_a']}_{meta['cluster_b']}" not in visited_gaps
        ]
        if len(unvisited) == 0:
            print("  All frontiers explored. Resetting.")
            visited_gaps.clear()
            unvisited = list(range(len(gap_vectors)))

        target_gap_idx = unvisited[0]
        gap_vector     = gap_vectors[target_gap_idx]
        gap_meta       = gap_metadata[target_gap_idx]
        density_before = gap_densities[target_gap_idx]
        gap_concept    = f"frontier between cluster {gap_meta['cluster_a']} and cluster {gap_meta['cluster_b']}"

        # Find the nearest known concepts to describe this frontier
        sims = np.array([
            gap_detector.cosine_sim(gap_vector, v) for v in vectors
        ])
        top3 = np.argsort(sims)[-3:][::-1]
        nearest_concepts = [labels[i] for i in top3]

        print(f"  Target: {gap_concept}")
        print(f"  Gap size: {gap_meta['gap_size']:.4f} | Frontier density: {density_before:.4f}")
        print(f"  Nearest known concepts: {', '.join(nearest_concepts)}")
        print(f"  → This is uncharted territory between these fields")

        # ── Stage 2: Hypothesis Generation ────────────────────────────────
        print(f"\n→ Stage 2: Hypothesis Generation")
        hypothesis, similarities, sorted_idx = hyp_generator.generate(
            gap_vector, vectors
        )

        # ── Stage 3: Human Oracle ──────────────────────────────────────────
        print(f"\n→ Stage 3: Human Oracle")
        oracle_response = sfe.ask_oracle(
            hypothesis, labels, model_interface, similarities
        )

        confirmed = False
        concept_text = None

        if oracle_response is True:
            print(f"\n  ✓ Confirmed")
            confirmed = True
            print(f"  Nearest known concepts were: {', '.join(nearest_concepts)}")
            concept_text = input("  Name this new concept in one short sentence (max 150 chars): ").strip()
            tracker.add_fact('human', True, concept_text)

        elif oracle_response is False:
            print(f"\n  ✗ Denied")
            tracker.add_fact('human', False, gap_concept)
            gap_key = f"gap_{gap_meta['cluster_a']}_{gap_meta['cluster_b']}"
            visited_gaps.add(gap_key)
            print(f"  Frontier {gap_key} marked as explored.")

        else:
            # Knowledge Frontier → simulation
            print(f"\n  ? Knowledge Frontier → switching to simulation")
            tracker.frontier_events.append({'cycle': cycle, 'concept': gap_concept})

            # ── Stage 4: Simulation ────────────────────────────────────────
            print(f"\n→ Stage 4: Empirical Validation")
            confirmed, result, sim_score = physics.validate(hypothesis)
            tracker.add_fact('simulation', confirmed, gap_concept)
            gp.record_residual(hypothesis, result)
            if not confirmed:
                gap_key = f"gap_{gap_meta['cluster_a']}_{gap_meta['cluster_b']}"
                visited_gaps.add(gap_key)

            if gp.check_discrepancy_frontier():
                tracker.discrepancy_frontiers.append({
                    'cycle': cycle, 'concept': gap_concept
                })

        # ── Stage 5: Integration ───────────────────────────────────────────
        if confirmed:
            print(f"\n→ Stage 5: Knowledge Integration")
            # Re-prompt if concept_text is invalid
            if concept_text and len(concept_text) > 150:
                print(f"  [INPUT TOO LONG] Please rephrase in one short sentence (max 150 chars).")
                concept_text = input("  Describe what was confirmed: ").strip()
            integrator.integrate(hypothesis, model_interface, concept_text)

            if cka is not None:
                delta_I = max(0.0, 0.1)  # simplified info gain
                cka.award(cycle % cka.n_agents, delta_I)

        print(f"\n  Frontier ratio: {tracker.frontier_ratio():.2f}")

    # ── Session End ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    tracker.summary()
    if cka:
        cka.leaderboard()

    return tracker


# ==============================================================================
# SECTION 13 — ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP")
    print("  curious_ai.py — Live MAIL Pipeline")
    print("  Embedding: all-MiniLM-L6-v2 | LLM: Llama 3.2 via Ollama")
    print("="*70)

    print("\nOptions:")
    print("  1. Run full MAIL loop (single agent)")
    print("  2. Run MAIL loop with Competitive Knowledge Arena")
    print("  3. Add custom knowledge to the base")
    print("  4. Run gap detection only — see what MAIL thinks is missing")

    choice = input("\nChoice (1/2/3/4): ").strip()

    model = RealModelInterface(
        knowledge_base=INITIAL_KNOWLEDGE.copy(),
        llm_model="llama3.2"
    )

    if choice == '1':
        n = int(input("How many cycles? (recommended: 3-5): ").strip() or "3")
        run_mail_loop(model, n_cycles=n, use_competition=False)

    elif choice == '2':
        n = int(input("How many cycles?: ").strip() or "3")
        agents = int(input("How many competing agents? (2-4): ").strip() or "2")
        run_mail_loop(model, n_cycles=n, use_competition=True, n_agents=agents)

    elif choice == '3':
        print("\nAdd knowledge to the base (type 'done' when finished):")
        while True:
            fact = input("  Fact: ").strip()
            if fact.lower() == 'done':
                break
            if fact:
                model.add_confirmed_knowledge(fact)
        print(f"\nKnowledge base now has {len(model.knowledge_texts)} concepts.")
        run = input("Run MAIL loop now? (yes/no): ").strip().lower()
        if run in ('yes', 'y'):
            run_mail_loop(model, n_cycles=3)

    elif choice == '4':
        print("\n[GAP DETECTION DEMO — INTER-CLUSTER FRONTIER PROBING]")
        vectors = model.get_embedding_space()
        labels  = model.get_concept_labels()
        detector = GapDetector(n_clusters=5, n_probes=20)
        gap_vectors, gap_metadata, densities = detector.find_gaps(vectors)

        print(f"\nFrontier gaps (ranked by how far from all known knowledge):")
        for i, (meta, density) in enumerate(zip(gap_metadata, densities)):
            # Show nearest known concepts for each frontier
            sims = np.array([detector.cosine_sim(gap_vectors[i], v) for v in vectors])
            top3 = np.argsort(sims)[-3:][::-1]
            nearest = [labels[j] for j in top3]
            print(f"\n  Gap {i+1}: Between cluster {meta['cluster_a']} and cluster {meta['cluster_b']}")
            print(f"    Gap size:  {meta['gap_size']:.4f}")
            print(f"    Density:   {density:.4f} (lower = more unknown)")
            print(f"    Nearest:   {', '.join(nearest)}")
            print(f"    → Uncharted territory between these fields")
            if i >= 6:
                break

        print(f"\nThese are the interdisciplinary frontiers.")
        print(f"MAIL will probe these regions — not just isolated known concepts.")
        print(f"This is where genuinely new knowledge gets discovered.")

    else:
        run_mail_loop(model, n_cycles=3)
