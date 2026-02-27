"""
curious_ai.py — The Initiative Gap: MAIL Framework Implementation
=================================================================
Meta-cognitive Autonomous Inquiry Loop (MAIL)

Full architectural implementation. Placeholders marked with:
    # [FRONTIER MODEL REQUIRED] — needs access to a model's weight/embedding architecture
    # [SIMULATION REQUIRED]     — needs a physics simulation environment at scale

Stage 1: Gap Detection          — k-NN density estimation over embedding space
Stage 2: Hypothesis Generation  — similarity-weighted vector interpolation
Stage 3: Sparse Feature Expl.   — top-10 / bottom-10 human interface
Stage 4: Empirical Validation   — simulation + Gaussian Process calibration
Stage 5: EKC Integration        — elastic weight consolidation

Author: [your name]
Date:   2026
Repo:   https://github.com/[your-username]/initiative-gap
"""

import numpy as np
from typing import Optional
import json
import time

# ── Optional imports (graceful fallback if not installed) ─────────────────────

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. Install with: pip install scikit-learn")

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[WARNING] PyBullet not installed. Install with: pip install pybullet")


# ==============================================================================
# SECTION 1 — FRONTIER MODEL INTERFACE
# Replace these placeholder functions with real calls to the model's internals.
# ==============================================================================

class FrontierModelInterface:
    """
    [FRONTIER MODEL REQUIRED]

    This class is the bridge between the MAIL loop and the underlying AI model.
    In production, this requires direct access to:
        - The model's embedding layer (to read and write vectors)
        - The attention weight matrices (for EKC Fisher Information computation)
        - An inference endpoint (to generate SFE queries in natural language)
        - An online weight update mechanism (for EKC integration)

    Replace each method below with real calls to the model's architecture.
    """

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        print(f"[FRONTIER MODEL REQUIRED] FrontierModelInterface initialised.")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Replace this class with real model access.\n")

    def get_embedding_space(self) -> np.ndarray:
        """
        [FRONTIER MODEL REQUIRED]
        Returns the full embedding matrix of the model — every concept vector.
        Shape: (num_concepts, embedding_dim)

        In production:
            return model.get_input_embeddings().weight.detach().cpu().numpy()
        """
        num_concepts = 500
        print(f"[PLACEHOLDER] Returning {num_concepts} random {self.embedding_dim}-dim vectors.")
        print(f"  In production: return model.get_input_embeddings().weight.detach().cpu().numpy()")
        return np.random.randn(num_concepts, self.embedding_dim).astype(np.float32)

    def get_concept_labels(self) -> list:
        """
        [FRONTIER MODEL REQUIRED]
        Returns human-readable labels for each dimension of the embedding space.
        Used by Sparse Feature Explanation to translate vector dimensions into words.

        In production:
            return tokenizer.convert_ids_to_tokens(range(vocab_size))
        """
        print(f"[PLACEHOLDER] Returning generic dimension labels.")
        return [f"concept_{i}" for i in range(self.embedding_dim)]

    def vector_to_natural_language(self, top_positive: list, top_negative: list) -> str:
        """
        [FRONTIER MODEL REQUIRED]
        Given the top-10 and bottom-10 concept labels, generate a natural language
        question for the human oracle.

        In production: call the model's language generation with a structured prompt
        describing the top and bottom concepts.
        """
        pos_str = ", ".join(top_positive[:5])
        neg_str = ", ".join(top_negative[:5])
        return (
            f"I think this concept is strongly related to: {pos_str}\n"
            f"And strongly unlike: {neg_str}\n"
            f"Is this approximately correct?"
        )

    def get_fisher_information(self) -> np.ndarray:
        """
        [FRONTIER MODEL REQUIRED]
        Returns the Fisher Information for each weight in the model.
        Used by EKC to determine which weights are important to preserve.
        Shape: same as model weight vector

        In production:
            Compute diagonal Fisher Information via:
            F_i = E[(d/dθ_i log p(y|x,θ))²]
            over a sample of the existing training data.
        """
        print(f"[PLACEHOLDER] Returning uniform Fisher Information.")
        print(f"  In production: compute diagonal Fisher Information Matrix over existing data.")
        return np.ones(self.embedding_dim, dtype=np.float32)

    def update_weights_ekc(self, confirmed_vector: np.ndarray,
                            anchor_weights: np.ndarray,
                            fisher_info: np.ndarray,
                            lambda_ekc: float = 0.5) -> bool:
        """
        [FRONTIER MODEL REQUIRED]
        Integrates a confirmed hypothesis vector into the model's weights
        using Elastic Knowledge Consolidation.

        Loss: L_EKC = L_new(θ) + λ × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²

        In production:
            1. Compute L_new as cross-entropy on the confirmed vector
            2. Add EKC regularisation term
            3. Run one gradient step
            4. Verify no catastrophic forgetting on held-out test set
        """
        print(f"[PLACEHOLDER] EKC weight update.")
        print(f"  Confirmed vector norm: {np.linalg.norm(confirmed_vector):.4f}")
        print(f"  Lambda EKC: {lambda_ekc}")
        print(f"  In production: gradient step with L_EKC = L_new + λ × Σ Fᵢ(θᵢ - θ*ᵢ)²")
        return True


# ==============================================================================
# SECTION 2 — STAGE 1: GAP DETECTION
# ==============================================================================

class GapDetector:
    """
    Stage 1 of MAIL: finds sparse regions in the embedding space.

    Uses k-Nearest Neighbour density estimation:
        ρ(x) = 1 / d_k(x)
    where d_k(x) is the average cosine distance to the k nearest neighbours.

    Low-density regions = knowledge gaps = targets for inquiry.
    """

    def __init__(self, k: int = 10, gap_percentile: float = 10.0):
        self.k = k
        self.gap_percentile = gap_percentile

    def compute_density(self, vectors: np.ndarray) -> np.ndarray:
        """Compute local density for each vector."""
        n = len(vectors)
        k = min(self.k, n - 1)

        if SKLEARN_AVAILABLE:
            nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
            nn.fit(vectors)
            distances, _ = nn.kneighbors(vectors)
            avg_distances = distances[:, 1:].mean(axis=1)
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-8, norms)
            normalised = vectors / norms
            similarity = normalised @ normalised.T
            dist_matrix = 1.0 - similarity
            avg_distances = np.zeros(n)
            for i in range(n):
                row = dist_matrix[i].copy()
                row[i] = np.inf
                avg_distances[i] = np.sort(row)[:k].mean()

        return 1.0 / (avg_distances + 1e-8)

    def find_gaps(self, vectors: np.ndarray) -> tuple:
        """Returns indices of vectors in gap regions (lowest density)."""
        density = self.compute_density(vectors)
        threshold = np.percentile(density, self.gap_percentile)
        gap_indices = np.where(density <= threshold)[0]

        print(f"\n[GAP DETECTION]")
        print(f"  Total vectors: {len(vectors)}")
        print(f"  Density threshold (bottom {self.gap_percentile}%): {threshold:.4f}")
        print(f"  Gaps found: {len(gap_indices)}")

        return gap_indices, density


# ==============================================================================
# SECTION 3 — STAGE 2: HYPOTHESIS VECTOR GENERATION
# ==============================================================================

class HypothesisGenerator:
    """
    Stage 2 of MAIL: generates a hypothesis vector for a gap region.

    Similarity-weighted vector interpolation:
        wᵢ = cos(vᵢ, G_direction)
        wᵢ = wᵢ / Σⱼ |wⱼ|
        H  = Σᵢ wᵢ × vᵢ

    Vectors similar to the gap pull H toward them.
    Vectors opposite to the gap push H away.
    """

    def __init__(self, n_neighbours: int = 20):
        self.n_neighbours = n_neighbours

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def generate(self, gap_vector: np.ndarray, all_vectors: np.ndarray) -> tuple:
        """Generate a hypothesis vector for a gap."""
        n = min(self.n_neighbours, len(all_vectors))

        similarities = np.array([
            self.cosine_similarity(gap_vector, v) for v in all_vectors
        ])

        sorted_indices = np.argsort(similarities)
        top_positive_idx = sorted_indices[-n:][::-1]
        top_negative_idx = sorted_indices[:n]

        weights = np.zeros(len(all_vectors))
        for idx in top_positive_idx:
            weights[idx] = similarities[idx]
        for idx in top_negative_idx:
            weights[idx] = similarities[idx]

        weight_sum = np.sum(np.abs(weights))
        if weight_sum > 0:
            weights = weights / weight_sum

        hypothesis = np.zeros_like(gap_vector)
        for i, w in enumerate(weights):
            if w != 0:
                hypothesis += w * all_vectors[i]

        print(f"\n[HYPOTHESIS GENERATION]")
        print(f"  Gap vector norm: {np.linalg.norm(gap_vector):.4f}")
        print(f"  Hypothesis vector norm: {np.linalg.norm(hypothesis):.4f}")
        print(f"  cos(gap, hypothesis): {self.cosine_similarity(gap_vector, hypothesis):.4f}")

        return hypothesis, similarities, sorted_indices


# ==============================================================================
# SECTION 4 — STAGE 3: SPARSE FEATURE EXPLANATION
# ==============================================================================

class SparseFeatureExplainer:
    """
    Stage 3 of MAIL: translates hypothesis vector into a human-readable query.

    Extracts top-k and bottom-k dimensions.
    Human answers yes / no / idk.
    idk = Knowledge Frontier detected → trigger simulation.
    """

    def __init__(self, k: int = 10):
        self.k = k

    def explain(self, hypothesis: np.ndarray, concept_labels: list) -> tuple:
        k = min(self.k, len(hypothesis))
        sorted_idx = np.argsort(hypothesis)
        top_positive_idx = sorted_idx[-k:][::-1]
        top_negative_idx = sorted_idx[:k]
        top_positive = [concept_labels[i] for i in top_positive_idx if i < len(concept_labels)]
        top_negative = [concept_labels[i] for i in top_negative_idx if i < len(concept_labels)]
        return top_positive, top_negative

    def ask_oracle(self, hypothesis: np.ndarray, concept_labels: list,
                   model_interface: FrontierModelInterface) -> Optional[bool]:
        """
        Returns: True (yes), False (no), None (idk → frontier)
        """
        top_positive, top_negative = self.explain(hypothesis, concept_labels)
        query = model_interface.vector_to_natural_language(top_positive, top_negative)

        print(f"\n[HUMAN ORACLE QUERY]")
        print(f"  {query}")
        print(f"\n  Most defining:     {', '.join(top_positive[:5])}")
        print(f"  Most contradicting: {', '.join(top_negative[:5])}")

        while True:
            answer = input("\n  Your answer (yes / no / idk): ").strip().lower()
            if answer in ('yes', 'y'):
                return True
            elif answer in ('no', 'n'):
                return False
            elif answer in ('idk', "i don't know", 'i dont know', 'dunno', '?'):
                return None
            else:
                print("  Please answer: yes, no, or idk")


# ==============================================================================
# SECTION 5 — STAGE 4: EMPIRICAL VALIDATION
# ==============================================================================

class PhysicsOracle:
    """
    Stage 4 of MAIL: post-frontier empirical validation via simulation.

    When oracle returns idk, system runs a physics simulation.
    Compares actual result R to hypothesis H: cos(H, R) > ε

    [SIMULATION REQUIRED] for meaningful results at scale.
    """

    def __init__(self, epsilon: float = 0.7):
        self.epsilon = epsilon
        self.obj_counter = 0

        if PYBULLET_AVAILABLE:
            self.client = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self.client)
            print("[PHYSICS] PyBullet initialised.")
        else:
            self.client = None

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def run_simulation(self, hypothesis: np.ndarray) -> np.ndarray:
        """
        [SIMULATION REQUIRED]
        Translate hypothesis → physical parameters → run simulation → embed result.

        In production:
            1. Map hypothesis dimensions to physical quantities
               (mass, force, temperature, viscosity, conductivity...)
            2. Build the simulation environment
            3. Run for N timesteps
            4. Extract measurements and embed as a result vector

        Placeholder: returns noisy hypothesis demonstrating the comparison mechanism.
        """
        print(f"\n[SIMULATION REQUIRED]")
        print(f"  Translate hypothesis vector → physical parameters → run sim → embed result")
        print(f"  Placeholder: noisy hypothesis as simulated result")

        if PYBULLET_AVAILABLE:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5,
                                         physicsClientId=self.client)
            self.obj_counter += 1
            body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col,
                                     basePosition=[0, 0, 5],
                                     physicsClientId=self.client)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.client)
            p.removeBody(body, physicsClientId=self.client)

        return hypothesis + np.random.randn(*hypothesis.shape) * 0.1

    def validate(self, hypothesis: np.ndarray) -> tuple:
        """Run simulation and compare to hypothesis."""
        result = self.run_simulation(hypothesis)
        similarity = self.cosine_similarity(hypothesis, result)
        confirmed = similarity >= self.epsilon

        print(f"  cos(H, R) = {similarity:.4f} | ε = {self.epsilon}")
        print(f"  {'CONFIRMED ✓' if confirmed else 'DENIED ✗'}")

        return confirmed, result, similarity


class GPCalibrator:
    """
    Gaussian Process surrogate for simulator calibration.
    Detects Discrepancy Frontier when no θ resolves the residual.

    [FRONTIER MODEL REQUIRED] — use GPyTorch or GPflow in production.
    δ(θ) ~ GP(μ(θ), k(θ, θ'))
    θ* = argmin_θ E[|δ(θ)|²]
    Discrepancy Frontier: min_θ E[|δ(θ)|²] > ε_dynamic
    """

    def __init__(self, epsilon_dynamic: float = 0.3):
        self.epsilon_dynamic = epsilon_dynamic
        self.residuals = []

    def record_residual(self, hypothesis: np.ndarray, result: np.ndarray):
        self.residuals.append(float(np.linalg.norm(hypothesis - result)))

    def check_discrepancy_frontier(self) -> bool:
        """
        [FRONTIER MODEL REQUIRED] — full GP implementation in production.
        Placeholder: flag frontier if recent residuals consistently high.
        """
        if len(self.residuals) < 5:
            return False
        avg = np.mean(self.residuals[-5:])
        if avg > self.epsilon_dynamic:
            print(f"\n[DISCREPANCY FRONTIER]")
            print(f"  Residual {avg:.4f} > ε_dynamic {self.epsilon_dynamic}")
            print(f"  No known calibration parameters resolve this.")
            print(f"  → Unknown physical property or force detected.")
            print(f"  → Flagged as highest-priority inquiry target.")
            return True
        return False


# ==============================================================================
# SECTION 6 — STAGE 5: ELASTIC KNOWLEDGE CONSOLIDATION
# ==============================================================================

class ElasticKnowledgeConsolidator:
    """
    Stage 5 of MAIL: integrates confirmed hypotheses without catastrophic forgetting.

    L_EKC = L_new(θ) + λ × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²

    Fᵢ = Fisher Information (weight importance)
    θ*ᵢ = anchor weights before integration
    λ   = consolidation strength

    [FRONTIER MODEL REQUIRED] for real weight updates.
    """

    def __init__(self, lambda_ekc: float = 0.5):
        self.lambda_ekc = lambda_ekc
        self.integrated_count = 0

    def integrate(self, confirmed_hypothesis: np.ndarray,
                  model_interface: FrontierModelInterface) -> bool:
        print(f"\n[EKC INTEGRATION]")
        anchor_weights = model_interface.get_embedding_space().mean(axis=0)
        fisher_info = model_interface.get_fisher_information()
        delta = confirmed_hypothesis - anchor_weights[:len(confirmed_hypothesis)]
        ekc_penalty = self.lambda_ekc * np.sum(fisher_info[:len(delta)] * delta**2)
        print(f"  EKC penalty: {ekc_penalty:.4f} | λ = {self.lambda_ekc}")
        success = model_interface.update_weights_ekc(
            confirmed_hypothesis, anchor_weights, fisher_info, self.lambda_ekc
        )
        if success:
            self.integrated_count += 1
            print(f"  Total integrated: {self.integrated_count}")
        return success


# ==============================================================================
# SECTION 7 — KNOWLEDGE TRACKER
# ==============================================================================

class KnowledgeTracker:
    """Tracks all knowledge acquired through the MAIL loop."""

    def __init__(self):
        self.facts = []
        self.frontier_events = []
        self.discrepancy_frontiers = []

    def add_human_fact(self, hypothesis: np.ndarray, confirmed: bool, query: str):
        self.facts.append({
            'source': 'human_oracle', 'confirmed': confirmed,
            'query': query, 'timestamp': time.time(),
            'vector_norm': float(np.linalg.norm(hypothesis))
        })

    def add_simulation_fact(self, hypothesis: np.ndarray,
                             confirmed: bool, similarity: float):
        self.facts.append({
            'source': 'simulation', 'confirmed': confirmed,
            'similarity': similarity, 'timestamp': time.time(),
            'vector_norm': float(np.linalg.norm(hypothesis))
        })

    def add_frontier_event(self, gap_idx: int):
        self.frontier_events.append({
            'gap_idx': gap_idx, 'timestamp': time.time(),
            'facts_at_time': len(self.facts)
        })

    def frontier_ratio(self, window: int = 20) -> float:
        recent = self.facts[-window:]
        if not recent:
            return 0.0
        return sum(1 for f in recent if f['source'] == 'simulation') / len(recent)

    def summary(self):
        confirmed = sum(1 for f in self.facts if f['confirmed'])
        human = sum(1 for f in self.facts if f['source'] == 'human_oracle')
        sim = sum(1 for f in self.facts if f['source'] == 'simulation')
        print(f"\n[KNOWLEDGE SUMMARY]")
        print(f"  Total facts processed:  {len(self.facts)}")
        print(f"  Confirmed: {confirmed} | Denied: {len(self.facts) - confirmed}")
        print(f"  From human oracle: {human} | From simulation: {sim}")
        print(f"  Frontier events: {len(self.frontier_events)}")
        print(f"  Discrepancy frontiers: {len(self.discrepancy_frontiers)}")


# ==============================================================================
# SECTION 8 — COMPETITIVE KNOWLEDGE ARENA
# ==============================================================================

class CompetitiveKnowledgeArena:
    """
    Multi-agent zero-sum epistemic competition.

    When agent i confirms a hypothesis:
        R_i = +ΔI_i
        R_j = -ΔI_i / (n-1)   for all j ≠ i

    Total reward across all agents = 0 at every timestep.

    [FRONTIER MODEL REQUIRED] — each agent needs its own model instance.
    """

    def __init__(self, n_agents: int = 4, embedding_dim: int = 768):
        self.n_agents = n_agents
        self.embedding_dim = embedding_dim
        self.scores = {i: 0.0 for i in range(n_agents)}
        self.knowledge_events = []
        print(f"\n[COMPETITIVE KNOWLEDGE ARENA]")
        print(f"  Agents: {n_agents} | Reward: zero-sum")
        print(f"  [FRONTIER MODEL REQUIRED] — each agent needs its own model instance\n")

    def compute_information_gain(self, density_before: float,
                                  density_after: float) -> float:
        return max(0.0, density_after - density_before)

    def award_scores(self, winning_agent: int, delta_I: float):
        self.scores[winning_agent] += delta_I
        penalty = delta_I / max(1, self.n_agents - 1)
        for j in range(self.n_agents):
            if j != winning_agent:
                self.scores[j] -= penalty
        self.knowledge_events.append({
            'agent': winning_agent, 'delta_I': delta_I,
            'scores_after': dict(self.scores), 'timestamp': time.time()
        })
        print(f"\n[CKA] Agent {winning_agent} confirmed (+{delta_I:.4f})")
        print(f"  Scores: { {k: f'{v:.3f}' for k, v in self.scores.items()} }")

    def leaderboard(self):
        print(f"\n[CKA LEADERBOARD]")
        for rank, (agent, score) in enumerate(
            sorted(self.scores.items(), key=lambda x: x[1], reverse=True), 1
        ):
            print(f"  #{rank} Agent {agent}: {score:.4f}")


# ==============================================================================
# SECTION 9 — THE FULL MAIL LOOP
# ==============================================================================

def run_mail_loop(model_interface: FrontierModelInterface,
                  n_cycles: int = 10,
                  frontier_threshold: float = 0.4,
                  use_competition: bool = False,
                  n_agents: int = 4):
    """
    The complete Meta-cognitive Autonomous Inquiry Loop.

    1. DETECT    → find gaps in embedding space
    2. GENERATE  → compute hypothesis vector
    3. QUERY     → ask human oracle via SFE
    4. VALIDATE  → confirm via human or simulation
    5. INTEGRATE → EKC weight update
    6. LOOP      → return to 1
    """

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP — MAIL Framework")
    print("  Meta-cognitive Autonomous Inquiry Loop")
    print("="*70)

    gap_detector   = GapDetector(k=10, gap_percentile=10.0)
    hyp_generator  = HypothesisGenerator(n_neighbours=20)
    sfe            = SparseFeatureExplainer(k=10)
    physics_oracle = PhysicsOracle(epsilon=0.7)
    gp_calibrator  = GPCalibrator(epsilon_dynamic=0.3)
    ekc            = ElasticKnowledgeConsolidator(lambda_ekc=0.5)
    tracker        = KnowledgeTracker()
    cka            = CompetitiveKnowledgeArena(n_agents, model_interface.embedding_dim) \
                     if use_competition else None
    concept_labels = model_interface.get_concept_labels()

    for cycle in range(1, n_cycles + 1):
        print(f"\n{'─'*70}")
        print(f"  MAIL CYCLE {cycle}/{n_cycles}")
        print(f"{'─'*70}")

        # Stage 1: Gap Detection
        print(f"\n→ Stage 1: Gap Detection")
        embedding_space = model_interface.get_embedding_space()
        gap_indices, density_scores = gap_detector.find_gaps(embedding_space)

        if len(gap_indices) == 0:
            print("  No gaps found.")
            break

        target_gap_idx = gap_indices[np.argmin(density_scores[gap_indices])]
        gap_vector = embedding_space[target_gap_idx]
        density_before = density_scores[target_gap_idx]

        # Stage 2: Hypothesis Generation
        print(f"\n→ Stage 2: Hypothesis Generation")
        hypothesis, similarities, sorted_indices = hyp_generator.generate(
            gap_vector, embedding_space
        )

        # Stage 3: Human Oracle
        print(f"\n→ Stage 3: Human Oracle Query")
        oracle_response = sfe.ask_oracle(hypothesis, concept_labels, model_interface)

        if oracle_response is True:
            print(f"\n  ✓ Confirmed by human oracle")
            tracker.add_human_fact(hypothesis, True, "confirmed")
            confirmed = True
            similarity = 1.0

        elif oracle_response is False:
            print(f"\n  ✗ Denied by human oracle")
            tracker.add_human_fact(hypothesis, False, "denied")
            confirmed = False
            similarity = 0.0

        else:
            # Knowledge Frontier → simulation
            print(f"\n  ? Knowledge Frontier detected → switching to simulation")
            tracker.add_frontier_event(target_gap_idx)

            # Stage 4: Simulation Validation
            print(f"\n→ Stage 4: Empirical Validation")
            confirmed, result_vector, similarity = physics_oracle.validate(hypothesis)
            tracker.add_simulation_fact(hypothesis, confirmed, similarity)

            gp_calibrator.record_residual(hypothesis, result_vector)
            if gp_calibrator.check_discrepancy_frontier():
                tracker.discrepancy_frontiers.append({
                    'cycle': cycle, 'gap_idx': target_gap_idx,
                    'timestamp': time.time()
                })

        # Stage 5: EKC Integration
        if confirmed:
            print(f"\n→ Stage 5: EKC Weight Integration")
            success = ekc.integrate(hypothesis, model_interface)

            if success and cka is not None:
                density_after = density_before * 1.1  # [SIMULATION REQUIRED] real density
                delta_I = cka.compute_information_gain(density_before, density_after)
                active_agent = cycle % cka.n_agents
                cka.award_scores(active_agent, delta_I)

        print(f"\n  Cycle {cycle} done | "
              f"Frontier ratio: {tracker.frontier_ratio():.2f} | "
              f"Threshold: {frontier_threshold}")

    print(f"\n{'='*70}")
    print(f"  MAIL LOOP COMPLETE")
    print(f"{'='*70}")
    tracker.summary()
    if cka is not None:
        cka.leaderboard()

    return tracker


# ==============================================================================
# SECTION 10 — ENTRY POINT
# ==============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("  THE INITIATIVE GAP")
    print("  curious_ai.py — MAIL Framework Architecture")
    print("="*70)
    print("\nThis file demonstrates the full MAIL architecture.")
    print("Placeholders marked [FRONTIER MODEL REQUIRED] need real model access.")
    print("See Section 15 of initiative_gap.docx for the experimental programme.\n")

    print("Options:")
    print("  1. Run MAIL loop (single agent)")
    print("  2. Run MAIL loop with Competitive Knowledge Arena")
    print("  3. Stage 1 demo — gap detection only")
    print("  4. Stage 2 demo — hypothesis generation only")

    choice = input("\nChoice (1/2/3/4): ").strip()

    # [FRONTIER MODEL REQUIRED] — replace embedding_dim with real model dimension
    model = FrontierModelInterface(embedding_dim=768)

    if choice == '1':
        run_mail_loop(model, n_cycles=5, frontier_threshold=0.4,
                      use_competition=False)

    elif choice == '2':
        run_mail_loop(model, n_cycles=5, frontier_threshold=0.4,
                      use_competition=True, n_agents=4)

    elif choice == '3':
        print("\n[STAGE 1 DEMO — Gap Detection]")
        vectors = model.get_embedding_space()
        detector = GapDetector(k=10, gap_percentile=10.0)
        gap_indices, density = detector.find_gaps(vectors)
        print(f"\nTop 5 gaps (lowest density):")
        top_gaps = gap_indices[np.argsort(density[gap_indices])[:5]]
        for i, idx in enumerate(top_gaps):
            print(f"  Gap {i+1}: index {idx}, density {density[idx]:.4f}")

    elif choice == '4':
        print("\n[STAGE 2 DEMO — Hypothesis Generation]")
        vectors = model.get_embedding_space()
        detector = GapDetector(k=10, gap_percentile=10.0)
        gap_indices, density = detector.find_gaps(vectors)
        gap_vector = vectors[gap_indices[0]]
        generator = HypothesisGenerator(n_neighbours=20)
        hypothesis, _, _ = generator.generate(gap_vector, vectors)
        explainer = SparseFeatureExplainer(k=10)
        labels = model.get_concept_labels()
        top_pos, top_neg = explainer.explain(hypothesis, labels)
        print(f"\nHypothesis most similar to: {', '.join(top_pos[:5])}")
        print(f"Hypothesis most unlike:     {', '.join(top_neg[:5])}")

    else:
        run_mail_loop(model, n_cycles=3)
