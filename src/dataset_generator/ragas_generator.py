"""RAGAS-based Dataset Generator implementation."""

import json
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
import random
import re
from collections import Counter
import numpy as np

from .interfaces import DatasetGenerator, EvaluationDataset, QualityReport


@dataclass
class KnowledgeGraphNode:
    """Represents a node in a knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]
    relationships: List[Dict[str, Any]]


@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation."""
    domain: str
    size: int
    diversity_threshold: float = 0.7
    quality_threshold: float = 0.8
    question_types: List[str] = None
    
    def __post_init__(self):
        if self.question_types is None:
            self.question_types = ["factual", "analytical", "comparative", "inferential"]


class RAGASDatasetGenerator(DatasetGenerator):
    """RAGAS-based implementation of DatasetGenerator."""
    
    def __init__(self, vector_store=None):
        """Initialize the RAGAS dataset generator.
        
        Args:
            vector_store: Optional VectorStore instance for storing embeddings
        """
        self.supported_formats = {"json", "csv", "jsonl", "parquet"}
        self.quality_validators = {
            "completeness": self._validate_completeness,
            "diversity": self._validate_diversity,
            "consistency": self._validate_consistency,
            "relevance": self._validate_relevance
        }
        self.vector_store = vector_store
    
    def store_dataset_vectors(self, dataset: EvaluationDataset, collection_name: str = None) -> Dict[str, Any]:
        """Store dataset embeddings in vector store.
        
        Args:
            dataset: Dataset to store vectors for
            collection_name: Name of collection (defaults to dataset_id)
            
        Returns:
            Dictionary with storage results
        """
        if not self.vector_store:
            return {
                "success": False,
                "error": "No vector store configured"
            }
        
        collection_name = collection_name or f"dataset_{dataset.dataset_id}"
        
        try:
            # Create simple embeddings for demo (in production, use proper embedding model)
            vectors = []
            metadata = []
            
            for i, (question, contexts, answer) in enumerate(zip(
                dataset.questions, dataset.contexts, dataset.ground_truth_answers
            )):
                # Simple embedding: convert text to vector (demo purposes)
                text = f"{question} {' '.join(contexts)} {answer}"
                # Create a simple hash-based vector (in production, use proper embeddings)
                vector = self._text_to_vector(text)
                vectors.append(vector)
                
                metadata.append({
                    "question": question,
                    "contexts": contexts,
                    "answer": answer,
                    "dataset_id": dataset.dataset_id,
                    "index": i
                })
            
            # Store in vector database
            from ..vector_store.interfaces import VectorConfig
            
            # Create collection if it doesn't exist
            try:
                vector_config = VectorConfig(
                    dimension=384,  # Match our simple embedding dimension
                    distance_metric="cosine",
                    index_type="hnsw"
                )
                self.vector_store.create_collection(collection_name, vector_config)
            except Exception:
                pass  # Collection might already exist
            
            # Store vectors
            storage_result = self.vector_store.store_embeddings(
                collection_name=collection_name,
                vectors=vectors,
                metadata=metadata
            )
            
            return {
                "success": storage_result.success,
                "collection_name": collection_name,
                "stored_count": storage_result.stored_count,
                "error": storage_result.error_message
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _text_to_vector(self, text: str, dimension: int = 384) -> np.ndarray:
        """Convert text to vector for demo purposes.
        
        In production, this would use a proper embedding model like:
        - sentence-transformers
        - OpenAI embeddings
        - Cohere embeddings
        
        Args:
            text: Text to convert
            dimension: Vector dimension
            
        Returns:
            Numpy array representing the text
        """
        # Simple hash-based embedding for demo
        import hashlib
        
        # Create deterministic vector from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers and normalize
        vector = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            vector.append(int(hex_pair, 16) / 255.0)
        
        # Pad or truncate to desired dimension
        while len(vector) < dimension:
            vector.extend(vector[:min(len(vector), dimension - len(vector))])
        
        vector = vector[:dimension]
        
        # Normalize to unit vector
        vector = np.array(vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def generate_from_knowledge_graph(self, graph_data: Dict[str, Any]) -> EvaluationDataset:
        """Generate evaluation dataset from knowledge graph.
        
        Args:
            graph_data: Dictionary containing nodes, edges, and metadata
            
        Returns:
            EvaluationDataset with questions, contexts, and ground truth answers
        """
        if not graph_data or "nodes" not in graph_data:
            raise ValueError("Invalid knowledge graph data: missing 'nodes' key")
        
        nodes = [KnowledgeGraphNode(**node) for node in graph_data.get("nodes", [])]
        edges = graph_data.get("edges", [])
        
        questions = []
        contexts = []
        ground_truth_answers = []
        
        # Generate questions from nodes
        for node in nodes:
            node_questions, node_contexts, node_answers = self._generate_from_node(node, nodes, edges)
            questions.extend(node_questions)
            contexts.extend(node_contexts)
            ground_truth_answers.extend(node_answers)
        
        # Generate relationship-based questions
        rel_questions, rel_contexts, rel_answers = self._generate_from_relationships(nodes, edges)
        questions.extend(rel_questions)
        contexts.extend(rel_contexts)
        ground_truth_answers.extend(rel_answers)
        
        dataset_id = str(uuid.uuid4())
        metadata = {
            "source": "knowledge_graph",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "generation_method": "ragas_transform"
        }
        
        # Calculate initial quality score
        temp_dataset = EvaluationDataset(
            dataset_id=dataset_id,
            questions=questions,
            contexts=contexts,
            ground_truth_answers=ground_truth_answers,
            metadata=metadata,
            quality_score=0.0
        )
        
        quality_report = self.validate_dataset_quality(temp_dataset)
        temp_dataset.quality_score = quality_report.overall_score
        
        return temp_dataset
    
    def create_synthetic_dataset(self, domain: str, size: int) -> EvaluationDataset:
        """Create synthetic evaluation dataset.
        
        Args:
            domain: Domain for dataset generation
            size: Number of question-answer pairs to generate
            
        Returns:
            EvaluationDataset with synthetic data
        """
        config = SyntheticDatasetConfig(domain=domain, size=size)
        
        questions = []
        contexts = []
        ground_truth_answers = []
        
        # Generate diverse question types
        questions_per_type = size // len(config.question_types)
        remainder = size % len(config.question_types)
        
        for i, question_type in enumerate(config.question_types):
            count = questions_per_type + (1 if i < remainder else 0)
            type_questions, type_contexts, type_answers = self._generate_synthetic_by_type(
                domain, question_type, count
            )
            questions.extend(type_questions)
            contexts.extend(type_contexts)
            ground_truth_answers.extend(type_answers)
        
        dataset_id = str(uuid.uuid4())
        metadata = {
            "source": "synthetic",
            "domain": domain,
            "generation_method": "ragas_synthetic",
            "diversity_threshold": config.diversity_threshold,
            "question_types": config.question_types
        }
        
        # Calculate quality score
        temp_dataset = EvaluationDataset(
            dataset_id=dataset_id,
            questions=questions,
            contexts=contexts,
            ground_truth_answers=ground_truth_answers,
            metadata=metadata,
            quality_score=0.0
        )
        
        quality_report = self.validate_dataset_quality(temp_dataset)
        temp_dataset.quality_score = quality_report.overall_score
        
        return temp_dataset
    
    def validate_dataset_quality(self, dataset: EvaluationDataset) -> QualityReport:
        """Validate the quality of a dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            QualityReport with quality metrics and recommendations
        """
        quality_metrics = {}
        issues_found = []
        recommendations = []
        
        # Run all quality validators
        for validator_name, validator_func in self.quality_validators.items():
            try:
                score, validator_issues, validator_recommendations = validator_func(dataset)
                quality_metrics[validator_name] = score
                issues_found.extend(validator_issues)
                recommendations.extend(validator_recommendations)
            except Exception as e:
                quality_metrics[validator_name] = 0.0
                issues_found.append(f"Validator {validator_name} failed: {str(e)}")
        
        # Calculate overall score
        overall_score = sum(quality_metrics.values()) / len(quality_metrics) if quality_metrics else 0.0
        
        return QualityReport(
            overall_score=overall_score,
            quality_metrics=quality_metrics,
            issues_found=issues_found,
            recommendations=recommendations
        )
    
    def export_dataset(self, dataset: EvaluationDataset, format: str) -> Dict[str, Any]:
        """Export dataset in specified format.
        
        Args:
            dataset: Dataset to export
            format: Export format (json, csv, jsonl, parquet)
            
        Returns:
            Dictionary containing exported data and metadata
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
        
        export_data = {
            "dataset_id": dataset.dataset_id,
            "format": format,
            "export_timestamp": str(uuid.uuid4()),  # Simplified timestamp
            "metadata": dataset.metadata
        }
        
        if format == "json":
            export_data["data"] = asdict(dataset)
        elif format == "jsonl":
            lines = []
            for i in range(len(dataset.questions)):
                line = {
                    "question": dataset.questions[i],
                    "contexts": dataset.contexts[i] if i < len(dataset.contexts) else [],
                    "ground_truth": dataset.ground_truth_answers[i] if i < len(dataset.ground_truth_answers) else ""
                }
                lines.append(line)
            export_data["data"] = lines
        elif format == "csv":
            # Simplified CSV representation
            rows = []
            for i in range(len(dataset.questions)):
                row = {
                    "question": dataset.questions[i],
                    "contexts": "|".join(dataset.contexts[i] if i < len(dataset.contexts) else []),
                    "ground_truth": dataset.ground_truth_answers[i] if i < len(dataset.ground_truth_answers) else ""
                }
                rows.append(row)
            export_data["data"] = rows
        elif format == "parquet":
            # Simplified parquet representation (would use actual parquet library in production)
            export_data["data"] = {
                "questions": dataset.questions,
                "contexts": dataset.contexts,
                "ground_truth_answers": dataset.ground_truth_answers,
                "schema": "parquet_compatible"
            }
        
        return export_data
    
    def _generate_from_node(self, node: KnowledgeGraphNode, all_nodes: List[KnowledgeGraphNode], 
                           edges: List[Dict[str, Any]]) -> tuple[List[str], List[List[str]], List[str]]:
        """Generate questions from a single knowledge graph node."""
        questions = []
        contexts = []
        answers = []
        
        # Generate factual question about node properties
        if node.properties:
            prop_key = list(node.properties.keys())[0]
            prop_value = node.properties[prop_key]
            
            question = f"What is the {prop_key} of {node.id}?"
            context = [f"{node.id} has {prop_key}: {prop_value}"]
            answer = str(prop_value)
            
            questions.append(question)
            contexts.append(context)
            answers.append(answer)
        
        return questions, contexts, answers
    
    def _generate_from_relationships(self, nodes: List[KnowledgeGraphNode], 
                                   edges: List[Dict[str, Any]]) -> tuple[List[str], List[List[str]], List[str]]:
        """Generate questions from knowledge graph relationships."""
        questions = []
        contexts = []
        answers = []
        
        for edge in edges[:3]:  # Limit to first 3 edges for simplicity
            source_id = edge.get("source")
            target_id = edge.get("target")
            relationship = edge.get("type", "related_to")
            
            if source_id and target_id:
                question = f"How is {source_id} related to {target_id}?"
                context = [f"{source_id} is {relationship} {target_id}"]
                answer = f"{source_id} is {relationship} {target_id}"
                
                questions.append(question)
                contexts.append(context)
                answers.append(answer)
        
        return questions, contexts, answers
    
    def _generate_synthetic_by_type(self, domain: str, question_type: str, 
                                  count: int) -> tuple[List[str], List[List[str]], List[str]]:
        """Generate synthetic questions by type."""
        questions = []
        contexts = []
        answers = []
        
        templates = {
            "factual": {
                "question": f"What is a key concept in {domain}?",
                "context": f"Key concepts in {domain} include various important elements.",
                "answer": f"A key concept in {domain} is fundamental knowledge."
            },
            "analytical": {
                "question": f"How does {domain} work?",
                "context": f"{domain} operates through systematic processes.",
                "answer": f"{domain} works through structured methodologies."
            },
            "comparative": {
                "question": f"What are the differences in {domain}?",
                "context": f"Different approaches exist within {domain}.",
                "answer": f"Differences in {domain} include methodological variations."
            },
            "inferential": {
                "question": f"What can be inferred about {domain}?",
                "context": f"Evidence suggests patterns in {domain}.",
                "answer": f"It can be inferred that {domain} follows predictable patterns."
            }
        }
        
        template = templates.get(question_type, templates["factual"])
        
        for i in range(count):
            # Add variation to avoid exact duplicates
            variation_suffix = f" (variant {i+1})" if i > 0 else ""
            
            questions.append(template["question"] + variation_suffix)
            contexts.append([template["context"] + variation_suffix])
            answers.append(template["answer"] + variation_suffix)
        
        return questions, contexts, answers
    
    def _validate_completeness(self, dataset: EvaluationDataset) -> tuple[float, List[str], List[str]]:
        """Validate dataset completeness."""
        issues = []
        recommendations = []
        
        # Check for empty fields
        if not dataset.questions:
            issues.append("No questions found in dataset")
        if not dataset.contexts:
            issues.append("No contexts found in dataset")
        if not dataset.ground_truth_answers:
            issues.append("No ground truth answers found in dataset")
        
        # Check for length mismatches
        lengths = [len(dataset.questions), len(dataset.contexts), len(dataset.ground_truth_answers)]
        if len(set(lengths)) > 1:
            issues.append(f"Length mismatch: questions={lengths[0]}, contexts={lengths[1]}, answers={lengths[2]}")
            recommendations.append("Ensure all lists have the same length")
        
        # Check for empty entries
        empty_questions = sum(1 for q in dataset.questions if not q.strip())
        empty_answers = sum(1 for a in dataset.ground_truth_answers if not a.strip())
        
        if empty_questions > 0:
            issues.append(f"{empty_questions} empty questions found")
        if empty_answers > 0:
            issues.append(f"{empty_answers} empty answers found")
        
        # Calculate score
        total_checks = 5
        failed_checks = len(issues)
        score = max(0.0, (total_checks - failed_checks) / total_checks)
        
        return score, issues, recommendations
    
    def _validate_diversity(self, dataset: EvaluationDataset) -> tuple[float, List[str], List[str]]:
        """Validate dataset diversity."""
        issues = []
        recommendations = []
        
        if not dataset.questions:
            return 0.0, ["No questions to validate diversity"], []
        
        # Check question diversity
        unique_questions = set(dataset.questions)
        question_diversity = len(unique_questions) / len(dataset.questions)
        
        # Check answer diversity
        unique_answers = set(dataset.ground_truth_answers)
        answer_diversity = len(unique_answers) / len(dataset.ground_truth_answers) if dataset.ground_truth_answers else 0
        
        # Check question length diversity
        question_lengths = [len(q.split()) for q in dataset.questions]
        length_variance = len(set(question_lengths)) / len(question_lengths) if question_lengths else 0
        
        diversity_score = (question_diversity + answer_diversity + length_variance) / 3
        
        if question_diversity < 0.7:
            issues.append(f"Low question diversity: {question_diversity:.2f}")
            recommendations.append("Increase variety in question formulation")
        
        if answer_diversity < 0.5:
            issues.append(f"Low answer diversity: {answer_diversity:.2f}")
            recommendations.append("Generate more diverse ground truth answers")
        
        return diversity_score, issues, recommendations
    
    def _validate_consistency(self, dataset: EvaluationDataset) -> tuple[float, List[str], List[str]]:
        """Validate dataset consistency."""
        issues = []
        recommendations = []
        
        if not dataset.questions or not dataset.ground_truth_answers:
            return 0.0, ["Insufficient data for consistency validation"], []
        
        # Check for consistent formatting
        question_patterns = set()
        for question in dataset.questions:
            # Simple pattern detection based on question words
            if question.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                question_patterns.add(question.split()[0].lower())
        
        pattern_consistency = len(question_patterns) / len(set(dataset.questions)) if dataset.questions else 0
        
        # Check answer length consistency (not too varied)
        answer_lengths = [len(answer.split()) for answer in dataset.ground_truth_answers]
        if answer_lengths:
            avg_length = sum(answer_lengths) / len(answer_lengths)
            length_variance = sum((length - avg_length) ** 2 for length in answer_lengths) / len(answer_lengths)
            length_consistency = 1.0 / (1.0 + length_variance / 100)  # Normalize variance
        else:
            length_consistency = 0.0
        
        consistency_score = (pattern_consistency + length_consistency) / 2
        
        if pattern_consistency < 0.3:
            issues.append("Inconsistent question formatting patterns")
            recommendations.append("Standardize question formats")
        
        return consistency_score, issues, recommendations
    
    def _validate_relevance(self, dataset: EvaluationDataset) -> tuple[float, List[str], List[str]]:
        """Validate dataset relevance."""
        issues = []
        recommendations = []
        
        if not dataset.questions or not dataset.contexts or not dataset.ground_truth_answers:
            return 0.0, ["Insufficient data for relevance validation"], []
        
        relevance_scores = []
        
        for i in range(min(len(dataset.questions), len(dataset.contexts), len(dataset.ground_truth_answers))):
            question = dataset.questions[i].lower()
            context = " ".join(dataset.contexts[i]).lower() if dataset.contexts[i] else ""
            answer = dataset.ground_truth_answers[i].lower()
            
            # Simple relevance check: look for common words
            question_words = set(question.split())
            context_words = set(context.split())
            answer_words = set(answer.split())
            
            # Calculate overlap
            q_c_overlap = len(question_words & context_words) / len(question_words) if question_words else 0
            q_a_overlap = len(question_words & answer_words) / len(question_words) if question_words else 0
            c_a_overlap = len(context_words & answer_words) / len(context_words) if context_words else 0
            
            relevance = (q_c_overlap + q_a_overlap + c_a_overlap) / 3
            relevance_scores.append(relevance)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        if avg_relevance < 0.3:
            issues.append(f"Low relevance score: {avg_relevance:.2f}")
            recommendations.append("Improve alignment between questions, contexts, and answers")
        
        return avg_relevance, issues, recommendations