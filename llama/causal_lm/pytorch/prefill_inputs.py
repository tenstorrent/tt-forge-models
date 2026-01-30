# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prefill inputs for Llama 3.2 1B testing at various sequence lengths and batch sizes.

Each input is designed to produce a real token count ~55-65% of target seq_len.
For batch_size > 1, we provide separate unique examples per batch item.

Structure: PREFILL_INPUTS[(seq_len, batch_size)] = [list of texts, one per batch item]
"""

PREFILL_INPUTS = {
    # ============================================================================
    # SEQ_LEN = 128
    # ============================================================================

    # 128 tokens target, batch 1 -> need ~75-85 real tokens (~400 chars)
    (128, 1): [
        """You are a helpful AI assistant. Please explain the concept of machine learning in simple terms that a beginner could understand. Focus on the key ideas like training data, models, and predictions. Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. The process involves collecting training data, choosing an algorithm, and iteratively improving the model's accuracy through optimization techniques like gradient descent.""",
    ],

    # 128 tokens target, batch 2 -> 2 separate examples of similar length
    (128, 2): [
        """You are a helpful AI assistant. Please explain the concept of machine learning in simple terms that a beginner could understand. Focus on the key ideas like training data, models, and predictions. Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. The process involves collecting training data, choosing an algorithm, and iteratively improving the model's accuracy through optimization techniques like gradient descent.""",
        """Neural networks are computational systems inspired by the human brain. They consist of layers of interconnected nodes that process information. Each connection has a weight that determines its importance. During training, these weights are adjusted using backpropagation to minimize errors. Deep learning uses many layers to learn hierarchical representations, enabling breakthroughs in image recognition, natural language processing, and game playing.""",
    ],

    # ============================================================================
    # SEQ_LEN = 512
    # ============================================================================

    # 512 tokens target, batch 1 -> need ~300-330 real tokens (~1600 chars)
    (512, 1): [
        """You are a helpful AI assistant specialized in computer science. Please provide a comprehensive explanation of neural networks and deep learning.

Start with the basics: what is a neural network? A neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines its importance. Information flows from the input layer through hidden layers to the output layer.

The key components include:
1. Neurons: Basic computational units that receive inputs, apply weights, sum them, and pass through an activation function
2. Layers: Input layer (receives data), hidden layers (process information), output layer (produces results)
3. Weights and biases: Learnable parameters that are adjusted during training
4. Activation functions: Non-linear functions like ReLU, sigmoid, or tanh that enable learning complex patterns

The learning process uses backpropagation, which calculates gradients of the loss function with respect to each weight. Gradient descent then updates the weights to minimize the loss. This iterative process continues until the model converges to an acceptable level of accuracy.

Different architectures serve different purposes: Convolutional Neural Networks (CNNs) excel at image processing, Recurrent Neural Networks (RNNs) handle sequential data, and Transformers power modern language models. Each architecture has unique characteristics optimized for specific types of data and tasks.""",
    ],

    # 512 tokens target, batch 2 -> 2 separate examples of similar length
    (512, 2): [
        """You are a helpful AI assistant specialized in computer science. Please provide a comprehensive explanation of neural networks and deep learning.

Start with the basics: what is a neural network? A neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers. Each connection has a weight that determines its importance. Information flows from the input layer through hidden layers to the output layer.

The key components include:
1. Neurons: Basic computational units that receive inputs, apply weights, sum them, and pass through an activation function
2. Layers: Input layer (receives data), hidden layers (process information), output layer (produces results)
3. Weights and biases: Learnable parameters that are adjusted during training
4. Activation functions: Non-linear functions like ReLU, sigmoid, or tanh that enable learning complex patterns

The learning process uses backpropagation, which calculates gradients of the loss function with respect to each weight. Gradient descent then updates the weights to minimize the loss. This iterative process continues until the model converges to an acceptable level of accuracy.

Different architectures serve different purposes: Convolutional Neural Networks (CNNs) excel at image processing, Recurrent Neural Networks (RNNs) handle sequential data, and Transformers power modern language models. Each architecture has unique characteristics optimized for specific types of data and tasks.""",

        """Explain the fundamentals of distributed systems and their importance in modern computing infrastructure. A distributed system is a collection of independent computers that appears to users as a single coherent system. These systems communicate through networks to coordinate their actions.

Key concepts in distributed systems include:
1. Consistency: Ensuring all nodes see the same data at the same time
2. Availability: Guaranteeing the system responds to every request
3. Partition Tolerance: Continuing to operate despite network failures

The CAP theorem states that a distributed system can only guarantee two of these three properties simultaneously. This fundamental trade-off drives architectural decisions in systems like databases, message queues, and cloud platforms.

Common distributed system patterns include leader election for coordination, consensus algorithms like Raft and Paxos, and eventual consistency models. Load balancing distributes requests across servers, while replication provides fault tolerance. Sharding partitions data horizontally for scalability.

Modern applications rely heavily on distributed systems: cloud computing platforms, content delivery networks, social media platforms, and streaming services all depend on these principles. Understanding distributed systems is essential for building reliable, scalable applications.""",
    ],

    # ============================================================================
    # SEQ_LEN = 1024
    # ============================================================================

    # 1024 tokens target, batch 1 -> need ~600-680 real tokens (~3200 chars)
    (1024, 1): [
        """You are a senior machine learning engineer writing comprehensive documentation. Please provide an in-depth technical guide on building and deploying large language models.

## Chapter 1: Introduction to Large Language Models

Large Language Models (LLMs) represent a significant advancement in natural language processing. These models, typically based on the Transformer architecture, have revolutionized how machines understand and generate human language. They are trained on massive datasets containing billions of tokens from books, websites, and other text sources.

### 1.1 Historical Context

The evolution of language models spans several key milestones:
- Statistical language models using n-grams provided early approaches but struggled with long-range dependencies
- Word embeddings like Word2Vec and GloVe captured semantic relationships between words in dense vector spaces
- Recurrent Neural Networks (LSTMs, GRUs) processed sequences but faced vanishing gradient problems
- The Transformer architecture (Vaswani et al., 2017) introduced self-attention, enabling parallel processing
- Pre-trained models like BERT and GPT demonstrated the power of transfer learning
- Modern LLMs (GPT-3, GPT-4, LLaMA, Claude) scale to hundreds of billions of parameters

### 1.2 Transformer Architecture Deep Dive

The Transformer architecture consists of several key components that work together:

**Self-Attention Mechanism:**
The attention mechanism computes weighted sums of value vectors based on query-key similarities. For each position i, we compute queries, keys, and values through learned linear projections. The attention scores are computed as softmax(QK^T / sqrt(d_k)) * V, where the scaling factor prevents gradient issues.

**Multi-Head Attention:**
Instead of performing a single attention function, multi-head attention runs multiple attention operations in parallel with different learned projections. This allows the model to capture different types of relationships and attend to information from different representation subspaces.

**Position-wise Feed-Forward Networks:**
Each transformer layer contains a fully connected feed-forward network applied to each position independently. This typically consists of two linear transformations with a ReLU or GELU activation in between, expanding and then contracting the dimension.

**Layer Normalization and Residual Connections:**
These techniques are crucial for training stability. Residual connections allow gradients to flow directly through the network, while layer normalization stabilizes the distribution of activations.

### 1.3 Training Infrastructure

Training LLMs requires massive computational resources including thousands of GPUs, distributed training strategies like data parallelism and model parallelism, and efficient data loading pipelines that can feed data fast enough to keep the accelerators busy.""",
    ],

    # 1024 tokens target, batch 2 -> 2 separate examples of similar length
    (1024, 2): [
        """You are a senior machine learning engineer writing comprehensive documentation. Please provide an in-depth technical guide on building and deploying large language models.

## Chapter 1: Introduction to Large Language Models

Large Language Models (LLMs) represent a significant advancement in natural language processing. These models, typically based on the Transformer architecture, have revolutionized how machines understand and generate human language. They are trained on massive datasets containing billions of tokens from books, websites, and other text sources.

### 1.1 Historical Context

The evolution of language models spans several key milestones:
- Statistical language models using n-grams provided early approaches but struggled with long-range dependencies
- Word embeddings like Word2Vec and GloVe captured semantic relationships between words in dense vector spaces
- Recurrent Neural Networks (LSTMs, GRUs) processed sequences but faced vanishing gradient problems
- The Transformer architecture (Vaswani et al., 2017) introduced self-attention, enabling parallel processing
- Pre-trained models like BERT and GPT demonstrated the power of transfer learning
- Modern LLMs (GPT-3, GPT-4, LLaMA, Claude) scale to hundreds of billions of parameters

### 1.2 Transformer Architecture Deep Dive

The Transformer architecture consists of several key components that work together:

**Self-Attention Mechanism:**
The attention mechanism computes weighted sums of value vectors based on query-key similarities. For each position i, we compute queries, keys, and values through learned linear projections. The attention scores are computed as softmax(QK^T / sqrt(d_k)) * V, where the scaling factor prevents gradient issues.

**Multi-Head Attention:**
Instead of performing a single attention function, multi-head attention runs multiple attention operations in parallel with different learned projections. This allows the model to capture different types of relationships and attend to information from different representation subspaces.

**Position-wise Feed-Forward Networks:**
Each transformer layer contains a fully connected feed-forward network applied to each position independently. This typically consists of two linear transformations with a ReLU or GELU activation in between, expanding and then contracting the dimension.

**Layer Normalization and Residual Connections:**
These techniques are crucial for training stability. Residual connections allow gradients to flow directly through the network, while layer normalization stabilizes the distribution of activations.

### 1.3 Training Infrastructure

Training LLMs requires massive computational resources including thousands of GPUs, distributed training strategies like data parallelism and model parallelism, and efficient data loading pipelines that can feed data fast enough to keep the accelerators busy.""",

        """Write a comprehensive guide on modern database systems and their evolution from relational to distributed architectures.

## Chapter 1: Database Fundamentals and Evolution

Database management systems have undergone remarkable transformations over the past five decades. From hierarchical and network models to the relational revolution, and now to distributed NoSQL and NewSQL systems, databases continue to evolve to meet changing requirements.

### 1.1 The Relational Model

Edgar Codd's seminal 1970 paper introduced the relational model, which became the foundation of modern database systems. Key principles include:
- Data organized in tables (relations) with rows (tuples) and columns (attributes)
- Structured Query Language (SQL) for data manipulation and retrieval
- ACID properties ensuring transaction reliability: Atomicity, Consistency, Isolation, Durability
- Normalization techniques to eliminate redundancy and maintain data integrity

Major relational databases like PostgreSQL, MySQL, Oracle, and SQL Server have powered enterprise applications for decades, providing robust transactional guarantees and rich query capabilities.

### 1.2 The NoSQL Movement

The rise of web-scale applications in the 2000s exposed limitations of traditional relational databases. NoSQL systems emerged to address:
- Horizontal scalability across commodity hardware clusters
- Schema flexibility for rapidly evolving data structures
- High availability through replication and eventual consistency
- Optimized performance for specific access patterns

NoSQL categories include document stores (MongoDB, CouchDB), key-value stores (Redis, DynamoDB), column-family stores (Cassandra, HBase), and graph databases (Neo4j, Amazon Neptune). Each category optimizes for different use cases and query patterns.

### 1.3 Distributed Database Architecture

Modern distributed databases must handle partitioning, replication, and consistency challenges. Consensus protocols like Raft and Paxos coordinate distributed transactions, while techniques like multi-version concurrency control enable isolation without excessive locking.

NewSQL databases combine the scalability of NoSQL with ACID guarantees. Systems like CockroachDB, TiDB, and Google Spanner demonstrate that distributed transactions at scale are achievable through careful engineering.""",
    ],

    # ============================================================================
    # SEQ_LEN = 2048
    # ============================================================================

    # 2048 tokens target, batch 1 -> need ~1200-1400 real tokens (~6500 chars)
    (2048, 1): [
        """You are a principal research scientist writing an extensive technical report on modern AI systems, their architectures, training methodologies, and deployment considerations.

# Comprehensive Guide to Modern Artificial Intelligence Systems

## Executive Summary

This document provides a thorough examination of contemporary artificial intelligence systems, with particular focus on large language models, their underlying architectures, training methodologies, optimization techniques, and practical deployment considerations. We cover both theoretical foundations and practical implementation details essential for building production-ready AI systems.

## Part I: Foundational Concepts

### Chapter 1: The Evolution of Neural Networks

#### 1.1 Historical Perspective

The journey of artificial neural networks spans several decades of research and development, marked by periods of intense optimism and challenging setbacks:

**The Perceptron Era (1950s-1960s):**
Frank Rosenblatt's perceptron represented one of the earliest attempts to create learning machines. This simple model could learn to classify linearly separable patterns. Despite initial enthusiasm about creating truly intelligent machines, Minsky and Papert's 1969 analysis of perceptron limitations led to the first "AI winter," a period of reduced funding and interest.

**Backpropagation Revival (1980s):**
The rediscovery and popularization of backpropagation by Rumelhart, Hinton, and Williams in 1986 reignited interest in neural networks. Multi-layer perceptrons could now learn complex, non-linear functions by propagating error gradients backward through the network layers. This era saw the development of fundamental concepts still used today.

**Deep Learning Revolution (2010s):**
Several factors converged to enable the deep learning revolution that transformed AI:
- Availability of large datasets like ImageNet provided the training data needed for complex models
- GPU computing enabled parallel matrix operations essential for neural network training
- Algorithmic improvements including dropout, batch normalization, and ReLU activation functions
- Architectural innovations like ResNet's skip connections enabled training of very deep networks
- The attention mechanism and Transformer architecture revolutionized sequence modeling

#### 1.2 Mathematical Foundations

**Linear Algebra Fundamentals:**
Neural networks heavily rely on linear algebra operations. Matrix multiplication performs layer transformations, mapping inputs through learned weight matrices. Understanding concepts like eigenvalue decomposition helps analyze network behavior and optimization landscapes.

**Calculus and Optimization:**
Training neural networks requires computing gradients through automatic differentiation. The chain rule enables backpropagation, computing how changes in each parameter affect the final loss. Understanding optimization landscapes, including the challenges of saddle points and local minima, is crucial for effective training.

**Probability and Statistics:**
Many aspects of machine learning involve probabilistic reasoning. Maximum likelihood estimation provides a framework for training models. Bayesian inference offers principled uncertainty quantification. Information theory concepts like entropy and KL divergence measure distributional differences.

### Chapter 2: The Transformer Architecture

#### 2.1 Attention Mechanisms

The attention mechanism is the cornerstone of modern language models, enabling dynamic weighting of input elements based on their relevance:

**Scaled Dot-Product Attention:**
The fundamental attention operation computes attention weights and applies them to values. Queries and keys determine the attention distribution, while values carry the information to be aggregated. The scaling factor of 1/sqrt(d_k) prevents the dot products from becoming too large, which would push the softmax into regions with very small gradients.

**Multi-Head Attention:**
Multiple attention heads allow the model to attend to different representation subspaces simultaneously. Each head can focus on different aspects of the input - some might attend to syntactic relationships while others capture semantic similarities.

#### 2.2 Position Encoding

Since transformers process all positions in parallel, they need explicit position information:

**Sinusoidal Position Encoding:** Uses fixed frequency patterns to encode absolute positions in a continuous manner.

**Rotary Position Embedding (RoPE):** A modern approach that encodes relative position through rotation matrices, enabling better length generalization beyond training sequence lengths.

### Chapter 3: Training Methodologies

#### 3.1 Data Preparation

Training data quality is crucial for LLM performance. Key considerations include careful data collection and curation, deduplication to prevent memorization, quality filtering to remove low-quality content, and appropriate tokenization strategies using methods like BPE or SentencePiece.

#### 3.2 Distributed Training

Large models require distributed training across many devices. Data parallelism replicates the model across devices with each processing different batches. Tensor parallelism splits individual layers across devices. Pipeline parallelism distributes layers across devices with micro-batch scheduling.""",
    ],

    # 2048 tokens target, batch 2 -> 2 separate examples of similar length
    (2048, 2): [
        """You are a principal research scientist writing an extensive technical report on modern AI systems, their architectures, training methodologies, and deployment considerations.

# Comprehensive Guide to Modern Artificial Intelligence Systems

## Executive Summary

This document provides a thorough examination of contemporary artificial intelligence systems, with particular focus on large language models, their underlying architectures, training methodologies, optimization techniques, and practical deployment considerations. We cover both theoretical foundations and practical implementation details essential for building production-ready AI systems.

## Part I: Foundational Concepts

### Chapter 1: The Evolution of Neural Networks

#### 1.1 Historical Perspective

The journey of artificial neural networks spans several decades of research and development, marked by periods of intense optimism and challenging setbacks:

**The Perceptron Era (1950s-1960s):**
Frank Rosenblatt's perceptron represented one of the earliest attempts to create learning machines. This simple model could learn to classify linearly separable patterns. Despite initial enthusiasm about creating truly intelligent machines, Minsky and Papert's 1969 analysis of perceptron limitations led to the first "AI winter," a period of reduced funding and interest.

**Backpropagation Revival (1980s):**
The rediscovery and popularization of backpropagation by Rumelhart, Hinton, and Williams in 1986 reignited interest in neural networks. Multi-layer perceptrons could now learn complex, non-linear functions by propagating error gradients backward through the network layers. This era saw the development of fundamental concepts still used today.

**Deep Learning Revolution (2010s):**
Several factors converged to enable the deep learning revolution that transformed AI:
- Availability of large datasets like ImageNet provided the training data needed for complex models
- GPU computing enabled parallel matrix operations essential for neural network training
- Algorithmic improvements including dropout, batch normalization, and ReLU activation functions
- Architectural innovations like ResNet's skip connections enabled training of very deep networks
- The attention mechanism and Transformer architecture revolutionized sequence modeling

#### 1.2 Mathematical Foundations

**Linear Algebra Fundamentals:**
Neural networks heavily rely on linear algebra operations. Matrix multiplication performs layer transformations, mapping inputs through learned weight matrices. Understanding concepts like eigenvalue decomposition helps analyze network behavior and optimization landscapes.

**Calculus and Optimization:**
Training neural networks requires computing gradients through automatic differentiation. The chain rule enables backpropagation, computing how changes in each parameter affect the final loss. Understanding optimization landscapes, including the challenges of saddle points and local minima, is crucial for effective training.

**Probability and Statistics:**
Many aspects of machine learning involve probabilistic reasoning. Maximum likelihood estimation provides a framework for training models. Bayesian inference offers principled uncertainty quantification. Information theory concepts like entropy and KL divergence measure distributional differences.

### Chapter 2: The Transformer Architecture

#### 2.1 Attention Mechanisms

The attention mechanism is the cornerstone of modern language models, enabling dynamic weighting of input elements based on their relevance:

**Scaled Dot-Product Attention:**
The fundamental attention operation computes attention weights and applies them to values. Queries and keys determine the attention distribution, while values carry the information to be aggregated. The scaling factor of 1/sqrt(d_k) prevents the dot products from becoming too large, which would push the softmax into regions with very small gradients.

**Multi-Head Attention:**
Multiple attention heads allow the model to attend to different representation subspaces simultaneously. Each head can focus on different aspects of the input - some might attend to syntactic relationships while others capture semantic similarities.

#### 2.2 Position Encoding

Since transformers process all positions in parallel, they need explicit position information:

**Sinusoidal Position Encoding:** Uses fixed frequency patterns to encode absolute positions in a continuous manner.

**Rotary Position Embedding (RoPE):** A modern approach that encodes relative position through rotation matrices, enabling better length generalization beyond training sequence lengths.

### Chapter 3: Training Methodologies

#### 3.1 Data Preparation

Training data quality is crucial for LLM performance. Key considerations include careful data collection and curation, deduplication to prevent memorization, quality filtering to remove low-quality content, and appropriate tokenization strategies using methods like BPE or SentencePiece.

#### 3.2 Distributed Training

Large models require distributed training across many devices. Data parallelism replicates the model across devices with each processing different batches. Tensor parallelism splits individual layers across devices. Pipeline parallelism distributes layers across devices with micro-batch scheduling.""",

        """Write a comprehensive technical document covering software engineering best practices, system design principles, and modern development methodologies.

# Software Engineering Excellence: A Complete Guide

## Executive Summary

This document presents a thorough examination of software engineering principles, from foundational concepts to advanced architectural patterns. We cover design principles, development methodologies, testing strategies, and deployment practices that enable teams to build reliable, maintainable, and scalable software systems.

## Part I: Design Principles and Patterns

### Chapter 1: SOLID Principles

#### 1.1 Single Responsibility Principle

Every module or class should have one reason to change. This principle promotes cohesion and reduces coupling between components. When a class handles multiple responsibilities, changes to one responsibility may inadvertently affect others, leading to bugs and maintenance challenges.

#### 1.2 Open-Closed Principle

Software entities should be open for extension but closed for modification. This means adding new functionality should not require changing existing code. Techniques like inheritance, composition, and dependency injection enable this extensibility while protecting existing behavior.

#### 1.3 Liskov Substitution Principle

Subtypes must be substitutable for their base types without altering program correctness. Violations of this principle lead to fragile hierarchies where callers must check concrete types, defeating the purpose of abstraction.

#### 1.4 Interface Segregation Principle

Clients should not be forced to depend on interfaces they do not use. Large interfaces should be split into smaller, focused ones. This reduces coupling and makes systems more flexible and maintainable.

#### 1.5 Dependency Inversion Principle

High-level modules should not depend on low-level modules. Both should depend on abstractions. This inverts the traditional dependency structure, making systems more modular and testable.

### Chapter 2: Architectural Patterns

#### 2.1 Microservices Architecture

Microservices decompose applications into independently deployable services. Each service owns its data, communicates through well-defined APIs, and can be developed, deployed, and scaled independently. This enables organizational agility but introduces operational complexity around service discovery, distributed tracing, and eventual consistency.

#### 2.2 Event-Driven Architecture

Events represent facts about things that happened in the system. Event-driven architectures use message brokers to decouple producers from consumers, enabling asynchronous processing and system resilience. Event sourcing maintains system state as an append-only log of events, providing audit trails and temporal queries.

#### 2.3 Domain-Driven Design

DDD aligns software design with business domains through ubiquitous language, bounded contexts, and strategic patterns. Aggregates enforce consistency boundaries, while domain events communicate changes between contexts. This approach is particularly valuable for complex business domains.

## Part II: Development Practices

### Chapter 3: Testing Strategies

#### 3.1 Test Pyramid

The test pyramid recommends many unit tests, fewer integration tests, and even fewer end-to-end tests. Unit tests are fast and focused, providing rapid feedback. Integration tests verify component interactions. End-to-end tests validate complete workflows but are slower and more brittle.

#### 3.2 Test-Driven Development

TDD follows the red-green-refactor cycle: write a failing test, make it pass with minimal code, then refactor while keeping tests green. This approach drives design toward testability and provides comprehensive regression coverage.

### Chapter 4: Continuous Integration and Deployment

CI/CD pipelines automate building, testing, and deploying software. Continuous integration ensures code changes integrate frequently, catching issues early. Continuous deployment extends this to automatically release validated changes to production, reducing release risk through smaller, more frequent deployments.""",
    ],

    # ============================================================================
    # SEQ_LEN = 4096
    # ============================================================================

    # 4096 tokens target, batch 1 -> need ~2400-2800 real tokens (~13000 chars)
    (4096, 1): [
        """You are a distinguished professor writing a comprehensive textbook chapter on artificial intelligence and machine learning systems.

# Advanced Topics in Artificial Intelligence: A Comprehensive Study

## Volume I: Foundations and Architectures

### Preface

This comprehensive textbook provides an in-depth exploration of modern artificial intelligence systems, covering theoretical foundations, practical implementations, and cutting-edge research directions. The material is designed for advanced graduate students and practitioners seeking deep understanding of AI systems. We aim to bridge theory and practice, providing both mathematical rigor and practical insights.

## Part I: Theoretical Foundations

### Chapter 1: Mathematical Prerequisites

#### 1.1 Linear Algebra for Deep Learning

Linear algebra forms the mathematical backbone of neural network computations. Understanding these concepts deeply is essential for anyone working with modern AI systems.

**Vector Spaces and Transformations:**
Neural networks operate on high-dimensional vector spaces. Understanding linear transformations is essential for comprehending how data flows through network layers. A vector space V over a field F is equipped with addition and scalar multiplication operations satisfying specific axioms. In neural networks, layer operations can be understood as linear transformations followed by non-linear activations.

Key concepts include:
- Vector spaces and subspaces with their closure properties
- Linear independence and basis vectors that span the space
- Eigenvalues and eigenvectors that reveal intrinsic structure
- Singular Value Decomposition (SVD) for matrix analysis and compression

**Matrix Operations:**
The fundamental operations in neural networks include matrix multiplication for layer transformations (computing y = Wx + b), Hadamard products for element-wise operations like gating mechanisms, matrix transposition and inversion for various computations, and batch matrix operations that enable efficient GPU computation.

**Tensor Algebra:**
Modern deep learning extends beyond matrices to higher-order tensors. Understanding tensor operations is crucial for working with attention mechanisms and other multi-dimensional computations. Tensor contractions generalize matrix multiplication, tensor decompositions enable efficient representations, and efficient tensor operations are key to GPU performance.

#### 1.2 Probability and Statistics

**Probabilistic Foundations:**
Understanding uncertainty is crucial for machine learning. Many models are fundamentally probabilistic, and even deterministic models can be interpreted through a probabilistic lens.

Random variables and their distributions describe uncertainty in data and predictions. Conditional probability and Bayes' theorem provide the framework for updating beliefs based on evidence. Maximum likelihood estimation offers a principled approach to parameter estimation. Bayesian inference enables uncertainty quantification in model predictions.

**Information Theory:**
Key concepts from information theory provide deep insights into learning:
- Entropy measures uncertainty in random variables
- Cross-entropy quantifies the cost of using one distribution to encode another
- Kullback-Leibler divergence measures distributional differences
- Mutual information captures shared information between variables
- The information bottleneck principle guides representation learning

#### 1.3 Optimization Theory

**Convex Optimization:**
While neural networks involve non-convex optimization, convex theory provides useful insights and many sub-problems are convex. Key concepts include convex sets and functions, gradient descent convergence guarantees, Lagrange multipliers and duality for constrained optimization.

**Non-Convex Optimization:**
Deep learning optimization presents unique challenges that require specialized techniques:
- Loss landscape geometry with many local minima and saddle points
- Stochastic gradient descent provides noise that helps escape poor solutions
- Momentum and adaptive learning rates improve convergence
- Modern optimizers like Adam and AdamW combine multiple techniques

### Chapter 2: Neural Network Architectures

#### 2.1 Feedforward Networks

The foundational architecture for neural networks consists of layers of neurons with weighted connections:

**Multi-Layer Perceptrons:**
The universal approximation theorem guarantees that MLPs can approximate any continuous function given sufficient width. However, practical considerations involve trade-offs between depth and width. Deeper networks can represent hierarchical features more efficiently, while wider networks may be easier to optimize.

Initialization strategies like Xavier and He initialization are crucial for training deep networks, ensuring gradients flow properly through the layers during backpropagation.

**Regularization Techniques:**
Preventing overfitting is crucial for generalization:
- L1 and L2 weight regularization penalize large weights
- Dropout randomly zeros activations during training, providing ensemble-like benefits
- Batch and layer normalization stabilize training and provide regularization
- Early stopping prevents overfitting to training data

#### 2.2 Convolutional Neural Networks

CNNs are specialized architectures for spatial data like images:

**Convolution Operations:**
Convolutions exploit spatial locality and translation invariance. Key concepts include 1D, 2D, and 3D convolutions for different data types, stride and padding for controlling output dimensions, receptive field analysis for understanding what input regions affect each output, and depthwise separable convolutions for efficiency.

**Classic Architectures:**
The evolution of CNN designs shows progressive improvements:
- LeNet pioneered convolutional neural networks for digit recognition
- AlexNet demonstrated GPU-accelerated deep learning at scale
- VGG showed that depth matters, using small 3x3 filters stacked deeply
- ResNet introduced skip connections enabling training of very deep networks
- EfficientNet used neural architecture search to find optimal configurations

#### 2.3 Recurrent Neural Networks

RNNs process sequential data through recurrent connections:

**Basic RNN Structure:**
Vanilla RNNs maintain hidden state that evolves over time, but face vanishing and exploding gradient problems that limit their effective memory span.

**Advanced Architectures:**
- LSTM uses gating mechanisms to control information flow and maintain long-term dependencies
- GRU simplifies the gating structure while maintaining effectiveness
- Bidirectional processing allows access to future context
- Deep RNNs stack recurrent layers for hierarchical processing

#### 2.4 Transformer Architecture

The dominant architecture for modern AI systems:

**Self-Attention Mechanism:**
The core innovation enabling parallel processing of sequences. Query, Key, and Value projections transform the input, scaled dot-product attention computes weighted combinations, and multi-head attention captures diverse relationships.

**Positional Information:**
Encoding sequence order without recurrence:
- Sinusoidal encodings use fixed frequency patterns
- Learned position embeddings are trainable parameters
- Rotary Position Embedding (RoPE) encodes relative positions
- ALiBi adds linear attention biases based on distance

**Architectural Variants:**
Different configurations serve different purposes:
- Decoder-only (GPT-style) for autoregressive generation
- Encoder-only (BERT-style) for bidirectional understanding
- Encoder-Decoder (T5-style) for sequence-to-sequence tasks
- Mixture of Experts (MoE) for scaling with sparse computation

### Chapter 3: Large Language Models

#### 3.1 Pre-training Paradigms

**Autoregressive Language Modeling:**
The dominant approach trains models to predict the next token given all previous tokens. This simple objective scales remarkably well and leads to emergent capabilities as model size increases.

**Masked Language Modeling:**
BERT-style training masks random tokens and trains the model to reconstruct them, enabling bidirectional context understanding for tasks like classification and extraction.

#### 3.2 Scaling Laws

Understanding how model performance scales with compute, data, and parameters is crucial for efficient training:

**Chinchilla Scaling:**
Research has shown that optimal compute allocation balances model size and training data. The compute-optimal frontier suggests that models should be trained on approximately 20 tokens per parameter.

**Emergent Capabilities:**
Larger models exhibit capabilities not present in smaller ones, including in-context learning, chain-of-thought reasoning, and few-shot generalization.

#### 3.3 Fine-tuning and Alignment

Adapting pre-trained models to specific tasks and human preferences:
- Supervised fine-tuning on task-specific data
- Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences
- Direct Preference Optimization (DPO) simplifies preference learning
- Parameter-efficient methods like LoRA enable adaptation with minimal compute""",
    ],

    # 4096 tokens target, batch 2 -> 2 separate examples of similar length
    (4096, 2): [
        """You are a distinguished professor writing a comprehensive textbook chapter on artificial intelligence and machine learning systems.

# Advanced Topics in Artificial Intelligence: A Comprehensive Study

## Volume I: Foundations and Architectures

### Preface

This comprehensive textbook provides an in-depth exploration of modern artificial intelligence systems, covering theoretical foundations, practical implementations, and cutting-edge research directions. The material is designed for advanced graduate students and practitioners seeking deep understanding of AI systems. We aim to bridge theory and practice, providing both mathematical rigor and practical insights.

## Part I: Theoretical Foundations

### Chapter 1: Mathematical Prerequisites

#### 1.1 Linear Algebra for Deep Learning

Linear algebra forms the mathematical backbone of neural network computations. Understanding these concepts deeply is essential for anyone working with modern AI systems.

**Vector Spaces and Transformations:**
Neural networks operate on high-dimensional vector spaces. Understanding linear transformations is essential for comprehending how data flows through network layers. A vector space V over a field F is equipped with addition and scalar multiplication operations satisfying specific axioms. In neural networks, layer operations can be understood as linear transformations followed by non-linear activations.

Key concepts include:
- Vector spaces and subspaces with their closure properties
- Linear independence and basis vectors that span the space
- Eigenvalues and eigenvectors that reveal intrinsic structure
- Singular Value Decomposition (SVD) for matrix analysis and compression

**Matrix Operations:**
The fundamental operations in neural networks include matrix multiplication for layer transformations (computing y = Wx + b), Hadamard products for element-wise operations like gating mechanisms, matrix transposition and inversion for various computations, and batch matrix operations that enable efficient GPU computation.

**Tensor Algebra:**
Modern deep learning extends beyond matrices to higher-order tensors. Understanding tensor operations is crucial for working with attention mechanisms and other multi-dimensional computations. Tensor contractions generalize matrix multiplication, tensor decompositions enable efficient representations, and efficient tensor operations are key to GPU performance.

#### 1.2 Probability and Statistics

**Probabilistic Foundations:**
Understanding uncertainty is crucial for machine learning. Many models are fundamentally probabilistic, and even deterministic models can be interpreted through a probabilistic lens.

Random variables and their distributions describe uncertainty in data and predictions. Conditional probability and Bayes' theorem provide the framework for updating beliefs based on evidence. Maximum likelihood estimation offers a principled approach to parameter estimation. Bayesian inference enables uncertainty quantification in model predictions.

**Information Theory:**
Key concepts from information theory provide deep insights into learning:
- Entropy measures uncertainty in random variables
- Cross-entropy quantifies the cost of using one distribution to encode another
- Kullback-Leibler divergence measures distributional differences
- Mutual information captures shared information between variables
- The information bottleneck principle guides representation learning

#### 1.3 Optimization Theory

**Convex Optimization:**
While neural networks involve non-convex optimization, convex theory provides useful insights and many sub-problems are convex. Key concepts include convex sets and functions, gradient descent convergence guarantees, Lagrange multipliers and duality for constrained optimization.

**Non-Convex Optimization:**
Deep learning optimization presents unique challenges that require specialized techniques:
- Loss landscape geometry with many local minima and saddle points
- Stochastic gradient descent provides noise that helps escape poor solutions
- Momentum and adaptive learning rates improve convergence
- Modern optimizers like Adam and AdamW combine multiple techniques

### Chapter 2: Neural Network Architectures

#### 2.1 Feedforward Networks

The foundational architecture for neural networks consists of layers of neurons with weighted connections:

**Multi-Layer Perceptrons:**
The universal approximation theorem guarantees that MLPs can approximate any continuous function given sufficient width. However, practical considerations involve trade-offs between depth and width. Deeper networks can represent hierarchical features more efficiently, while wider networks may be easier to optimize.

Initialization strategies like Xavier and He initialization are crucial for training deep networks, ensuring gradients flow properly through the layers during backpropagation.

**Regularization Techniques:**
Preventing overfitting is crucial for generalization:
- L1 and L2 weight regularization penalize large weights
- Dropout randomly zeros activations during training, providing ensemble-like benefits
- Batch and layer normalization stabilize training and provide regularization
- Early stopping prevents overfitting to training data

#### 2.2 Convolutional Neural Networks

CNNs are specialized architectures for spatial data like images:

**Convolution Operations:**
Convolutions exploit spatial locality and translation invariance. Key concepts include 1D, 2D, and 3D convolutions for different data types, stride and padding for controlling output dimensions, receptive field analysis for understanding what input regions affect each output, and depthwise separable convolutions for efficiency.

**Classic Architectures:**
The evolution of CNN designs shows progressive improvements:
- LeNet pioneered convolutional neural networks for digit recognition
- AlexNet demonstrated GPU-accelerated deep learning at scale
- VGG showed that depth matters, using small 3x3 filters stacked deeply
- ResNet introduced skip connections enabling training of very deep networks
- EfficientNet used neural architecture search to find optimal configurations

#### 2.3 Recurrent Neural Networks

RNNs process sequential data through recurrent connections:

**Basic RNN Structure:**
Vanilla RNNs maintain hidden state that evolves over time, but face vanishing and exploding gradient problems that limit their effective memory span.

**Advanced Architectures:**
- LSTM uses gating mechanisms to control information flow and maintain long-term dependencies
- GRU simplifies the gating structure while maintaining effectiveness
- Bidirectional processing allows access to future context
- Deep RNNs stack recurrent layers for hierarchical processing

#### 2.4 Transformer Architecture

The dominant architecture for modern AI systems:

**Self-Attention Mechanism:**
The core innovation enabling parallel processing of sequences. Query, Key, and Value projections transform the input, scaled dot-product attention computes weighted combinations, and multi-head attention captures diverse relationships.

**Positional Information:**
Encoding sequence order without recurrence:
- Sinusoidal encodings use fixed frequency patterns
- Learned position embeddings are trainable parameters
- Rotary Position Embedding (RoPE) encodes relative positions
- ALiBi adds linear attention biases based on distance

**Architectural Variants:**
Different configurations serve different purposes:
- Decoder-only (GPT-style) for autoregressive generation
- Encoder-only (BERT-style) for bidirectional understanding
- Encoder-Decoder (T5-style) for sequence-to-sequence tasks
- Mixture of Experts (MoE) for scaling with sparse computation

### Chapter 3: Large Language Models

#### 3.1 Pre-training Paradigms

**Autoregressive Language Modeling:**
The dominant approach trains models to predict the next token given all previous tokens. This simple objective scales remarkably well and leads to emergent capabilities as model size increases.

**Masked Language Modeling:**
BERT-style training masks random tokens and trains the model to reconstruct them, enabling bidirectional context understanding for tasks like classification and extraction.

#### 3.2 Scaling Laws

Understanding how model performance scales with compute, data, and parameters is crucial for efficient training:

**Chinchilla Scaling:**
Research has shown that optimal compute allocation balances model size and training data. The compute-optimal frontier suggests that models should be trained on approximately 20 tokens per parameter.

**Emergent Capabilities:**
Larger models exhibit capabilities not present in smaller ones, including in-context learning, chain-of-thought reasoning, and few-shot generalization.

#### 3.3 Fine-tuning and Alignment

Adapting pre-trained models to specific tasks and human preferences:
- Supervised fine-tuning on task-specific data
- Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences
- Direct Preference Optimization (DPO) simplifies preference learning
- Parameter-efficient methods like LoRA enable adaptation with minimal compute""",

        """Write an extensive technical reference on computer systems architecture, from processors to distributed computing platforms.

# Computer Systems Architecture: A Complete Reference

## Volume I: From Transistors to Cloud Platforms

### Preface

This reference work provides comprehensive coverage of computer systems architecture at multiple levels of abstraction. From the physics of transistors to the design of warehouse-scale computers, we explore how modern computing systems are built and optimized. The material bridges hardware and software perspectives, essential for understanding system performance.

## Part I: Processor Architecture

### Chapter 1: Instruction Set Architecture

#### 1.1 RISC vs CISC

The debate between Reduced and Complex Instruction Set Computing shaped processor design for decades. RISC architectures like ARM and RISC-V use simple, fixed-length instructions that execute in single cycles. CISC architectures like x86 provide complex instructions that may take multiple cycles but reduce code size.

Modern processors blur these distinctions. x86 processors decode complex instructions into micro-operations internally, achieving RISC-like efficiency. ARM has added complex instructions for specific workloads. The key insight is that instruction set design involves trade-offs between code density, decode complexity, and execution efficiency.

**Instruction Formats:**
Instructions encode operations, operands, and addressing modes. Register-register operations are fastest, while memory operations involve address calculation and cache access. Immediate operands embed constants directly in instructions. Branch instructions specify targets through various addressing modes.

**Addressing Modes:**
Different addressing modes provide flexibility in accessing data:
- Register direct: operand in register
- Immediate: operand in instruction
- Direct: address in instruction
- Register indirect: address in register
- Indexed: base register plus offset
- PC-relative: offset from program counter

#### 1.2 Pipelining and Superscalar Execution

Pipelining overlaps instruction execution stages, improving throughput without reducing latency. Classic five-stage pipelines include fetch, decode, execute, memory, and writeback. Hazards occur when instructions depend on results not yet available.

**Pipeline Hazards:**
- Data hazards: instruction needs result from previous instruction
- Control hazards: branch target unknown when next instruction fetches
- Structural hazards: hardware resource conflict

**Hazard Resolution:**
- Forwarding: bypass results directly to dependent instructions
- Stalling: insert bubble cycles until data available
- Branch prediction: guess branch direction, flush on mispredict

Superscalar processors issue multiple instructions per cycle. Out-of-order execution finds independent instructions to execute while waiting for dependencies. Register renaming eliminates false dependencies from register reuse.

### Chapter 2: Memory Hierarchy

#### 2.1 Cache Design

Caches exploit temporal and spatial locality to bridge the processor-memory speed gap. Modern processors have multiple cache levels with increasing size and latency.

**Cache Organization:**
- Direct-mapped: each address maps to one cache line
- Set-associative: address maps to set, any line in set valid
- Fully associative: any line can hold any address

**Replacement Policies:**
When caches fill, policies determine which line to evict:
- LRU: evict least recently used
- Random: evict randomly selected line
- FIFO: evict oldest line

**Write Policies:**
- Write-through: update memory immediately
- Write-back: update memory on eviction

#### 2.2 Virtual Memory

Virtual memory provides isolation, protection, and the illusion of large address spaces. Page tables translate virtual to physical addresses. Translation lookaside buffers (TLBs) cache recent translations.

**Page Table Organization:**
Modern systems use hierarchical page tables to reduce memory overhead. Four or five level page tables support large address spaces efficiently. Huge pages reduce TLB pressure for large allocations.

**Memory Protection:**
Page table entries encode access permissions. Hardware enforces read, write, and execute permissions. Privileged operations require supervisor mode.

## Part II: Parallel Computing

### Chapter 3: Multicore Architecture

#### 3.1 Cache Coherence

When multiple cores cache shared data, coherence protocols ensure consistency. MESI protocol tracks line states: Modified, Exclusive, Shared, Invalid. Snooping protocols broadcast memory operations; directory protocols scale better.

**Coherence Challenges:**
False sharing occurs when different variables share cache lines. Padding separates frequently-modified data. NUMA systems have non-uniform memory access times based on processor-memory distance.

#### 3.2 Synchronization

Atomic operations enable thread coordination. Compare-and-swap and load-linked/store-conditional provide building blocks. Lock-free algorithms avoid blocking but require careful design.

**Memory Ordering:**
Different architectures provide different memory ordering guarantees. Fence instructions enforce ordering when needed. Understanding memory models is crucial for correct concurrent programming.

### Chapter 4: Distributed Systems

#### 4.1 Consensus and Replication

Distributed systems must handle partial failures and network partitions. Consensus protocols like Paxos and Raft ensure agreement despite failures. State machine replication applies consensus to build fault-tolerant services.

#### 4.2 Distributed Storage

Distributed storage systems shard data across nodes for scalability. Consistent hashing minimizes data movement when nodes join or leave. Replication provides durability; quorum protocols balance availability and consistency.""",
    ],

    # ============================================================================
    # SEQ_LEN = 8192
    # ============================================================================

    # 8192 tokens target, batch 1 -> need ~5000-5500 real tokens (~26000 chars)
    (8192, 1): [
        """You are a world-renowned AI researcher writing the definitive reference book on artificial intelligence systems.

# The Complete Encyclopedia of Artificial Intelligence: Theory, Practice, and Future Directions

## Volume I: Foundations of Machine Learning and Neural Networks

### Foreword

This encyclopedic reference represents the culmination of decades of research in artificial intelligence and machine learning. It is designed to serve as both a comprehensive introduction for newcomers and a detailed reference for experienced practitioners. The material covers theoretical foundations, practical implementations, and cutting-edge research directions that are shaping the future of AI. We hope this work will serve as a valuable resource for students, researchers, and practitioners alike.

### Part I: Historical and Philosophical Foundations

#### Chapter 1: The Origins of Artificial Intelligence

##### 1.1 Early Pioneers and Visionaries

The dream of creating intelligent machines predates modern computing by centuries. Ancient myths and legends speak of artificial beings endowed with intelligence, from the bronze automaton Talos in Greek mythology to the Golem of Jewish folklore. However, the scientific pursuit of artificial intelligence began in earnest in the mid-20th century with the advent of digital computers.

**Alan Turing and the Foundations of Computation:**
Alan Turing's seminal 1936 paper "On Computable Numbers" laid the theoretical groundwork for all of computer science. His conceptualization of the Turing machine demonstrated that any computation could be performed by a sufficiently simple abstract device, establishing the theoretical limits of what machines could compute. In 1950, Turing proposed the famous "Turing Test" as a criterion for machine intelligence, asking whether a machine could exhibit behavior indistinguishable from a human in open-ended conversation. This operationalization of intelligence, while controversial, provided a concrete goal for AI research.

**The Dartmouth Conference (1956):**
The field of artificial intelligence was formally established at the Dartmouth Summer Research Project on Artificial Intelligence. Organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, this workshop brought together researchers who would shape the field for decades to come. McCarthy coined the term "artificial intelligence" and the conference established AI as a distinct discipline with its own research agenda, methodologies, and aspirations.

**Early Optimism and Challenges:**
The decades following Dartmouth saw remarkable optimism about the potential for rapid progress. Researchers like Herbert Simon predicted that machines would be capable of any work a human could do within 20 years. Early successes included programs that could prove mathematical theorems, play checkers at a high level, and solve algebra word problems. These achievements seemed to validate the optimistic predictions.

However, the 1970s brought a reality check. The Lighthill Report in the UK criticized AI research for failing to achieve its ambitious goals, particularly in machine translation and general problem-solving. This criticism led to reduced funding across the field. This period, known as the first "AI winter," forced researchers to focus on more tractable problems and develop more rigorous methodologies.

##### 1.2 The Knowledge-Based Systems Era

**Expert Systems:**
The 1980s saw a resurgence of AI research focused on expert systems—programs that captured domain expertise in rule-based formats. Rather than pursuing general intelligence, researchers focused on narrow but practically useful applications.

Systems like MYCIN for medical diagnosis demonstrated that AI could provide valuable decision support in complex domains. DENDRAL for chemical analysis showed that expert knowledge could be encoded and applied systematically. These systems achieved commercial success and restored confidence in AI research.

**The Knowledge Acquisition Bottleneck:**
Despite commercial success, expert systems faced a fundamental limitation: acquiring and encoding expert knowledge was expensive and time-consuming. Human experts had to work with knowledge engineers to explicitly articulate their decision-making processes. This "knowledge acquisition bottleneck" limited scalability and motivated the search for systems that could learn from data rather than relying on hand-crafted rules. This limitation would eventually lead to the machine learning approaches that dominate modern AI.

##### 1.3 The Machine Learning Revolution

**Statistical Learning Theory:**
The development of rigorous statistical foundations for machine learning, particularly the work of Vladimir Vapnik on Support Vector Machines and the VC dimension, provided theoretical tools for understanding when and why learning algorithms work. This theory established fundamental limits on generalization and provided principled approaches to model selection. The mathematical rigor brought credibility to machine learning as a scientific discipline.

**The Deep Learning Renaissance:**
The modern era of deep learning began in earnest with the success of deep neural networks on challenging benchmarks. Key milestones transformed the field:
- AlexNet's victory in ImageNet 2012 demonstrated that deep CNNs could dramatically outperform traditional computer vision
- The introduction of dropout and batch normalization enabled training of deeper networks
- Residual networks enabled training of networks with hundreds of layers
- The Transformer architecture revolutionized natural language processing

### Part II: Mathematical Foundations

#### Chapter 2: Linear Algebra for Machine Learning

##### 2.1 Vector Spaces and Linear Transformations

**Abstract Vector Spaces:**
A vector space V over a field F is a set equipped with addition and scalar multiplication operations satisfying specific axioms. These axioms ensure that vectors can be added together and scaled in consistent ways. In machine learning, we primarily work with real vector spaces (F = R) where vectors represent data points, model parameters, or intermediate representations.

The fundamental operations in a vector space satisfy several properties:
- Commutativity of addition: u + v = v + u
- Associativity of addition: (u + v) + w = u + (v + w)
- Additive identity: there exists a zero vector such that v + 0 = v
- Additive inverse: for each v, there exists -v such that v + (-v) = 0
- Distributivity of scalar multiplication over vector addition
- Distributivity of scalar multiplication over field addition
- Associativity of scalar multiplication

Understanding these properties helps in reasoning about the behavior of neural network layers, which are fundamentally linear transformations between vector spaces.

**Linear Transformations:**
A linear transformation T: V → W between vector spaces preserves the vector space structure. Specifically, T(u + v) = T(u) + T(v) and T(αv) = αT(v) for all vectors and scalars. In neural networks, each layer performs a linear transformation followed by a non-linear activation.

The composition of linear transformations is itself linear. This is why neural networks without activation functions collapse to single linear layers - the composition of multiple linear layers is equivalent to a single linear transformation.

**Eigenvalue Decomposition:**
For a square matrix A, eigenvalues λ and eigenvectors v satisfy Av = λv. The eigenvectors represent directions that are only scaled (not rotated) by the transformation. The eigendecomposition provides insights into matrix behavior, including conditioning for numerical stability, principal directions in data via PCA, and characteristics of optimization landscapes in neural network training.

**Singular Value Decomposition:**
Every matrix M can be decomposed as M = UΣV^T, where U contains left singular vectors, Σ contains singular values on the diagonal, and V contains right singular vectors. This decomposition has numerous applications in machine learning, including low-rank approximation for dimensionality reduction, matrix compression for efficient storage and computation, and latent semantic analysis for text processing.

##### 2.2 Matrix Calculus

**Gradients and Jacobians:**
The gradient of a scalar function f with respect to a vector x is a vector of partial derivatives. The Jacobian matrix of a vector function generalizes this to functions with multiple outputs.

**Backpropagation as Chain Rule Application:**
The chain rule enables computing gradients through composed functions. For a composition of functions, the derivative is the product of individual derivatives. This principle underlies automatic differentiation in deep learning frameworks, where gradients are computed by accumulating local gradients through the computational graph.

#### Chapter 3: Probability Theory and Information Theory

##### 3.1 Probabilistic Foundations

**Random Variables and Distributions:**
A random variable X maps outcomes from a sample space to real numbers. The probability distribution describes the likelihood of different values, either through a probability mass function for discrete variables or a probability density function for continuous variables.

**Key Distributions in Machine Learning:**
Different distributions model different types of phenomena:
- Bernoulli distribution models binary outcomes like classification decisions
- Categorical distribution extends to multi-class classification
- Gaussian distribution models continuous data and is used extensively in latent space representations
- Dirichlet distribution provides priors over probability distributions, useful in topic modeling

**Bayes' Theorem:**
The foundation of probabilistic inference relates prior beliefs to posterior beliefs given observed data. Bayesian approaches provide principled uncertainty quantification, though they often require approximations for computational tractability.

##### 3.2 Information Theory

**Entropy:**
The entropy of a random variable quantifies uncertainty or information content. Higher entropy indicates more uncertainty. For a discrete distribution, entropy is H(X) = -Σ p(x) log p(x). The base of the logarithm determines the units (bits for base 2, nats for base e).

**Cross-Entropy and KL Divergence:**
Cross-entropy measures the expected number of bits needed to encode data from distribution p using a code optimized for distribution q. The KL divergence measures how much extra information is needed, and is always non-negative. These concepts are central to training neural networks - cross-entropy loss is the standard objective for classification.

**Mutual Information:**
Mutual information quantifies the amount of information shared between two variables. It measures how much knowing one variable reduces uncertainty about the other. Applications include feature selection, representation learning through information maximization, and understanding what neural networks learn.

### Part III: Neural Network Architectures

#### Chapter 4: Deep Feedforward Networks

##### 4.1 Architecture Design

**Layer Structure:**
A deep network computes a sequence of transformations. Each layer applies a linear transformation followed by a non-linear activation function. The depth of the network (number of layers) and width (neurons per layer) are key architectural choices.

**Activation Functions:**
Non-linear activations enable networks to learn complex functions:
- ReLU (max(0, x)) is simple, computationally efficient, and promotes sparsity
- GELU provides a smooth approximation used in transformers
- Swish (x * sigmoid(βx)) can be learned or fixed
- Mish (x * tanh(softplus(x))) offers smooth gradients throughout

**Initialization:**
Proper initialization prevents vanishing or exploding gradients at the start of training. Xavier initialization sets variance inversely proportional to layer size, while He initialization accounts for ReLU's effect on variance.

##### 4.2 Regularization

**Dropout:**
Randomly zeroing activations during training provides regularization through implicit ensemble averaging. Variants include DropConnect for weights, spatial dropout for CNNs, and DropBlock for contiguous regions.

**Normalization:**
Normalizing activations stabilizes training and provides regularization. Batch normalization normalizes over the batch dimension, layer normalization over features, and group normalization offers a middle ground.

#### Chapter 5: The Transformer Architecture

##### 5.1 Self-Attention Mechanism

**Scaled Dot-Product Attention:**
The core operation computes attention weights from query-key similarities and applies them to values. The scaling factor prevents dot products from becoming too large, which would saturate the softmax.

**Multi-Head Attention:**
Parallel attention heads with different learned projections capture diverse relationships. Each head can specialize in different aspects of the input, such as syntactic vs. semantic relationships.

**Efficient Attention Variants:**
The quadratic complexity of attention motivates efficient alternatives:
- Flash Attention reorders computation to minimize memory access
- Linear attention uses kernel approximations
- Sparse attention patterns combine local and global attention
- Sliding window approaches limit the context window

##### 5.2 Positional Information

**Absolute Positional Encodings:**
Sinusoidal encodings use fixed frequency patterns, while learned embeddings are optimized during training. Both have limitations for sequences longer than those seen during training.

**Relative Position Encodings:**
RoPE rotates query and key vectors based on position, encoding relative distances. ALiBi adds linear biases to attention scores based on distance, enabling extrapolation to longer sequences.

##### 5.3 Architectural Variants

**Decoder-Only Models:**
GPT-style models use causal masking and are trained with next-token prediction. This simple objective scales remarkably well and enables few-shot learning.

**Encoder-Only Models:**
BERT-style models use bidirectional attention and masked language modeling. They excel at understanding tasks like classification and extraction.

**Encoder-Decoder Models:**
T5-style models use cross-attention between encoder and decoder. This flexible format handles various sequence-to-sequence tasks.

### Part IV: Large Language Models

#### Chapter 6: Pre-training and Scaling

##### 6.1 Training Objectives

**Causal Language Modeling:**
The autoregressive objective maximizes the probability of each token given all previous tokens. Despite its simplicity, this objective gives rise to remarkable capabilities when combined with scale.

##### 6.2 Scaling Laws

Research has revealed predictable relationships between model performance, compute, data, and parameters. The Chinchilla scaling laws suggest that optimal training balances model size with dataset size, recommending approximately 20 tokens per parameter.

**Emergent Capabilities:**
Larger models exhibit qualitatively new behaviors not present at smaller scales, including in-context learning, chain-of-thought reasoning, and instruction following.

##### 6.3 Fine-tuning and Alignment

Pre-trained models are adapted through supervised fine-tuning on curated examples, RLHF to align with human preferences, and parameter-efficient methods like LoRA that enable adaptation with minimal compute.

#### Chapter 7: Inference and Deployment

##### 7.1 Prefill and Decode Phases

**Prefill Phase:**
The prompt is processed in parallel to build the key-value cache. This phase is compute-bound, dominated by matrix multiplications.

**Decode Phase:**
Tokens are generated autoregressively, each step using the cached keys and values. This phase is memory-bound, limited by the bandwidth to read the cache.

##### 7.2 Optimization Techniques

**Quantization:**
Reducing numerical precision decreases memory and improves throughput. INT8 provides 2x memory reduction with minimal quality loss, while INT4 enables 4x reduction with careful calibration.

**KV-Cache Optimization:**
Grouped-query attention shares keys and values across head groups. Paged attention manages memory like virtual memory. Sliding window limits the context length.

This comprehensive reference provides the foundation for understanding modern AI systems from first principles through cutting-edge applications and deployment strategies.""",
    ],

    # 8192 tokens target, batch 2 -> 2 separate examples of similar length
    (8192, 2): [
        """You are a world-renowned AI researcher writing the definitive reference book on artificial intelligence systems.

# The Complete Encyclopedia of Artificial Intelligence: Theory, Practice, and Future Directions

## Volume I: Foundations of Machine Learning and Neural Networks

### Foreword

This encyclopedic reference represents the culmination of decades of research in artificial intelligence and machine learning. It is designed to serve as both a comprehensive introduction for newcomers and a detailed reference for experienced practitioners. The material covers theoretical foundations, practical implementations, and cutting-edge research directions that are shaping the future of AI. We hope this work will serve as a valuable resource for students, researchers, and practitioners alike.

### Part I: Historical and Philosophical Foundations

#### Chapter 1: The Origins of Artificial Intelligence

##### 1.1 Early Pioneers and Visionaries

The dream of creating intelligent machines predates modern computing by centuries. Ancient myths and legends speak of artificial beings endowed with intelligence, from the bronze automaton Talos in Greek mythology to the Golem of Jewish folklore. However, the scientific pursuit of artificial intelligence began in earnest in the mid-20th century with the advent of digital computers.

**Alan Turing and the Foundations of Computation:**
Alan Turing's seminal 1936 paper "On Computable Numbers" laid the theoretical groundwork for all of computer science. His conceptualization of the Turing machine demonstrated that any computation could be performed by a sufficiently simple abstract device, establishing the theoretical limits of what machines could compute. In 1950, Turing proposed the famous "Turing Test" as a criterion for machine intelligence, asking whether a machine could exhibit behavior indistinguishable from a human in open-ended conversation. This operationalization of intelligence, while controversial, provided a concrete goal for AI research.

**The Dartmouth Conference (1956):**
The field of artificial intelligence was formally established at the Dartmouth Summer Research Project on Artificial Intelligence. Organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, this workshop brought together researchers who would shape the field for decades to come. McCarthy coined the term "artificial intelligence" and the conference established AI as a distinct discipline with its own research agenda, methodologies, and aspirations.

**Early Optimism and Challenges:**
The decades following Dartmouth saw remarkable optimism about the potential for rapid progress. Researchers like Herbert Simon predicted that machines would be capable of any work a human could do within 20 years. Early successes included programs that could prove mathematical theorems, play checkers at a high level, and solve algebra word problems. These achievements seemed to validate the optimistic predictions.

However, the 1970s brought a reality check. The Lighthill Report in the UK criticized AI research for failing to achieve its ambitious goals, particularly in machine translation and general problem-solving. This criticism led to reduced funding across the field. This period, known as the first "AI winter," forced researchers to focus on more tractable problems and develop more rigorous methodologies.

##### 1.2 The Knowledge-Based Systems Era

**Expert Systems:**
The 1980s saw a resurgence of AI research focused on expert systems—programs that captured domain expertise in rule-based formats. Rather than pursuing general intelligence, researchers focused on narrow but practically useful applications.

Systems like MYCIN for medical diagnosis demonstrated that AI could provide valuable decision support in complex domains. DENDRAL for chemical analysis showed that expert knowledge could be encoded and applied systematically. These systems achieved commercial success and restored confidence in AI research.

**The Knowledge Acquisition Bottleneck:**
Despite commercial success, expert systems faced a fundamental limitation: acquiring and encoding expert knowledge was expensive and time-consuming. Human experts had to work with knowledge engineers to explicitly articulate their decision-making processes. This "knowledge acquisition bottleneck" limited scalability and motivated the search for systems that could learn from data rather than relying on hand-crafted rules. This limitation would eventually lead to the machine learning approaches that dominate modern AI.

##### 1.3 The Machine Learning Revolution

**Statistical Learning Theory:**
The development of rigorous statistical foundations for machine learning, particularly the work of Vladimir Vapnik on Support Vector Machines and the VC dimension, provided theoretical tools for understanding when and why learning algorithms work. This theory established fundamental limits on generalization and provided principled approaches to model selection. The mathematical rigor brought credibility to machine learning as a scientific discipline.

**The Deep Learning Renaissance:**
The modern era of deep learning began in earnest with the success of deep neural networks on challenging benchmarks. Key milestones transformed the field:
- AlexNet's victory in ImageNet 2012 demonstrated that deep CNNs could dramatically outperform traditional computer vision
- The introduction of dropout and batch normalization enabled training of deeper networks
- Residual networks enabled training of networks with hundreds of layers
- The Transformer architecture revolutionized natural language processing

### Part II: Mathematical Foundations

#### Chapter 2: Linear Algebra for Machine Learning

##### 2.1 Vector Spaces and Linear Transformations

**Abstract Vector Spaces:**
A vector space V over a field F is a set equipped with addition and scalar multiplication operations satisfying specific axioms. These axioms ensure that vectors can be added together and scaled in consistent ways. In machine learning, we primarily work with real vector spaces (F = R) where vectors represent data points, model parameters, or intermediate representations.

The fundamental operations in a vector space satisfy several properties:
- Commutativity of addition: u + v = v + u
- Associativity of addition: (u + v) + w = u + (v + w)
- Additive identity: there exists a zero vector such that v + 0 = v
- Additive inverse: for each v, there exists -v such that v + (-v) = 0
- Distributivity of scalar multiplication over vector addition
- Distributivity of scalar multiplication over field addition
- Associativity of scalar multiplication

Understanding these properties helps in reasoning about the behavior of neural network layers, which are fundamentally linear transformations between vector spaces.

**Linear Transformations:**
A linear transformation T: V → W between vector spaces preserves the vector space structure. Specifically, T(u + v) = T(u) + T(v) and T(αv) = αT(v) for all vectors and scalars. In neural networks, each layer performs a linear transformation followed by a non-linear activation.

The composition of linear transformations is itself linear. This is why neural networks without activation functions collapse to single linear layers - the composition of multiple linear layers is equivalent to a single linear transformation.

**Eigenvalue Decomposition:**
For a square matrix A, eigenvalues λ and eigenvectors v satisfy Av = λv. The eigenvectors represent directions that are only scaled (not rotated) by the transformation. The eigendecomposition provides insights into matrix behavior, including conditioning for numerical stability, principal directions in data via PCA, and characteristics of optimization landscapes in neural network training.

**Singular Value Decomposition:**
Every matrix M can be decomposed as M = UΣV^T, where U contains left singular vectors, Σ contains singular values on the diagonal, and V contains right singular vectors. This decomposition has numerous applications in machine learning, including low-rank approximation for dimensionality reduction, matrix compression for efficient storage and computation, and latent semantic analysis for text processing.

##### 2.2 Matrix Calculus

**Gradients and Jacobians:**
The gradient of a scalar function f with respect to a vector x is a vector of partial derivatives. The Jacobian matrix of a vector function generalizes this to functions with multiple outputs.

**Backpropagation as Chain Rule Application:**
The chain rule enables computing gradients through composed functions. For a composition of functions, the derivative is the product of individual derivatives. This principle underlies automatic differentiation in deep learning frameworks, where gradients are computed by accumulating local gradients through the computational graph.

#### Chapter 3: Probability Theory and Information Theory

##### 3.1 Probabilistic Foundations

**Random Variables and Distributions:**
A random variable X maps outcomes from a sample space to real numbers. The probability distribution describes the likelihood of different values, either through a probability mass function for discrete variables or a probability density function for continuous variables.

**Key Distributions in Machine Learning:**
Different distributions model different types of phenomena:
- Bernoulli distribution models binary outcomes like classification decisions
- Categorical distribution extends to multi-class classification
- Gaussian distribution models continuous data and is used extensively in latent space representations
- Dirichlet distribution provides priors over probability distributions, useful in topic modeling

**Bayes' Theorem:**
The foundation of probabilistic inference relates prior beliefs to posterior beliefs given observed data. Bayesian approaches provide principled uncertainty quantification, though they often require approximations for computational tractability.

##### 3.2 Information Theory

**Entropy:**
The entropy of a random variable quantifies uncertainty or information content. Higher entropy indicates more uncertainty. For a discrete distribution, entropy is H(X) = -Σ p(x) log p(x). The base of the logarithm determines the units (bits for base 2, nats for base e).

**Cross-Entropy and KL Divergence:**
Cross-entropy measures the expected number of bits needed to encode data from distribution p using a code optimized for distribution q. The KL divergence measures how much extra information is needed, and is always non-negative. These concepts are central to training neural networks - cross-entropy loss is the standard objective for classification.

**Mutual Information:**
Mutual information quantifies the amount of information shared between two variables. It measures how much knowing one variable reduces uncertainty about the other. Applications include feature selection, representation learning through information maximization, and understanding what neural networks learn.

### Part III: Neural Network Architectures

#### Chapter 4: Deep Feedforward Networks

##### 4.1 Architecture Design

**Layer Structure:**
A deep network computes a sequence of transformations. Each layer applies a linear transformation followed by a non-linear activation function. The depth of the network (number of layers) and width (neurons per layer) are key architectural choices.

**Activation Functions:**
Non-linear activations enable networks to learn complex functions:
- ReLU (max(0, x)) is simple, computationally efficient, and promotes sparsity
- GELU provides a smooth approximation used in transformers
- Swish (x * sigmoid(βx)) can be learned or fixed
- Mish (x * tanh(softplus(x))) offers smooth gradients throughout

**Initialization:**
Proper initialization prevents vanishing or exploding gradients at the start of training. Xavier initialization sets variance inversely proportional to layer size, while He initialization accounts for ReLU's effect on variance.

##### 4.2 Regularization

**Dropout:**
Randomly zeroing activations during training provides regularization through implicit ensemble averaging. Variants include DropConnect for weights, spatial dropout for CNNs, and DropBlock for contiguous regions.

**Normalization:**
Normalizing activations stabilizes training and provides regularization. Batch normalization normalizes over the batch dimension, layer normalization over features, and group normalization offers a middle ground.

#### Chapter 5: The Transformer Architecture

##### 5.1 Self-Attention Mechanism

**Scaled Dot-Product Attention:**
The core operation computes attention weights from query-key similarities and applies them to values. The scaling factor prevents dot products from becoming too large, which would saturate the softmax.

**Multi-Head Attention:**
Parallel attention heads with different learned projections capture diverse relationships. Each head can specialize in different aspects of the input, such as syntactic vs. semantic relationships.

**Efficient Attention Variants:**
The quadratic complexity of attention motivates efficient alternatives:
- Flash Attention reorders computation to minimize memory access
- Linear attention uses kernel approximations
- Sparse attention patterns combine local and global attention
- Sliding window approaches limit the context window

##### 5.2 Positional Information

**Absolute Positional Encodings:**
Sinusoidal encodings use fixed frequency patterns, while learned embeddings are optimized during training. Both have limitations for sequences longer than those seen during training.

**Relative Position Encodings:**
RoPE rotates query and key vectors based on position, encoding relative distances. ALiBi adds linear biases to attention scores based on distance, enabling extrapolation to longer sequences.

##### 5.3 Architectural Variants

**Decoder-Only Models:**
GPT-style models use causal masking and are trained with next-token prediction. This simple objective scales remarkably well and enables few-shot learning.

**Encoder-Only Models:**
BERT-style models use bidirectional attention and masked language modeling. They excel at understanding tasks like classification and extraction.

**Encoder-Decoder Models:**
T5-style models use cross-attention between encoder and decoder. This flexible format handles various sequence-to-sequence tasks.

### Part IV: Large Language Models

#### Chapter 6: Pre-training and Scaling

##### 6.1 Training Objectives

**Causal Language Modeling:**
The autoregressive objective maximizes the probability of each token given all previous tokens. Despite its simplicity, this objective gives rise to remarkable capabilities when combined with scale.

##### 6.2 Scaling Laws

Research has revealed predictable relationships between model performance, compute, data, and parameters. The Chinchilla scaling laws suggest that optimal training balances model size with dataset size, recommending approximately 20 tokens per parameter.

**Emergent Capabilities:**
Larger models exhibit qualitatively new behaviors not present at smaller scales, including in-context learning, chain-of-thought reasoning, and instruction following.

##### 6.3 Fine-tuning and Alignment

Pre-trained models are adapted through supervised fine-tuning on curated examples, RLHF to align with human preferences, and parameter-efficient methods like LoRA that enable adaptation with minimal compute.

#### Chapter 7: Inference and Deployment

##### 7.1 Prefill and Decode Phases

**Prefill Phase:**
The prompt is processed in parallel to build the key-value cache. This phase is compute-bound, dominated by matrix multiplications.

**Decode Phase:**
Tokens are generated autoregressively, each step using the cached keys and values. This phase is memory-bound, limited by the bandwidth to read the cache.

##### 7.2 Optimization Techniques

**Quantization:**
Reducing numerical precision decreases memory and improves throughput. INT8 provides 2x memory reduction with minimal quality loss, while INT4 enables 4x reduction with careful calibration.

**KV-Cache Optimization:**
Grouped-query attention shares keys and values across head groups. Paged attention manages memory like virtual memory. Sliding window limits the context length.

This comprehensive reference provides the foundation for understanding modern AI systems from first principles through cutting-edge applications and deployment strategies.""",

        """Write a comprehensive technical encyclopedia covering the complete landscape of modern software systems engineering.

# The Complete Encyclopedia of Software Systems Engineering

## Volume I: From Fundamentals to Enterprise Scale

### Foreword

This comprehensive reference work provides an exhaustive exploration of software systems engineering, from foundational principles to enterprise-scale deployment patterns. It serves as both an introduction for newcomers entering the field and a detailed reference for experienced practitioners. The material covers programming paradigms, architectural patterns, distributed systems, and modern DevOps practices that define contemporary software engineering.

### Part I: Programming Language Theory and Paradigms

#### Chapter 1: Foundations of Programming Languages

##### 1.1 Type Systems and Their Role

Type systems are fundamental mechanisms for preventing errors and expressing programmer intent. Static type systems check types at compile time, catching errors before execution. Dynamic type systems defer checking to runtime, offering flexibility at the cost of runtime errors.

**Type Safety and Soundness:**
A type-safe language prevents undefined behavior from type errors. Soundness ensures that type checking rejects all programs that could exhibit type errors. Languages like Rust combine strong static typing with ownership concepts to eliminate entire classes of memory errors at compile time.

**Polymorphism:**
Parametric polymorphism (generics) allows functions and data structures to work with any type while maintaining type safety. Subtype polymorphism enables treating derived types as base types. Ad-hoc polymorphism (overloading) provides different implementations for different types with the same interface.

**Type Inference:**
Modern languages like TypeScript, Kotlin, and Rust infer types from context, reducing annotation burden while maintaining type safety. Hindley-Milner type inference provides complete inference for languages without subtyping, while more powerful systems require annotations in strategic locations.

##### 1.2 Memory Management Paradigms

**Manual Memory Management:**
Languages like C require explicit allocation and deallocation. This provides control but creates risks: use-after-free, double-free, and memory leaks. Disciplined coding and tools like sanitizers and static analyzers help detect errors.

**Garbage Collection:**
Automatic memory management tracks object references and reclaims unreachable memory. Generational collectors optimize for the observation that most objects die young. Concurrent collectors minimize pause times. Languages like Go and Java use sophisticated GC algorithms.

**Ownership and Borrowing:**
Rust's ownership system prevents data races and memory errors at compile time through unique ownership of values, borrowing for temporary access with compile-time checked lifetimes, and move semantics that transfer ownership. This approach achieves memory safety without garbage collection overhead.

##### 1.3 Concurrency Models

**Shared Memory Concurrency:**
Threads share memory directly, requiring synchronization through mutexes, semaphores, and atomic operations. Data races occur when unsynchronized access includes at least one write. Memory models define visibility guarantees across threads.

**Message Passing:**
Communicating Sequential Processes (CSP) and Actor models isolate state in processes that communicate through channels or messages. Go's channels and Erlang's actors exemplify this approach, making concurrent code easier to reason about.

**Async/Await:**
Cooperative multitasking with async/await syntax enables concurrent I/O without thread overhead. Runtimes like Tokio (Rust) and asyncio (Python) provide the underlying execution. This model excels for I/O-bound workloads.

### Part II: Software Architecture

#### Chapter 2: Architectural Patterns

##### 2.1 Layered Architecture

The classic n-tier architecture separates concerns into presentation, business logic, and data access layers. This separation enables independent evolution but can lead to rigid boundaries that complicate cross-cutting concerns.

**Dependency Inversion:**
Inverting dependencies makes high-level modules independent of low-level details. Abstractions defined in higher layers are implemented in lower layers, enabling testing and substitution.

##### 2.2 Domain-Driven Design

DDD aligns software structure with business domains, providing ubiquitous language as shared vocabulary between developers and domain experts, bounded contexts as explicit boundaries with independent models, aggregates as consistency boundaries with a single root entity, and domain events that capture significant occurrences.

**Strategic Design:**
Context mapping identifies relationships between bounded contexts. Integration patterns include shared kernel, customer-supplier, and anticorruption layer. This strategic view guides system decomposition.

##### 2.3 Microservices Architecture

Decomposing applications into independently deployable services enables team autonomy and technology flexibility. Each service owns its data and exposes capabilities through APIs.

**Service Communication:**
- Synchronous: REST or gRPC for request-response patterns
- Asynchronous: Message brokers enable event-driven integration
- Service mesh: Infrastructure layer handles cross-cutting concerns

**Operational Challenges:**
Distributed systems bring complexity. Service discovery locates running instances. Circuit breakers prevent cascade failures. Distributed tracing follows requests across services. Proper observability is essential.

### Part III: Distributed Systems

#### Chapter 3: Consistency and Consensus

##### 3.1 CAP Theorem and Trade-offs

The CAP theorem states that distributed systems can provide at most two of three properties simultaneously: Consistency (all nodes see the same data), Availability (requests receive responses), and Partition tolerance (system operates despite network failures).

Since partitions are inevitable in distributed systems, the real choice is between consistency and availability during partitions. Different systems make different trade-offs based on use case requirements.

##### 3.2 Consensus Algorithms

**Raft:**
A understandable consensus algorithm with leader election through randomized timeouts, log replication from leader to followers, and safety guarantees ensuring consistency. Raft is used in systems like etcd and CockroachDB.

**Paxos:**
The classic consensus algorithm proves that distributed consensus is achievable with a majority of nodes. While theoretically elegant, Paxos is notoriously difficult to implement correctly.

##### 3.3 Eventual Consistency

Many distributed systems accept temporary inconsistency for improved availability and performance. Conflict resolution strategies include last-writer-wins based on timestamps, vector clocks tracking causal relationships, CRDTs (Conflict-free Replicated Data Types) that merge automatically, and application-specific resolution logic.

#### Chapter 4: Distributed Storage

##### 4.1 Partitioning Strategies

**Range Partitioning:**
Data is split by key ranges, enabling efficient range queries but risking hot spots for sequential access patterns.

**Hash Partitioning:**
Consistent hashing distributes data evenly and minimizes movement when nodes join or leave. Virtual nodes improve balance further.

##### 4.2 Replication

**Leader-Based Replication:**
A single leader accepts writes and replicates to followers. Synchronous replication ensures durability at the cost of latency; asynchronous improves performance but risks data loss.

**Multi-Leader Replication:**
Multiple nodes accept writes, requiring conflict resolution. This improves write availability but complicates consistency.

**Leaderless Replication:**
Quorum reads and writes provide tunable consistency. Systems like Cassandra and DynamoDB use this approach for high availability.

### Part IV: Modern Development Practices

#### Chapter 5: Continuous Integration and Delivery

##### 5.1 Build Automation

Automated builds ensure reproducibility. Key practices include declarative build definitions in version control, incremental builds for development speed, hermetic builds eliminating environmental dependencies, and artifact caching to avoid repeated work.

##### 5.2 Testing Strategies

**Test Levels:**
- Unit tests: isolated component verification
- Integration tests: component interaction
- System tests: end-to-end validation
- Performance tests: latency and throughput validation

**Continuous Testing:**
Tests run automatically on every change. Fast feedback enables frequent integration. Flaky test management prevents false negatives.

##### 5.3 Deployment Automation

**Deployment Patterns:**
- Blue-green: parallel environments with instant switchover
- Canary: gradual rollout with monitoring
- Feature flags: decouple deployment from release
- GitOps: declarative infrastructure from version control

#### Chapter 6: Observability

##### 6.1 The Three Pillars

**Metrics:**
Numerical measurements over time track system health and performance. RED (Rate, Errors, Duration) and USE (Utilization, Saturation, Errors) methodologies guide metric selection.

**Logs:**
Structured logging captures contextual information for debugging and audit. Centralized log aggregation enables search and analysis at scale.

**Traces:**
Distributed tracing follows requests across service boundaries, revealing latency bottlenecks and failure points. Standards like OpenTelemetry provide vendor-neutral instrumentation.

##### 6.2 Alerting and Incident Response

Effective alerting requires defining actionable conditions tied to user impact, avoiding alert fatigue through appropriate thresholds, providing runbooks for common incidents, and continuous improvement through post-mortems.

This comprehensive encyclopedia provides the foundational knowledge and practical guidance for building reliable, scalable, and maintainable software systems at any scale.""",
    ],
}


def get_prefill_input(seq_len: int, batch_size: int = 1) -> list:
    """Get the prefill input texts for a given sequence length and batch size.

    Args:
        seq_len: Target sequence length
        batch_size: Batch size (number of separate examples needed)

    Returns:
        List of input text strings, one per batch item

    Raises:
        KeyError: If (seq_len, batch_size) combination is not in PREFILL_INPUTS
    """
    key = (seq_len, batch_size)
    if key not in PREFILL_INPUTS:
        available_seq_lens = sorted(set(k[0] for k in PREFILL_INPUTS.keys()))
        available_batches = sorted(set(k[1] for k in PREFILL_INPUTS.keys()))
        raise KeyError(
            f"No prefill input defined for seq_len={seq_len}, batch_size={batch_size}. "
            f"Available seq_lens: {available_seq_lens}, batch_sizes: {available_batches}"
        )
    return PREFILL_INPUTS[key]


def get_available_configs() -> list:
    """Get all available (seq_len, batch_size) configurations.

    Returns:
        List of (seq_len, batch_size) tuples
    """
    return sorted(PREFILL_INPUTS.keys())
