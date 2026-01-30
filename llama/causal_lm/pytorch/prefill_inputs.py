# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prefill input texts for different sequence lengths and batch sizes.

Dictionary structure: PREFILL_TEXTS[seq_len][batch_idx] = text
Each text is designed to tokenize to approximately seq_len tokens to minimize padding.
For batch_size > 1, use multiple texts from the same seq_len bucket.
"""

# Texts designed to be approximately 100-120 tokens (for seq_len=128)
_TEXT_128_0 = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain."""

_TEXT_128_1 = """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention."""

# Texts designed to be approximately 900-1000 tokens (for seq_len=1024)
_TEXT_1024_0 = """The development of artificial intelligence has been one of the most transformative technological achievements of the modern era. From its theoretical foundations laid by Alan Turing in the 1950s to the sophisticated neural networks of today, AI has evolved through several distinct phases, each marked by breakthrough discoveries and persistent challenges.

The field began with the ambitious goal of creating machines that could think like humans. Early pioneers such as John McCarthy, Marvin Minsky, and Claude Shannon believed that human intelligence could be precisely described and replicated in machines. This optimism led to significant early achievements, including programs that could play chess, prove mathematical theorems, and engage in simple conversations.

However, the 1970s brought what is now called the "AI Winter," a period of reduced funding and interest following the realization that early AI systems were far more limited than anticipated. The complexity of natural language understanding, common-sense reasoning, and real-world perception proved to be enormous obstacles that simple symbolic approaches could not overcome.

The resurgence of AI came with the development of machine learning, particularly neural networks. Inspired by the structure of biological brains, these systems learn from data rather than following explicitly programmed rules. The backpropagation algorithm, refined in the 1980s, allowed neural networks to be trained effectively, opening new possibilities for pattern recognition and classification tasks.

The 2010s witnessed an explosion in AI capabilities driven by deep learning. With access to massive datasets and powerful computing hardware, deep neural networks achieved superhuman performance in image recognition, speech processing, and game playing. The victory of AlphaGo over world champion Go player Lee Sedol in 2016 demonstrated that AI could master tasks requiring intuition and strategic thinking.

Today, large language models represent the cutting edge of AI research. These models, trained on vast corpora of text, can generate coherent and contextually appropriate responses across a wide range of topics. They have found applications in customer service, content creation, code generation, and scientific research. However, they also raise important questions about reliability, bias, and the nature of understanding.

The transformer architecture, introduced in 2017, revolutionized natural language processing. By using self-attention mechanisms, transformers can capture long-range dependencies in text more effectively than previous recurrent approaches. This architecture forms the foundation of models like GPT, BERT, and their successors.

As AI systems become more capable, society faces important decisions about their deployment and governance. Questions of accountability, transparency, and alignment with human values become increasingly urgent. Researchers are actively working on techniques for making AI systems more interpretable, robust, and aligned with human intentions.

The future of AI promises even greater capabilities and challenges. Multimodal systems that can seamlessly process text, images, audio, and video are already emerging. The integration of AI with robotics, biotechnology, and other fields opens possibilities that were once the domain of science fiction. Yet realizing this potential requires careful attention to safety, ethics, and the broader societal implications of these powerful technologies."""

_TEXT_1024_1 = """Computer science is the study of computation, automation, and information. It spans theoretical disciplines such as algorithms, theory of computation, and information theory to practical disciplines including the design and implementation of hardware and software. Computer science is generally considered an area of academic research and distinct from computer programming.

The history of computer science began long before our modern discipline existed. Thinkers throughout history have been fascinated by computation and the automation of reasoning. The ancient Greeks developed sophisticated mechanical devices, including the Antikythera mechanism, which computed astronomical positions. In the 17th century, Blaise Pascal and Gottfried Wilhelm Leibniz invented mechanical calculators that could perform arithmetic operations.

The theoretical foundations of modern computer science were established in the early 20th century. Kurt Gödel's incompleteness theorems revealed fundamental limitations of formal systems. Alan Turing's conceptualization of the Turing machine provided a formal model of computation that remains central to theoretical computer science. Alonzo Church developed lambda calculus, an alternative model that proved equivalent to Turing machines.

The electronic computer emerged during World War II, driven by military needs for code-breaking and ballistic calculations. ENIAC, completed in 1945, was one of the first general-purpose electronic computers. The subsequent decades saw rapid advances in hardware, with vacuum tubes giving way to transistors and then integrated circuits. Moore's Law, the observation that computing power doubles approximately every two years, has held remarkably true for decades.

Software development evolved alongside hardware improvements. Early programming was done in machine code, directly manipulating the computer's internal representations. Assembly languages provided a slightly more human-readable format. High-level languages like FORTRAN and COBOL abstracted away hardware details, making programming accessible to a broader audience. Today, programmers can choose from hundreds of languages, each with its own strengths and paradigms.

Operating systems emerged to manage computer resources and provide a platform for applications. From early batch processing systems to interactive time-sharing systems to modern graphical interfaces, operating systems have continuously evolved to meet user needs. Unix, developed at Bell Labs in the 1970s, introduced many concepts that remain fundamental today, including hierarchical file systems and the shell.

Networking transformed computers from isolated machines into interconnected systems. The ARPANET, funded by the U.S. Department of Defense, pioneered packet switching and laid the groundwork for the Internet. The World Wide Web, invented by Tim Berners-Lee in 1989, made networked information accessible to everyone. Today, the Internet connects billions of devices and enables communication, commerce, and collaboration on a global scale.

Database systems provide organized storage and retrieval of information. Relational databases, based on Edgar Codd's work in the 1970s, use tables and structured query language to manage data. More recently, NoSQL databases have emerged to handle the scale and variety of modern data. Data science and machine learning increasingly drive insights from the vast amounts of information being collected.

Security has become a critical concern as computers handle sensitive information. Cryptography protects data from unauthorized access. Authentication systems verify user identities. Firewalls and intrusion detection systems guard against attacks. Despite these measures, cybersecurity remains an ongoing challenge as adversaries continuously develop new techniques.

The future of computer science promises continued rapid change. Quantum computing may solve problems intractable for classical computers. Brain-computer interfaces could create new modes of human-machine interaction. Edge computing brings processing closer to data sources. Whatever specific technologies emerge, the fundamental principles of computer science will continue to guide their development."""

# Texts designed to be approximately 1800-2000 tokens (for seq_len=2048)
_TEXT_2048_0 = _TEXT_1024_0 + """

The intersection of artificial intelligence and scientific discovery represents one of the most exciting frontiers in modern research. AI systems are now being used to accelerate drug discovery, predict protein structures, and analyze astronomical data. The AlphaFold system, developed by DeepMind, solved the protein folding problem that had challenged biologists for decades. This breakthrough demonstrates how AI can tackle problems that were previously considered intractable.

In materials science, machine learning models are being used to predict the properties of new materials before they are synthesized. This approach dramatically accelerates the discovery of materials with desired characteristics, such as superconductors, battery materials, and catalysts. The traditional trial-and-error approach to materials discovery is being replaced by intelligent search guided by learned patterns.

Climate science has also benefited from AI advances. Machine learning models can process the vast amounts of data generated by climate sensors, satellites, and simulations. These models help scientists understand complex climate dynamics, improve weather predictions, and assess the impacts of various intervention strategies. The ability to process and find patterns in massive datasets has opened new avenues for climate research.

The healthcare industry is undergoing a transformation driven by AI technologies. Medical imaging systems powered by deep learning can detect diseases with accuracy matching or exceeding human experts. Natural language processing systems can extract insights from medical records, helping clinicians make more informed decisions. Personalized medicine, which tailors treatments to individual patients based on their genetic and health data, is becoming increasingly practical.

Autonomous systems represent another area of rapid AI advancement. Self-driving cars, once the domain of science fiction, are now being tested on public roads. Autonomous drones are being used for delivery, agriculture, and search and rescue operations. Robots powered by AI are becoming more capable in manufacturing, logistics, and even household tasks. These systems combine perception, planning, and control in increasingly sophisticated ways.

The economics of AI are reshaping industries and labor markets. Automation powered by AI is changing the nature of work, displacing some jobs while creating others. Companies that effectively leverage AI gain competitive advantages, leading to market concentration in some sectors. Policymakers are grappling with questions about how to ensure that the benefits of AI are widely shared while managing the disruptions it causes."""

_TEXT_2048_1 = _TEXT_1024_1 + """

The evolution of programming paradigms has profoundly influenced how we think about and construct software. Structured programming, introduced in the 1960s and 1970s, emphasized the use of control structures like loops and conditionals rather than arbitrary jumps. This approach made programs easier to understand and maintain, establishing principles that remain relevant today.

Object-oriented programming emerged as a dominant paradigm in the 1980s and 1990s. By organizing code around objects that combine data and behavior, this approach facilitated the development of complex systems. Languages like C++, Java, and Python brought object-oriented concepts to mainstream software development. Design patterns, reusable solutions to common problems, became an important part of software engineering practice.

Functional programming, with roots in lambda calculus, has gained renewed interest in recent years. By treating computation as the evaluation of mathematical functions without side effects, functional programming offers advantages for concurrent and parallel systems. Languages like Haskell, Scala, and Clojure have brought functional concepts to practical software development. Even traditionally imperative languages have incorporated functional features.

The rise of web development created new challenges and opportunities for software engineering. The client-server model evolved into sophisticated web applications with rich user interfaces. JavaScript, initially designed for simple browser scripting, became a full-featured language powering both front-end and back-end development. Frameworks and libraries abstracted away low-level details, enabling rapid application development.

Mobile computing introduced additional complexity with its constraints on power, bandwidth, and screen size. Developing applications that work across different devices and platforms became a significant challenge. Native development, web-based approaches, and cross-platform frameworks each offered different tradeoffs. The app economy created new business models and distribution channels for software.

Cloud computing transformed how software is deployed and operated. Rather than managing physical servers, developers can provision virtual resources on demand. Services like compute instances, databases, and message queues are available as commodities. This shift enabled new architectures, including microservices and serverless computing, that emphasize scalability and resilience. DevOps practices emerged to manage the complexity of continuous deployment and operation."""

# Texts designed to be approximately 3500-4000 tokens (for seq_len=4096)
_TEXT_4096_0 = _TEXT_2048_0 + """

The philosophical implications of artificial intelligence extend far beyond its technical achievements. Questions about consciousness, understanding, and the nature of mind have taken on new urgency as AI systems become more sophisticated. When a language model generates coherent text, is there any sense in which it understands what it is saying? This question connects to ancient philosophical debates about the relationship between language and thought.

The Chinese Room argument, proposed by philosopher John Searle, challenges the notion that syntactic manipulation of symbols can constitute genuine understanding. In this thought experiment, a person who does not understand Chinese follows rules to produce appropriate responses to Chinese inputs. The person produces correct outputs without understanding Chinese. Searle argued that computer programs, no matter how sophisticated, similarly lack genuine understanding.

Defenders of artificial intelligence have offered various responses to this argument. Some argue that understanding emerges from the system as a whole rather than any single component. Others suggest that the distinction between genuine and simulated understanding may not be as clear as it seems. The debate continues to evolve as AI systems become more capable.

The question of machine consciousness is even more challenging. Consciousness involves subjective experience, the qualitative feel of what it is like to be something. Whether machines could ever have such experiences is deeply uncertain. Some theorists argue that consciousness requires biological substrates that cannot be replicated in silicon. Others suggest that consciousness might emerge from certain types of information processing regardless of the underlying medium.

These philosophical questions have practical implications for how we design and deploy AI systems. If AI systems could have experiences, we would face ethical obligations toward them. The treatment of potentially conscious machines would become a moral concern. Even without resolving these deep questions, we must make decisions about how AI systems should behave and how they should be integrated into society.

The alignment problem has become a central concern in AI safety research. How do we ensure that AI systems pursue goals that are beneficial to humanity? As systems become more capable, the consequences of misalignment become more severe. A superintelligent system with subtly wrong objectives could cause catastrophic harm while pursuing those objectives with great effectiveness.

Researchers are developing various approaches to the alignment problem. Inverse reinforcement learning attempts to infer human preferences from observed behavior. Constitutional AI provides systems with explicit principles to guide their behavior. Debate and amplification techniques aim to leverage AI systems themselves to identify and correct problematic behaviors. Despite these efforts, ensuring alignment remains an open and critical challenge.

The social impacts of AI extend throughout society. AI systems are increasingly used in high-stakes decisions about lending, hiring, and criminal justice. These applications raise concerns about fairness, accountability, and transparency. Biases present in training data can be amplified by AI systems, leading to discriminatory outcomes. Ensuring that AI systems treat people fairly has become an active area of research and regulation.

The concentration of AI capabilities in a small number of large technology companies raises concerns about power and competition. Training state-of-the-art AI models requires enormous computational resources that few organizations can afford. This concentration could lead to reduced innovation and increased market power. Policymakers are considering various interventions to promote competition and ensure broad access to AI capabilities.

International competition in AI has become a significant geopolitical issue. Nations view AI leadership as essential to economic competitiveness and national security. This competition has led to increased investment in AI research and development, but also to concerns about an AI arms race. The governance of AI across national boundaries presents complex challenges that existing international institutions may not be equipped to handle.

Education and workforce development must adapt to a world increasingly shaped by AI. Some skills become less valuable as AI systems automate tasks that previously required human expertise. Other skills, particularly those that complement AI capabilities, become more valuable. Lifelong learning becomes essential as the pace of technological change accelerates. Educational institutions are experimenting with new approaches to prepare students for an uncertain future."""

_TEXT_4096_1 = _TEXT_2048_1 + """

The architecture of modern software systems reflects decades of accumulated wisdom about how to build reliable, scalable, and maintainable applications. Layered architectures separate concerns into distinct levels of abstraction. Presentation layers handle user interaction, business logic layers implement domain rules, and data access layers manage persistent storage. This separation facilitates testing, maintenance, and evolution of each layer independently.

Distributed systems have become the norm for applications that must handle significant scale. Rather than running on a single machine, these systems spread computation across multiple nodes. This distribution introduces new challenges around consistency, availability, and partition tolerance. The CAP theorem, formulated by Eric Brewer, establishes fundamental tradeoffs that distributed system designers must navigate.

Microservices architecture decomposes applications into small, independently deployable services. Each service handles a specific capability and communicates with others through well-defined APIs. This approach enables different parts of an application to evolve at different rates and be developed by different teams. However, it also introduces complexity in coordination, monitoring, and debugging distributed interactions.

Event-driven architectures have gained popularity for building responsive and scalable systems. Rather than direct synchronous calls, components communicate through events that capture significant occurrences. Message brokers and event streams decouple producers from consumers, enabling flexible and resilient system designs. This approach is particularly well-suited for applications that must react to changing conditions in real time.

Containerization and orchestration have transformed how software is packaged and deployed. Containers provide lightweight, isolated environments that include application code and all its dependencies. Orchestration platforms like Kubernetes manage the deployment, scaling, and operation of containerized applications across clusters of machines. These technologies have become foundational to modern cloud-native development.

Observability encompasses the practices and tools for understanding the behavior of running systems. Metrics provide quantitative measurements of system performance and health. Logs capture detailed records of events and errors. Distributed tracing follows requests as they flow through complex systems. Together, these capabilities enable operators to detect problems, diagnose root causes, and continuously improve system reliability.

Site reliability engineering, pioneered at Google, applies software engineering practices to operations. SREs use automation to eliminate repetitive manual work and improve system reliability. Service level objectives define target levels of reliability that balance user needs with engineering costs. Blameless postmortems learn from incidents without assigning blame, fostering a culture of continuous improvement.

The practice of continuous integration and continuous deployment has accelerated the pace of software delivery. Automated tests run with every code change, catching problems early in the development cycle. Automated pipelines build, test, and deploy software with minimal manual intervention. Feature flags allow new functionality to be released gradually and rolled back quickly if problems emerge.

Infrastructure as code brings software engineering practices to the management of computing infrastructure. Declarative configurations describe the desired state of infrastructure resources. Version control tracks changes and enables rollback to previous configurations. Automated tools ensure that actual infrastructure matches the declared state. This approach reduces errors, improves reproducibility, and enables infrastructure to evolve safely.

The security of software systems requires attention throughout the development and operation lifecycle. Secure development practices identify and mitigate vulnerabilities early in the development process. Static and dynamic analysis tools detect common security flaws. Dependency management ensures that third-party components are kept up to date and free of known vulnerabilities. Security monitoring detects and responds to attacks on running systems."""

# Texts designed to be approximately 7000-8000 tokens (for seq_len=8192)
_TEXT_8192_0 = _TEXT_4096_0 + """

The history of computing is marked by periodic paradigm shifts that fundamentally changed how we think about and use computers. The transition from mainframes to personal computers democratized access to computing power. The rise of the internet transformed computers from isolated tools into nodes in a global network. Mobile computing made computing ubiquitous and personal. Each shift created new opportunities and disrupted existing business models.

The current era is defined by the convergence of artificial intelligence, cloud computing, and ubiquitous connectivity. AI capabilities that once required specialized expertise are now accessible through APIs and pre-trained models. Cloud platforms provide elastic compute resources that scale with demand. The Internet of Things connects billions of devices, generating vast streams of data. These technologies are combining in ways that enable new applications and reshape existing industries.

Edge computing represents an important evolution in how computing resources are deployed. Rather than sending all data to centralized cloud data centers, edge computing processes data closer to where it is generated. This approach reduces latency, conserves bandwidth, and enables operation even when network connectivity is limited. Applications like autonomous vehicles, industrial automation, and augmented reality benefit from edge computing capabilities.

Quantum computing promises to solve certain problems exponentially faster than classical computers. Quantum bits, or qubits, can exist in superpositions of states, enabling quantum algorithms to explore many possibilities simultaneously. Algorithms like Shor's algorithm for factoring and Grover's algorithm for search demonstrate the potential of quantum computing. However, current quantum computers are limited by decoherence and error rates.

The development of practical quantum computers faces significant technical challenges. Qubits must be isolated from environmental disturbances that cause decoherence. Error correction requires many physical qubits for each logical qubit. Scaling to the number of qubits needed for useful computation remains an active area of research. Multiple approaches, including superconducting circuits, trapped ions, and photonic systems, are being pursued.

Quantum computing will not replace classical computing for most applications. Instead, quantum computers will serve as specialized accelerators for problems where they offer advantages. Optimization, simulation of quantum systems, and certain machine learning tasks are promising applications. Organizations are already exploring how quantum computing might benefit their operations, even as they wait for the technology to mature.

The biological and digital worlds are increasingly intertwined. Synthetic biology applies engineering principles to living systems, enabling the design and construction of new biological parts, devices, and systems. DNA can be used as a storage medium, potentially offering density and durability far beyond current technologies. Brain-computer interfaces enable direct communication between nervous systems and digital devices.

These biotechnologies raise profound ethical questions. The ability to modify genomes, including human genomes, requires careful consideration of benefits, risks, and values. Privacy concerns arise when biological data can be analyzed to reveal sensitive information. The line between therapy and enhancement becomes blurred as our capabilities expand. Society must develop ethical frameworks and governance mechanisms appropriate to these powerful technologies.

The environmental impact of computing has become a significant concern. Data centers consume enormous amounts of electricity, contributing to carbon emissions. The manufacture and disposal of electronic devices generate environmental pollution. As computing becomes ever more pervasive, its environmental footprint grows. Researchers and practitioners are working on green computing approaches that reduce energy consumption and environmental impact.

Sustainability considerations are driving changes in how computing systems are designed and operated. More efficient hardware reduces energy consumption for a given level of performance. Renewable energy sources power an increasing fraction of data center operations. Extending the useful life of devices reduces the environmental cost of manufacturing and disposal. Carbon-aware computing schedules workloads to minimize their climate impact.

The governance of emerging technologies presents challenges for existing institutions. Technological change often outpaces regulatory development, creating gaps in oversight. International coordination is difficult when countries have different values and interests. Private companies make consequential decisions about technology deployment with limited public input. New mechanisms for technology governance are needed to ensure that emerging technologies serve the public interest.

Participatory approaches to technology governance seek to include diverse voices in decision-making. Citizens' assemblies and public consultations can inform policy development. Impact assessments evaluate the potential effects of technologies before deployment. Algorithmic auditing examines how automated systems make decisions. These mechanisms aim to make technology development more responsive to public values and concerns.

The future of computing is being shaped by decisions made today. Research priorities determine which technologies receive investment and attention. Educational programs prepare the next generation of computing professionals. Policy choices shape the incentives and constraints that guide technology development. By engaging thoughtfully with these decisions, we can help ensure that the future of computing reflects our highest aspirations.

Computing has always been about more than just machines and algorithms. It is fundamentally about amplifying human capabilities and addressing human needs. The most successful technologies are those that empower people to do things they could not do before. As we develop ever more powerful computing technologies, we must keep this human-centered perspective at the forefront of our work."""

_TEXT_8192_1 = _TEXT_4096_1 + """

The art and science of software engineering encompasses not just technical skills but also practices for organizing human effort. Software projects fail more often from organizational and communication problems than from purely technical challenges. Understanding how teams work effectively, how requirements are gathered and refined, and how quality is assured are essential aspects of software engineering practice.

Requirements engineering is the process of discovering, analyzing, and documenting what software should do. This process is challenging because stakeholders often have difficulty articulating their needs, requirements evolve over time, and different stakeholders may have conflicting objectives. Techniques like user stories, use cases, and prototyping help elicit and validate requirements. Traceability links requirements to design decisions and test cases.

Software architecture provides the high-level structure that shapes all other development activities. Architectural decisions determine how a system is divided into components, how those components interact, and what qualities the system will exhibit. These decisions are difficult to change later and have long-lasting consequences. Architects must balance multiple concerns including functionality, quality attributes, and organizational constraints.

Testing is essential for building confidence that software works correctly. Unit tests verify the behavior of individual components in isolation. Integration tests check that components work together correctly. End-to-end tests validate complete user scenarios. Performance tests ensure that systems meet their speed and scalability requirements. Security tests identify vulnerabilities before they can be exploited.

Test-driven development inverts the traditional relationship between tests and code. Rather than writing tests after code is complete, developers write tests first, then implement code to make the tests pass. This approach ensures that code is testable and helps clarify requirements. The resulting test suite serves as documentation and enables confident refactoring.

Code review is a practice where developers examine each other's code before it is integrated. Reviews catch bugs, improve code quality, and spread knowledge across the team. Automated tools can check for style violations and common errors. The social aspect of review encourages developers to write cleaner code knowing it will be read by peers.

Technical debt is a metaphor for the accumulated cost of shortcuts and suboptimal decisions in software development. Like financial debt, technical debt can be useful when incurred deliberately and managed carefully. However, excessive technical debt makes systems difficult to understand, modify, and operate. Regular refactoring pays down technical debt and keeps systems healthy.

Documentation serves multiple purposes in software development. User documentation helps people use software effectively. Developer documentation explains how to build, modify, and extend software. Architectural documentation captures important decisions and their rationale. Documentation requires ongoing maintenance to remain useful as software evolves.

Version control systems track changes to code and other artifacts over time. They enable multiple developers to work on the same codebase without overwriting each other's changes. Branching strategies manage parallel lines of development for features, releases, and experiments. The history preserved by version control supports debugging, auditing, and understanding how software has evolved.

Agile methodologies have transformed how software is developed and delivered. Rather than following a detailed plan created at the start of a project, agile approaches embrace change and focus on delivering value incrementally. Scrum, Kanban, and other frameworks provide structures for organizing agile development. Daily standups, retrospectives, and demonstrations foster communication and continuous improvement.

The relationship between developers and operations teams has evolved significantly. Traditional approaches separated these functions, often leading to friction when software was thrown over the wall to operations. DevOps culture emphasizes collaboration and shared responsibility for delivering and operating software. Practices like infrastructure as code and continuous deployment blur the line between development and operations.

Leadership in software organizations requires technical understanding combined with people skills. Technical leaders must make difficult decisions about architecture, technology choices, and technical investments. They must also build effective teams, develop talent, and create environments where people can do their best work. The transition from individual contributor to leader requires developing new skills and perspectives.

The economics of software development differ from those of physical products. Software can be copied at essentially zero marginal cost, enabling business models based on services, subscriptions, and network effects. Open source software creates shared value that can be built upon by many parties. The economics of platforms and ecosystems create powerful dynamics that shape competition and innovation.

Software ethics has become an increasingly important topic as software systems take on greater responsibility. Decisions encoded in software affect people's lives in domains from healthcare to criminal justice. Engineers face ethical choices about what to build, how to build it, and whether to participate in projects they find problematic. Professional codes of ethics provide guidance, but individual judgment remains essential.

The craft of software development is a continuous journey of learning and improvement. New languages, frameworks, and tools emerge regularly. Best practices evolve as the field gains experience. Individual developers must invest in their skills to remain effective. Organizations must create environments that support learning and experimentation.

Software development is ultimately a human endeavor, shaped by the people who participate in it. Diversity and inclusion matter for both ethical reasons and because diverse teams produce better outcomes. Psychological safety enables people to take risks, raise concerns, and learn from failures. Building software well requires attending to both technical and human dimensions of the work."""

# Dictionary mapping (seq_len) -> list of texts for different batch indices
PREFILL_TEXTS = {
    128: [_TEXT_128_0, _TEXT_128_1],
    1024: [_TEXT_1024_0, _TEXT_1024_1],
    2048: [_TEXT_2048_0, _TEXT_2048_1],
    4096: [_TEXT_4096_0, _TEXT_4096_1],
    8192: [_TEXT_8192_0, _TEXT_8192_1],
}


def get_prefill_text(seq_len: int, batch_idx: int = 0) -> str:
    """Get prefill text for a given sequence length and batch index.
    
    Args:
        seq_len: Target sequence length (128, 1024, etc.)
        batch_idx: Index within the batch (0 or 1 for batch_size <= 2)
    
    Returns:
        Text string designed to tokenize close to seq_len tokens.
    
    Raises:
        KeyError: If seq_len is not supported.
        IndexError: If batch_idx exceeds available texts.
    """
    if seq_len not in PREFILL_TEXTS:
        raise KeyError(f"seq_len {seq_len} not supported. Available: {list(PREFILL_TEXTS.keys())}")
    texts = PREFILL_TEXTS[seq_len]
    if batch_idx >= len(texts):
        raise IndexError(f"batch_idx {batch_idx} exceeds available texts ({len(texts)}) for seq_len {seq_len}")
    return texts[batch_idx]


def get_prefill_texts_for_batch(seq_len: int, batch_size: int) -> list:
    """Get list of prefill texts for a given sequence length and batch size.
    
    Args:
        seq_len: Target sequence length
        batch_size: Number of texts needed
    
    Returns:
        List of text strings for the batch.
    """
    return [get_prefill_text(seq_len, i % len(PREFILL_TEXTS[seq_len])) for i in range(batch_size)]

