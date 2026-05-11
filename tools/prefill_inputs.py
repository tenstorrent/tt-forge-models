# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prefill input texts for different sequence lengths and batch sizes.

Dictionary structure: PREFILL_TEXTS[seq_len][batch_idx] = text
Each text is designed to tokenize to approximately seq_len tokens to minimize padding.
For batch_size > 1, use multiple texts from the same seq_len bucket.
"""

_TEXT_128_0 = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain."""

_TEXT_128_1 = """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention."""

_TEXT_128_2 = """The theory of general relativity, published by Albert Einstein in 1915, fundamentally changed our understanding of gravity, space, and time. Rather than treating gravity as a force acting at a distance, Einstein described it as a curvature of spacetime caused by mass and energy. This revolutionary framework predicted phenomena such as the bending of light around massive objects, the existence of black holes, and gravitational waves. These predictions have been confirmed through numerous experiments and observations, solidifying general relativity as one of the pillars of modern physics."""

_TEXT_128_3 = """The Renaissance was a cultural movement that began in Italy during the late Middle Ages and later spread to the rest of Europe. This period, spanning roughly from the fourteenth to the seventeenth century, witnessed a renewed interest in classical Greek and Roman culture. Artists such as Leonardo da Vinci, Michelangelo, and Raphael produced masterpieces that continue to inspire generations. The invention of the printing press by Johannes Gutenberg around 1440 accelerated the spread of knowledge and ideas, fundamentally transforming education and intellectual discourse across the continent."""

_TEXT_128_4 = """Cryptography is the practice and study of techniques for securing communication and data in the presence of adversaries. Modern cryptographic methods rely on mathematical algorithms and computational hardness assumptions to ensure confidentiality, integrity, and authenticity of information. Public key cryptography, introduced by Diffie and Hellman in 1976, allows two parties to establish a shared secret over an insecure channel. Today, cryptographic protocols underpin the security of internet communications, financial transactions, and digital identities, making them essential infrastructure for the digital age."""

_TEXT_128_5 = """Marine biology is the scientific study of organisms that inhabit the ocean and other saltwater environments. The ocean covers more than seventy percent of the Earth's surface and contains an extraordinary diversity of life, from microscopic phytoplankton to the largest animal ever known, the blue whale. Coral reefs, sometimes called the rainforests of the sea, support approximately twenty-five percent of all marine species despite covering less than one percent of the ocean floor. Understanding marine ecosystems is critical for addressing challenges such as overfishing, ocean acidification, and the loss of biodiversity."""

_TEXT_128_6 = """The development of the transistor at Bell Labs in 1947 by John Bardeen, Walter Brattain, and William Shockley marked a turning point in the history of electronics. Transistors replaced bulky and unreliable vacuum tubes, enabling the miniaturization of electronic circuits. This breakthrough paved the way for integrated circuits, microprocessors, and the modern computing revolution. Today, a single microprocessor chip can contain billions of transistors, providing computational power that would have been unimaginable to the pioneers who first demonstrated that a semiconductor device could amplify electrical signals."""

_TEXT_128_7 = """Music theory is the study of the practices and possibilities of music. It encompasses the analysis of rhythm, harmony, melody, form, and texture. Western music theory traces its roots to ancient Greece, where philosophers like Pythagoras explored the mathematical relationships between musical intervals. Over centuries, these ideas evolved into the tonal system that underpins most classical and popular music today. Understanding music theory provides musicians with a vocabulary for discussing composition and improvisation, enabling deeper creative expression and more effective collaboration among performers."""


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


_TEXT_1024_2 = """Robotics is an interdisciplinary field that integrates computer science, mechanical engineering, and electrical engineering to design, build, and operate machines capable of performing tasks autonomously or semi-autonomously. The history of robotics stretches back to ancient civilizations that created mechanical automata, but the modern era began in the mid-twentieth century with industrial robots used in manufacturing.

Early industrial robots were simple programmable arms that performed repetitive tasks like welding and painting on assembly lines. These machines operated in structured environments with precisely known geometries. Their success in manufacturing demonstrated the economic value of automation, driving continued investment in robotic technology.

The field underwent a transformation with the development of sensor technology and computer vision. Robots equipped with cameras, lidar, and other sensors could perceive their environment and adapt to changing conditions. This capability opened new applications beyond the factory floor, including search and rescue, medical surgery, and space exploration.

Mobile robotics presented unique challenges in navigation and path planning. Autonomous vehicles must build maps of their environment, localize themselves within those maps, and plan collision-free paths to their destinations. The SLAM problem, simultaneous localization and mapping, became a central research topic that combined insights from probability theory, optimization, and sensor fusion.

Humanoid robots represent one of the most ambitious goals in robotics research. Building machines that can walk, manipulate objects, and interact naturally with humans requires advances in mechanics, control theory, and artificial intelligence. Projects like Boston Dynamics Atlas and Honda ASIMO have demonstrated impressive physical capabilities, though general-purpose humanoid robots remain a work in progress.

Soft robotics is an emerging subfield that draws inspiration from biological organisms. Unlike traditional rigid robots, soft robots are made from compliant materials that can deform and adapt to their environment. These robots excel at tasks that require gentle manipulation, such as handling delicate objects or operating in confined spaces. Applications range from minimally invasive surgery to underwater exploration.

The integration of artificial intelligence with robotics has accelerated progress in recent years. Deep learning enables robots to learn complex behaviors from demonstration and experience rather than explicit programming. Reinforcement learning allows robots to discover optimal strategies through trial and error. These approaches are particularly valuable for tasks that are difficult to specify mathematically.

Human-robot interaction has become increasingly important as robots move from isolated industrial settings into everyday life. Social robots must understand and respond to human emotions, gestures, and language. Designing robots that people trust and feel comfortable working with requires insights from psychology, anthropology, and design. The ethical implications of increasingly capable robots also demand careful consideration."""

_TEXT_1024_3 = """Space exploration has been one of humanity's greatest scientific and technological endeavors. From the launch of Sputnik in 1957 to current missions exploring Mars and the outer solar system, our ventures into space have expanded our understanding of the universe and our place within it.

The Space Race between the United States and the Soviet Union during the Cold War drove rapid advances in rocketry and spacecraft design. The Soviet Union achieved many firsts, including the first satellite, the first human in space with Yuri Gagarin, and the first spacewalk. The United States responded with the Apollo program, which culminated in the historic Moon landing on July 20, 1969, when Neil Armstrong and Buzz Aldrin walked on the lunar surface.

After the Apollo era, space agencies shifted focus to reusable spacecraft and space stations. The Space Shuttle program operated from 1981 to 2011, providing a versatile platform for satellite deployment, scientific research, and construction of the International Space Station. The ISS has been continuously inhabited since November 2000, serving as a laboratory for research in microgravity, biology, and materials science.

Robotic exploration has complemented human spaceflight. Mars rovers including Sojourner, Spirit, Opportunity, and Curiosity have traversed the Martian surface, analyzing rocks and soil for evidence of past water and potential biosignatures. The Perseverance rover, which landed in 2021, collected samples for eventual return to Earth and carried the Ingenuity helicopter, demonstrating powered flight on another planet.

The outer solar system has been explored by spacecraft including Voyager, Cassini, and Juno. These missions revealed the complexity of gas giant atmospheres, the geysers of Enceladus, and the methane lakes of Titan. The Voyager probes, launched in 1977, have now entered interstellar space, carrying golden records with sounds and images from Earth.

Commercial space companies have transformed the economics of space access. SpaceX developed reusable rockets that dramatically reduced launch costs. Blue Origin and other companies are pursuing their own reusable launch vehicles. This commercial revolution has enabled a new generation of space ventures, from satellite internet constellations to space tourism.

The search for extraterrestrial life drives many space science missions. Astrobiologists study extremophiles on Earth to understand the limits of life. Telescopes like the James Webb Space Telescope observe exoplanet atmospheres for biosignature gases. Whether life exists beyond Earth remains one of the most profound unanswered questions in science.

Future plans include returning humans to the Moon through the Artemis program, establishing permanent lunar bases, and eventually sending crews to Mars. These ambitious goals require advances in propulsion, life support, radiation protection, and in-situ resource utilization. International cooperation and commercial partnerships will be essential for realizing these visions."""

_TEXT_1024_4 = """The human brain is the most complex organ in the known universe, containing approximately eighty-six billion neurons connected by trillions of synapses. Understanding how this intricate network gives rise to perception, thought, emotion, and consciousness is one of the greatest challenges in science.

Neuroscience has made remarkable progress since Santiago Ramon y Cajal first described the neuron as the basic unit of the nervous system in the late nineteenth century. Modern techniques including functional magnetic resonance imaging, electroencephalography, and optogenetics allow researchers to observe and manipulate brain activity with unprecedented precision.

The cerebral cortex, the outermost layer of the brain, is responsible for higher cognitive functions. It is organized into distinct regions that specialize in processing different types of information. The visual cortex processes images from the eyes, the auditory cortex handles sound, and the prefrontal cortex supports planning and decision-making. However, these regions do not work in isolation but communicate through complex networks.

Memory is one of the most studied cognitive functions. The hippocampus plays a crucial role in forming new memories and consolidating them from short-term to long-term storage. Different types of memory, including episodic, semantic, and procedural, appear to involve distinct neural circuits. Sleep plays an important role in memory consolidation, with specific sleep stages contributing to different aspects of memory processing.

Neuroplasticity, the brain's ability to reorganize itself by forming new neural connections, has transformed our understanding of brain function. This capacity allows the brain to adapt to new experiences, recover from injury, and learn throughout life. Research on neuroplasticity has led to improved rehabilitation therapies for stroke and brain injury patients.

Mental health disorders affect hundreds of millions of people worldwide. Depression, anxiety, schizophrenia, and other conditions involve complex interactions between genetic, environmental, and neurochemical factors. Advances in understanding the neural basis of these disorders have led to new pharmacological and therapeutic approaches, though much remains to be learned.

Brain-computer interfaces represent a frontier where neuroscience meets engineering. These devices record neural activity and translate it into commands for external devices. Paralyzed patients have used brain-computer interfaces to control robotic arms, type messages, and even regain limited movement. As these technologies mature, they promise to restore function and enhance human capabilities.

The relationship between the brain and artificial neural networks has been mutually illuminating. Insights from neuroscience have inspired the architecture of deep learning systems, while computational models have provided new ways to understand brain function. This interdisciplinary exchange continues to drive progress in both fields, with implications for medicine, technology, and our understanding of intelligence itself."""

_TEXT_1024_5 = """Renewable energy technologies have emerged as critical tools in addressing climate change and ensuring sustainable development. Solar, wind, hydroelectric, and other clean energy sources are rapidly displacing fossil fuels in electricity generation, driven by falling costs and supportive policies.

Solar photovoltaic technology converts sunlight directly into electricity using semiconductor materials. The efficiency of commercial solar cells has improved steadily, from around six percent in early devices to over twenty-five percent in modern monocrystalline panels. Manufacturing scale has driven dramatic cost reductions, making solar the cheapest source of electricity in many regions. Perovskite solar cells represent a promising next-generation technology that could further reduce costs.

Wind energy harnesses the kinetic energy of moving air through turbines. Onshore wind farms are now a mature technology deployed at scale around the world. Offshore wind offers access to stronger and more consistent winds, though at higher installation costs. Floating wind platforms are extending offshore wind to deeper waters where fixed foundations are impractical.

Energy storage is essential for managing the intermittency of solar and wind power. Lithium-ion batteries have become the dominant storage technology, with costs declining by more than ninety percent over the past decade. Alternative storage approaches including flow batteries, compressed air, and green hydrogen are being developed for longer-duration applications. Grid-scale storage enables utilities to shift renewable generation to match demand patterns.

Smart grid technologies are modernizing electricity infrastructure to accommodate distributed renewable generation. Advanced sensors, communication networks, and control systems enable real-time monitoring and optimization of power flows. Demand response programs incentivize consumers to shift electricity use to periods of abundant renewable generation. These digital capabilities improve grid reliability and reduce the need for fossil fuel backup.

The electrification of transportation represents a major opportunity for emissions reduction. Electric vehicles powered by renewable electricity produce zero tailpipe emissions and are increasingly cost-competitive with conventional vehicles. Charging infrastructure is expanding rapidly, and battery technology improvements are extending driving range. Heavy-duty transport, shipping, and aviation present greater challenges but are also seeing electrification and alternative fuel progress.

Geothermal energy taps heat from the Earth's interior. Conventional geothermal plants operate at sites with naturally occurring hot water or steam. Enhanced geothermal systems could expand access by creating artificial reservoirs in hot dry rock formations. This technology could provide baseload renewable power independent of weather conditions.

The transition to renewable energy creates both opportunities and challenges for communities worldwide. New jobs are being created in manufacturing, installation, and maintenance of clean energy systems. However, fossil fuel workers and communities face displacement that requires proactive policy support. Ensuring an equitable energy transition that benefits all communities is both a moral imperative and a practical necessity for sustained political support."""

_TEXT_1024_6 = """Philosophy of mind is a branch of philosophy that examines the nature of consciousness, mental states, and the relationship between mind and body. These questions have occupied thinkers for millennia and have gained new urgency with advances in neuroscience and artificial intelligence.

The mind-body problem, first articulated clearly by Rene Descartes in the seventeenth century, asks how mental phenomena relate to physical processes. Descartes proposed substance dualism, arguing that mind and body are fundamentally different kinds of substance. While few philosophers today accept Cartesian dualism, the intuition that consciousness cannot be fully explained by physical processes remains influential.

Physicalism holds that everything that exists is physical or depends on the physical. Identity theory identifies mental states with brain states, while functionalism defines mental states by their causal roles rather than their physical composition. These materialist approaches face challenges from thought experiments like Mary's Room, which suggests that knowing all physical facts about color vision might not capture the subjective experience of seeing red.

The hard problem of consciousness, articulated by David Chalmers, distinguishes between easy and hard problems in the science of mind. Easy problems concern how the brain processes information, discriminates stimuli, and controls behavior. The hard problem asks why these processes are accompanied by subjective experience at all. Why does information processing feel like something from the inside?

Intentionality, the mind's capacity to be about or directed toward objects and states of affairs, is another central topic. When you think about Paris, your thought has content that represents the city. How physical systems can have such representational properties is deeply puzzling. Theories range from causal accounts that link mental representations to their real-world referents to interpretationist views that ground meaning in patterns of behavior.

Free will is a perennial concern at the intersection of philosophy of mind and ethics. If our actions are determined by prior physical causes, including brain states we did not choose, in what sense are we free? Compatibilists argue that freedom is compatible with determinism, defining it as acting in accordance with one's desires without external coercion. Libertarians about free will maintain that genuine freedom requires causal indeterminism.

The problem of other minds asks how we can know that other beings have conscious experiences. We have direct access only to our own consciousness and must infer mental states in others from behavior and neural activity. This problem extends to questions about animal consciousness and the possibility of machine consciousness.

Advances in artificial intelligence have reinvigorated these philosophical debates. If a machine behaves indistinguishably from a conscious being, should we attribute consciousness to it? The Turing test and Chinese Room argument represent contrasting perspectives on this question. As AI systems become more sophisticated, the practical implications of these philosophical questions grow more pressing."""

_TEXT_1024_7 = """The history of mathematics spans thousands of years and encompasses contributions from civilizations around the world. From the earliest counting systems to the abstract structures of modern algebra and topology, mathematics has been essential to scientific progress and technological innovation.

Ancient Mesopotamian civilizations developed sophisticated numerical systems and mathematical techniques for commerce, surveying, and astronomy. The Babylonians used a base-sixty number system and could solve quadratic equations. Egyptian mathematicians developed practical geometry for land measurement and construction of monuments including the pyramids.

Greek mathematics emphasized proof and abstraction. Euclid's Elements, compiled around 300 BCE, organized geometry into a deductive system built from axioms and postulates. This axiomatic method became the model for rigorous mathematical reasoning. Archimedes developed methods for calculating areas and volumes that anticipated integral calculus. The discovery of irrational numbers challenged Pythagorean beliefs about the harmony of whole numbers.

During the medieval period, Islamic mathematicians made fundamental advances. Al-Khwarizmi's work on algebra gave the field its name. Persian and Arab scholars preserved and extended Greek mathematical knowledge, developing trigonometry and advancing number theory. Indian mathematicians independently developed the decimal place-value system and the concept of zero.

The seventeenth century saw the invention of calculus by Newton and Leibniz, one of the most important developments in mathematical history. Calculus provided tools for analyzing motion, change, and continuous quantities. It became the mathematical foundation for physics, engineering, and eventually economics and biology.

The nineteenth century brought a revolution in mathematical rigor and abstraction. Cauchy and Weierstrass placed calculus on a firm logical foundation. Galois and Abel developed group theory, revealing deep structures underlying polynomial equations. Riemann generalized geometry beyond Euclidean space, providing the mathematical framework that Einstein would later use for general relativity.

Set theory, developed by Cantor in the late nineteenth century, provided a foundation for all of mathematics but also revealed paradoxes that shook the foundations of mathematical logic. Russell's paradox and Godel's incompleteness theorems showed that any consistent formal system powerful enough to express arithmetic must contain true statements that cannot be proved within the system.

Modern mathematics is characterized by increasing abstraction and interconnection. Category theory provides a language for describing mathematical structures and their relationships. The Langlands program seeks to unify disparate areas of mathematics through deep correspondences. Applied mathematics has expanded into new domains including data science, computational biology, and financial modeling, demonstrating the continuing relevance of mathematical thinking to human endeavors."""


_TEXT_2048_0 = (
    _TEXT_1024_0
    + """

The intersection of artificial intelligence and scientific discovery represents one of the most exciting frontiers in modern research. AI systems are now being used to accelerate drug discovery, predict protein structures, and analyze astronomical data. The AlphaFold system, developed by DeepMind, solved the protein folding problem that had challenged biologists for decades. This breakthrough demonstrates how AI can tackle problems that were previously considered intractable.

In materials science, machine learning models are being used to predict the properties of new materials before they are synthesized. This approach dramatically accelerates the discovery of materials with desired characteristics, such as superconductors, battery materials, and catalysts. The traditional trial-and-error approach to materials discovery is being replaced by intelligent search guided by learned patterns.

Climate science has also benefited from AI advances. Machine learning models can process the vast amounts of data generated by climate sensors, satellites, and simulations. These models help scientists understand complex climate dynamics, improve weather predictions, and assess the impacts of various intervention strategies. The ability to process and find patterns in massive datasets has opened new avenues for climate research.

The healthcare industry is undergoing a transformation driven by AI technologies. Medical imaging systems powered by deep learning can detect diseases with accuracy matching or exceeding human experts. Natural language processing systems can extract insights from medical records, helping clinicians make more informed decisions. Personalized medicine, which tailors treatments to individual patients based on their genetic and health data, is becoming increasingly practical.

Autonomous systems represent another area of rapid AI advancement. Self-driving cars, once the domain of science fiction, are now being tested on public roads. Autonomous drones are being used for delivery, agriculture, and search and rescue operations. Robots powered by AI are becoming more capable in manufacturing, logistics, and even household tasks. These systems combine perception, planning, and control in increasingly sophisticated ways.

The economics of AI are reshaping industries and labor markets. Automation powered by AI is changing the nature of work, displacing some jobs while creating others. Companies that effectively leverage AI gain competitive advantages, leading to market concentration in some sectors. Policymakers are grappling with questions about how to ensure that the benefits of AI are widely shared while managing the disruptions it causes."""
)

_TEXT_2048_1 = (
    _TEXT_1024_1
    + """

The evolution of programming paradigms has profoundly influenced how we think about and construct software. Structured programming, introduced in the 1960s and 1970s, emphasized the use of control structures like loops and conditionals rather than arbitrary jumps. This approach made programs easier to understand and maintain, establishing principles that remain relevant today.

Object-oriented programming emerged as a dominant paradigm in the 1980s and 1990s. By organizing code around objects that combine data and behavior, this approach facilitated the development of complex systems. Languages like C++, Java, and Python brought object-oriented concepts to mainstream software development. Design patterns, reusable solutions to common problems, became an important part of software engineering practice.

Functional programming, with roots in lambda calculus, has gained renewed interest in recent years. By treating computation as the evaluation of mathematical functions without side effects, functional programming offers advantages for concurrent and parallel systems. Languages like Haskell, Scala, and Clojure have brought functional concepts to practical software development. Even traditionally imperative languages have incorporated functional features.

The rise of web development created new challenges and opportunities for software engineering. The client-server model evolved into sophisticated web applications with rich user interfaces. JavaScript, initially designed for simple browser scripting, became a full-featured language powering both front-end and back-end development. Frameworks and libraries abstracted away low-level details, enabling rapid application development.

Mobile computing introduced additional complexity with its constraints on power, bandwidth, and screen size. Developing applications that work across different devices and platforms became a significant challenge. Native development, web-based approaches, and cross-platform frameworks each offered different tradeoffs. The app economy created new business models and distribution channels for software.

Cloud computing transformed how software is deployed and operated. Rather than managing physical servers, developers can provision virtual resources on demand. Services like compute instances, databases, and message queues are available as commodities. This shift enabled new architectures, including microservices and serverless computing, that emphasize scalability and resilience. DevOps practices emerged to manage the complexity of continuous deployment and operation."""
)


_TEXT_2048_2 = (
    _TEXT_1024_2
    + """

Swarm robotics draws inspiration from social insects like ants and bees to coordinate large numbers of simple robots. Rather than relying on a central controller, swarm systems use local interactions between robots to produce emergent collective behavior. This approach offers robustness and scalability advantages for tasks like environmental monitoring, search operations, and distributed construction.

Agricultural robotics is transforming farming practices through precision agriculture. Autonomous tractors, drone-based crop monitoring, and robotic harvesting systems reduce labor requirements while improving efficiency and sustainability. Computer vision enables robots to identify and selectively treat individual plants, reducing pesticide use and environmental impact.

Surgical robots have revolutionized minimally invasive procedures. Systems like the da Vinci surgical robot provide surgeons with enhanced dexterity, precision, and visualization. Teleoperation allows specialists to perform procedures remotely. Research into autonomous surgical capabilities promises to further improve outcomes and expand access to expert-level surgical care.

The ethical dimensions of robotics are becoming increasingly important. Questions about autonomous weapons, job displacement, privacy, and liability require thoughtful consideration by engineers, policymakers, and society at large. Establishing governance frameworks that promote beneficial uses of robotics while mitigating risks is essential for responsible technological development.

Service robots are entering homes and public spaces in growing numbers. Robotic vacuum cleaners, lawn mowers, and pool cleaners handle household chores. Delivery robots navigate sidewalks and corridors to transport packages and meals. Social companion robots provide assistance and companionship to elderly and disabled individuals. As these robots become more capable and affordable, they will play an increasingly significant role in daily life.

The materials used in robot construction are evolving rapidly. Advanced composites provide high strength-to-weight ratios for mobile robots. Shape-memory alloys enable actuators that mimic biological muscles. Biodegradable materials are being explored for robots designed for environmental applications where retrieval may not be feasible."""
)

_TEXT_2048_3 = (
    _TEXT_1024_3
    + """

Asteroid mining represents a potential paradigm shift in resource extraction. Near-Earth asteroids contain vast quantities of precious metals, water, and other valuable materials. Water extracted from asteroids could be converted to rocket propellant, establishing refueling stations in space and dramatically reducing the cost of deep space missions.

Space debris has become a growing concern as the number of objects in orbit increases. Collisions between satellites and debris can generate thousands of new fragments, potentially triggering a cascade known as the Kessler syndrome. Active debris removal technologies, including nets, harpoons, and laser systems, are being developed to address this threat to the orbital environment.

The search for habitable exoplanets has yielded remarkable discoveries. The Kepler and TESS missions have identified thousands of planets orbiting other stars, including many in the habitable zone where liquid water could exist. Atmospheric characterization of these worlds using spectroscopy may reveal signs of biological activity, addressing the fundamental question of whether life exists beyond our solar system.

Space-based solar power is an ambitious concept that could provide clean energy to Earth. Solar panels in orbit would collect sunlight continuously without atmospheric interference and beam energy to ground stations using microwaves or lasers. While the engineering challenges are formidable, advances in launch costs and solar technology are making this concept more feasible.

International space law, primarily based on the Outer Space Treaty of 1967, establishes principles for the peaceful use of space. However, increasing commercial activity and the prospect of resource extraction raise questions about property rights and governance that existing treaties do not fully address. Developing appropriate legal frameworks for the new space economy is an active area of international discussion.

Interstellar travel remains a distant goal but one that inspires significant research. Concepts including solar sails, nuclear pulse propulsion, and laser-powered spacecraft offer theoretical paths to reaching nearby stars within human timescales. The Breakthrough Starshot initiative aims to send tiny probes to Alpha Centauri using powerful ground-based lasers, demonstrating that interstellar exploration may be achievable with near-term technology."""
)

_TEXT_2048_4 = (
    _TEXT_1024_4
    + """

Computational neuroscience uses mathematical models and simulations to understand brain function. Large-scale brain simulations aim to replicate the activity of millions of neurons and their connections. The Human Brain Project and the Allen Brain Atlas are ambitious efforts to map brain structure and function at unprecedented resolution.

Neurodegenerative diseases including Alzheimer's, Parkinson's, and Huntington's disease cause progressive loss of neurons and cognitive function. Research into the molecular mechanisms of these diseases has identified protein misfolding and aggregation as common pathological features. Developing effective treatments remains one of the most important challenges in biomedical research.

The gut-brain axis describes the bidirectional communication between the gastrointestinal tract and the central nervous system. The gut microbiome influences brain function and behavior through neural, hormonal, and immune pathways. This connection has implications for understanding mood disorders, neurodevelopmental conditions, and the effects of diet on mental health.

Developmental neuroscience studies how the brain forms and matures from embryonic stages through adulthood. The precise orchestration of neuronal migration, axon guidance, and synapse formation creates the complex circuits that underlie behavior. Critical periods of heightened plasticity during childhood enable rapid learning of language and other skills.

Neuroethics examines the ethical implications of advances in brain science. Issues include the privacy of neural data, the fairness of cognitive enhancement technologies, and the legal implications of brain-based evidence. As neurotechnology becomes more powerful, these ethical considerations will become increasingly important for policymakers and society.

The connectome, a comprehensive map of neural connections in the brain, is a major goal of modern neuroscience. Complete connectomes have been mapped for simple organisms like C. elegans, but mapping the human connectome at synaptic resolution remains an enormous challenge. Partial connectomes and statistical models of brain connectivity are providing valuable insights into how brain structure supports function."""
)

_TEXT_2048_5 = (
    _TEXT_1024_5
    + """

Green hydrogen, produced by electrolysis of water using renewable electricity, is emerging as a key enabler of deep decarbonization. Hydrogen can store energy over long periods and serve as a feedstock for industrial processes that are difficult to electrify directly. Steel manufacturing, cement production, and chemical synthesis are among the sectors where green hydrogen could replace fossil fuels.

Nuclear energy provides low-carbon baseload electricity and is being reconsidered by many countries as a tool for climate change mitigation. Small modular reactors offer the potential for factory-built standardized units that could reduce construction times and costs. Advanced reactor designs, including molten salt and high-temperature gas reactors, promise improved safety and waste characteristics.

Carbon capture and storage technologies aim to reduce emissions from hard-to-abate sources by capturing carbon dioxide and permanently storing it underground. Direct air capture goes further by removing carbon dioxide that has already been emitted. While currently expensive, these technologies may be necessary to achieve net-zero emissions targets.

Bioenergy uses organic materials including agricultural residues, dedicated energy crops, and waste streams to produce heat, electricity, and transportation fuels. When combined with carbon capture and storage, bioenergy can achieve negative emissions by removing more carbon from the atmosphere than it releases. Sustainable sourcing of biomass feedstocks is essential to ensure that bioenergy delivers genuine climate benefits.

Ocean energy technologies harness the power of waves, tides, and ocean thermal gradients. Tidal stream generators and wave energy converters are being deployed at demonstration scale in favorable locations. While ocean energy is less mature than wind and solar, it offers the advantage of high predictability and energy density.

Energy efficiency improvements reduce the total amount of energy needed to deliver goods and services. Building insulation, efficient appliances, and industrial process optimization can significantly cut energy demand and emissions. Efficiency measures are often the most cost-effective climate solutions, delivering economic benefits alongside environmental improvements."""
)

_TEXT_2048_6 = (
    _TEXT_1024_6
    + """

Panpsychism, the view that consciousness or experience is a fundamental feature of the physical world, has gained renewed attention in recent philosophical discourse. On this view, even basic physical entities have some form of experience, and complex consciousness arises from the combination of these fundamental experiential properties. While counterintuitive, panpsychism avoids the hard problem by denying that consciousness must emerge from wholly non-conscious matter.

Phenomenology, founded by Edmund Husserl, approaches consciousness through careful description of lived experience. Rather than reducing mental life to brain processes, phenomenologists analyze the structures of experience as they appear to the subject. Concepts like intentionality, temporality, and embodiment have been developed through phenomenological investigation and have influenced psychology, cognitive science, and artificial intelligence research.

The extended mind thesis, proposed by Andy Clark and David Chalmers, challenges the assumption that cognition occurs entirely within the skull. When we use a notebook to remember information or a calculator to perform arithmetic, these external tools may literally constitute part of our cognitive system. This thesis has implications for understanding the boundary between mind and environment.

Eliminative materialism, advocated by philosophers like Paul and Patricia Churchland, argues that common-sense psychological categories like beliefs and desires may be fundamentally mistaken. Just as folk theories of physics were replaced by Newtonian mechanics and then relativity, our folk psychology may eventually be replaced by a mature neuroscience. This radical position remains controversial but highlights the potential for scientific progress to revise our self-understanding.

The global workspace theory of consciousness, developed by Bernard Baars, proposes that consciousness arises when information is broadcast widely across the brain. Information that enters the global workspace becomes available to multiple cognitive processes including attention, memory, and decision-making. This theory has generated testable predictions and has been supported by neuroimaging evidence.

Integrated information theory, developed by Giulio Tononi, proposes a mathematical measure of consciousness called phi. According to this theory, consciousness corresponds to integrated information, and any system with non-zero phi has some degree of experience. This framework provides a principled approach to assessing consciousness in biological and artificial systems, though measuring phi in complex systems remains computationally challenging."""
)

_TEXT_2048_7 = (
    _TEXT_1024_7
    + """

Probability theory and statistics have become indispensable tools across science and technology. The foundations of probability were laid by Pascal and Fermat in the seventeenth century through their analysis of games of chance. Bayesian statistics, named after Reverend Thomas Bayes, provides a framework for updating beliefs in light of new evidence and has become increasingly important in machine learning and data science.

Topology studies properties of spaces that are preserved under continuous deformations. The Euler characteristic, fundamental group, and homology groups are topological invariants that classify spaces up to deformation equivalence. Topological methods have found surprising applications in data analysis, materials science, and quantum computing.

Number theory, once considered the purest branch of mathematics with no practical applications, has become central to modern cryptography. The difficulty of factoring large numbers underpins the RSA encryption algorithm. Elliptic curve cryptography uses the algebraic structure of elliptic curves over finite fields. The potential development of quantum computers that could break these cryptographic systems motivates research into post-quantum cryptographic algorithms.

Combinatorics studies the enumeration and arrangement of discrete structures. Graph theory, a major branch of combinatorics, models relationships between objects and has applications in network analysis, scheduling, and circuit design. The four color theorem, proved with computer assistance in 1976, demonstrated that any map can be colored with at most four colors such that no two adjacent regions share a color.

Mathematical logic has deep connections to computer science. The lambda calculus of Alonzo Church and the Turing machines of Alan Turing formalized the concept of computation. Type theory provides a foundation for programming language design. Proof assistants and automated theorem provers use logical frameworks to verify mathematical proofs and software correctness.

The interface between mathematics and biology is a growing area of research. Mathematical models describe population dynamics, disease spread, gene regulatory networks, and protein folding. Computational biology uses algorithms and statistical methods to analyze genomic data. The increasing availability of biological data is creating new opportunities for mathematical analysis and discovery."""
)

_TEXT_4096_0 = (
    _TEXT_2048_0
    + """

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
)

_TEXT_4096_1 = (
    _TEXT_2048_1
    + """

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
)


_TEXT_4096_2 = (
    _TEXT_2048_2
    + """

Underwater robotics operates in one of the most challenging environments on Earth. Autonomous underwater vehicles must contend with extreme pressure, limited visibility, corrosive saltwater, and unreliable communication. Despite these challenges, underwater robots are essential tools for ocean science, offshore energy, telecommunications cable maintenance, and deep-sea exploration.

The development of robotic manipulators has paralleled advances in control theory. Early robots used simple position control, while modern systems employ force and impedance control to handle delicate tasks. Compliant control strategies allow robots to interact safely with humans and adapt to uncertain environments. Learning-based control methods are increasingly being used to handle tasks that are difficult to model analytically.

Robot perception has advanced significantly with the development of deep learning. Convolutional neural networks enable robots to recognize objects and scene elements in images. Point cloud processing networks analyze three-dimensional sensor data from lidar and depth cameras. Multimodal perception systems combine information from multiple sensor types to build rich representations of the environment.

The concept of digital twins, virtual replicas of physical robots and their environments, is transforming robot development and deployment. Engineers can test control algorithms and plan operations in simulation before deploying them on physical hardware. Digital twins enable predictive maintenance by modeling wear and degradation. This approach reduces development time and risk while improving operational efficiency.

Collaborative robots, or cobots, are designed to work alongside humans in shared workspaces. Unlike traditional industrial robots that operate behind safety barriers, cobots use force limiting, speed reduction, and compliance to ensure safe physical interaction. Cobots are particularly valuable for small and medium enterprises that need flexible automation without the cost of redesigning their work environment.

Legged locomotion enables robots to traverse terrain that wheeled vehicles cannot navigate. Quadruped robots like Boston Dynamics' Spot can climb stairs, walk over rubble, and maintain balance on uneven surfaces. Research into dynamic locomotion has produced robots capable of running, jumping, and even performing backflips. Biologically inspired gaits and reflexes contribute to robust locomotion in challenging environments.

The economics of robotics are evolving as hardware costs decrease and software capabilities increase. Robotics-as-a-service models allow companies to access robotic capabilities without large capital investments. Open-source software frameworks like ROS provide standardized tools for robot development. Cloud robotics leverages remote computing resources for perception, planning, and learning.

The future of robotics lies in systems that are more adaptable, capable, and integrated into daily life. Robots that can learn continuously from their experiences, communicate effectively with humans, and operate reliably in unstructured environments will unlock new applications across healthcare, agriculture, logistics, and personal assistance. Achieving this vision requires continued advances in hardware, software, and our understanding of intelligence itself."""
)

_TEXT_4096_3 = (
    _TEXT_2048_3
    + """

The development of in-space manufacturing could enable the construction of structures too large or fragile to launch from Earth. Three-dimensional printing in microgravity has been demonstrated on the International Space Station. Future space factories could produce precision optical components, pharmaceuticals, and advanced materials that benefit from the unique conditions of space.

Planetary defense against asteroid impacts has become an active area of research and policy. The DART mission successfully demonstrated kinetic impact deflection by altering the orbit of the asteroid Dimorphos. Future planetary defense capabilities may include gravity tractors, ion beam deflection, and nuclear devices. Early detection through surveys like the Vera Rubin Observatory is essential for providing sufficient warning time.

Space weather, driven primarily by solar activity, poses risks to satellites, power grids, and astronaut health. Coronal mass ejections can damage satellite electronics, disrupt GPS signals, and induce dangerous currents in electrical infrastructure. Space weather forecasting relies on solar observatories and models of the solar wind to provide advance warning of geomagnetic storms.

The commercialization of low Earth orbit is accelerating as private companies develop space stations, manufacturing facilities, and research platforms. Axiom Space and other companies plan commercial space stations that will complement and eventually succeed the International Space Station. These facilities will support scientific research, tourism, and industrial production in microgravity.

Life support systems for long-duration space missions must recycle air, water, and waste with high efficiency. Current systems on the ISS recover about ninety percent of water from humidity and urine. Future missions to Mars will require closed-loop systems that minimize resupply from Earth. Bioregenerative life support, which uses plants and microorganisms to regenerate resources, is being studied for very long missions.

The psychological challenges of long-duration space flight are as significant as the technical ones. Crew members on Mars missions will face isolation, confinement, communication delays, and the stress of living in a hostile environment for months or years. Research in analog environments like Antarctic stations and underwater habitats provides insights into crew selection, training, and support strategies.

Space telescopes have revolutionized astronomy by observing the universe above the distorting effects of Earth's atmosphere. The Hubble Space Telescope has operated for over three decades, producing iconic images and groundbreaking science. The James Webb Space Telescope, with its much larger mirror and infrared capabilities, is revealing the earliest galaxies and studying the atmospheres of potentially habitable exoplanets.

International cooperation has been a hallmark of space exploration, but geopolitical tensions increasingly affect space activities. The partnership model of the ISS faces challenges as national space programs pursue independent capabilities. Establishing norms of behavior and cooperative frameworks for space activities is important for preventing conflict and ensuring sustainable use of the space environment."""
)

_TEXT_4096_4 = (
    _TEXT_2048_4
    + """

Neurotechnology encompasses devices and methods that interface directly with the nervous system. Deep brain stimulation uses implanted electrodes to modulate neural activity and has proven effective for treating Parkinson's disease, essential tremor, and some forms of depression. Non-invasive neurostimulation techniques including transcranial magnetic stimulation and transcranial direct current stimulation offer lower-risk alternatives.

The study of sleep has revealed its essential role in brain function and health. Sleep consists of multiple stages including rapid eye movement sleep and various stages of non-REM sleep. Each stage serves distinct functions including memory consolidation, metabolic waste clearance, and neural circuit maintenance. Chronic sleep deprivation is associated with cognitive impairment, mood disorders, and increased risk of neurodegenerative disease.

Addiction involves complex changes in brain reward circuits that make it difficult for affected individuals to control substance use despite harmful consequences. The mesolimbic dopamine system plays a central role in the rewarding effects of drugs and natural rewards. Understanding the neurobiological basis of addiction has led to pharmacological treatments and has informed public health approaches that treat addiction as a medical condition rather than a moral failing.

Language is one of the most remarkable capabilities of the human brain. The neural basis of language involves a distributed network including Broca's area for production and Wernicke's area for comprehension. Modern neuroimaging studies have revealed that language processing is more distributed and dynamic than classical models suggested. Bilingualism has been shown to enhance executive function and may delay the onset of dementia.

Artificial neural networks, while inspired by biological neurons, differ from their biological counterparts in important ways. Biological neurons communicate through complex temporal patterns of electrical and chemical signals. Synaptic plasticity involves multiple molecular mechanisms operating at different timescales. Understanding these differences may reveal principles that could improve artificial systems.

The cerebellum, often associated primarily with motor coordination, has been found to contribute to cognitive and emotional functions as well. Its massive computational capacity and connections with the cerebral cortex suggest a role in timing, prediction, and learning across multiple domains. Damage to the cerebellum can produce cognitive deficits in addition to the well-known motor impairments.

Epigenetics, the study of heritable changes in gene expression that do not involve changes to the DNA sequence, is revealing how environmental factors influence brain development and function. Stress, nutrition, and social experience can alter gene expression through DNA methylation and histone modification. These epigenetic changes can affect neural development and may even be transmitted across generations.

Consciousness research has become increasingly interdisciplinary, drawing on philosophy, neuroscience, psychology, and computer science. Theories of consciousness are being tested through carefully designed experiments that compare brain activity during conscious and unconscious processing. The development of reliable measures of consciousness has practical implications for anesthesia, disorders of consciousness, and the assessment of non-human and artificial systems."""
)

_TEXT_4096_5 = (
    _TEXT_2048_5
    + """

The circular economy model aims to eliminate waste and maximize resource utilization by designing products and systems for reuse, repair, and recycling. In the energy sector, this means designing renewable energy equipment for end-of-life recovery of valuable materials. Recycling solar panels, wind turbine blades, and batteries is becoming an important industrial capability as early installations reach the end of their operational lives.

Microgrids are small-scale power systems that can operate independently or in connection with the main electrical grid. They combine distributed generation, storage, and intelligent controls to provide reliable power to communities, campuses, and military installations. Microgrids improve resilience against grid outages and can integrate high fractions of renewable energy.

The water-energy nexus describes the interdependence between water and energy systems. Electricity generation requires water for cooling and hydropower. Water treatment and distribution consume significant amounts of energy. Climate change is affecting both water availability and energy demand. Understanding and managing these interconnections is essential for sustainable resource management.

Community-owned renewable energy projects enable local stakeholders to invest in and benefit from clean energy development. Energy cooperatives, community solar gardens, and locally owned wind farms distribute economic benefits more broadly than developer-owned projects. These models can increase public acceptance of renewable energy and contribute to energy democracy.

Advances in power electronics are enabling more efficient and flexible energy systems. Wide-bandgap semiconductors like silicon carbide and gallium nitride offer lower losses and higher switching frequencies than conventional silicon devices. These materials are improving the performance of solar inverters, electric vehicle chargers, and grid-scale power converters.

The role of financial markets in accelerating the energy transition has grown significantly. Green bonds, sustainability-linked loans, and carbon markets direct capital toward clean energy investments. Institutional investors are increasingly incorporating climate risk into their decision-making. The Task Force on Climate-related Financial Disclosures has established frameworks for reporting climate-related financial risks.

Digital technologies are creating new opportunities for energy system optimization. Artificial intelligence and machine learning are being applied to weather forecasting for renewable generation prediction, predictive maintenance of energy infrastructure, and optimization of building energy management. The Internet of Things enables real-time monitoring of energy flows at unprecedented granularity.

Just transition policies recognize that the shift to clean energy must address the social and economic impacts on workers and communities dependent on fossil fuel industries. Retraining programs, economic diversification support, and community investment funds can help ensure that the benefits of the energy transition are widely shared. International cooperation is needed to support developing countries in their clean energy pathways."""
)

_TEXT_4096_6 = (
    _TEXT_2048_6
    + """

Higher-order theories of consciousness propose that a mental state is conscious when it is the object of a higher-order representation. On this view, a perceptual experience becomes conscious when you are aware of having that experience. Higher-order thought theory and higher-order perception theory offer different accounts of how this meta-awareness is achieved. Critics argue that these theories face difficulties explaining how the higher-order representation adds phenomenal character to the first-order state.

The philosophical zombie thought experiment imagines a being physically identical to a conscious person but lacking subjective experience. If such a being is conceivable, it seems to show that consciousness is not entailed by physical facts alone. This argument has been used to support property dualism, the view that consciousness involves non-physical properties. Physicalists respond by questioning whether zombies are truly conceivable or by arguing that conceivability does not establish metaphysical possibility.

Embodied cognition challenges the traditional view that the mind is essentially a computational device manipulating abstract symbols. Instead, proponents argue that cognition is deeply shaped by the body's morphology, sensory systems, and motor capabilities. Our understanding of concepts like warmth, weight, and distance is grounded in bodily experience. This perspective has influenced robotics, artificial intelligence, and educational practice.

Predictive processing theories propose that the brain is fundamentally a prediction machine that continuously generates models of the world and updates them based on sensory input. Perception occurs when top-down predictions are compared with bottom-up sensory signals. Prediction errors drive learning and attention. This framework has been applied to understanding perception, action, emotion, and psychopathology.

The relationship between attention and consciousness is a topic of ongoing investigation. Some researchers argue that attention is necessary and sufficient for consciousness, while others maintain that the two can be dissociated. Inattentional blindness demonstrates that unattended stimuli can go unnoticed even when in plain view. Whether these stimuli are genuinely unconscious or merely unreportable remains debated.

Animal consciousness raises important ethical and scientific questions. Comparative neuroanatomy and behavioral studies suggest that many animals have rich subjective experiences. The Cambridge Declaration on Consciousness affirmed that many non-human animals possess the neurological substrates that generate consciousness. This recognition has implications for animal welfare legislation and our understanding of the evolution of mind.

The relationship between language and thought has been debated throughout the history of philosophy. The Sapir-Whorf hypothesis, in its strong form, holds that language determines thought. While the strong version is generally rejected, research demonstrates that language influences perception, categorization, and memory. The diversity of human languages provides a natural laboratory for studying these effects.

Consciousness science increasingly employs formal mathematical frameworks to develop testable theories. Information-theoretic approaches, dynamical systems theory, and category theory have all been applied to the study of consciousness. While no complete theory has been achieved, the mathematical formalization of consciousness theories represents progress toward a more rigorous science of subjective experience."""
)

_TEXT_4096_7 = (
    _TEXT_2048_7
    + """

Differential equations are fundamental tools for modeling dynamic systems across science and engineering. Ordinary differential equations describe systems that change with respect to a single variable, while partial differential equations model systems with multiple independent variables. The Navier-Stokes equations for fluid dynamics, Maxwell's equations for electromagnetism, and the Schrodinger equation for quantum mechanics are all partial differential equations whose solutions describe physical phenomena.

Abstract algebra studies algebraic structures including groups, rings, and fields. Group theory classifies symmetries and has applications in physics, chemistry, and crystallography. Ring theory generalizes arithmetic operations beyond the integers. Field theory provides the algebraic framework for Galois theory, which explains why there is no general formula for solving polynomial equations of degree five or higher.

Geometry has evolved from Euclid's study of shapes and sizes to a rich collection of interrelated disciplines. Differential geometry studies curved spaces using calculus and provides the mathematical framework for general relativity. Algebraic geometry studies geometric objects defined by polynomial equations and has deep connections to number theory. Computational geometry develops algorithms for geometric problems that arise in computer graphics, robotics, and geographic information systems.

Analysis, the rigorous study of limits, continuity, and convergence, has expanded far beyond its origins in calculus. Functional analysis studies infinite-dimensional vector spaces and operators, providing the mathematical foundation for quantum mechanics. Harmonic analysis decomposes functions into component frequencies and underpins signal processing and data compression. Complex analysis studies functions of complex variables and has surprising connections to number theory and physics.

Mathematical optimization seeks to find the best solution from a set of alternatives subject to constraints. Linear programming optimizes linear objective functions subject to linear constraints and has widespread applications in logistics, economics, and engineering. Convex optimization generalizes linear programming and provides efficient algorithms with guaranteed convergence. Non-convex optimization, which includes many machine learning problems, is more challenging but critically important.

Dynamical systems theory studies the long-term behavior of systems that evolve over time. Chaos theory revealed that deterministic systems can exhibit unpredictable behavior due to sensitive dependence on initial conditions. Strange attractors, fractals, and bifurcations are key concepts in this field. Applications range from weather prediction to population biology to financial market modeling.

Information theory, founded by Claude Shannon in 1948, provides a mathematical framework for quantifying information and the limits of communication. Shannon entropy measures the uncertainty in a random variable. Channel capacity determines the maximum rate at which information can be reliably transmitted over a noisy communication channel. These concepts have applications far beyond communications, including statistical physics, biology, and machine learning.

The development of computational mathematics has been driven by the increasing power of digital computers. Numerical methods approximate solutions to problems that cannot be solved analytically. Finite element methods are essential tools for engineering simulation. Monte Carlo methods use random sampling to estimate quantities that are difficult to compute directly. The interplay between mathematical theory and computational practice continues to drive advances in both areas."""
)

_TEXT_8192_0 = (
    _TEXT_4096_0
    + """

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
)

_TEXT_8192_1 = (
    _TEXT_4096_1
    + """

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
)

_TEXT_8192_2 = (
    _TEXT_4096_2
    + """

The field of human-robot collaboration is evolving beyond simple coexistence to genuine partnership. Shared autonomy systems allow humans and robots to jointly control complex tasks, with each partner contributing their strengths. The human provides high-level goals and contextual understanding, while the robot contributes precision, speed, and tireless execution. This paradigm is particularly effective in assembly tasks, surgical procedures, and teleoperation scenarios.

Natural language interaction with robots is becoming more practical thanks to advances in large language models and multimodal AI. Robots that can understand spoken instructions, ask clarifying questions, and explain their actions in natural language are more accessible to non-expert users. Grounding language in physical actions and perceptions remains a challenging research problem that bridges natural language processing and embodied intelligence.

The verification and validation of robotic systems for safety-critical applications requires rigorous methodologies. Formal methods can prove that robot control software satisfies safety specifications. Simulation-based testing evaluates robot behavior across millions of scenarios. Runtime monitoring systems detect anomalous behavior and trigger safe fallback actions. These approaches are essential for certifying autonomous systems in healthcare, transportation, and aerospace.

Robot learning from demonstration allows non-expert users to teach robots new tasks by showing them what to do. Learning from demonstration captures not just the trajectory of motion but also the intent and constraints behind it. Generalization from a small number of demonstrations to new situations remains challenging but critical for making robots truly flexible and useful in unstructured environments.

The miniaturization of robots has opened new application domains. Microrobots can navigate through blood vessels to deliver drugs or perform minimally invasive procedures. Insect-scale flying robots can access confined spaces for inspection and monitoring. Nanorobots are envisioned for targeted therapy at the cellular level. Manufacturing and controlling robots at these scales requires novel actuators, sensors, and power sources.

Multi-robot systems coordinate the actions of multiple robots to accomplish tasks that exceed the capabilities of any individual robot. Coordination strategies range from centralized planning to fully distributed approaches. Multi-robot systems are used for warehouse automation, environmental monitoring, and disaster response. The scalability and robustness of these systems are key advantages over single-robot solutions.

The integration of robots into education is providing new ways for students to learn about science, technology, engineering, and mathematics. Educational robots make abstract concepts tangible and engaging. Programming robots teaches computational thinking and problem-solving skills. Research suggests that robots can serve as effective tutoring partners, adapting to individual student needs and providing patient, consistent instruction.

Ethical frameworks for robotics must address questions of accountability, transparency, and human dignity. When a robot causes harm, who is responsible: the designer, manufacturer, owner, or operator? How transparent should robotic decision-making be? How do we ensure that robots respect human autonomy and privacy? These questions require input from engineers, ethicists, lawyers, policymakers, and the public to develop appropriate governance structures."""
)

_TEXT_8192_3 = (
    _TEXT_4096_3
    + """

The study of planetary atmospheres provides insights into climate processes on Earth and the potential for life elsewhere. Venus's runaway greenhouse effect demonstrates the extreme consequences of atmospheric carbon dioxide accumulation. Mars's thin atmosphere and evidence of past water offer clues about planetary evolution. The thick atmospheres of gas giants contain complex weather systems including massive storms like Jupiter's Great Red Spot.

Gravitational wave astronomy, inaugurated by LIGO's first detection in 2015, has opened a new window on the universe. Gravitational waves are produced by the most violent events in the cosmos, including merging black holes and neutron stars. The observation of gravitational waves from a neutron star merger in 2017, combined with electromagnetic observations, demonstrated the power of multi-messenger astronomy.

The cosmic microwave background radiation, discovered in 1965, provides a snapshot of the universe approximately 380,000 years after the Big Bang. Precise measurements of its temperature fluctuations by satellites including COBE, WMAP, and Planck have revealed the geometry, composition, and age of the universe. These observations support the inflationary model of the early universe and indicate that ordinary matter constitutes only about five percent of the total energy content.

Dark matter and dark energy together account for about ninety-five percent of the universe's energy budget, yet their nature remains mysterious. Dark matter's gravitational effects are observed in galaxy rotation curves and gravitational lensing. Dark energy drives the accelerating expansion of the universe. Experiments including direct detection searches, particle accelerator experiments, and cosmological surveys are working to illuminate these fundamental mysteries.

Astrobiology, the study of the origin, evolution, and distribution of life in the universe, draws on biology, chemistry, geology, and astronomy. Extremophiles on Earth demonstrate that life can thrive in conditions once thought inhospitable. The discovery of subsurface oceans on moons like Europa and Enceladus has expanded the range of potentially habitable environments in our solar system.

Space medicine addresses the health challenges of spaceflight including bone loss, muscle atrophy, cardiovascular deconditioning, and radiation exposure. Countermeasures including exercise protocols, pharmacological interventions, and radiation shielding are being developed. Understanding the long-term effects of space radiation on human health is critical for planning missions to Mars and beyond.

The legal and governance frameworks for space activities are evolving to address new challenges. Questions about the mining of celestial resources, the management of satellite mega-constellations, and the prevention of harmful interference in space require international cooperation and updated regulatory frameworks. The Artemis Accords represent one approach to establishing norms for lunar exploration and resource utilization.

The democratization of space through small satellites and rideshare launches has enabled universities, startups, and developing nations to participate in space activities. CubeSats and other small satellites provide affordable platforms for technology demonstration, Earth observation, and communications. This broader participation enriches the space community and accelerates innovation across diverse applications."""
)

_TEXT_8192_4 = (
    _TEXT_4096_4
    + """

Optogenetics has revolutionized neuroscience by enabling precise control of neuronal activity using light. Genetically modified neurons express light-sensitive proteins called opsins that can activate or silence neural firing in response to specific wavelengths of light. This technique allows researchers to establish causal relationships between neural activity and behavior with unprecedented temporal and spatial precision.

The default mode network, a set of brain regions active during rest and mind-wandering, has become a major focus of neuroscience research. This network is involved in self-referential thought, mental simulation, and memory retrieval. Abnormal default mode network activity has been associated with depression, schizophrenia, and Alzheimer's disease. Understanding this network's function and dysfunction has important implications for mental health.

Neuromorphic computing aims to design computer hardware that mimics the architecture and computational principles of biological neural networks. Neuromorphic chips process information using artificial neurons and synapses that operate with high energy efficiency. Intel's Loihi and IBM's TrueNorth are examples of neuromorphic processors. These chips excel at pattern recognition, sensory processing, and other tasks where biological brains outperform conventional computers.

The microbiome-gut-brain axis is emerging as a critical factor in mental health. Gut bacteria produce neurotransmitters and short-chain fatty acids that influence brain function through the vagus nerve and immune system. Clinical trials are exploring whether probiotics and dietary interventions can alleviate symptoms of anxiety, depression, and autism spectrum disorder. This research suggests that mental health treatment may need to consider the whole body rather than just the brain.

Pain neuroscience has advanced significantly with the identification of specific molecular receptors and neural pathways involved in pain perception. The gate control theory of pain, proposed by Melzack and Wall, explained how non-painful stimuli can modulate pain perception. Modern understanding encompasses peripheral sensitization, central sensitization, and the role of cognitive and emotional factors in pain experience.

Neuroeducation applies findings from neuroscience to improve teaching and learning. Research on memory consolidation suggests that spaced practice is more effective than massed practice. Understanding the neural basis of attention can inform classroom design and instructional strategies. The development of educational interventions based on neuroscience is a growing field, though care must be taken to distinguish established findings from neuromyths.

The development of high-density neural recording technologies is enabling researchers to simultaneously monitor thousands of neurons. Neuropixels probes can record from hundreds of neurons across multiple brain regions. Calcium imaging allows visualization of neural activity across large populations of neurons. These technologies are revealing the distributed nature of neural computations and how information is represented across neural populations.

The intersection of neuroscience and law is creating the field of neurolaw. Brain imaging has been introduced as evidence in criminal cases to support claims about mental state, competency, and culpability. The reliability and interpretation of neuroscientific evidence in legal contexts raise important questions about privacy, free will, and the boundaries of scientific expertise in the courtroom."""
)

_TEXT_8192_5 = (
    _TEXT_4096_5
    + """

Agrivoltaics, the co-location of solar panels and agricultural production, offers a promising approach to addressing competing demands for land use. Partial shading from solar panels can benefit certain crops by reducing water stress and heat damage. Livestock can graze beneath elevated solar panels. Research is quantifying the conditions under which agrivoltaics provides benefits for both energy production and agricultural yield.

The development of long-duration energy storage technologies is critical for achieving high penetrations of renewable energy. Compressed air energy storage, liquid air energy storage, and gravity-based storage systems are being developed for durations of hours to days. Seasonal storage using hydrogen or ammonia could address the challenge of matching renewable generation with demand across months and seasons.

Electric aviation is progressing from small experimental aircraft toward commercial passenger service. Battery-electric aircraft are viable for short-range regional routes. Hybrid-electric designs extend range by combining electric motors with conventional engines. Hydrogen fuel cells offer another pathway to zero-emission flight, particularly for longer routes where battery weight is prohibitive.

The role of indigenous knowledge in sustainable energy development is increasingly recognized. Many indigenous communities have sophisticated understanding of local environments that can inform the siting and design of renewable energy projects. Meaningful consultation and benefit-sharing with indigenous communities is both an ethical obligation and a practical necessity for successful project development.

Heat pumps are emerging as a key technology for decarbonizing building heating. By extracting heat from air, ground, or water sources, heat pumps can deliver multiple units of heat for each unit of electricity consumed. When powered by renewable electricity, heat pumps eliminate direct emissions from building heating. Policy support through incentives and building codes is accelerating heat pump adoption in many countries.

The development of next-generation batteries promises improvements in energy density, cost, and safety. Solid-state batteries replace liquid electrolytes with solid materials, potentially eliminating fire risk and enabling higher energy density. Sodium-ion batteries use abundant materials instead of lithium. Iron-air batteries offer extremely low cost for long-duration storage applications.

Energy access remains a critical challenge in many developing regions. Off-grid solar systems and mini-grids are providing electricity to communities without connection to centralized power networks. Pay-as-you-go financing models and mobile payment platforms are making these systems affordable. Reliable electricity enables improved healthcare, education, and economic opportunity.

The integration of renewable energy with industrial processes is opening new pathways for decarbonization. Electric arc furnaces powered by renewable electricity are replacing coal-fired blast furnaces in steelmaking. Electrochemical processes are being developed for cement and chemical production. The electrification of industrial heat using heat pumps and electric furnaces is becoming economically viable as renewable electricity costs continue to decline."""
)

_TEXT_8192_6 = (
    _TEXT_4096_6
    + """

Moral psychology investigates the psychological processes underlying moral judgment and behavior. Dual-process theories propose that moral judgments arise from both fast, intuitive emotional responses and slower, deliberative reasoning. The trolley problem and its variants have been used extensively to probe the factors that influence moral intuitions. Cross-cultural research reveals both universal patterns and cultural variation in moral psychology.

The philosophy of personal identity asks what makes a person the same individual over time. Physical continuity theories emphasize the persistence of the body or brain. Psychological continuity theories focus on the persistence of memories, personality, and other mental characteristics. Thought experiments involving brain transplants, teleportation, and fission challenge our intuitions and reveal the complexity of identity.

Epistemology, the study of knowledge and justified belief, addresses questions about what we can know and how we can know it. Foundationalism, coherentism, and reliabilism offer different accounts of epistemic justification. The Gettier problem challenged the traditional definition of knowledge as justified true belief. Virtue epistemology shifts focus from properties of beliefs to intellectual virtues of the believer.

Social epistemology examines the social dimensions of knowledge production and dissemination. Testimony is a fundamental source of knowledge, yet it raises questions about trust and credibility. The epistemology of disagreement asks how we should respond when equally competent peers reach different conclusions. The spread of misinformation in the digital age has made social epistemology urgently relevant.

Aesthetics examines the nature of beauty, art, and aesthetic experience. What makes something beautiful? Is beauty in the eye of the beholder or an objective property? What distinguishes art from non-art? These questions have occupied philosophers from Plato to the present. Contemporary aesthetics engages with conceptual art, environmental aesthetics, and the aesthetics of everyday life.

Political philosophy addresses fundamental questions about justice, liberty, equality, and the legitimate exercise of power. John Rawls's theory of justice as fairness proposed that just institutions are those that would be chosen behind a veil of ignorance. Robert Nozick's libertarianism emphasized individual rights and minimal state intervention. The debate between these and other perspectives continues to shape political thought and policy.

Philosophy of science examines the nature of scientific knowledge, explanation, and methodology. The demarcation problem asks what distinguishes science from non-science. Scientific realism holds that successful scientific theories describe the world approximately as it is. Instrumentalism treats theories merely as tools for prediction. The underdetermination of theory by evidence suggests that multiple incompatible theories can be consistent with the same observations.

The relationship between philosophy and artificial intelligence is bidirectional and increasingly productive. AI raises philosophical questions about intelligence, consciousness, and agency. Philosophical analysis helps clarify concepts and assumptions underlying AI systems. Conversely, AI research provides new perspectives on traditional philosophical problems by constructing systems that instantiate different theoretical proposals. This productive interaction promises continued mutual enrichment as both fields advance."""
)

_TEXT_8192_7 = (
    _TEXT_4096_7
    + """

Stochastic processes model systems that evolve randomly over time. Markov chains, where the future state depends only on the present state, are widely used in modeling queues, genetic drift, and financial markets. Brownian motion models the random movement of particles and underlies the Black-Scholes model for option pricing. Martingale theory provides a mathematical framework for fair games and has applications in probability and finance.

Algebraic topology uses algebraic structures to classify topological spaces. Homotopy theory studies continuous deformations of mappings between spaces. Homology theory assigns algebraic invariants that capture holes in different dimensions. K-theory connects topology with algebra and has applications in physics, particularly in classifying topological phases of matter.

Partial differential equations arise naturally in mathematical models of physical phenomena. The heat equation describes the diffusion of thermal energy. The wave equation models vibrations and propagation of waves. Existence and uniqueness theorems establish when solutions to these equations exist and are well-determined. Numerical methods including finite differences, finite elements, and spectral methods approximate solutions that cannot be found analytically.

Representation theory studies how algebraic structures act on vector spaces. The representation theory of finite groups has applications in chemistry for analyzing molecular symmetries. Lie group representations are essential in particle physics for classifying elementary particles. The Langlands program, which seeks deep connections between representation theory and number theory, has been called a grand unified theory of mathematics.

Discrete mathematics studies structures that are fundamentally countable rather than continuous. Boolean algebra provides the mathematical foundation for digital logic and computer circuits. Coding theory develops error-correcting codes for reliable data transmission. Algorithmic complexity theory classifies computational problems by the resources required to solve them, with the P versus NP problem standing as one of the greatest open questions in mathematics.

Mathematical biology has become an increasingly quantitative field. Differential equation models describe the dynamics of predator-prey systems, epidemic spread, and neural networks. Stochastic models capture the inherent randomness in biological processes at the molecular level. Phylogenetic methods use mathematical models to reconstruct evolutionary relationships from genetic data.

The philosophy of mathematics addresses fundamental questions about the nature of mathematical objects and the reliability of mathematical knowledge. Platonism holds that mathematical objects exist independently of human minds. Formalism treats mathematics as manipulation of symbols according to rules. Intuitionism requires that mathematical proofs be constructive. These foundational perspectives influence how mathematicians approach their work and understand their discoveries.

Applied mathematics continues to expand into new domains as data availability and computational power increase. Machine learning algorithms are grounded in optimization theory, statistical learning theory, and approximation theory. Financial mathematics develops models for pricing derivatives and managing risk. Climate modeling uses numerical methods to simulate complex Earth system dynamics. The breadth of applied mathematics reflects the universal applicability of mathematical thinking to understanding and shaping the world."""
)

# Dictionary mapping (seq_len) -> list of texts for different batch indices
PREFILL_TEXTS = {
    128: [
        _TEXT_128_0,
        _TEXT_128_1,
        _TEXT_128_2,
        _TEXT_128_3,
        _TEXT_128_4,
        _TEXT_128_5,
        _TEXT_128_6,
        _TEXT_128_7,
    ],
    1024: [
        _TEXT_1024_0,
        _TEXT_1024_1,
        _TEXT_1024_2,
        _TEXT_1024_3,
        _TEXT_1024_4,
        _TEXT_1024_5,
        _TEXT_1024_6,
        _TEXT_1024_7,
    ],
    2048: [
        _TEXT_2048_0,
        _TEXT_2048_1,
        _TEXT_2048_2,
        _TEXT_2048_3,
        _TEXT_2048_4,
        _TEXT_2048_5,
        _TEXT_2048_6,
        _TEXT_2048_7,
    ],
    4096: [
        _TEXT_4096_0,
        _TEXT_4096_1,
        _TEXT_4096_2,
        _TEXT_4096_3,
        _TEXT_4096_4,
        _TEXT_4096_5,
        _TEXT_4096_6,
        _TEXT_4096_7,
    ],
    8192: [
        _TEXT_8192_0,
        _TEXT_8192_1,
        _TEXT_8192_2,
        _TEXT_8192_3,
        _TEXT_8192_4,
        _TEXT_8192_5,
        _TEXT_8192_6,
        _TEXT_8192_7,
    ],
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
        raise KeyError(
            f"seq_len {seq_len} not supported. Available: {list(PREFILL_TEXTS.keys())}"
        )
    texts = PREFILL_TEXTS[seq_len]
    if batch_idx >= len(texts):
        raise IndexError(
            f"batch_idx {batch_idx} exceeds available texts ({len(texts)}) for seq_len {seq_len}"
        )
    return texts[batch_idx]


def get_prefill_texts_for_batch(seq_len: int, batch_size: int) -> list:
    """Get list of prefill texts for a given sequence length and batch size.

    Args:
        seq_len: Target sequence length
        batch_size: Number of texts needed

    Returns:
        List of text strings for the batch.
    """
    return [
        get_prefill_text(seq_len, i % len(PREFILL_TEXTS[seq_len]))
        for i in range(batch_size)
    ]
