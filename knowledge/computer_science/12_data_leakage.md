# Data leakage in evaluation

**ConceptLens original teaching note; demonstration content, not a vetted textbook extract.**

Data leakage happens when information unavailable at real deployment time influences model training or selection. Examples include fitting transformations using test data, copying near-duplicate examples across train and test, or inserting test answer labels into a RAG knowledge base. Keep held-out examples isolated from decisions that tune the system.
