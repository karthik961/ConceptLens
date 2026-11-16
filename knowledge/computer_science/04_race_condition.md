# Race conditions in shared data

**ConceptLens original teaching note; demonstration content, not a vetted textbook extract.**

A race condition occurs when program behavior depends on the relative timing of concurrent operations, often because shared mutable data is accessed without suitable synchronization. A read-modify-write sequence is not necessarily atomic. A mutex can protect a critical section, but only when all relevant accesses follow the locking discipline.
