# Hash tables and collisions

**ConceptLens original teaching note; demonstration content, not a vetted textbook extract.**

A hash table maps keys to locations using a hash function. Distinct keys can map to the same slot, creating a collision; implementations handle collisions with strategies such as chaining or probing. Average-case lookup can be near constant time under suitable assumptions, but worst-case behavior can be slower.
