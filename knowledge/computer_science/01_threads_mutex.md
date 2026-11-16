# Threads, shared state and mutual exclusion

**Concept notes written for the ConceptLens starter demo.**

Threads are independent sequences of execution within a process. An operating system can interleave their instructions, and multiple cores can execute threads in parallel. Therefore one cannot assume two threads always take turns in a safe order.

If two threads update the same mutable variable without synchronization, one update can overwrite another. A race condition occurs when the result depends on uncontrolled execution ordering. A mutex allows one thread at a time to enter a protected critical section, provided all participating code follows the locking discipline.

A mutex does not make all programs correct automatically. Locks can introduce deadlocks if acquired inconsistently, and carefully scoped critical sections are important for performance and safety.
