# Processes versus threads

**ConceptLens original teaching note; demonstration content, not a vetted textbook extract.**

A process normally has its own virtual address space and operating-system resources. Threads within a process share much of that address space but have separate execution state such as stacks and registers. Shared memory makes communication efficient but introduces synchronization risks.
