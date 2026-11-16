# Deadlock and lock ordering

**ConceptLens original teaching note; demonstration content, not a vetted textbook extract.**

Deadlock can arise when threads hold resources while waiting for other resources in a cycle, so none can proceed. A common prevention technique is acquiring multiple locks in a consistent global order. Mutual exclusion is not identical to deadlock; a correctly designed critical section can use locks without becoming deadlocked.
