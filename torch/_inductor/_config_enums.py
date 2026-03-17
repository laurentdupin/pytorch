import enum


class PreGradPassTiming(str, enum.Enum):
    # Run after cache lookup, only on cache miss.
    LATE = "late"
    # Run before cache lookup so the cache key reflects the
    # already-transformed graph and passes always execute.
    EARLY = "early"
