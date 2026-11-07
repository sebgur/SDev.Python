from functools import lru_cache


class ConventionRepository:
    def __init(self):
        pass


@lru_cache(maxsize=1)
def convention_repository() -> ConventionRepository:
    """ Return the convention repository """
    return ConventionRepository()
