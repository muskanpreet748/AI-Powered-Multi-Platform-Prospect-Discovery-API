"""
Section 3 - Duplicate detection and profile consolidation (entity resolution).

The same person can appear on several platforms. We link two profiles only
when there is a *strong identity signal*:

  1. Same canonical URL           (found twice, e.g. by two different queries)
  2. Same email / phone / WhatsApp (contact details are personal identifiers)
  3. Explicit cross-link           (her website or bio links to that Instagram profile)
  4. Same username on two platforms AND a similar display name

A similar *name alone* is never enough - "John Smith" on LinkedIn and
"John Smith" on Instagram are usually different people.

Linked profiles are grouped with a Union-Find (disjoint set) structure: if
A~B and B~C then A, B and C all end up in one group, efficiently.
"""

from difflib import SequenceMatcher

from app.models import ProspectGroup, RawProfile
from app.utils.normalizers import normalize_handle, normalize_name


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]  # path compression
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def names_similar(a: str, b: str) -> bool:
    if a.startswith("@") or b.startswith("@"):
        return False  # a bare handle is not a name - no evidence of identity
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if SequenceMatcher(None, na, nb).ratio() >= 0.85:
        return True
    ta, tb = set(na.split()), set(nb.split())
    # "Jane Doe" vs "Jane A. Doe": all words of the shorter name appear in the longer one
    return min(len(ta), len(tb)) >= 2 and (ta <= tb or tb <= ta)


def _merge_into(target: RawProfile, other: RawProfile) -> None:
    if target.name.startswith("@") and not other.name.startswith("@"):
        target.name = other.name
    if other.text and other.text not in target.text:
        target.text += f" \n{other.text}"
    target.emails |= other.emails
    target.phones |= other.phones
    target.whatsapp |= other.whatsapp
    target.linked_urls |= other.linked_urls
    target.country_code = target.country_code or other.country_code


def merge_same_url(profiles: list[RawProfile]) -> list[RawProfile]:
    """Rule 1: the same profile found several times becomes one record."""
    by_url: dict[str, RawProfile] = {}
    for profile in profiles:
        if profile.url in by_url:
            _merge_into(by_url[profile.url], profile)
        else:
            by_url[profile.url] = profile
    return list(by_url.values())


def consolidate(profiles: list[RawProfile]) -> list[ProspectGroup]:
    unique = merge_same_url(profiles)
    uf = UnionFind(len(unique))
    index_by_url = {p.url: i for i, p in enumerate(unique)}

    # Rule 2 (shared contact) and Rule 3 (explicit cross-links)
    contact_owner: dict[str, int] = {}
    for i, profile in enumerate(unique):
        for contact in profile.emails | profile.phones | profile.whatsapp:
            if contact in contact_owner:
                uf.union(contact_owner[contact], i)
            else:
                contact_owner[contact] = i
        for linked_url in profile.linked_urls:
            j = index_by_url.get(linked_url)
            if j is not None:
                uf.union(i, j)

    # Rule 4: same handle on different platforms + similar name
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            a, b = unique[i], unique[j]
            if a.platform == b.platform or not a.handle or not b.handle:
                continue
            handle_a, handle_b = normalize_handle(a.handle), normalize_handle(b.handle)
            if len(handle_a) >= 4 and handle_a == handle_b and names_similar(a.name, b.name):
                uf.union(i, j)

    groups: dict[int, list[RawProfile]] = {}
    for i, profile in enumerate(unique):
        groups.setdefault(uf.find(i), []).append(profile)
    return [ProspectGroup(profiles=members) for members in groups.values()]
