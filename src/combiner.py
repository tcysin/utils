"""
This module contains functionality for working with extracted data
(foregrounds, apartments, etc) and various checks.
"""

from collections import defaultdict
from copy import deepcopy
from math import isclose
from typing import Any, Iterable, Mapping, Sequence, Set, Tuple, Union

import pandas as pd


# TODO comments
# TODO better tag management (non-hardcoded list of tags, explanations, etc)


class Region:
    def __init__(self, coords: Sequence[int], score: float, label: str):
        self.coords = tuple(coords)  # (x_min, y_min, x_max, y_max)
        self.score = score
        self.label = label

    def __repr__(self):
        return f"Region(coords={self.coords}, score={self.score}, label={self.label})"


class DigitRegion(Region):
    def __init__(
        self,
        coords: Sequence[int],
        score: float,
        label: str,
        value: Union[float, int],
    ):
        super().__init__(coords, score, label)
        self.value = value

    def __repr__(self):
        s = (
            f"DigitRegion(coords={self.coords}, score={self.score}, "
            f"label={self.label}, value={self.value})"
        )
        return s


class Room(DigitRegion):
    def __init__(
        self,
        coords: Sequence[int],
        score: float,
        label: str,
        m2: float,
    ):
        super().__init__(coords, score, label, m2)
        self.m2 = self.value

    def __repr__(self):
        s = (
            f"Room(coords={self.coords}, score={self.score}, "
            f"label={self.label}, m2={self.m2})"
        )
        return s


class Infobox(Region):
    _indent = 4

    def __init__(
        self,
        coords: Sequence[int],
        score: float,
        label: str,
        digit_regions: Iterable[DigitRegion] = (),
        adjust_regions=False,
    ):
        super().__init__(coords, score, label)
        # TODO digit region coordinates are absolute w.r.t. to cropped roi, NOT orig pic
        # TODO verify digit regions are within the infobox
        regions = deepcopy(list(digit_regions))

        if adjust_regions:
            self._adjust_regions(regions)

        self.habitable = self._get_region("habitable", regions)
        self.inside = self._get_region("inside", regions)
        self.total = self._get_region("total", regions)

    def _adjust_regions(self, digit_regions: Iterable[DigitRegion]):
        """
        Translate coordinates of each digit region in-place.

        X-axis and Y-axis translation values come from x_min and y_min
        coordinates of infobox.
        """

        x_offset, y_offset, *_ = self.coords

        for region in digit_regions:
            x_min, y_min, x_max, y_max = region.coords
            region.coords = (
                x_min + x_offset,
                y_min + y_offset,
                x_max + x_offset,
                y_max + y_offset,
            )

    @staticmethod
    def _get_region(label: str, regions: Iterable[DigitRegion]):
        """
        Return first region from regions with matching label.

        Return None if no regions match provided label.
        """
        # TODO maybe select highest-scoring region?

        for region in regions:
            if region.label == label:
                return region

        return None

    def __repr__(self):
        s_rs = ",\n".join(
            " " * self._indent + repr(r)
            for r in [self.habitable, self.inside, self.total]
        )
        s = (
            f"Infobox(coords={self.coords}, score={self.score}, "
            f"label={self.label}, regions=[\n"
        )
        s_final = s + s_rs + "\n])"
        return s_final


class Apartment:
    _indent = 2

    def __init__(
        self,
        filename: str,
        id_region: DigitRegion,
        infobox: Infobox,
        rooms: Iterable[Room],
        tags: Iterable = None,
    ):
        self.filename = filename
        self.id_region = id_region
        self.infobox = infobox
        self.rooms = frozenset(rooms)
        self.tags: Set = set(tags) if tags is not None else set()

    def __repr__(self):
        s = "Apartment\n"
        s += " " * self._indent + f"filename: {self.filename}\n"
        s += " " * self._indent + "id region: " + repr(self.id_region) + "\n"
        s += " " * self._indent + "infobox: " + repr(self.infobox) + "\n"
        s += " " * self._indent + "tags: " + repr(sorted(self.tags)) + "\n"
        s += " " * self._indent + "rooms: [\n"
        s += ",\n".join(" " * 2 * self._indent + repr(r) for r in self.rooms)
        s += "\n])"
        return s


class ApartmentChecker:
    """
    Helper class to facilitate apartment checks.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        """
        Set up from config dictionary.

        Config dictionary must contain the following keys:
            - `classes_habitable` is a list of labels which belong to
                habitable apartment spaces
            - `classes_inside` is a list of labels which belong to spaces
                inside the apartment
            - `count_range` maps labels to a pair of `(mincount, maxcount)`
                possible total counts for objects with corresponding label
            - `m2_range` maps labels to a pair of `(minval, maxval)`
                possible m2 values
            - `tolerance` describes by how much can m2 totals differ during
                final comparison; must be float
        """
        self.cfg = cfg
        self.count_range: Mapping[str, Tuple[int, int]] = cfg["count_range"]
        self.m2_range: Mapping[str, Tuple[float, float]] = cfg["m2_range"]

    def _setup(self, apartment: Apartment) -> None:
        """
        Set up the checker for this apartment.

        Creates `tags` set, `self.label2rooms` and `self.label2counts`
        mappings for this apartment to facilitate further checks.
        """

        self.apartment = apartment
        self.tags = set()
        label2rooms = defaultdict(list)

        for room in apartment.rooms:
            label2rooms[room.label].append(room)

        self.label2rooms: Mapping[str, Sequence[Room]] = dict(label2rooms)
        self.label2counts: Mapping[str, int] = {
            key: len(val) for key, val in self.label2rooms.items()
        }

    def __call__(self, apartment: Apartment) -> bool:
        """
        Return True if an apartment passes all checks, False otherwise.

        Checks include:
            - ID and infobox
            - general (maximum counts and m2 values for each room object)
            - presence of obligatory space types (entrance, habitable, kitchen, wc)
            - exclusivity
            - required pairs
            - total m2 areas match between rooms and infobox

        If an apartment fails any checks, adds a tag corresponding to failed
        check to a set of tags.
        """

        # TODO better messages in debug
        # TODO maybe make check functions accept some explicit params instead? easier to test?
        # TODO add debug calls
        # TODO see if you need to add teardown() and wrap this whole thing
        self._setup(apartment)

        id_ok = self.check_id()
        infobox_ok = self.check_infobox()

        # general checks for *detected* objects
        counts_ok = self.check_counts()
        vals_ok = self.check_room_values()
        general_ok = counts_ok and vals_ok

        # obligatory space type (presence) checks
        entrance_present = self.check_entrance()
        habitable_present = self.check_habitable()
        kitchen_spaces_present = self.check_kitchen_spaces()
        wc_spaces_present = self.check_wc_spaces()
        spaces_ok = (
            entrance_present
            and habitable_present
            and kitchen_spaces_present
            and wc_spaces_present
        )
        # TODO make sure no extra labels in rooms

        # exclusivity checks
        kitchen_space_exclusive_ok = self.check_kitchen_space_exclusive()
        lrwk_exclusive_ok = self.check_lrwk_exclusive()
        exclusivity_ok = kitchen_space_exclusive_ok and lrwk_exclusive_ok

        # required pair checks
        kn_and_lr_paired = self.check_kn_lr_paired()
        paired_ok = kn_and_lr_paired

        # check that areas match between rooms and infobox
        cross_check_ok = self.check_matching_totals()

        ok = (
            id_ok
            and infobox_ok
            and general_ok
            and spaces_ok
            and exclusivity_ok
            and paired_ok
            and cross_check_ok
        )

        return ok

    @staticmethod
    def get_total_m2(rooms: Iterable[Room], labels: Iterable[str] = None) -> float:
        """
        Return the sum of m2 values for rooms.

        If labels are provided, return the sum only of those rooms with
        corresponding labels or zero if no rooms match provided labels.
        """

        if labels is not None:
            labels = set(labels)
            return sum(room.m2 for room in rooms if room.label in labels)
        else:
            return sum(room.m2 for room in rooms)

    def check_id(self) -> bool:
        """
        Return True if apartment ID is valid, False otherwise.

        Checks for ranges of ID values and bounding box areas.
        """

        # id_region = self.apartment.id_region
        # value = id_region.value
        # x_min, y_min, x_max, y_max = id_region.coords

        # TODO implement min / max id value check
        range_ok = True

        # TODO implement area check
        # area = (x_max - x_min) * (y_max - y_min)
        area_ok = True

        return range_ok and area_ok

    def check_infobox(self) -> bool:
        """
        Return True if infobox is valid, False otherwise.

        Checks:
            - number of regions in this infobox (habitable, inside, total)
            - detected m2 areas in in order: `habitable <= inside <= total`
            - range of m2 values
        """

        habitable = self.apartment.infobox.habitable
        inside = self.apartment.infobox.inside
        total = self.apartment.infobox.total
        present_regions = [i for i in [habitable, inside, total] if i is not None]

        # check num regions: at least 1 and at most 3
        # TODO how about putting this into config?
        num_regions = len(present_regions)
        length_ok = 0 < num_regions <= 3
        if not length_ok:
            self.tags.add("infobox_bad_regions_count")
            return False

        # check habitable <= inside <= total
        values = [region.value for region in present_regions]
        non_decreasing_ok = non_decreasing(values)
        if not non_decreasing_ok:
            self.tags.add("infobox_incorrect_regions")

        # TODO implemenbt maximum values check for DigitRegions
        vmax_ok = True

        return non_decreasing_ok and vmax_ok

    def check_counts(self) -> bool:
        """
        Return True if counts for each object type are valid, False otherwise.

        For example, there must be between 0 and 2 wardrobes, 1 id region. etc.
        Note: this check makes sure that counts of detected rooms are within
        limits; it does not check for obligatory presence.
        """

        for label, count in self.label2counts.items():
            mincount = min(self.count_range[label])
            maxcount = max(self.count_range[label])
            count_ok = mincount <= count <= maxcount

            if not count_ok:
                self.tags.add(f"bad_count_{label}")
                return False

        return True

    def check_room_values(self) -> bool:
        """
        Return True if m2 area values for each room are valid, False otherwise.

        For example, a wardrobe must take between 0 and 5 m2, etc.
        """

        # label2rooms are guranteed to have only rooms (no ID / infobox objects)
        for label, rooms in self.label2rooms.items():
            minval, maxval = min(self.m2_range[label]), max(self.m2_range[label])

            for room in rooms:
                m2_ok = minval <= room.m2 <= maxval

                if not m2_ok:
                    self.tags.add(f"bad_value_{label}")
                    return False

        return True

    def check_entrance(self) -> bool:
        """
        Return True if lobby is present, False otherwise.
        """

        entrance_ok = "lobby" in self.label2counts
        if not entrance_ok:
            self.tags.add("missing_lobby")

        return entrance_ok

    def check_habitable(self) -> bool:
        """
        Return True if at least one room for sleeping is present, False otherwise.

        Such rooms include bedroom, living room and living room with kitchen.
        """

        # TODO move definition of habitable spaces to config
        habitable_ok = (
            "bedroom" in self.label2counts
            or "living_room" in self.label2counts
            or "living_room_with_kitchen" in self.label2counts
        )
        if not habitable_ok:
            self.tags.add("missing_habitable_spaces")

        return habitable_ok

    def check_kitchen_spaces(self) -> bool:
        """
        Return True if at least one kitchen space is present, False otherwise.

        Such rooms include kitchen, kitchen-niche and living room with kitchen.
        """

        # TODO move definition to config
        kitchen_spaces_ok = (
            "kitchen" in self.label2counts
            or "kitchen_niche" in self.label2counts
            or "living_room_with_kitchen" in self.label2counts
        )
        if not kitchen_spaces_ok:
            self.tags.add("missing_kitchen_spaces")

        return kitchen_spaces_ok

    def check_wc_spaces(self) -> bool:
        """
        Return True if at least one wc space is present, False otherwise.

        Such rooms include wc and combination of toilet + bathroom.
        """

        wc_spaces_ok = "wc" in self.label2counts or (
            "bathroom" in self.label2counts and "toilet" in self.label2counts
        )
        if not wc_spaces_ok:
            self.tags.add("missing_wc_spaces")

        return wc_spaces_ok

    def check_kitchen_space_exclusive(self) -> bool:
        """
        Return True if only one type of kitchen space is present, False otherwise.

        Such rooms include kitchen, kitchen-niche and living room with kitchen.
        """

        # TODO we can also add these to config as exclusive sets
        # we are guaranteed to have at least one of three at this point
        # only one out of three must be present
        flags = [
            "kitchen" in self.label2counts,
            "kitchen_niche" in self.label2counts,
            "living_room_with_kitchen" in self.label2counts,
        ]
        ok = sum(flags) == 1
        if not ok:
            self.tags.add("non_exclusive_kitchen_spaces")

        return ok

    def check_lrwk_exclusive(self) -> bool:
        """
        Return False if both living room with kitchen and living room are present.

        Return True otherwise.
        """

        flags = [
            "living_room_with_kitchen" in self.label2counts,
            "living_room" in self.label2counts,
        ]
        ok = sum(flags) != 2
        if not ok:
            self.tags.add("non_exclusive_lrwk_lr")

        return ok

    def check_kn_lr_paired(self) -> bool:
        """
        Return True if both kitchen-niche and living room are present or absent.

        Return False if one is present, but the other is absent.

        FIXME this fails for apartments with kitchen AND living room
        """

        # either both present or both absent
        flags = [
            "kitchen_niche" in self.label2counts,
            "living_room" in self.label2counts,
        ]
        ok = sum(flags) != 1
        if not ok:
            self.tags.add("bad_pair_kn_lr")

        return ok

    def check_matching_totals(self) -> bool:
        """
        Return True if habitable, inside and total m2 area sums match between
        rooms and infobox.

        Return False otherwise.

        This should be done after we verified that rooms and infobox regions
        are ok themselves.
        """

        # at this point, the apartment passed all previous checks
        apartment = self.apartment
        infobox = apartment.infobox
        rooms = apartment.rooms

        flags = []

        # TODO better naming conventions (esp habitable / etc.)
        if infobox.habitable is not None:
            habitable_m2 = self.get_total_m2(rooms, self.cfg["classes_habitable"])
            habitable_ok = isclose(
                habitable_m2, infobox.habitable.value, abs_tol=self.cfg["tolerance"]
            )
            if not habitable_ok:
                self.tags.add("habitable_not_close")
            flags.append(habitable_ok)

        if infobox.inside is not None:
            inside_m2 = self.get_total_m2(rooms, self.cfg["classes_inside"])
            inside_ok = isclose(
                inside_m2, infobox.inside.value, abs_tol=self.cfg["tolerance"]
            )
            if not inside_ok:
                self.tags.add("inside_not_close")
            flags.append(inside_ok)

        if infobox.total is not None:
            total_m2 = self.get_total_m2(rooms)
            total_ok = isclose(
                total_m2, infobox.total.value, abs_tol=self.cfg["tolerance"]
            )
            if not total_ok:
                self.tags.add("total_not_close")
            flags.append(total_ok)

        return len(flags) > 0 and all(flags)


class ApartmentConverter:
    """
    Helper class for conversion of iterable of apartments to pandas DataFrame.

    Can re-map room labels given new mapping.
    """

    def __init__(
        self, mapper_dict: Mapping[str, str] = {}, with_tags: bool = False
    ) -> None:
        """
        Initialize the converter instance.

        Args:
            mapper_dict: mapping between original and new labels for rooms.
            with_tags: whether to add a list of apartment tags as additional dict key.
        """
        self.mapper_dict = mapper_dict
        self.with_tags = with_tags

    def to_df(self, apartments: Iterable[Apartment]) -> pd.DataFrame:
        """
        Convert a sequence of apartments into pandas DataFrame and return it.
        """

        apartment_dicts = []

        for apartment in apartments:
            d = {}

            # id region
            if apartment.id_region is not None:
                d["id"] = apartment.id_region.value
            else:
                d["id"] = None

            # infobox info
            # TODO infobox classes are hard-coded
            if apartment.infobox.habitable is not None:
                d["area_habitable"] = apartment.infobox.habitable.value
            else:
                d["area_habitable"] = None

            if apartment.infobox.inside is not None:
                d["area_inside"] = apartment.infobox.inside.value
            else:
                d["area_inside"] = None

            if apartment.infobox.total is not None:
                d["area_total"] = apartment.infobox.total.value
            else:
                d["area_total"] = None

            # convert each room into a pair {label_name: m2}
            l2rs = defaultdict(list)

            for room in apartment.rooms:
                l2rs[room.label].append(room.m2)

            for label, m2s in l2rs.items():
                # try to map to corresponding label; use original if impossible
                # TODO feels a bit hacky?
                stem = self.mapper_dict.get(label, label)

                # take care of multiple rooms with the same label
                for idx, m2 in enumerate(sorted(m2s), start=1):
                    name = stem + str(idx)
                    d[name] = m2

            # add tags key if needed
            if self.with_tags:
                d["tags"] = ", ".join(sorted(apartment.tags))

            apartment_dicts.append(d)

        df = pd.DataFrame(apartment_dicts)

        return df


def non_decreasing(seq: Sequence):
    """
    Return True if a sequence is in non-decreasing order, False otherwise.
    """
    return all(x <= y for x, y in zip(seq, seq[1:]))
