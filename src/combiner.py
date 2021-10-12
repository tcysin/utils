"""
This module contains functionality for working with extracted data
(foregrounds, apartments, etc) and various checks.
"""

import logging
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from operator import attrgetter
from typing import Any, Iterable, Mapping, Sequence


class Region:
    def __init__(self, coords: Sequence[int], score: float, label: int):
        self.coords = tuple(coords)  # (x_min, y_min, x_max, y_max)
        self.score = score
        self.label = label
        # TODO boxtype

    def __repr__(self):
        return f"Region(coords={self.coords}, score={self.score}, label={self.label})"


class DigitRegion(Region):
    def __init__(
        self,
        coords: Sequence[int],
        score: float,
        label: int,
        value: Decimal,
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
        label: int,
        m2: Decimal,
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
    # TODO labels for infoboxes are different from labels for rooms
    # TODO do verification right here
    _indent = 4

    def __init__(
        self,
        coords: Sequence[int],
        score: float,
        label: int,
        digit_regions: Iterable[DigitRegion] = (),
        adjust_regions=False,
    ):
        super().__init__(coords, score, label)
        # TODO digit region coordinates are absolute w.r.t. to cropped roi, NOT orig pic
        # TODO verify digit regions are within the infobox
        self.regions = frozenset(deepcopy(digit_regions))

        if adjust_regions:
            self._adjust_regions()

    def _adjust_regions(self):
        x_offset, y_offset, *_ = self.coords

        for digit_region in self.regions:
            x_min, y_min, x_max, y_max = digit_region.coords
            digit_region.coords = (
                x_min + x_offset,
                y_min + y_offset,
                x_max + x_offset,
                y_max + y_offset,
            )

    def __repr__(self):
        s_rs = ",\n".join(" " * self._indent + repr(r) for r in self.regions)
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
        rooms: Iterable[Room] = (),
    ):
        self.filename = filename
        self.id_region = id_region
        self.infobox = infobox
        self.rooms = frozenset(rooms)

    def __repr__(self):
        s = "Apartment\n"
        s += " " * self._indent + f"filename: {self.filename}\n"
        s += " " * self._indent + "id region: " + repr(self.id_region) + "\n"
        s += " " * self._indent + "infobox: " + repr(self.infobox) + "\n"
        s += " " * self._indent + "rooms: [\n"
        s += ",\n".join(" " * 2 * self._indent + repr(r) for r in self.rooms)
        s += "\n])"
        return s


def non_decreasing(seq: Sequence):
    return all(x <= y for x, y in zip(seq, seq[1:]))


class ApartmentChecker:
    """Helper class to facilitate apartment checks."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.label2name = self.cfg["thing_classes"]
        self.name2label = {name: label for label, name in enumerate(self.label2name)}
        self.count_range = {
            self.name2label[label_name]: range_pair
            for label_name, range_pair in cfg["count_range"].items()
        }
        self.m2_range = {
            self.name2label[label_name]: range_pair
            for label_name, range_pair in cfg["m2_range"].items()
        }

    def _setup(self, apartment: Apartment) -> None:
        self.apartment = apartment
        label2rooms = defaultdict(list)

        for room in apartment.rooms:
            label2rooms[room.label].append(room)

        self.label2rooms: Mapping[int, Sequence[Room]] = dict(label2rooms)
        self.label2counts: Mapping[int, int] = {
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
        """

        logging.debug(f'checking {apartment.filename}')
        # TODO maybe make check functions accept some self params instead? easier to test?
        # TODO add debug calls
        # TODO see if you need to add teardown() and wrap this whole thing
        self._setup(apartment)

        # TODO optimize this one?
        id_ok = self.check_id()
        infobox_ok = self.check_infobox()

        # general checks for *detected* objects
        counts_ok = self.check_counts()
        vals_ok = self.check_vals()
        general_ok = counts_ok and vals_ok

        # obligatory space type (presence) checks TODO any one present
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
        # make sure no extra labels in rooms  # TODO

        # exclusivity checks  TODO only one of present
        kitchen_space_exclusive_ok = self.check_kitchen_space_exclusive()
        lrwk_exclusive_ok = self.check_lrwk_exclusive()
        exclusivity_ok = kitchen_space_exclusive_ok and lrwk_exclusive_ok

        # required pair checks  TODO either all or none present (not xor)
        kn_and_lr_paired = self.check_kn_lr_paired()  # kn -> lr  TODO
        paired_ok = kn_and_lr_paired

        ok = (
            id_ok
            and infobox_ok
            and general_ok
            and spaces_ok
            and exclusivity_ok
            and paired_ok
        )

        if not ok:
            logging.debug(f"problem with apartment: {apartment}")

        return ok

    def check_id(self) -> bool:
        """
        Return True if apartment ID is valid, False otherwise.

        Checks for ranges of ID values and bounding box areas.
        """

        # id_region = self.apartment.id_region
        # value = id_region.value
        # x_min, y_min, x_max, y_max = id_region.coords

        # FIXME TODO min / max id value check
        range_ok = True

        # TODO area check
        # area = (x_max - x_min) * (y_max - y_min)
        area_ok = True  # area >= 100  # FIXME TODO

        return range_ok and area_ok

    def check_infobox(self) -> bool:
        """
        Return True if infobox is valid, False otherwise.

        Checks:
            - number of regions in this infobox (habitable, inside, total)
            - detected m2 areas in in order: `habitable <= inside <= total`
            - range of m2 values
        """

        infobox = self.apartment.infobox

        # check num regions: at least 1 and at most 3  TODO put this into config
        num_regions = len(infobox.regions)
        length_ok = 0 < num_regions <= 3
        if not length_ok:
            logging.debug(f"infobox not ok: {num_regions} regions")
            return False

        # check uniqueness
        unique_labels = set(map(attrgetter("label"), infobox.regions))
        unique_ok = num_regions == len(unique_labels)

        # check habitable <= inside <= total
        regions = sorted(
            infobox.regions, key=attrgetter("label")
        )  # put regions in order [habitable, .. inside, .. total, ..]
        values = [i.value for i in regions]
        non_decreasing_ok = non_decreasing(values)
        if not non_decreasing_ok:
            logging.debug(f"infobox not ok: {values} are out of order")

        # TODO check maximum values for DigitRegions
        vmax_ok = True  # FIXME TODO read vmax area for an apartment from somewhere

        return unique_ok and non_decreasing_ok and vmax_ok

    def check_counts(self) -> bool:
        """
        Return True if counts for each toom type are valid, False otherwise.

        For example, there must be between 0 and 2 wardrobes, etc.
        Note: this check makes sure that counts of detected rooms are within
        limits; it does not check for obligatory presence.
        """

        # TODO re-do with flags and label to name conversions to see whats the problem

        for label, count in self.label2counts.items():
            mincount = min(self.count_range[label])
            maxcount = max(self.count_range[label])
            count_ok = mincount <= count <= maxcount

            if not count_ok:
                name = self.label2name[label]
                logging.debug(
                    f"count not ok: label {name} [{label}], "
                    f"count {count}, range ({mincount}, {maxcount})"
                )
                return False

        return True

    def check_vals(self) -> bool:
        """
        Return True if m2 areas for each room are valid, False otherwise.

        For example, a wardrobe must take between 0 and 5 m2, etc.
        """

        # label2rooms are guranteed to have only rooms (no ID / infobox objects)
        for label, rooms in self.label2rooms.items():
            minval, maxval = min(self.m2_range[label]), max(self.m2_range[label])

            for room in rooms:
                m2_ok = minval <= room.m2 <= maxval

                if not m2_ok:
                    logging.debug(
                        f"m2 not ok: label {label}, m2 {room.m2}, range ({minval}, {maxval})"
                    )
                    return False

        return True

    def check_entrance(self) -> bool:
        """
        Return True if lobby is present, False otherwise.
        """

        entrance_ok = self.name2label["lobby"] in self.label2counts
        if not entrance_ok:
            logging.debug("entrance not ok: lobby is missing")

        return entrance_ok

    def check_habitable(self) -> bool:
        """
        Return True if at least one room for sleeping is present, False otherwise.

        Such rooms include bedroom, living room and living room with kitchen.
        """

        # TODO move definition of habitable spaces to config
        habitable_ok = (
            self.name2label["bedroom"] in self.label2counts
            or self.name2label["living_room"] in self.label2counts
            or self.name2label["living_room_with_kitchen"] in self.label2counts
        )
        if not habitable_ok:
            logging.debug("habitable spaces not ok")

        return habitable_ok

    def check_kitchen_spaces(self) -> bool:
        """
        Return True if at least one kitchen space is present, False otherwise.

        Such rooms include kitchen, kitchen-niche and living room with kitchen.
        """

        # TODO move definition to config
        # TODO structure is very similar to other presence checks; refactor?
        kitchen_spaces_ok = (
            self.name2label["kitchen"] in self.label2counts
            or self.name2label["kitchen_niche"] in self.label2counts
            or self.name2label["living_room_with_kitchen"] in self.label2counts
        )
        if not kitchen_spaces_ok:
            logging.debug("kitchen spaces not ok")

        return kitchen_spaces_ok

    def check_wc_spaces(self) -> bool:
        """
        Return True if at least one wc space is present, False otherwise.

        Such rooms include wc and combination of toilet + bathroom.
        """

        # TODO refactor to use check_one_of + check_all
        wc_spaces_ok = self.name2label["wc"] in self.label2counts or (
            self.name2label["bathroom"] in self.label2counts
            and self.name2label["toilet"] in self.label2counts
        )
        if not wc_spaces_ok:
            logging.debug("wc spaces not ok")

        return wc_spaces_ok

    def check_kitchen_space_exclusive(self) -> bool:
        """
        Return True if only one type of kitchen space is present, False otherwise.

        Such rooms include kitchen, kitchen-niche and living room with kitchen.
        """

        # TODO we can also add these to config as exclusive sets
        # we are guaranteed to have at least one of three at this point
        # three-way XOR- only one out of three must be present
        flags = [
            self.name2label["kitchen"] in self.label2counts,
            self.name2label["kitchen_niche"] in self.label2counts,
            self.name2label["living_room_with_kitchen"] in self.label2counts,
        ]
        ok = sum(flags) == 1
        if not ok:
            logging.debug("kitchen spaces are not exclusive")

        return ok

    def check_lrwk_exclusive(self) -> bool:
        """
        Return False if both living room with kitchen and living room are present.

        Return True otherwise.
        """

        flags = [
            self.name2label["living_room_with_kitchen"] in self.label2counts,
            self.name2label["living_room"] in self.label2counts,
        ]
        ok = sum(flags) != 2
        if not ok:
            logging.debug(
                "not ok: both living_room_with_kitchen and living_room present"
            )

        return ok

    def check_kn_lr_paired(self) -> bool:
        """
        Return True if both kitchen-niche and living room are present or absent.

        Return False if one is present, but the other is absent.
        """

        # either both present or both absent
        flags = [
            self.name2label["kitchen_niche"] in self.label2counts,
            self.name2label["living_room"] in self.label2counts,
        ]
        ok = sum(flags) != 1
        if not ok:
            logging.debug("not ok: either kitchen_niche or living_room are missing")

        return ok


def combine(digits: Sequence[int], symbol_pos: int) -> Decimal:
    """
    Combine extracted digits list and symbol position integer into a
    decimal number.

    Return None if there are no valid digits.
    """

    digits = [x for x in digits if x != 10]

    if not digits:
        return None

    if symbol_pos != 0 and symbol_pos < len(digits):
        digits.insert(symbol_pos, ".")

    s = "".join(map(str, digits))

    return Decimal(s)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
    )

    cfg = {
        "count_range": {
            "balcony": [0, 3],
            "bathroom": [0, 3],
            "bedroom": [0, 5],
            "corridor": [0, 1],
            "id": [1, 1],
            "infobox": [1, 1],
            "kitchen": [0, 1],
            "kitchen_niche": [0, 1],
            "laundry": [0, 2],
            "living_room": [0, 1],
            "living_room_with_kitchen": [0, 1],
            "lobby": [1, 1],
            "storage": [0, 2],
            "toilet": [0, 3],
            "wardrobe": [0, 2],
            "wc": [0, 2],
        },
        "m2_range": {
            "balcony": [0, 5],
            "bathroom": [0, 5],
            "bedroom": [0, 30],
            "corridor": [0, 20],
            "kitchen": [0, 15],
            "kitchen_niche": [0, 10],
            "laundry": [0, 8],
            "living_room": [0, 30],
            "living_room_with_kitchen": [0, 50],
            "lobby": [0, 15],
            "storage": [0, 10],
            "toilet": [0, 8],
            "wardrobe": [0, 6],
            "wc": [0, 10],
        },
        "thing_classes": [
            "balcony",
            "bathroom",
            "bedroom",
            "corridor",
            "id",
            "infobox",
            "kitchen",
            "kitchen_niche",
            "laundry",
            "living_room",
            "living_room_with_kitchen",
            "lobby",
            "storage",
            "toilet",
            "wardrobe",
            "wc",
        ],
    }

    # TODO play around with apartment check
    apt = Apartment(
        filename="fg11.jpg",
        id_region=DigitRegion((226, 486, 301, 514), 0.999, 4, Decimal("7")),
        infobox=Infobox(
            coords=(177, 492, 341, 616),
            score=1.0,
            label=5,
            digit_regions=[
                DigitRegion(
                    coords=(266, 523, 320, 546),
                    score=0.993,
                    label=0,
                    value=Decimal("35.10"),
                ),
                DigitRegion(
                    coords=(264, 553, 323, 577),
                    score=0.998,
                    label=1,
                    value=Decimal("53.87"),
                ),
                DigitRegion(
                    coords=(265, 584, 324, 609),
                    score=1.0,
                    label=2,
                    value=Decimal("53.87"),
                ),
            ],
            adjust_regions=False,
        ),
        rooms=[
            Room(coords=(40, 230, 69, 252), score=0.944, label=7, m2=Decimal("4.6")),
            Room(
                coords=(154, 230, 187, 252), score=0.943, label=9, m2=Decimal("10.82")
            ),
            Room(coords=(197, 53, 222, 73), score=0.999, label=1, m2=Decimal("2.91")),
            Room(coords=(333, 113, 364, 135), score=1.0, label=2, m2=Decimal("11.61")),
            Room(coords=(54, 54, 81, 74), score=1.0, label=11, m2=Decimal("3.34")),
            Room(coords=(92, 54, 120, 75), score=0.73, label=13, m2=Decimal("1.83")),
            Room(coords=(217, 92, 244, 113), score=0.981, label=3, m2=Decimal("6.01")),
            Room(coords=(335, 230, 369, 251), score=0.998, label=2, m2=Decimal("12.6")),
        ],
    )

    checker = ApartmentChecker(cfg)

    checker(apt)
    print('End')
