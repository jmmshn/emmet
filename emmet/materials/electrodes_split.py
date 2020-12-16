import operator
from collections import namedtuple
from itertools import groupby, chain
from typing import Iterable, Dict, List, Any

from maggma.builders import Builder
from maggma.stores import MongoStore
from numpy import unique
from pymatgen import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.core import Structure
from datetime import datetime

__author__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"


def s_hash(el):
    return el.data["comp_delith"]


TaskStruct = namedtuple("task_id_structure", ["task_id", "structure"])

REDOX_ELEMENTS = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Nb",
    "Mo",
    "Sn",
    "Sb",
    "W",
    "Re",
    "Bi",
    "C",
    "Hf",
]

WORKING_IONS = ["Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba"]

MAT_PROPS = [
    "structure",
    "task_id",
    "formula_pretty",
]

sg_fields = ["number", "hall_number", "international", "hall", "choice"]


def generic_groupby(list_in, comp=operator.eq):
    """
    Group a list of unsortable objects
    Args:
        list_in: A list of generic objects
        comp: (Default value = operator.eq) The comparator
    Returns:
        [int] list of labels for the input list
    """
    list_out = [None] * len(list_in)
    label_num = 0
    for i1, ls1 in enumerate(list_out):
        if ls1 is not None:
            continue
        list_out[i1] = label_num
        for i2, ls2 in list(enumerate(list_out))[i1 + 1 :]:
            if comp(list_in[i1], list_in[i2]):
                if list_out[i2] is None:
                    list_out[i2] = list_out[i1]
                else:
                    list_out[i1] = list_out[i2]
                    label_num -= 1
        label_num += 1
    return list_out


class StructureGroupBuilder(Builder):
    def __init__(
        self,
        materials: MongoStore,
        groups: MongoStore,
        working_ion: str,
        query: dict = None,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        **kwargs,
    ):
        """
        Calculates physical parameters of battery materials the battery entries using
        groups of ComputedStructureEntry and the entry for the most stable version of the working_ion in the system
        Args:
            materials (Store): Store of materials documents that contains the structures
            groups (Store): Store of grouped material ids
            query (dict): dictionary to limit materials to be analyzed ---
                            only applied to the materials when we need to group structures
                            the phase diagram is still constructed with the entire set
        """
        self.materials = materials
        self.groups = groups
        self.working_ion = working_ion
        self.query = query if query else {}
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        super().__init__(sources=[materials], targets=[groups], **kwargs)

    def prechunk(self, number_splits: int) -> Iterable[Dict]:
        """
        Only used in distributed runs
        """
        all_chemsys = self.materials.distinct(
            "chemsys",
            criteria={
                "$and": [
                    {"elements": {"$nin": [self.working_ion]}},
                    {"elements": {"$in": REDOX_ELEMENTS}},
                ]
            },
        )
        for chemsys in all_chemsys:
            chemsys_wi = "-".join(sorted(chemsys.split("-") + [self.working_ion]))
            yield {"query": {"chemsys": {"$in": [chemsys, chemsys_wi]}}}

    def get_items(self):
        chemsys_query = {
            "$and": [
                {"elements": {"$nin": [self.working_ion]}},
                {"elements": {"$in": REDOX_ELEMENTS}},
                self.query.copy(),
            ]
        }

        all_chemsys = self.materials.distinct("chemsys", criteria=chemsys_query)

        for chemsys in all_chemsys:
            chemsys_wi = "-".join(sorted(chemsys.split("-") + [self.working_ion]))
            all_distinct_formula = self.materials.distinct(
                "formula_pretty", criteria={"chemsys": {"$in": [chemsys_wi, chemsys]}}
            )

            self.logger.debug(
                f"Grouping these chemical formulas based on similarity : {all_distinct_formula}"
            )

            for form_group in self._get_simlar_formula_in_group(all_distinct_formula):
                self.logger.debug(f"Goup of similar formula : {form_group}")
                if len(form_group) > 1:  # possibility of insertion reaction
                    mat_list = list(
                        self.materials.query(
                            {"formula_pretty": {"$in": form_group}},
                            properties=MAT_PROPS + [self.materials.last_updated_field],
                        )
                    )
                    framework = self._host_comp(form_group[0])
                    grouped_struct_times = self.groups.distinct(
                        self.groups.last_updated_field,
                        criteria={"framework": framework},
                    )
                    if not grouped_struct_times:
                        yield {"framework": framework, "materials": mat_list}
                    else:
                        mat_times = [
                            mat_doc[self.materials.last_updated_field]
                            for mat_doc in mat_list
                        ]
                        if max(mat_times) < min(grouped_struct_times):
                            continue  # no work is needed
                    self.groups.remove_docs(criteria={"framework": framework})
                    yield {"framework": framework, "materials": mat_list}

    def update_targets(self, items: List):
        items = list(filter(None, chain.from_iterable(items)))
        if len(items) > 0:
            self.logger.info("Updating {} groups documents".format(len(items)))
            for k in items:
                k[self.groups.last_updated_field] = datetime.utcnow()
            self.groups.update(docs=items, key=["task_id"])
        else:
            self.logger.info("No items to update")

    def process_item(self, item: Any) -> Any:
        sm = StructureMatcher(
            comparator=ElementComparator(),
            primitive_cell=True,
            ignored_species=[self.working_ion],
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
        )

        task_id_vs_struct = [
            TaskStruct(
                task_id=mat_doc["task_id"],
                structure=Structure.from_dict(mat_doc["structure"]),
            )
            for mat_doc in item["materials"]
        ]
        results = []
        ungrouped_structures = []
        for g in self._group_struct(task_id_vs_struct, sm):
            different_comps = set([ts_.structure.composition for ts_ in g])
            if len(different_comps) > 1:
                ids = [ts_.task_id for ts_ in g]
                lowest_id = sorted(ids, key=get_id_num)[0]
                d_ = {
                    "task_id": f"{lowest_id}_{self.working_ion}",
                    "has_distinct_compositions": True,
                    "grouped_task_ids": ids,
                    "framework": item["framework"],
                    "working_ion": self.working_ion,
                }
                results.append(d_)
            else:
                ungrouped_structures.extend(g)

        if ungrouped_structures:
            ids = [ts_.task_id for ts_ in ungrouped_structures]
            lowest_id = sorted(ids, key=get_id_num)[0]
            results.append(
                {
                    "task_id": f"{lowest_id}_{self.working_ion}",
                    "has_distinct_compositions": False,
                    "grouped_task_ids": ids,
                    "framework": item["framework"],
                    self.groups.last_updated_field: datetime.utcnow(),
                    "working_ion": self.working_ion,
                }
            )
        return results

    def _group_struct(self, g, sm):
        """
        group the entries together based on similarity of the delithiated primitive cells
        Args:
            g: a list of entries
        Returns:
            subgroups: subgroups that are grouped together based on structure
        """
        labs = generic_groupby(
            g, comp=lambda x, y: sm.fit(x.structure, y.structure, symmetric=True)
        )
        for ilab in unique(labs):
            sub_g = [g[itr] for itr, jlab in enumerate(labs) if jlab == ilab]
            yield [el for el in sub_g]

    def _host_comp(self, formula):
        dd_ = Composition(formula).as_dict()
        if self.working_ion in dd_:
            dd_.pop(self.working_ion)
        return Composition.from_dict(dd_).reduced_formula

    def _get_simlar_formula_in_group(self, formula_group):
        for k, g in groupby(formula_group, self._host_comp):
            yield list(g)  # Store group iterator as a list


def get_id_num(task_id):
    if isinstance(task_id, int):
        return task_id
    if isinstance(task_id, str) and "-" in task_id:
        return int(task_id.split("-")[-1])
    else:
        raise ValueError("TaskID needs to be either a number or of the form xxx-#####")
