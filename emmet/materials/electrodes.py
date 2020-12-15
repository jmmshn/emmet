from pymatgen.core import Structure, Element
from maggma.builders import Builder, MapBuilder
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.apps.battery.plotter import VoltageProfilePlotter

from pymatgen.analysis.phase_diagram import PhaseDiagram, PhaseDiagramError
from pymatgen.transformations.standard_transformations import (
    PrimitiveCellTransformation,
)
from itertools import chain, combinations
from itertools import groupby
from pymatgen.entries.computed_entries import ComputedStructureEntry, ComputedEntry
from pymatgen.apps.battery.insertion_battery import InsertionElectrode
from pymatgen.apps.battery.conversion_battery import ConversionElectrode
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen import Composition
from emmet.materials.thermo import chemsys_permutations
from pymatgen.analysis.structure_analyzer import oxide_type
from numpy import unique
import operator

__author__ = "Jimmy Shen"
__email__ = "jmmshn@lbl.gov"


def s_hash(el):
    return el.data["comp_delith"]


redox_els = [
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
mat_props = [
    "structure",
    "calc_settings",
    "task_id",
    "_sbxn",
    "entries",
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


class ElectrodeSummaryBuilder(MapBuilder):
    def __init__(self, source, target, **kwargs):
        source.key = 'battery_id'
        target.key = 'battery_id'
        super(ElectrodeSummaryBuilder, self).__init__(source=source, target=target, **kwargs)

    def unary_function(self, item):
        ie = InsertionElectrode.from_dict()
        ce = InsertionElectrode.from_dict()
        d = ie.get_summary_dict()
        # plot data
        for xaxis in ['capacity_grav', "x_form"]:
            vp = VoltageProfilePlotter(xaxis)
            vp.add_electrode(ie, label="Insertion Profile")
            vp.add_electrode(ce, label="Conversion Profile")
            fig = vp.get_plotly_figure(term_zero = False)
            fig.layout.pop("template")

            xx, _ = vp.get_plot_data(ie, term_zero = False)
            xmin, xmax = xx[0], xx[-1]
            xlim = xmax - xmin

            fig.update_layout(title_x=0.5, xaxis={'range': (xmin-0.05*xlim, xmax + xlim * 0.05)})
            d[f'plot_data_{xaxis}'] = fig.to_json()
        return d

class ElectrodesBuilder(Builder):
    def __init__(
        self,
        materials,
        electro,
        working_ion,
        query=None,
        compatibility=None,
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        **kwargs,
    ):
        """
        Calculates physical parameters of battery materials the battery entries using
        groups of ComputedStructureEntry and the entry for the most stable version of the working_ion in the system
        Args:
            materials (Store): Store of materials documents that contains the structures
            electro (Store): Store of insertion electrodes data such as voltage and capacity
            query (dict): dictionary to limit materials to be analyzed ---
                            only applied to the materials when we need to group structures
                            the phase diagram is still constructed with the entire set
            compatibility (PymatgenCompatability): Compatability module
                to ensure energies are compatible
        """
        self.materials = materials
        self.electro = electro
        self.working_ion = working_ion
        self.query = query if query else {}
        self.compatibility = (
            compatibility
            if compatibility
            else MaterialsProjectCompatibility("Advanced")
        )
        self.completed_tasks = set()
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol
        super().__init__(sources=[materials], targets=[electro], **kwargs)

    def get_items(self):
        """
        Get all entries by first obtaining the distinct chemical systems then
        sorting them by their composition (sans the working ion)

        Returns:
            list of dictionaries with keys 'chemsys' 'elec_entries' and 'pd_entries'
            the entries in 'elec_entries' contain all of the structures for insertion electrode analysis
            the entries in 'pd_entries' contain the information to generate the phase diagram
        """

        # We only need the working_ion_entry once
        # working_ion_entries = self.materials.query(criteria={"chemsys": self.working_ion}, properties=mat_props)
        # working_ion_entries = self._mat_doc2comp_entry(working_ion_entries, store_struct=False)
        #
        # if working_ion_entries:
        #     self.working_ion_entry = min(working_ion_entries, key=lambda e: e.energy_per_atom)

        self.logger.info(
            "Grabbing the relavant chemical systems containing the current working ion and a single redox element."
        )
        q = dict()
        q.update(
            {
                "$and": [
                    {"elements": {"$in": [self.working_ion]}},
                    {"elements": {"$in": redox_els}},
                ]
            }
        )
        q.update(self.query)

        chemsys_names = self.materials.distinct("chemsys", q)
        self.logger.debug(f"chemsys_names: {chemsys_names}")
        for chemsys in chemsys_names:
            self.logger.debug(f"Calculating the phase diagram for: {chemsys}")
            # get the phase diagram from using the chemsys
            pd_q = {
                "chemsys": {"$in": list(chemsys_permutations(chemsys))},
                "deprecated": False,
            }
            self.logger.debug(f"pd_q: {pd_q}")
            pd_docs = list(self.materials.query(properties=mat_props, criteria=pd_q))
            pd_ents = self._mat_doc2comp_entry(pd_docs, is_structure_entry=True)
            pd_ents = list(filter(None.__ne__, pd_ents))

            for item in self.get_hashed_entries_from_chemsys(chemsys):
                item.update({"pd_entries": pd_ents})

                ids_all_ents = {ient.entry_id for ient in item["elec_entries"]}
                ids_pd = {ient.entry_id for ient in item["pd_entries"]}
                assert ids_all_ents.issubset(ids_pd)
                self.logger.debug(
                    f"all_ents [{[ient.composition.reduced_formula for ient in item['elec_entries']]}]"
                )
                self.logger.debug(
                    f"pd_entries [{[ient.composition.reduced_formula for ient in item['pd_entries']]}]"
                )
                yield item

    def get_hashed_entries_from_chemsys(self, chemsys):
        """
        Read the entries from the materials database and group them based on the reduced composition
        of the framework material (without working ion).
        Args:
            chemsys(string): the chemical system string to be queried
        returns:
            (chemsys, [group]): entry contains a list of entries the materials together by composition
        """
        # return the entries grouped by composition
        # then we will sort them
        elements = set(chemsys.split("-"))
        chemsys_w_wo_ion = {
            "-".join(sorted(c)) for c in [elements, elements - {self.working_ion}]
        }
        self.logger.info("chemsys list: {}".format(chemsys_w_wo_ion))
        q = {
            "$and": [
                {
                    "chemsys": {"$in": list(chemsys_w_wo_ion)},
                    "formula_pretty": {"$ne": self.working_ion},
                    "deprecated": False,
                },
                self.query,
            ]
        }
        self.logger.debug(f"q: {q}")
        docs = self.materials.query(q, mat_props)
        entries = self._mat_doc2comp_entry(docs)
        entries = list(filter(lambda x: x is not None, entries))
        self.logger.debug(
            f"entries found using q [{[ient.composition.reduced_formula for ient in entries]}]"
        )
        self.logger.info("Found {} entries in the database".format(len(entries)))
        entries = list(filter(None.__ne__, entries))

        if len(entries) > 1:
            # ignore systems with only one entry
            # group entries together by their composition sans the working ion
            entries = sorted(entries, key=s_hash)
            for _, g in groupby(entries, key=s_hash):
                g = list(g)
                self.logger.debug(
                    "The full group of entries found based on chemical formula alone: {}".format(
                        [el.name for el in g]
                    )
                )
                if len(g) > 1:
                    yield {"chemsys": chemsys, "elec_entries": g}

    def process_item(self, item):
        """
        Read the entries from the thermo database and group them based on the reduced composition
        of the framework material (without working ion).
        Args:
            chemsys(string): the chemical system string to be queried
        returns:
            (chemsys, [group]): entry contains a list of entries the materials together by composition
        """
        sm = StructureMatcher(
            comparator=ElementComparator(),
            primitive_cell=True,
            ignored_species=[self.working_ion],
            ltol=self.ltol,
            stol=self.stol,
            angle_tol=self.angle_tol,
        )
        # sort the entries intro subgroups
        # then perform PD analysis
        elec_entries = item["elec_entries"]
        pd_ents = item["pd_entries"]
        phdi = PhaseDiagram(pd_ents)

        # The working ion entries
        ents_wion = list(
            filter(
                lambda x: x.composition.get_integer_formula_and_factor()[0]
                == self.working_ion,
                pd_ents,
            )
        )
        working_ion_entry = min(ents_wion, key=lambda e: e.energy_per_atom)
        assert working_ion_entry is not None

        grouped_entries = list(self.get_sorted_subgroups(elec_entries, sm))
        docs = []  # results

        for group in grouped_entries:
            self.logger.debug(
                f"Grouped entries in {', '.join([en.name for en in group])}"
            )
            for en in group:
                # skip this d_muO2 stuff if you do note have oxygen
                if Element("O") in en.composition.elements:
                    d_muO2 = [
                        {
                            "reaction": str(itr["reaction"]),
                            "chempot": itr["chempot"],
                            "evolution": itr["evolution"],
                        }
                        for itr in phdi.get_element_profile("O", en.composition)
                    ]
                else:
                    d_muO2 = None
                en.data["muO2"] = d_muO2
                en.data["decomposition_energy"] = phdi.get_e_above_hull(en)

            # sort out the sandboxes
            # for each sandbox core+sandbox will both contribute entries

            # Need more than one level of lithiation to define a electrode
            # material

            ie = InsertionElectrode.from_entries(group, working_ion_entry, strip_structures=True)
            if ie.num_steps < 1:
                self.logger.warn(
                    f"Not able to generate a hull using the following entires-- \
                        {', '.join([str(en.entry_id) for en in group])}"
                )
                continue
            ce = self.get_competing_conversion_electrode(
                ie.framework, phase_diagram=phdi
            )

            d = {
                'insertion_electrode': ie.as_dict(),
                'converion_electrode': ce.as_dict(),
            }

            # Get the battery_id using the lowest numerical value
            ids = [entry.entry_id for entry in ie.get_all_entries()]
            lowest_id = sorted(ids, key=get_id_num)[0]
            d["battery_id"] = str(lowest_id) + "_" + self.working_ion

            host_structure = group[0].structure.copy()
            host_structure.remove_species([self.working_ion])
            d["host_structure"] = host_structure.as_dict()

            spacegroup = SpacegroupAnalyzer(host_structure)
            d["spacegroup"] = {k: spacegroup._space_group_data[k] for k in sg_fields}
            docs.append(d)

        return docs

    def update_targets(self, items):
        items = list(filter(None, chain.from_iterable(items)))
        if len(items) > 0:
            self.logger.info("Updating {} electro documents".format(len(items)))
            self.electro.update(docs=items, key=["battery_id"])
        else:
            self.logger.info("No items to update")

    def get_sorted_subgroups(self, group, sm):
        matching_subgroups = list(self.group_entries(group, sm))
        if matching_subgroups:
            for subg in matching_subgroups:
                wion_conc = set()
                for el in subg:
                    wion_conc.add(
                        el.composition.fractional_composition[self.working_ion]
                    )
                if len(wion_conc) > 1:
                    yield subg
                else:
                    del subg

    def group_entries(self, g, sm):
        """
        group the structures together based on similarity of the delithiated primitive cells
        Args:
            g: a list of entries
        Returns:
            subgroups: subgroups that are grouped together based on structure
        """
        labs = generic_groupby(
            g,
            comp=lambda x, y: any(
                [sm.fit(x.structure, y.structure), sm.fit(y.structure, x.structure)]
            ),
        )  # because fit is not commutitive
        for ilab in unique(labs):
            sub_g = [g[itr] for itr, jlab in enumerate(labs) if jlab == ilab]
            if len(sub_g) > 1:
                yield [el for el in sub_g]

    def _chemsys_delith(self, chemsys):
        # get the chemsys with the working ion removed from the set
        elements = set(chemsys.split("-"))
        return {"-".join(sorted(c)) for c in [elements, elements - {self.working_ion}]}

    def _mat_doc2comp_entry(self, docs, is_structure_entry=True):
        def get_prim_host(struct):
            """
            Get the primitive structure with all of the lithiums removed
            """
            structure = struct.copy()
            structure.remove_species([self.working_ion])
            prim = PrimitiveCellTransformation()
            return prim.apply_transformation(structure)

        entries = []

        for d in docs:
            struct = Structure.from_dict(d["structure"])
            # get the calc settings
            entry_type = "gga_u" if "gga_u" in d["entries"] else "gga"
            d["entries"][entry_type]["correction"] = 0.0
            if is_structure_entry:
                d["entries"][entry_type]["structure"] = struct
                en = ComputedStructureEntry.from_dict(d["entries"][entry_type])
            else:
                en = ComputedEntry.from_dict(d["entries"][entry_type])

            en.data["_sbxn"] = d.get("_sbxn", [])

            if en.composition.reduced_formula != self.working_ion:
                dd = en.composition.as_dict()
                if self.working_ion in dd:
                    dd.pop(self.working_ion)
                en.data["comp_delith"] = Composition.from_dict(dd).reduced_formula

            en.data["oxide_type"] = oxide_type(struct)

            try:
                entries.append(self.compatibility.process_entry(en))
            except BaseException:
                self.logger.warn(
                    "unable to process material with task_id: {}".format(en.entry_id)
                )
        return entries

    def get_competing_conversion_electrode(self, comp, phase_diagram):
        """
        Use the present phase diagram to constructure the completing conversion electrode
        """

        return ConversionElectrode.from_composition_and_pd(
            comp=comp,
            pd=phase_diagram,
            working_ion_symbol=self.working_ion,
            allow_unstable=True,
        )


def get_id_num(task_id):
    if isinstance(task_id, int):
        return task_id
    if isinstance(task_id, str) and "-" in task_id:
        return int(task_id.split("-")[-1])
    else:
        raise ValueError("TaskID needs to be either a number or of the form xxx-#####")
