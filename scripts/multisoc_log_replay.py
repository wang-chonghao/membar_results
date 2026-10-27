"""Build canonical VF JSON from the checked CAModel instruction evidence.

The replay preserves fetched order (dynamic ID, not issue time), register RAWs,
predicate definitions, and VLOOP structure. Scalar vector-control instructions
are retained in the evidence audit, but are not timed by the current Core.
"""

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
import re

from api.common.instruction_catalog import DEFAULT_INSTRUCTION_CATALOG
from api.common.schema import (
    AccessKind, AffineExpression, CanonicalInstruction, CanonicalLoop,
    CanonicalMembar, CanonicalOperand, CanonicalStorageObject, CanonicalValue,
    CanonicalVfInfo, InductionVariable, InstructionClass, LoopCarriedValue,
    MemoryAccess, OperandRole, SourceLocation, StorageKind,
)


LINE = re.compile(
    r"\[(\d+)\].*?\(PC:\s*(0x[0-9a-fA-F]+)\).*?"
    r"\(ID:\s*(\d+)\)\s+(\w+)\s*(.*)"
)
SCALAR_CONTROL = {"RV_SMOV", "RV_SMOVI", "RV_SMOVK", "RV_SMIN", "RV_SSUB", "RV_SSUBI",
                  "RV_SADDI", "RV_SADD", "RV_SMUL", "RV_SOR", "RV_SSHLI", "RV_SZEROEXT",
                  "RV_SNOP"}
ALIASES = {"RV_VABS_FP": "VABS", "RV_VSHLI": "VSHLS"}


@dataclass(frozen=True)
class Event:
    cycle: int
    pc: int
    ident: int
    op: str
    detail: str

    def register(self, label):
        match = re.search(r"\b" + re.escape(label) + r"\[(\d+)\]", self.detail)
        return int(match[1]) if match else None

    def scalar(self, label):
        match = re.search(r"\b" + re.escape(label) + r"\[(\d+)\]=(0x[0-9a-fA-F]+|\d+)", self.detail)
        return (int(match[1]), int(match[2], 0)) if match else None


def read_events(path):
    events = {}
    for line in path.read_text().splitlines():
        match = LINE.search(line)
        if match:
            event = Event(int(match[1]), int(match[2], 16), int(match[3]), match[4], match[5])
            # Outer flow-control annotations can reuse an RV dynamic ID.
            # Deduplicate repeated port reports, not unrelated instructions.
            events.setdefault((event.ident,event.pc,event.op), event)
    return sorted(events.values(), key=lambda e: e.ident)


def vf_streams(case_dir):
    popped = read_events(case_dir / "cce/core0.veccore0.instr_popped_log.dump")
    completed = {(e.ident,e.pc,e.op): e for e in read_events(case_dir / "cce/core0.veccore0.instr_log.dump")}
    result = []
    for vf in [e for e in popped if e.op == "VF"]:
        match = re.search(r"addr:\s*(0x[0-9a-fA-F]+),instr_num:\s*(0x[0-9a-fA-F]+)", vf.detail)
        if not match:
            raise ValueError("Missing VF address/length")
        done = completed[(vf.ident,vf.pc,vf.op)]
        addr, words = int(match[1], 16), int(match[2], 16)
        rows = [e for e in popped if e.op.startswith("RV_") and addr <= e.pc < addr + words * 4
                and vf.cycle <= e.cycle <= done.cycle]
        if not rows:
            raise ValueError("VF contains no decoded instructions")
        result.append((vf, done, addr, words, rows))
    return result


def operation(event):
    if event.op in SCALAR_CONTROL or event.op in {"RV_SEND", "RV_VLOOPv2"}:
        return None
    if event.op == "RV_SMEM_BAR":
        return "MEMBAR"
    if event.op == "RV_VSTUI":
        return "VSTUS"
    if event.op == "RV_VSTAI":
        return "VSTAS"
    if event.op == "RV_VLDAS":
        return "VLDAS"
    if event.op == "RV_VLDUI":
        return "VLDUS"
    if event.op in {"RV_VLD", "RV_VLDI"}:
        return "VLDS"
    if event.op in {"RV_VST", "RV_VSTI"}:
        return "VSTS"
    if event.op in {"RV_PLD", "RV_PLDI"}:
        return "PLDS"
    if event.op in {"RV_PST", "RV_PSTI"}:
        return "PSTS"
    if event.op.startswith("RV_VCVT_"):
        conversion = re.search(r"type:\s*([A-Z0-9]+)", event.detail)
        if conversion is None:
            raise ValueError("Missing conversion type")
        mapping = {"F162F32": "VCVT_F16_TO_F32", "F322F16": "VCVT_F32_TO_F16",
                   "S322F32": "VCVT_S32_TO_F32", "F322S32": "VCVT_F32_TO_S32"}
        if conversion[1] not in mapping:
            raise ValueError(f"Unsupported conversion {conversion[1]}")
        return mapping[conversion[1]]
    return ALIASES.get(event.op, event.op.removeprefix("RV_"))


def immediate_delta_bytes(event):
    immediate = re.search(r"#offset=(-?\d+)", event.detail)
    offset = int(immediate[1]) if immediate else 0
    if 128 <= offset <= 255:
        offset -= 256
    unit = 4 if "dist:BRC_B32" in event.detail else 2 if "dist:BRC_B16" in event.detail else 32
    return offset * unit


def scalar_byte_offset(event):
    scalar = event.scalar("Sm")
    if scalar is None:
        raise ValueError("Missing scalar-indexed vector memory offset")
    high = event.scalar("Sm2")
    value = scalar[1] | ((high[1] if high else 0) << 16)
    return value - (1 << 32) if value >= (1 << 31) else value


def update_delta_bytes(event):
    if event.op in {"RV_VLDS", "RV_VSTS", "RV_PLDS", "RV_PSTS"}:
        return scalar_byte_offset(event)
    return immediate_delta_bytes(event)


def memory_address(event):
    base=event.scalar("Sn")
    if base is None:
        raise ValueError("Missing UB base address")
    high = event.scalar("Sn2")
    base_address = base[1] | ((high[1] if high else 0) << 16)
    if event.op in {"RV_VLDAS", "RV_VLDUI"}:
        # Unaligned loads report the access pointer; #offset is the transfer
        # size/update amount, not an additional effective-address offset.
        return base_address
    if event.op in {"RV_VSTUI", "RV_VSTAI"}:
        offset = int(re.search(r"#offset=(-?\d+)", event.detail)[1])
        return base_address - offset
    if event.op in {"RV_VLDS", "RV_VSTS", "RV_PLDS", "RV_PSTS"}:
        offset = scalar_byte_offset(event)
        return base_address - offset if re.search(r"#p=1\b", event.detail) else base_address + offset
    address=re.search(r"Am\[\d+\]=(0x[0-9a-fA-F]+|\d+)",event.detail)
    if address:
        return base_address+int(address[1],0)
    delta = immediate_delta_bytes(event)
    # CAModel prints the updated scalar pointer for immediate POST_UPDATE.
    # Recover the access address before the update; do not add delta twice.
    post=re.search(r"#p=(\d+)",event.detail)
    return base_address-delta if post and post[1]=="1" else base_address+delta


class Replay:
    def __init__(self, name, index, source_path):
        self.name = name
        self.index = index
        self.source_path = source_path
        self.values = {}
        self.environment = {}
        self.serial = 0
        self.node_serial = 0
        self.missing_semantics = set()
        # The harness uses at most 192 KiB; compiler VF stack lives above it.
        self.objects = {"ub": CanonicalStorageObject("ub", StorageKind.UB, (256 * 1024,))}

    def node_id(self, event):
        ident = f"{self.name}.vf{self.index}.pc{event.pc:x}.{self.node_serial}"
        self.node_serial += 1
        return ident

    def define(self, logical, storage, dtype, producer=None):
        definition = f"{logical}.def{self.serial}"
        self.serial += 1
        value = CanonicalValue(
            definition, logical, storage, dtype,
            (128 if dtype == "fp16" else 64,) if storage == StorageKind.REGISTER else (),
            producer, "ub" if storage == StorageKind.UB else None,
        )
        self.values[definition] = value
        self.environment[logical] = definition
        return definition

    def read(self, logical, storage, dtype):
        if logical not in self.environment:
            if storage in {StorageKind.REGISTER, StorageKind.PREDICATE_REGISTER}:
                raise ValueError(f"Undefined compiled register input {logical}; refusing a fabricated live-in")
            self.define(logical, storage, dtype)
        return self.environment[logical]

    def vector(self, index):
        return self.read(f"v{index}", StorageKind.REGISTER, "fp32")

    def predicate(self, index):
        return self.read(f"p{index}", StorageKind.PREDICATE_REGISTER, "bool")

    def scalar_input(self, event, label="Sm", paired="Sm2", dtype="uint32"):
        item = event.scalar(label)
        if item is None:
            raise ValueError(f"Missing scalar operand {label}: {event.detail}")
        high = event.scalar(paired)
        bits = item[1] | ((high[1] if high else 0) << 16)
        logical = f"constant.{dtype}.{bits:x}"
        return self.read(logical, StorageKind.SCALAR, dtype)

    def instruction(self, event, offsets=None):
        opcode = operation(event)
        if opcode is None:
            return None
        ident = self.node_id(event)
        location = SourceLocation(source="CAModel decoded instruction", path=str(self.source_path))
        if opcode == "MEMBAR":
            return CanonicalMembar(ident, event.detail.split("type:", 1)[1].strip(), source_location=location)
        attrs = {"rv_opcode": event.op, "rv_pc": hex(event.pc), "rv_detail": event.detail,
                 "replay_dynamic_id": event.ident}
        raw_dtype=re.search(r'Dtype:\s*(\w+)',event.detail)
        if raw_dtype:
            attrs['rv_dtype']=raw_dtype[1]
        inputs, outputs = [], []
        form = "fp32"
        cls = InstructionClass.COMPUTE
        dtype = "fp32"

        def source(value, role=OperandRole.SOURCE, memory=None):
            inputs.append(CanonicalOperand(value, role, memory_access=memory))

        def destination(logical, storage=StorageKind.REGISTER, value_dtype="fp32", memory=None):
            value = self.define(logical, storage, value_dtype, ident)
            outputs.append(CanonicalOperand(value, OperandRole.DESTINATION if memory is None else OperandRole.MEMORY,
                                            memory_access=memory))

        if opcode in {"VLDAS", "VLDUS"}:
            cls, form = InstructionClass.LOAD, "int32"
            state = event.register("Uld")
            if state is None:
                raise ValueError("Missing unaligned-load state")
            attrs.update(align_state_id=f"uld{state}",
                         align_state_operation="load_init" if opcode == "VLDAS" else "load_use")
            span = 64
            if opcode == "VLDUS":
                width = re.search(r"#offset=(\d+)", event.detail)
                if width is None or int(width[1]) <= 0 or int(width[1]) % 4:
                    raise ValueError("Invalid unaligned-load byte count")
                span = int(width[1]) // 4
            access = MemoryAccess("ub", AffineExpression(constant=memory_address(event) // 4), AccessKind.READ, span)
            source(self.read("memory.fp32", StorageKind.UB, "fp32"), OperandRole.MEMORY, access)
            if opcode == "VLDUS":
                destination(f"v{event.register('Vd')}")
        elif opcode in {"VSTUS", "VSTAS"}:
            cls, form = InstructionClass.STORE, "fp32"
            state = event.register("Ust")
            attrs.update(align_state_id=f"ust{state}",
                         align_state_operation="append" if opcode == "VSTUS" else "consume")
            if opcode == "VSTUS":
                source(self.vector(event.register("Vd")))
                byte_count = int(re.search(r"#offset=(\d+)", event.detail)[1])
                if byte_count <= 0 or byte_count % 4:
                    raise ValueError("VSTUS must append a positive byte count")
                span = byte_count // 4
            else:
                span = None  # Runtime state seals the preceding VSTUS ranges.
            if memory_address(event) % 4:
                raise ValueError("FP32 align-store byte address is not element aligned")
            access = MemoryAccess("ub", AffineExpression(memory_address(event) // 4), AccessKind.WRITE, span)
            destination("memory.fp32", StorageKind.UB, "fp32", access)
        elif opcode in {"PLDS", "PSTS"}:
            cls = InstructionClass.LOAD if opcode == "PLDS" else InstructionClass.STORE
            form, dtype = "b8", "uint8"
            addr = memory_address(event)
            access = MemoryAccess("ub", AffineExpression(addr),
                                  AccessKind.READ if opcode == "PLDS" else AccessKind.WRITE,
                                  32, address_unit_bytes=1)
            logical = "memory.uint8"
            if opcode == "PLDS":
                source(self.read(logical, StorageKind.UB, dtype), OperandRole.MEMORY, access)
                destination(f"p{event.register('Pd')}", StorageKind.PREDICATE_REGISTER, "bool")
            else:
                source(self.predicate(event.register("Pd")), OperandRole.PREDICATE)
                destination(logical, StorageKind.UB, dtype, access)
        elif opcode in {"VLDS", "VSTS"}:
            cls = InstructionClass.LOAD if opcode == "VLDS" else InstructionClass.STORE
            mode_match = re.search(r"dist:([A-Z0-9_]+)", event.detail)
            if mode_match is None:
                raise ValueError("Missing memory mode")
            mode = mode_match[1].replace("NORAML", "NORM")
            if mode in {"UNPK_B16", "PK_B32"}:
                dtype, form = "fp16", "fp16"
                attrs["catalog_mode"] = mode
                span, unit = 64, 2
            else:
                form = "fp32"
                span, unit = (1 if mode == "BRC_B32" else 64), 4
            addr = memory_address(event)
            expression = (offsets or {}).get(event.pc, AffineExpression(constant=addr // unit))
            # span and fitted offsets above are in pointer elements. Catalog
            # normalization may subsequently convert them to explicit bytes.
            access = MemoryAccess("ub", expression, AccessKind.READ if opcode == "VLDS" else AccessKind.WRITE,
                                  span, address_unit_bytes=unit if "catalog_mode" in attrs else None)
            logical = f"memory.{dtype}"
            if opcode == "VLDS":
                source(self.read(logical, StorageKind.UB, dtype), OperandRole.MEMORY, access)
                destination(f"v{event.register('Vd')}", value_dtype=dtype)
            else:
                source(self.vector(event.register("Vd")))
                if event.register("Pg") is not None:
                    source(self.predicate(event.register("Pg")), OperandRole.PREDICATE)
                destination(logical, StorageKind.UB, dtype, access)
        elif opcode == "PSET":
            width = re.search(r"Dtype:\s*B(\d+)", event.detail)[1]
            opcode, form = f"PSET_B{width}", f"b{width}"
            attrs["pattern"] = "PAT_ALL"
            destination(f"p{event.register('Pd')}", StorageKind.PREDICATE_REGISTER, "bool")
        elif opcode == "PLT":
            width = int(re.search(r"Dtype:\s*B(\d+)", event.detail)[1])
            form = f"b{width}"
            scalar = event.scalar("Sn")
            logical = f"counter.s{scalar[0]}"
            source(self.read(logical, StorageKind.SCALAR, "uint32"), OperandRole.SCALAR)
            destination(f"p{event.register('Pd')}", StorageKind.PREDICATE_REGISTER, "bool")
            destination(logical, StorageKind.SCALAR, "uint32")
            attrs["scalar_counter_step"] = 2048 // width
            attrs['first_observed_count_before_update']=scalar[1]+2048//width
        elif opcode == "VBR":
            form = "b32"
            source(self.scalar_input(event, "Sn", "Sn2"), OperandRole.SCALAR)
            destination(f"v{event.register('Vd')}")
        elif opcode == "VDUP":
            form = "b32"
            source(self.vector(event.register("Vn")))
            source(self.predicate(event.register("Pg")), OperandRole.PREDICATE)
            destination(f"v{event.register('Vd')}")
        elif opcode == "VDUPS":
            form = "b32"
            source(self.scalar_input(event, "Sn", "Sn2"))
            source(self.predicate(event.register("Pg")), OperandRole.PREDICATE)
            destination(f"v{event.register('Vd')}")
        else:
            spec = DEFAULT_INSTRUCTION_CATALOG.lookup(opcode)
            if spec is None:
                self.missing_semantics.add(opcode)
            if opcode.startswith("VCVT_"):
                form = spec.fixed_form
                dtype = "fp16" if opcode.endswith("TO_F16") else "int32" if opcode.endswith("TO_S32") else "fp32"
            elif opcode in {"PAND", "POR", "PNOT", "PXOR"}:
                form = "b8"
            elif opcode in {"VSHLS", "VAND", "VXOR"}:
                form = "b32"
            if opcode in {"VMULA", "VMADD", "VAXPY"}:
                source(self.vector(event.register("Vd")))
            if opcode in {"PAND", "POR", "PNOT", "PXOR"}:
                for label in ("Pn", "Pm"):
                    if event.register(label) is not None:
                        source(self.predicate(event.register(label)), OperandRole.PREDICATE)
            else:
                for label in ("Vn", "Vm"):
                    if event.register(label) is not None:
                        source(self.vector(event.register(label)))
                if event.scalar("Sm") is not None:
                    source(self.scalar_input(event, dtype="fp32"), OperandRole.SCALAR)
                if opcode == "VSHLS":
                    shift = re.search(r"#sa=(\d+)", event.detail)[1]
                    source(self.read(f"constant.int32.{shift}", StorageKind.SCALAR, "int32"), OperandRole.SCALAR)
            if event.register("Pg") is not None:
                source(self.predicate(event.register("Pg")), OperandRole.PREDICATE)
            if event.register("Pd") is not None:
                destination(f"p{event.register('Pd')}", StorageKind.PREDICATE_REGISTER, "bool")
            else:
                if event.register('Vd') is None:
                    raise ValueError(f'No vector/predicate destination contract for {event.op}')
                destination(f"v{event.register('Vd')}", value_dtype=dtype)
        return CanonicalInstruction(ident, opcode, cls, form, tuple(inputs), tuple(outputs),
                                    attributes=attrs, source_location=location)

    def region(self, static, lo, hi):
        nodes = []
        pcs = sorted(pc for pc in static if lo <= pc < hi)
        skip_until = lo
        for pc in pcs:
            if pc < skip_until:
                continue
            event = static[pc]
            if event.op != "RV_VLOOPv2":
                node = self.instruction(event)
                if node is not None:
                    nodes.append(node)
                continue
            count = event.scalar("Sn")[1]
            words = int(re.search(r"instr_num\s*:\s*(\d+)", event.detail)[1])
            end = pc + 4 + words * 4
            if end > hi or count <= 0:
                raise ValueError("Invalid compiled VLOOP extent/count")
            before = dict(self.environment)
            ident = self.node_id(event)
            variable = f"iter_{pc:x}"
            body = self.region(static, pc + 4, end)
            updated = {k: v for k, v in self.environment.items() if before.get(k) != v}
            first_reads = {}
            defined=set()
            def reads(children):
                for child in children:
                    if isinstance(child, CanonicalInstruction):
                        for operand in child.inputs:
                            value=self.values[operand.value_id]
                            if value.logical_id not in defined:
                                first_reads.setdefault(value.logical_id,operand.value_id)
                        defined.update(self.values[o.value_id].logical_id for o in child.outputs)
                    elif isinstance(child, CanonicalLoop):
                        for carried in child.carried_values:
                            if carried.entry_value_id is not None and carried.logical_id not in defined:
                                first_reads.setdefault(carried.logical_id,carried.entry_value_id)
                        defined.update(c.logical_id for c in child.carried_values)
            reads(body)
            carried = []
            for logical, back in updated.items():
                value = self.values[back]
                entry = first_reads.get(logical)
                exit_id = self.define(logical, value.storage, value.dtype, ident)
                carried.append(LoopCarriedValue(logical, entry, back, exit_id))
            nodes.append(CanonicalLoop(ident, InductionVariable(variable), count, body=tuple(body), carried_values=tuple(carried)))
            skip_until = end
        return nodes

    def build(self, address, words, events):
        static = {}
        for event in events:
            static.setdefault(event.pc, event)
        nodes = self.region(static, address, address + words * 4)
        vf = CanonicalVfInfo(
            schema_version=2, context=tuple(nodes), values=dict(self.values),
            storage_objects=self.objects, source={"adapter": "compiled_cce_replay", "case": self.name,
                                                 "vf_index": self.index, "scalar_timing_modeled": False},
        )
        return vf


def expanded_count(nodes):
    counts = Counter()
    for node in nodes:
        if isinstance(node, CanonicalLoop):
            for op, count in expanded_count(node.body).items():
                counts[op] += int(node.count) * count
        elif isinstance(node, CanonicalMembar):
            counts["MEMBAR"] += 1
        else:
            counts[node.opcode] += 1
    return counts


def expanded_instructions(nodes, indices=()):
    for node in nodes:
        if isinstance(node,CanonicalLoop):
            for index in range(int(node.count)):
                yield from expanded_instructions(node.body,indices+((node.induction.variable_id,index),))
        else:
            yield node,indices


def affine_fit(samples, variables):
    # Solve the small exact system; count-one dimensions have coefficient zero.
    matrix=[[Fraction(1),*[Fraction(dict(indices).get(v,0)) for v in variables],Fraction(value)]
            for indices,value in samples]
    pivot_rows={};row=0
    for col in range(len(variables)+1):
        pivot=next((i for i in range(row,len(matrix)) if matrix[i][col]),None)
        if pivot is None:
            continue
        matrix[row],matrix[pivot]=matrix[pivot],matrix[row]
        divisor=matrix[row][col]
        matrix[row]=[x/divisor for x in matrix[row]]
        for i in range(len(matrix)):
            if i!=row and matrix[i][col]:
                factor=matrix[i][col]
                matrix[i]=[a-factor*b for a,b in zip(matrix[i],matrix[row])]
        pivot_rows[col]=row;row+=1
    result=[matrix[pivot_rows[c]][-1] if c in pivot_rows else Fraction(0) for c in range(len(variables)+1)]
    if any(value.denominator!=1 for value in result):
        raise ValueError("Non-integral affine UB address")
    result=list(map(int,result))
    if any(result[0]+sum(result[i+1]*dict(indices).get(v,0) for i,v in enumerate(variables))!=value for indices,value in samples):
        raise ValueError("Dynamic UB address is not affine in recovered loop indices")
    from api.common.schema import AffineTerm
    return AffineExpression(result[0],tuple(AffineTerm(v,result[i+1]) for i,v in enumerate(variables) if result[i+1]))


def add_memory_offsets(canonical, events):
    observed=[e for e in events if operation(e) is not None]
    expanded=list(expanded_instructions(canonical.context))
    if len(observed)!=len(expanded):
        raise ValueError("Dynamic instruction count differs from the checked CAModel stream")
    samples={};post_states={};state_initial={};state_current={};metadata={}
    for event in observed:
        if operation(event) in {"VLDS","VSTS","PLDS","PSTS"} and re.search(r"#p=1\b",event.detail):
            post_states[event.scalar("Sn")[0]]=True
    for (node,indices),event in zip(expanded,observed):
        pc=node.attributes.get("rv_pc") if isinstance(node,CanonicalInstruction) else None
        if isinstance(node,CanonicalMembar):
            if operation(event)!="MEMBAR":
                raise ValueError("Membar position differs from compiled stream")
            continue
        if pc!=hex(event.pc) or node.attributes["rv_opcode"]!=event.op:
            raise ValueError("Dynamic PC/opcode order differs from compiled stream")
        if node.opcode not in {"VLDS","VSTS","PLDS","PSTS","VSTUS","VSTAS","VLDAS","VLDUS"}:
            continue
        unit=1 if node.opcode in {"PLDS","PSTS"} else 2 if node.form=="fp16" else 4
        addr=memory_address(event)
        if addr%unit:
            raise ValueError("Unaligned decoded UB address")
        scalar_index,pointer=event.scalar('Sn')
        high=event.scalar('Sn2')
        pointer |= (high[1] if high else 0) << 16
        is_post=bool(re.search(r'#p=1\b',event.detail))
        delta=update_delta_bytes(event) if is_post else 0
        if scalar_index in post_states:
            before=pointer-delta if is_post else pointer
            state_initial.setdefault(scalar_index,before)
            state_current.setdefault(scalar_index,before)
            if state_current[scalar_index]!=before:
                raise ValueError('Compiled pointer reset requires an explicit state-reset event')
            # Offset describes the pointer initializer plus the access offset.
            # The dynamic post-update engine accounts for accumulated deltas.
            value=(state_initial[scalar_index]+addr-before)//unit
            state_current[scalar_index]=before+delta
            # All PCs sharing this physical pointer use one stable state ID.
            metadata[node.instruction_id]=(f'pointer.s{scalar_index}',is_post,delta)
        else:
            value=addr//unit
        samples.setdefault(node.instruction_id,[]).append((indices,value))
    expressions={}
    for ident,rows in samples.items():
        variables=[v for v,_ in rows[0][0]]
        expressions[ident]=affine_fit(rows,variables)
    def update(nodes):
        result=[]
        for node in nodes:
            if isinstance(node,CanonicalLoop):
                node=replace(node,body=tuple(update(node.body)))
            elif isinstance(node,CanonicalInstruction) and node.instruction_id in expressions:
                def operand(item):
                    if item.memory_access:
                        access=replace(item.memory_access,offset=expressions[node.instruction_id])
                        if node.instruction_id in metadata:
                            state,post,delta=metadata[node.instruction_id]
                            access=replace(access,address_state_id=state,update_mode='post_update' if post else 'none',
                                           post_update_delta_bytes=AffineExpression(delta) if post else None)
                        return replace(item,memory_access=access)
                    return item
                node=replace(node,inputs=tuple(map(operand,node.inputs)),outputs=tuple(map(operand,node.outputs)))
            result.append(node)
        return result
    return replace(canonical,context=tuple(update(canonical.context)))
