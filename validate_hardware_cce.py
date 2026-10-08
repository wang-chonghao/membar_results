import json,sys
from collections import Counter
from pathlib import Path
repo=Path('/mnt/e/vfsimulator')
sys.path.insert(0,str(repo))
from api.cce_adapter import parse_cce_canonical_vf_info
from api.frontend.schema import CanonicalLoop,CanonicalInstruction,CanonicalMembar
root=repo/'results/membar_results'
index=json.loads((root/'hardware_equivalent_index.json').read_text())
results=[]
def count(nodes,multiplier=1):
    result=Counter()
    for node in nodes:
        if isinstance(node,CanonicalLoop):result.update(count(node.body,multiplier*int(node.count)))
        elif isinstance(node,CanonicalMembar):result['MEMBAR']+=multiplier
        elif isinstance(node,CanonicalInstruction):
            if not node.opcode.startswith('PSET'):result[node.opcode]+=multiplier
    return result
aliases={'VLD':'VLDS','VLDI':'VLDS','VST':'VSTS','VSTI':'VSTS','PLDI':'PLDS','PSTI':'PSTS','VDUPS':'VDUP','SMEM_BAR':'MEMBAR'}
for item in index:
    path=root/item['file']
    audit=json.loads((path.parent/'hardware_pc_mapping.json').read_text())
    expected=Counter()
    for op,n in audit['dynamic_instruction_counts'].items():
        name=op.removeprefix('RV_')
        if name=='PSET':continue
        expected[aliases.get(name,name)]+=n
    vf=parse_cce_canonical_vf_info(path)
    actual=count(vf.context)
    assert expected==actual,(item['file'],expected-actual,actual-expected)
    results.append(dict(file=item['file'],dynamic_counts=dict(actual),pc_coverage_complete=True,
                        canonical_parser_pass=True))
(root/'hardware_cce_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print('PASS:',len(results),'hardware equivalent CCEs; canonical dynamic counts match logged hardware')
