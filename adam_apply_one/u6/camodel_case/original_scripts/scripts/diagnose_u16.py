import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path('/mnt/e/vfsimulator')
sys.path.insert(0, str(ROOT))
import main

CASE = ROOT/'results/ub_address_dependency_experiment/adam_apply_one_fused_i48_unroll/u16'
original = main.run_simulation

def diagnose(**kwargs):
    kwargs['params'] = dict(kwargs['params'], max_cycles=1000)
    try:
        return original(**kwargs)
    except RuntimeError as error:
        ooo, idu = kwargs['ooo'], kwargs['idu']
        snapshot = {
            'error': str(error),
            'diagnostic_max_cycles': 1000,
            'physical_register_capacity': ooo.preg_num,
            'physical_registers_free': ooo.get_free_preg(),
            'dispatched': dict(Counter(event['op'] for event in idu.dispatch_log)),
            'idu_window': list(idu.window),
            'last_dispatch': idu.dispatch_log[-4:],
            'shq': len(ooo.SHQ), 'lsq': len(ooo.LSQ), 'rob': len(ooo.ROB),
        }
        (CASE/'vfsim_blocked_diagnostic.json').write_text(json.dumps(snapshot, indent=2)+'\n')
        print(json.dumps({k:v for k,v in snapshot.items() if k not in ('idu_window','last_dispatch')}))
        print('IDU head:', snapshot['idu_window'][0])
        return None

main.run_simulation = diagnose
sys.argv = ['main.py','--cce',str(CASE/'source/adam_i48_u16.cce'),
            '--cce-kernel','adam_fused_vf','--soc=dav-3510',
            '--out_dir',str(CASE/'vfsim_diagnostic')]
try:
    main.main()
except TypeError:
    if not (CASE/'vfsim_blocked_diagnostic.json').exists():
        raise
